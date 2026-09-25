"""Merge blocking passes (A: sparse tokens, B: bi-encoder kNN) into the final candidate set and attach
the bi-encoder cosine similarity to every pair (also those found only by pass A).

Memory: ~175M pairs on the full train split. String ids cost ~130 bytes/pair in pandas, so pairs are
held as int32 row positions into s1 / s23 (columns `i1`, `i2`). Strings return only when writing outputs.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config


def ids_to_rows(ids, index: pd.Index) -> np.ndarray:
    r = index.get_indexer(ids)
    assert (r >= 0).all(), "id not found in split"
    return r.astype(np.int32)


def load_pass(path, s1_index: pd.Index, s23_index: pd.Index, batch=8_000_000) -> pd.DataFrame:
    """Read a candidate parquet (s1_id, cand_id, ...) converting ids to row positions batch by batch."""
    pf = pq.ParquetFile(path)
    parts = []
    for b in pf.iter_batches(batch_size=batch):
        d = b.to_pandas()
        d.insert(0, "i1", ids_to_rows(d.pop("s1_id").values, s1_index))
        d.insert(1, "i2", ids_to_rows(d.pop("cand_id").values, s23_index))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def merge_candidates(split: str, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    s1_index, s23_index = pd.Index(s1.entity_id), pd.Index(s23.entity_id)
    A = load_pass(config.ART_DIR / f"{split}_cands_A.parquet", s1_index, s23_index)
    pb = config.ART_DIR / f"{split}_cands_B.parquet"
    B = load_pass(pb, s1_index, s23_index) if pb.exists() else A.iloc[:0][["i1", "i2"]]
    n2 = np.int64(len(s23))
    ka = A.i1.values.astype(np.int64) * n2 + A.i2.values
    kb = B.i1.values.astype(np.int64) * n2 + B.i2.values
    keys = np.unique(np.concatenate([ka, kb]))          # sorted union
    print(f"cands A {len(A):,}  B {len(B):,}  union {len(keys):,}", flush=True)
    m = pd.DataFrame({"i1": (keys // n2).astype(np.int32), "i2": (keys % n2).astype(np.int32)})
    pa, pb_ = np.searchsorted(keys, ka), np.searchsorted(keys, kb)
    del keys, ka, kb

    def scatter(pos, vals, fill, dtype):
        out = np.full(len(m), fill, dtype=dtype)
        out[pos] = vals
        return out

    m["blk_score"] = scatter(pa, A.blk_score.values, 0, np.float32)
    m["blk_rank_fwd"] = scatter(pa, A.blk_rank_fwd.values, 99, np.int16)
    m["blk_rank_rev"] = scatter(pa, A.blk_rank_rev.values, 99, np.int16)
    m["found_A"] = scatter(pa, 1, 0, np.int8)
    if len(B):
        m["emb_rank_fwd"] = scatter(pb_, B.emb_rank_fwd.values, 99, np.int16)
        m["emb_rank_rev"] = scatter(pb_, B.emb_rank_rev.values, 99, np.int16)
        m["found_B"] = scatter(pb_, 1, 0, np.int8)
    del A, B
    e1p, e2p = config.ART_DIR / f"{split}_emb_s1.npy", config.ART_DIR / f"{split}_emb_s23.npy"
    if e1p.exists():
        m["emb_cos"] = pair_cosine(m.i1.values, m.i2.values, np.load(e1p, mmap_mode="r"), np.load(e2p, mmap_mode="r"))
    return m


def pair_cosine(i1, i2, E1, E2, chunk=4_000_000):
    """Pairs are sorted by i1, so E1 reads are sequential; E2 is fully loaded (random access)."""
    E2 = np.asarray(E2)
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        a = np.asarray(E1[i1[s:s + chunk]], dtype=np.float32); b = E2[i2[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = np.einsum("ij,ij->i", a, b)
    return out


def rows_to_pairs(i1, i2, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Back to string ids (for outputs / metrics on small tables)."""
    return pd.DataFrame({"s1_id": s1.entity_id.values[i1], "cand_id": s23.entity_id.values[i2]})
