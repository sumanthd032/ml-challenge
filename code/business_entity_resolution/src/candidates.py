"""Merge blocking passes (A: sparse tokens, B: bi-encoder kNN) into the final candidate set and attach
the bi-encoder cosine similarity to every pair (also those found only by pass A)."""
import numpy as np
import pandas as pd

import config


def merge_candidates(split: str, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    A = pd.read_parquet(config.ART_DIR / f"{split}_cands_A.parquet")
    pb = config.ART_DIR / f"{split}_cands_B.parquet"
    if pb.exists():
        B = pd.read_parquet(pb)
        m = A.merge(B, on=["s1_id", "cand_id"], how="outer")
        print(f"cands A {len(A):,}  B {len(B):,}  union {len(m):,}", flush=True)
    else:
        m = A
    m["blk_score"] = m.blk_score.fillna(0).astype(np.float32)
    for c in ("blk_rank_fwd", "blk_rank_rev", "emb_rank_fwd", "emb_rank_rev"):
        if c in m:
            m[c] = m[c].fillna(99).astype(np.int16)
    m["found_A"] = m.blk_rank_fwd.lt(99) | m.blk_rank_rev.lt(99)
    if "emb_rank_fwd" in m:
        m["found_B"] = m.emb_rank_fwd.lt(99) | m.emb_rank_rev.lt(99)
    m[["found_A"] + (["found_B"] if "found_B" in m else [])] = m[["found_A"] + (["found_B"] if "found_B" in m else [])].astype(np.int8)
    e1p, e2p = config.ART_DIR / f"{split}_emb_s1.npy", config.ART_DIR / f"{split}_emb_s23.npy"
    if e1p.exists():
        m["emb_cos"] = pair_cosine(m, s1, s23, np.load(e1p, mmap_mode="r"), np.load(e2p, mmap_mode="r"))
        m = m.drop(columns=["emb_score"], errors="ignore")
    return m.reset_index(drop=True)


def pair_cosine(pairs, s1, s23, E1, E2, chunk=2_000_000):
    i1 = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.s1_id.values].values
    i2 = pd.Series(np.arange(len(s23)), index=s23.entity_id).loc[pairs.cand_id.values].values
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), chunk):
        a = np.asarray(E1[i1[s:s + chunk]], dtype=np.float32); b = np.asarray(E2[i2[s:s + chunk]], dtype=np.float32)
        out[s:s + chunk] = (a * b).sum(1)
    return out
