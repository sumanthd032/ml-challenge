"""Sharded, resumable feature store for one split (FEAT_DIR/<split>_store/).

The dev machine has a failing RAM stick that hard-resets it under heavy memory load (D-012), so the
feature stage is cut into small checkpoints. Every file is written under a temp name and renamed,
so a reset never leaves a half-written file; re-running the same command skips what is on disk.

  pairs.parquet     candidate pairs: i1, i2 (row positions), blocking columns, emb_cos.
                    train: pairs of ghost S1s are already removed (D-007).
  base_KKKK.npy     pairwise features (FEAT_NAMES) of rows [K*SHARD, (K+1)*SHARD), float32
  group.npy         competition features (GROUP_NAMES) of all rows, float32, one country at a time
  meta.json         inputs the store was built from; a mismatch means the store is stale
"""
import json
import os
import time

import numpy as np
import pandas as pd

import config
from candidates import merge_candidates
from features import FEAT_NAMES, FIELDS, GROUP_INPUTS, add_group_features, build_idf, make_pool, pair_features

SHARD = 4_000_000


def store_dir(split: str):
    d = config.FEAT_DIR / f"{split}_store"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_npy(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _meta(split: str) -> dict:
    m = {"feat_names": FEAT_NAMES, "shard": SHARD}
    for p in ("cands_A.parquet", "cands_B.parquet", "emb_s1.npy"):
        f = config.ART_DIR / f"{split}_{p}"
        m[p] = f.stat().st_size if f.exists() else None
    if split == "train":
        m["ghost_frac"] = config.GHOST_FRAC
    return m


def _check_meta(d, split):
    path, cur = d / "meta.json", _meta(split)
    if path.exists():
        old = json.load(open(path))
        if old != cur:
            raise RuntimeError(f"{d} was built from different inputs; delete it to rebuild.\nold={old}\nnew={cur}")
    else:
        json.dump(cur, open(path, "w"), indent=1)


def build(split: str, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Build whatever is missing from the store and return the pairs table."""
    d = store_dir(split)
    _check_meta(d, split)
    t = time.time()
    pairs_p = d / "pairs.parquet"
    if not pairs_p.exists():
        pairs = merge_candidates(split, s1, s23)
        if split == "train":
            keep = s1.entity_id.map(config.s1_role).values != "ghost"
            pairs = pairs[keep[pairs.i1.values]].reset_index(drop=True)
        tmp = d / "pairs.tmp.parquet"
        pairs.to_parquet(tmp, index=False)
        os.replace(tmp, pairs_p)
        print(f"[store] pairs saved: {len(pairs):,} ({time.time() - t:.0f}s)", flush=True)
    else:
        pairs = pd.read_parquet(pairs_p)
        print(f"[store] pairs loaded: {len(pairs):,}", flush=True)

    n_shards = -(-len(pairs) // SHARD)
    todo = [k for k in range(n_shards) if not (d / f"base_{k:04d}.npy").exists()]
    if todo:
        print(f"[store] base features: {len(todo)}/{n_shards} shards to compute", flush=True)
        colsA = [s1[c].values for c in FIELDS]
        colsB = [s23[c].values for c in FIELDS]
        ia, ib = pairs.i1.values, pairs.i2.values
        with make_pool(build_idf(s1, s23)) as pool:
            for k in todo:
                t0 = time.time()
                sl = slice(k * SHARD, (k + 1) * SHARD)
                _save_npy(d / f"base_{k:04d}.npy", pair_features(pool, ia[sl], ib[sl], colsA, colsB))
                print(f"    shard {k + 1}/{n_shards} ({time.time() - t0:.0f}s, total {time.time() - t:.0f}s)", flush=True)

    if not (d / "group.npy").exists():
        _build_group(d, pairs, s1, n_shards)
        print(f"[store] group features done ({time.time() - t:.0f}s)", flush=True)
    return pairs


def _base_columns(d, n_shards, names) -> np.ndarray:
    idx = [FEAT_NAMES.index(c) for c in names]
    return np.concatenate([np.load(d / f"base_{k:04d}.npy", mmap_mode="r")[:, idx] for k in range(n_shards)])


def _build_group(d, pairs, s1, n_shards):
    """Competition features. Groups (same S1 / same candidate) never cross countries, so each country
    is processed on its own to keep the pandas groupby peak small."""
    inp = _base_columns(d, n_shards, GROUP_INPUTS)
    country = s1.country.values[pairs.i1.values]
    G, names, tmp = None, None, d / "group.tmp.npy"
    for c in np.unique(country):
        rows = np.flatnonzero(country == c)
        df = pd.DataFrame({"i1": pairs.i1.values[rows], "i2": pairs.i2.values[rows],
                           "blk_score": pairs.blk_score.values[rows]})
        if "emb_cos" in pairs:
            df["emb_cos"] = pairs.emb_cos.values[rows]
        for j, col in enumerate(GROUP_INPUTS):
            df[col] = inp[rows, j]
        before = set(df.columns)
        add_group_features(df)
        new = [col for col in df.columns if col not in before]
        if G is None:     # disk-backed, so the full matrix never has to sit in RAM
            names = new
            G = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(pairs), len(names)))
        G[rows] = df[names].to_numpy(np.float32)
        print(f"    group features {c}: {len(rows):,} pairs", flush=True)
        del df
    G.flush()
    del G
    json.dump(names, open(d / "group_names.json", "w"))
    os.replace(tmp, d / "group.npy")


def s3_flags(s23: pd.DataFrame) -> np.ndarray:
    return s23.entity_id.str.startswith("S3").values.astype(np.int8)


def load_rows(split: str, pairs: pd.DataFrame, s3: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Full feature table (pairs columns + base + is_s3 + group) for the sorted row positions `rows`.
    s3: s3_flags(s23)."""
    d = store_dir(split)
    out = pairs.take(rows).reset_index(drop=True)
    X = np.empty((len(rows), len(FEAT_NAMES)), dtype=np.float32)
    shard_of = rows // SHARD
    for k in np.unique(shard_of):
        m = shard_of == k
        X[m] = np.load(d / f"base_{k:04d}.npy", mmap_mode="r")[rows[m] - k * SHARD]
    for j, n in enumerate(FEAT_NAMES):
        out[n] = X[:, j]
    del X
    out["is_s3"] = s3[out.i2.values]
    Gr = np.load(d / "group.npy", mmap_mode="r")[rows]
    for j, n in enumerate(json.load(open(d / "group_names.json"))):
        out[n] = Gr[:, j]
    return out
