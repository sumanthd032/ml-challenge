"""Step 6 - test inference: features on test candidates, LightGBM scores, decision, output files.

Features come from the resumable store (store.py): re-running after a crash continues where it stopped.

Usage: python predict.py --model lgb_v1.txt [--thr 0.7 | --alpha 1.0 --min-p 0.05]
Writes OUT_DIR/matching_results.tsv and OUT_DIR/candidate_pairs.tsv.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from data import load_split
from decide import expected_f_select
import store


def write_lists(i1, i2, s1: pd.DataFrame, s23: pd.DataFrame, col: str, path):
    """One row per S1 entity (all of them), comma-joined unique ids, empty when none. i1/i2: row positions."""
    pairs = pd.DataFrame({"i1": i1, "i2": i2}).drop_duplicates()
    pairs["cand_id"] = s23.entity_id.values[pairs.i2.values]
    lists = pairs.groupby("i1").cand_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1.entity_id.values})
    out[col] = pd.Series(np.arange(len(s1))).map(lists).fillna("").values
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in zip(out.source1_entity_id, out[col]):
            f.write(f"{a}\t{b}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgb_v1.txt")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=None,
                    help="flat probability threshold instead of expected-F0.5 selection")
    args = ap.parse_args()
    t = time.time()
    s1, s23 = load_split("test")
    pairs = store.build("test", s1, s23)
    print(f"features for {len(pairs):,} pairs ({time.time() - t:.0f}s)", flush=True)
    model = lgb.Booster(model_file=str(config.ART_DIR / args.model))
    cols = model.feature_name()
    s3 = store.s3_flags(s23)
    p = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), store.SHARD):          # one shard at a time: small peak memory
        X = store.load_rows("test", pairs, s3, np.arange(s, min(s + store.SHARD, len(pairs))))
        p[s:s + len(X)] = model.predict(X[cols], num_threads=config.N_JOBS)
    del X
    scored = pd.DataFrame({"s1_id": pairs.i1.values, "cand_id": pairs.i2.values, "p": p})
    scored.to_parquet(config.FEAT_DIR / f"test_scores_{args.model.removesuffix('.txt')}.parquet", index=False)
    del pairs
    if args.thr is not None:
        sel = expected_f_select(scored, mode="thr", thr=args.thr, alpha=1.0)
    else:
        sel = expected_f_select(scored, alpha=args.alpha, min_p=args.min_p)
    write_lists(scored.s1_id.values, scored.cand_id.values, s1, s23, "candidate_entity_ids",
                config.OUT_DIR / "candidate_pairs.tsv")
    res = write_lists(sel.s1_id.values, sel.cand_id.values, s1, s23, "matched_entity_ids",
                      config.OUT_DIR / "matching_results.tsv")
    n = (res.matched_entity_ids != "").sum()
    country = s1.country.values[sel.s1_id.values]
    stats = {"s1": len(res), "with_match": int(n), "pred_pairs": len(sel), "pairs_per_s1": len(sel) / len(res),
             "pred_pairs_by_country": {k: int(v) for k, v in pd.Series(country).value_counts().items()},
             "s1_by_country": {k: int(v) for k, v in s1.country.value_counts().items()}}
    print(json.dumps(stats, indent=1))
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
