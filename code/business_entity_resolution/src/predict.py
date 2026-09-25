"""Step 6 - test inference: features on test candidates, LightGBM scores, decision, output files.

Usage: python predict.py --model lgb_v1.txt [--alpha 1.0 --min-p 0.05]
Writes OUT_DIR/matching_results.tsv and OUT_DIR/candidate_pairs.tsv.
"""
import argparse
import json
import time

import lightgbm as lgb
import pandas as pd

import config
from data import load_split
from decide import expected_f_select
from features import compute_features, build_idf
from candidates import merge_candidates


def write_lists(pairs: pd.DataFrame, s1_ids, col: str, path):
    """One row per S1 entity (all of them), comma-joined unique ids, empty when none."""
    lists = pairs.drop_duplicates(["s1_id", "cand_id"]).groupby("s1_id").cand_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": list(s1_ids)})
    out[col] = out.source1_entity_id.map(lists).fillna("")
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
    ap.add_argument("--reuse-feats", action="store_true")
    args = ap.parse_args()
    t = time.time()
    s1, s23 = load_split("test")
    feats_path = config.ART_DIR / "feats_test.parquet"
    if args.reuse_feats:
        feats = pd.read_parquet(feats_path)
    else:
        cands = merge_candidates("test", s1, s23)
        feats = compute_features(cands, s1, s23, build_idf(s1, s23))
        feats.to_parquet(feats_path, index=False)
    print(f"features for {len(feats):,} pairs ({time.time() - t:.0f}s)", flush=True)
    model = lgb.Booster(model_file=str(config.ART_DIR / args.model))
    cols = model.feature_name()
    feats["p"] = model.predict(feats[cols], num_threads=config.N_JOBS)
    sel = expected_f_select(feats[["s1_id", "cand_id", "p"]], alpha=args.alpha, min_p=args.min_p)
    write_lists(feats[["s1_id", "cand_id"]], s1.entity_id, "candidate_entity_ids", config.OUT_DIR / "candidate_pairs.tsv")
    res = write_lists(sel, s1.entity_id, "matched_entity_ids", config.OUT_DIR / "matching_results.tsv")
    n = (res.matched_entity_ids != "").sum()
    stats = {"s1": len(res), "with_match": int(n), "pred_pairs": len(sel), "pairs_per_s1": len(sel) / len(res),
             "by_country": feats.merge(s1[["entity_id", "country"]], left_on="s1_id", right_on="entity_id")
             .groupby("country").p.apply(lambda x: float((x > 0.5).mean())).to_dict()}
    print(json.dumps(stats, indent=1))
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
