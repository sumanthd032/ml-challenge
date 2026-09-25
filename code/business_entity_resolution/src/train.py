"""Step 4 - build training pairs on the train split, fit LightGBM, validate with macro F0.5.

Validation protocol (see docs/DECISIONS.md, D-007):
  * S1 entities are split into train / valid by entity id (no S1 appears in both).
  * A "ghost" fraction of S1 entities is removed from the S1 pool entirely: their S2/S3 records
    stay as distractors whose true S1 does not exist, mimicking the higher S2/S3-per-S1 ratio of
    the test set (5.76 vs 4.68).
  * Every metric is computed against the FULL ground truth of the validation S1s, so blocking
    misses count as recall loss.

Usage: python train.py --cands train_cands_A.parquet [--max-train-s1 400000]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from data import load_split, load_gt_pairs
from decide import expected_f_select
from features import compute_features, build_idf, FEAT_NAMES
from metrics import macro_f05, blocking_recall

DROP = {"s1_id", "cand_id", "label"}


def feature_columns(df):
    return [c for c in df.columns if c not in DROP]


def make_feature_table(cands_file: str, ghost_frac: float, max_s1: int, tag: str):
    s1, s23 = load_split("train")
    cands = pd.read_parquet(config.ART_DIR / cands_file)
    ids = np.array(sorted(cands.s1_id.unique()))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(ids)
    n_ghost = int(len(ids) * ghost_frac)
    ghost = set(ids[:n_ghost])
    keep = ids[n_ghost:]
    if max_s1 and len(keep) > max_s1:
        keep = keep[:max_s1]
    cands = cands[cands.s1_id.isin(set(keep))].reset_index(drop=True)
    print(f"S1 kept {len(keep):,} (ghost {len(ghost):,}); pairs {len(cands):,}", flush=True)
    t = time.time()
    idf = build_idf(s1, s23)
    feats = compute_features(cands, s1, s23, idf)
    print(f"features in {time.time() - t:.0f}s", flush=True)
    truth = load_gt_pairs()
    truth["label"] = 1
    feats = feats.merge(truth, on=["s1_id", "cand_id"], how="left")
    feats["label"] = feats.label.fillna(0).astype(np.int8)
    feats.to_parquet(config.ART_DIR / f"feats_{tag}.parquet", index=False)
    pd.Series(sorted(keep)).to_csv(config.ART_DIR / f"s1_keep_{tag}.csv", index=False)
    return feats, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="train_cands_A.parquet")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--ghost-frac", type=float, default=0.2)
    ap.add_argument("--max-s1", type=int, default=0)
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    if args.reuse:
        feats = pd.read_parquet(config.ART_DIR / f"feats_{args.tag}.parquet")
        keep = pd.read_csv(config.ART_DIR / f"s1_keep_{args.tag}.csv").iloc[:, 0].values
    else:
        feats, keep = make_feature_table(args.cands, args.ghost_frac, args.max_s1, args.tag)
    keep = np.array(sorted(keep))
    rng = np.random.default_rng(config.SEED + 1)
    va_ids = set(rng.choice(keep, int(len(keep) * args.valid_frac), replace=False))
    is_va = feats.s1_id.isin(va_ids).values
    cols = feature_columns(feats)
    print(f"{len(cols)} features; train pairs {(~is_va).sum():,} valid pairs {is_va.sum():,}; "
          f"pos rate {feats.label.mean():.3f}", flush=True)

    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)
    dtr = lgb.Dataset(feats.loc[~is_va, cols], feats.label[~is_va])
    dva = lgb.Dataset(feats.loc[is_va, cols], feats.label[is_va], reference=dtr)
    t = time.time()
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"trained in {time.time() - t:.0f}s, best iter {model.best_iteration}", flush=True)
    model.save_model(str(config.ART_DIR / f"lgb_{args.tag}.txt"))

    va = feats.loc[is_va, ["s1_id", "cand_id", "label"]].copy()
    va["p"] = model.predict(feats.loc[is_va, cols], num_iteration=model.best_iteration)
    truth = load_gt_pairs()
    truth_va = truth[truth.s1_id.isin(va_ids)]
    print("blocking recall (valid):", blocking_recall(va, truth_va, va_ids))
    results = {}
    for mode, kw in [("thr", dict(thr=0.5)), ("thr", dict(thr=0.6)), ("thr", dict(thr=0.7)), ("thr", dict(thr=0.8)),
                     ("expf", dict(alpha=1.0)), ("expf", dict(alpha=1.2)), ("expf", dict(alpha=0.8)),
                     ("expf", dict(alpha=1.0, min_p=0.3))]:
        sel = expected_f_select(va, mode=mode, **kw)
        r = macro_f05(sel, truth_va, va_ids)
        key = f"{mode} {kw}"
        results[key] = r
        print(f"{key:40s} F0.5={r['f05']:.4f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f} "
              f"sing={r['f05_singletons']:.4f}", flush=True)
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:\n", (imp / imp.sum()).head(25).round(4).to_string())
    json.dump({"results": results, "best_iter": model.best_iteration, "features": cols},
              open(config.ART_DIR / f"train_report_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
