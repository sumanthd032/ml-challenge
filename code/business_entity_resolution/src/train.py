"""Step 4 - build training pairs on the train split, fit LightGBM, validate with macro F0.5.

Validation protocol (see docs/DECISIONS.md, D-007):
  * S1 entities are split into train / valid by entity id (no S1 appears in both).
  * A "ghost" fraction of S1 entities is removed from the S1 pool entirely: their S2/S3 records
    stay as distractors whose true S1 does not exist, mimicking the higher S2/S3-per-S1 ratio of
    the test set (5.76 vs 4.68).
  * Every metric is computed against the FULL ground truth of the validation S1s, so blocking
    misses count as recall loss.

Internally pairs are int32 row positions (i1 into S1, i2 into S2+S3); metrics run on those codes.

Features come from the resumable store (store.py): re-running after a crash continues where it stopped.

Usage: python train.py --tag v1 [--max-train-rows 20000000]
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
import store
from metrics import macro_f05, blocking_recall
from candidates import ids_to_rows

DROP = {"i1", "i2", "label", "is_valid"}


def feature_columns(df):
    return [c for c in df.columns if c not in DROP]


def truth_rows(s1, s23) -> pd.DataFrame:
    gt = load_gt_pairs()
    return pd.DataFrame({"s1_id": ids_to_rows(gt.s1_id.values, pd.Index(s1.entity_id)),
                         "cand_id": ids_to_rows(gt.cand_id.values, pd.Index(s23.entity_id))})


def make_feature_table(max_train_rows: int):
    t = time.time()
    s1, s23 = load_split("train")
    role = s1.entity_id.map(config.s1_role).values
    # ghost S1s are removed from the pool (inside the store): their S2/S3 remain as distractors
    pairs = store.build("train", s1, s23)
    truth = truth_rows(s1, s23)
    n2 = np.int64(len(s23))
    tk = np.sort(truth.s1_id.values.astype(np.int64) * n2 + truth.cand_id.values)
    pk = pairs.i1.values.astype(np.int64) * n2 + pairs.i2.values
    pos = np.minimum(np.searchsorted(tk, pk), len(tk) - 1)
    label = (tk[pos] == pk).astype(np.int8)
    del pk, pos
    is_valid = (role[pairs.i1.values] == "valid").astype(np.int8)
    # rows: all validation pairs + a random subsample of fit pairs (features were computed on the full pool)
    fit_idx = np.flatnonzero(is_valid == 0)
    if len(fit_idx) > max_train_rows:
        fit_idx = np.random.default_rng(config.SEED).choice(fit_idx, max_train_rows, replace=False)
    rows = np.sort(np.concatenate([fit_idx, np.flatnonzero(is_valid == 1)]))
    feats = store.load_rows("train", pairs, store.s3_flags(s23), rows)
    del pairs
    feats["label"] = label[rows]
    feats["is_valid"] = is_valid[rows]
    print(f"feature table {feats.shape} pos rate {feats.label.mean():.3f} ({time.time() - t:.0f}s)", flush=True)
    return feats, s1, s23, role, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max-train-rows", type=int, default=20_000_000)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()

    feats, s1, s23, role, truth = make_feature_table(args.max_train_rows)
    is_va = feats.is_valid.values == 1
    cols = feature_columns(feats)
    print(f"{len(cols)} features; train pairs {(~is_va).sum():,} valid pairs {is_va.sum():,}", flush=True)

    params = dict(objective="binary", learning_rate=args.lr, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)
    dtr = lgb.Dataset(feats.loc[~is_va, cols], feats.label.values[~is_va], free_raw_data=True)
    # early stopping on a 3M-row slice of validation (full validation is scored afterwards)
    es_idx = np.flatnonzero(is_va)
    es_idx = np.sort(np.random.default_rng(1).choice(es_idx, min(3_000_000, len(es_idx)), replace=False))
    dva = lgb.Dataset(feats.loc[es_idx, cols], feats.label.values[es_idx], reference=dtr)
    t = time.time()
    model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    print(f"trained in {time.time() - t:.0f}s, best iter {model.best_iteration}", flush=True)
    model.save_model(str(config.ART_DIR / f"lgb_{args.tag}.txt"))
    del dtr, dva

    va = feats.loc[is_va, ["i1", "i2", "label", "found_A"] + (["found_B"] if "found_B" in feats else [])]
    va = va.rename(columns={"i1": "s1_id", "i2": "cand_id"})
    va_rows = np.flatnonzero(is_va)
    p = np.empty(len(va_rows), dtype=np.float32)
    for s in range(0, len(va_rows), 5_000_000):     # chunked: predict() copies its input to float64
        p[s:s + 5_000_000] = model.predict(feats.iloc[va_rows[s:s + 5_000_000]][cols],
                                           num_iteration=model.best_iteration, num_threads=config.N_JOBS)
    va["p"] = p
    del feats, p
    va_ids = np.flatnonzero(role == "valid")
    truth_va = truth[role[truth.s1_id.values] == "valid"]
    report = {"blocking_union": blocking_recall(va, truth_va, va_ids)}
    for f in ("found_A", "found_B"):
        if f in va:
            report[f"blocking_{f}"] = blocking_recall(va[va[f] == 1], truth_va, va_ids)
    for k, v in report.items():
        print(k, v, flush=True)
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
    va[["s1_id", "cand_id", "label", "p"]].to_parquet(config.FEAT_DIR / f"val_scores_{args.tag}.parquet", index=False)
    json.dump({"blocking": report, "results": results, "best_iter": model.best_iteration, "features": cols},
              open(config.ART_DIR / f"train_report_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
