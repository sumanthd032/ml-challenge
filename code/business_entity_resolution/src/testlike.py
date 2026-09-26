"""Test-like validation (D-020): score validation the way the test would see it.

The ghost validation (D-007) keeps records of absent S1s as distractors; the test has none (its no-address rates
match address-bearing distractors only), and in the test every owner S1 is scored and competes. So:
* records owned by ghost S1s and by fit S1s (not scored in the val table) are removed;
* real confusions between validation S1s stay;
* false positives on owner-less distractors count W_NOOWNER times (the test has 1.94x more unmatched records per S1).

Usage: python testlike.py val_scores2_v1cel12xdg.parquet [--w 1.94]   (prints F0.5 by threshold and competition)
"""
import argparse

import numpy as np
import pandas as pd

import config
from data import load_split
from decide import compete, expected_f_select
from train import truth_rows


def setup():
    s1, s23 = load_split("train")
    role = s1.entity_id.map(config.s1_role).values
    truth = truth_rows(s1, s23)
    owner = np.full(len(s23), -1, dtype=np.int64)
    owner[truth.cand_id.values] = truth.s1_id.values
    orole = np.where(owner >= 0, role[np.maximum(owner, 0)], "none")
    va_ids = np.flatnonzero(role == "valid")
    tv = truth[role[truth.s1_id.values] == "valid"]
    return owner, orole, va_ids, tv


def f05(sel, owner, orole, va_ids, tv, w_noowner=1.94):
    """Macro F0.5 over validation S1s; sel: selected (s1_id, cand_id) row positions."""
    s1, c = sel.s1_id.values, sel.cand_id.values
    tp_m = owner[c] == s1
    fp_real = (~tp_m) & (orole[c] == "valid")
    fp_none = orole[c] == "none"
    idx = pd.Index(va_ids)
    pos = idx.get_indexer(s1)
    n = len(va_ids)
    tp = np.bincount(pos, weights=tp_m, minlength=n)
    npred = np.bincount(pos, weights=tp_m + fp_real + w_noowner * fp_none, minlength=n)
    ntrue = np.bincount(idx.get_indexer(tv.s1_id.values), minlength=n)
    den = 0.25 * ntrue + npred
    f = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-9), 1.0)
    return f.mean(), tp.sum() / max(npred.sum(), 1), tp.sum() / ntrue.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--w", type=float, default=1.94)
    a = ap.parse_args()
    owner, orole, va_ids, tv = setup()
    v = pd.read_parquet(config.FEAT_DIR / a.scores, columns=["s1_id", "cand_id", "p"])
    v = v[np.isin(orole[v.cand_id.values], ["valid", "none"])]
    for thr in (0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
        for cw in (1.0, 2.0):
            m = f05(expected_f_select(compete(v, cw), mode="thr", thr=thr), owner, orole, va_ids, tv, a.w)
            print(f"thr {thr} compete {cw}: F0.5 {m[0]:.5f}  P {m[1]:.4f}  R {m[2]:.4f}")


if __name__ == "__main__":
    main()
