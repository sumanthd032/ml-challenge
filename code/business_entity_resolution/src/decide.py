"""Step 5 - turn pair probabilities into final match lists.

1. One-to-many constraint: every S2/S3 record is linked to at most one S1 (true in 100% of train
   labels), so each candidate keeps only its highest-probability S1.
2. Per-S1 expected-F0.5 maximization. For an S1 with sorted probabilities p_1 >= p_2 >= ...:
       E[F | predict top-k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * sum_i p_i + k)      (k >= 1)
       E[F | predict nothing] ~= prod_i (1 - p_i)                                 (true singleton)
   and we choose the k with the largest value. A hard floor `min_p` guards against
   low-confidence picks.
"""
import numpy as np
import pandas as pd


def assign_best_s1(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each candidate id, only its best-scoring S1 pair."""
    idx = scored.groupby("cand_id").p.idxmax()
    return scored.loc[idx]


def expected_f_select(scored: pd.DataFrame, alpha: float = 1.0, min_p: float = 0.05,
                      mode: str = "expf", thr: float = 0.5) -> pd.DataFrame:
    """Return selected (s1_id, cand_id) pairs. scored needs columns s1_id, cand_id, p."""
    df = assign_best_s1(scored)[["s1_id", "cand_id", "p"]].copy()
    df["p"] = df.p.clip(1e-6, 1 - 1e-6) ** alpha
    if mode == "thr":
        return df[df.p >= thr][["s1_id", "cand_id"]]
    df = df.sort_values(["s1_id", "p"], ascending=[True, False]).reset_index(drop=True)
    g = df.groupby("s1_id", sort=False)
    df["k"] = g.cumcount() + 1
    df["cum_p"] = g.p.cumsum()
    df["sum_p"] = g.p.transform("sum")
    df["ef"] = 1.25 * df.cum_p / (0.25 * df.sum_p + df.k)
    df["log1m"] = np.log1p(-df.p)
    e0 = np.exp(g.log1m.transform("sum"))
    best_ef = g.ef.transform("max")
    # k* = first k achieving the max expected F; select all rows with k <= k*
    df["is_best"] = df.ef >= best_ef - 1e-12
    kstar = df[df.is_best].groupby("s1_id").k.min()
    df["kstar"] = df.s1_id.map(kstar)
    sel = df[(df.k <= df.kstar) & (best_ef > e0) & (df.p >= min_p)]
    return sel[["s1_id", "cand_id"]]
