"""Official metric re-implementation: macro F0.5 over Source-1 entities (singletons included)."""
import numpy as np
import pandas as pd


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> dict:
    """pred/truth: long tables (s1_id, cand_id). s1_ids: all S1 entities in the evaluation set.

    Per entity: F0.5 = 1.25*TP / (0.25*n_true + n_pred); both empty -> 1.0; one empty -> 0.0.
    """
    s1_ids = pd.Index(pd.unique(pd.Series(list(s1_ids))))
    pred = pred[pred.s1_id.isin(s1_ids)]
    truth = truth[truth.s1_id.isin(s1_ids)]
    n_pred = pred.groupby("s1_id").size().reindex(s1_ids, fill_value=0).values
    n_true = truth.groupby("s1_id").size().reindex(s1_ids, fill_value=0).values
    tp_df = pred.merge(truth, on=["s1_id", "cand_id"])
    tp = tp_df.groupby("s1_id").size().reindex(s1_ids, fill_value=0).values
    denom = 0.25 * n_true + n_pred
    f = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-9), 1.0)
    prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), np.nan)
    rec = np.where(n_true > 0, tp / np.maximum(n_true, 1), np.nan)
    sing = n_true == 0
    return {"f05": float(f.mean()), "precision_macro": float(np.nanmean(prec)), "recall_macro": float(np.nanmean(rec)),
            "f05_singletons": float(f[sing].mean()) if sing.any() else float("nan"),
            "f05_nonsingletons": float(f[~sing].mean()), "n_entities": int(len(s1_ids)),
            "micro_precision": float(tp.sum() / max(n_pred.sum(), 1)), "micro_recall": float(tp.sum() / max(n_true.sum(), 1))}


def blocking_recall(cands: pd.DataFrame, truth: pd.DataFrame, s1_ids=None) -> dict:
    if s1_ids is not None:
        truth = truth[truth.s1_id.isin(s1_ids)]
        cands = cands[cands.s1_id.isin(s1_ids)]
    hit = truth.merge(cands[["s1_id", "cand_id"]], on=["s1_id", "cand_id"]).shape[0]
    n_s1 = len(set(s1_ids)) if s1_ids is not None else cands.s1_id.nunique()
    return {"pair_recall": hit / max(len(truth), 1), "pairs": len(cands), "pairs_per_s1": len(cands) / max(n_s1, 1)}
