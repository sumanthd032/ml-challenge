"""Step 5b - second-stage re-scoring with cluster context.

Stage 1 scores every (S1, candidate) pair on its own. Many missed matches are obvious once the S1's other,
confident matches are known: a record with a blank address whose name equals a confident sibling's, or a
record with a replaced name whose address equals one. Stage 2 adds, for each pair (a, c):
  * S1 context: best / sum / count of stage-1 probabilities of a, rank of this pair within a
  * sibling context: similarity of c to a's confident candidates ("anchors", p >= ANCHOR_P, c excluded):
    embedding cosine, fuzzy name and address similarity, exact address / name equality
Only within-S1 context is used: on validation the pool of competing S1s is restricted to validation S1s,
so candidate-side statistics would not transfer to test.

Stage 2 is trained on the validation pairs, whose stage-1 scores are out-of-sample exactly like test scores.
`fit` reports 2-fold (by S1) out-of-fold F0.5 against stage 1, then trains on all validation S1s.

Usage: python stage2.py fit --tag v1        (reads val_scores_v1.parquet, writes lgb2_v1.txt)
       python stage2.py apply --tag v1      (reads test_scores_lgb_v1.parquet, writes test_scores2_v1.parquet)
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config
from data import load_split
from decide import expected_f_select
from metrics import macro_f05

MIN_P = 0.003       # pairs below this keep their stage-1 score (never selected anyway)
ANCHOR_P = 0.9      # confident siblings
STEP = 2_000_000


def _row_cos(emb, a, b):
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), STEP):
        x = emb[a[s:s + STEP]].astype(np.float32)
        y = emb[b[s:s + STEP]].astype(np.float32)
        out[s:s + STEP] = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-6)
    return out


def context_features(sc: pd.DataFrame, s23: pd.DataFrame, emb) -> pd.DataFrame:
    """sc: s1_id, cand_id, p (stage 1). Returns the rows with p >= MIN_P plus the context features."""
    t = time.time()
    df = sc[sc.p >= MIN_P].reset_index(drop=True)
    g = df.groupby("s1_id").p
    df["s1_max"] = g.transform("max")
    df["s1_sum"] = g.transform("sum")
    df["s1_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df["p_gap"] = df.p - df.s1_max
    for name, lo in (("s1_n_hi", ANCHOR_P), ("s1_n_mid", 0.3)):
        df[name] = (df.p >= lo).groupby(df.s1_id).transform("sum").astype(np.float32)

    anc = df.loc[df.p >= ANCHOR_P, ["s1_id", "cand_id", "p"]].rename(columns={"cand_id": "sib", "p": "sib_p"})
    m = df[["s1_id", "cand_id"]].reset_index().merge(anc, on="s1_id")
    m = m[m.cand_id.values != m.sib.values].reset_index(drop=True)
    print(f"  [stage2] {len(df):,} pairs, {len(m):,} sibling comparisons", flush=True)
    c, s = m.cand_id.values, m.sib.values
    m["cos"] = _row_cos(emb, c, s)
    addr, name, raw = s23.addr.values, s23.name.values, s23.raw_addr.values
    m["addr_sim"] = cpdist(addr[c], addr[s], scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    m["name_sim"] = cpdist(name[c], name[s], scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    has_a = (raw[c] != "") & (raw[s] != "")
    m["addr_eq"] = (has_a & (addr[c] == addr[s])).astype(np.float32)
    m["name_eq"] = ((name[c] != "") & (name[c] == name[s])).astype(np.float32)
    m.loc[~has_a, "addr_sim"] = np.nan
    agg = m.groupby("index").agg(sib_n=("sib", "size"), sib_cos_max=("cos", "max"), sib_cos_mean=("cos", "mean"),
                                 sib_addr_max=("addr_sim", "max"), sib_name_max=("name_sim", "max"),
                                 sib_addr_eq=("addr_eq", "max"), sib_name_eq=("name_eq", "max"))
    df = df.join(agg)
    df["sib_n"] = df.sib_n.fillna(0)
    df["c_no_addr"] = (raw[df.cand_id.values] == "").astype(np.float32)
    print(f"  [stage2] context features done ({time.time() - t:.0f}s)", flush=True)
    return df


FEATS = ["p", "s1_max", "s1_sum", "s1_rank", "p_gap", "s1_n_hi", "s1_n_mid", "sib_n", "sib_cos_max", "sib_cos_mean",
         "sib_addr_max", "sib_name_max", "sib_addr_eq", "sib_name_eq", "c_no_addr"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=config.N_JOBS, verbose=-1,
              seed=config.SEED)


def rescored(sc, df, p2):
    """Full score table with stage-2 probabilities replacing stage 1 where stage 2 ran."""
    out = sc[["s1_id", "cand_id", "p"]].copy()
    k = sc.s1_id.values.astype(np.int64) * (1 << 32) + sc.cand_id.values
    kd = df.s1_id.values.astype(np.int64) * (1 << 32) + df.cand_id.values
    order = np.argsort(k)
    pos = order[np.searchsorted(k[order], kd)]
    p = out.p.values.copy()
    p[pos] = p2
    out["p"] = p
    return out


def fit(tag, rounds):
    s1, s23 = load_split("train")
    role = s1.entity_id.map(config.s1_role).values
    from train import truth_rows
    truth = truth_rows(s1, s23)
    truth_va = truth[role[truth.s1_id.values] == "valid"]
    va_ids = np.flatnonzero(role == "valid")
    sc = pd.read_parquet(config.FEAT_DIR / f"val_scores_{tag}.parquet")
    emb = np.load(config.ART_DIR / "train_emb_s23.npy", mmap_mode="r")
    df = context_features(sc, s23, emb)
    y = df.label.values
    fold = (pd.util.hash_array(df.s1_id.values.astype(np.int64)) % 2).astype(int)
    oof = np.empty(len(df), dtype=np.float32)
    iters = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        dtr = lgb.Dataset(df.loc[tr, FEATS], y[tr])
        dte = lgb.Dataset(df.loc[te, FEATS], y[te], reference=dtr)
        mdl = lgb.train(PARAMS, dtr, rounds, valid_sets=[dte], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        oof[te] = mdl.predict(df.loc[te, FEATS], num_iteration=mdl.best_iteration)
        iters.append(mdl.best_iteration)
    s2 = rescored(sc, df, oof)
    res = {}
    for name, table in (("stage1", sc), ("stage2", s2)):
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
            r = macro_f05(expected_f_select(table, mode="thr", thr=thr), truth_va, va_ids)
            res[f"{name} thr{thr}"] = r
            print(f"{name} thr {thr}: F0.5={r['f05']:.4f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f}", flush=True)
    n_iter = int(np.mean(iters) * 1.1)
    mdl = lgb.train(PARAMS, lgb.Dataset(df[FEATS], y), n_iter)
    mdl.save_model(str(config.ART_DIR / f"lgb2_{tag}.txt"))
    imp = pd.Series(mdl.feature_importance("gain"), index=FEATS)
    print("importance:\n", (imp / imp.sum()).sort_values(ascending=False).round(4).to_string())
    json.dump({"results": res, "iters": iters, "final_iter": n_iter}, open(config.ART_DIR / f"stage2_report_{tag}.json", "w"), indent=1)


def apply(tag, model_tag, n_chunks=12):
    """Test re-scoring in chunks of whole S1s (all context is within-S1, so chunking is exact).

    Each chunk's stage-2 probabilities are checkpointed to FEAT_DIR/stage2_test_parts/, so a machine reset
    (D-012) only loses the chunk in progress, and the peak memory is ~1/n_chunks of a single pass.
    """
    import os
    t = time.time()
    s1, s23 = load_split("test")
    s23 = s23[["name", "addr", "raw_addr"]]
    del s1
    sc = pd.read_parquet(config.FEAT_DIR / f"test_scores_lgb_{tag}.parquet")
    pos = np.flatnonzero(sc.p.values >= MIN_P)                       # rows stage 2 re-scores
    sub = sc.iloc[pos].reset_index(drop=True)
    emb = np.load(config.ART_DIR / "test_emb_s23.npy", mmap_mode="r")
    mdl = lgb.Booster(model_file=str(config.ART_DIR / f"lgb2_{model_tag}.txt"))
    d = config.FEAT_DIR / "stage2_test_parts"
    d.mkdir(exist_ok=True)
    # chunk boundaries on S1 id, so every S1's rows land in one chunk
    edges = np.quantile(sub.s1_id.values, np.linspace(0, 1, n_chunks + 1)).astype(np.int64)
    edges[0], edges[-1] = -1, sub.s1_id.max()
    for k in range(n_chunks):
        f = d / f"part_{k:03d}_of_{n_chunks:03d}.npy"
        if f.exists():
            continue
        m = (sub.s1_id.values > edges[k]) & (sub.s1_id.values <= edges[k + 1])
        idx = np.flatnonzero(m)
        p2 = np.empty(0, dtype=np.float32)
        if len(idx):
            ch = sub.iloc[idx].reset_index(drop=True)
            df = context_features(ch, s23, emb)
            assert len(df) == len(ch)                                # sub already has p >= MIN_P
            p2 = mdl.predict(df[FEATS], num_threads=config.N_JOBS).astype(np.float32)
            del ch, df
        np.save(str(f) + ".tmp.npy", np.stack([idx.astype(np.float64), p2.astype(np.float64)]))
        os.replace(str(f) + ".tmp.npy", f)
        print(f"  [stage2] chunk {k + 1}/{n_chunks} saved ({time.time() - t:.0f}s)", flush=True)
    p = sc.p.values.astype(np.float32)
    for k in range(n_chunks):
        idx, p2 = np.load(d / f"part_{k:03d}_of_{n_chunks:03d}.npy")
        p[pos[idx.astype(np.int64)]] = p2
    out = pd.DataFrame({"s1_id": sc.s1_id.values, "cand_id": sc.cand_id.values, "p": p})
    del sc
    out.to_parquet(config.FEAT_DIR / f"test_scores2_{model_tag}.parquet", index=False)
    print("saved", config.FEAT_DIR / f"test_scores2_{model_tag}.parquet", f"({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "apply"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.tag, args.rounds)
    else:
        apply(args.tag, args.tag)


if __name__ == "__main__":
    main()
