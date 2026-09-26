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
       add --ce to use the cross-encoder score (crossenc.py, ce_scores_<split>.parquet) as a feature:
       the model / output tag becomes v1ce
       add --ce2 <t> to also use a second cross-encoder (ce_scores_<split>_<t>.parquet): tag v1ce<t>
       add --x for name-ambiguity + twin features (name_features, twin.py): tag suffix x
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


def name_codes(s1: pd.DataFrame, s23: pd.DataFrame, pool: np.ndarray) -> dict:
    """Per name field: integer codes of (country, name) for S1 and S2/S3 rows, and how many pool S1s carry each."""
    out = {}
    for col in ("name", "core"):
        keys = np.concatenate([(s1.country + "|" + s1[col]).values, (s23.country + "|" + s23[col]).values])
        codes = pd.factorize(keys)[0]
        k1, k2 = codes[:len(s1)], codes[len(s1):]
        out[col] = (k1, k2, np.bincount(k1[pool], minlength=codes.max() + 1), s23[col].values == "")
    return out


def name_features(df: pd.DataFrame, codes: dict) -> pd.DataFrame:
    """Name ambiguity. A record without an address can only be linked by its name: it is a safe match when this S1
    is the only pool S1 with that name, and a coin flip when several S1s share it (the legal form often decides)."""
    a, c = df.s1_id.values, df.cand_id.values
    for col, (k1, k2, cnt, empty2) in codes.items():
        eq = (k1[a] == k2[c]) & ~empty2[c]
        nc = np.where(empty2[c], -1, cnt[k2[c]]).astype(np.float32)
        df[f"{col}_eq"] = eq.astype(np.float32)
        df[f"n_{col}_c"] = nc                                        # pool S1s sharing the candidate's name
        df[f"n_{col}_a"] = cnt[k1[a]].astype(np.float32)             # pool S1s sharing this S1's name
        df[f"{col}_eq_uniq"] = (eq & (nc == 1)).astype(np.float32)
    return df


def density_codes(split: str, s1: pd.DataFrame, s23: pd.DataFrame, pool: np.ndarray) -> dict:
    """Neighbourhood density inputs (D-018): city of every record (geo.py) and, per (country, city, core name) and per
    (country, street+city tokens), how many pool S1s carry it. France packs 259k S1s into 15 cities, so a candidate's
    name often fits several S1s of one city; India/US dense areas teach stage 2 what that does to match odds."""
    import geo
    c1, c23 = geo.city_arrays(split, s1, s23, config.ART_DIR)
    k_cc = np.concatenate([s1.country.values + "|" + c1.astype(str) + "|" + s1.core.values,
                           s23.country.values + "|" + c23.astype(str) + "|" + s23.core.values])
    ok_cc = np.concatenate([(c1 != "") & (s1.core.values != ""), (c23 != "") & (s23.core.values != "")])
    code_cc = pd.factorize(k_cc)[0]
    n1 = len(s1)
    cnt_cc = np.bincount(code_cc[:n1][pool & ok_cc[:n1]], minlength=code_cc.max() + 1)
    k_st = s1.country.values + "|" + s1.toks.values
    code_st = pd.factorize(k_st)[0]
    cnt_st = np.bincount(code_st[pool & (s1.toks.values != "")], minlength=code_st.max() + 1)
    return {"c1": c1, "c23": c23, "cc1": code_cc[:n1], "cc23": code_cc[n1:], "ok1": ok_cc[:n1], "ok23": ok_cc[n1:],
            "cnt_cc": cnt_cc, "st1": code_st, "st_ok": s1.toks.values != "", "cnt_st": cnt_st}


DENS_FEATS = ["n_cc_a", "n_cc_c", "city_eq", "n_st_a", "pt_name", "pt_num", "pt_st"]


def density_features(df: pd.DataFrame, d: dict, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    import pairtype
    a, c = df.s1_id.values, df.cand_id.values
    df["n_cc_a"] = np.where(d["ok1"][a], d["cnt_cc"][d["cc1"][a]], -1).astype(np.float32)
    df["n_cc_c"] = np.where(d["ok23"][c], d["cnt_cc"][d["cc23"][c]], -1).astype(np.float32)
    ca, cc = d["c1"][a], d["c23"][c]
    df["city_eq"] = np.where(cc == "", -1, np.where(ca == cc, 1, 0)).astype(np.float32)
    df["n_st_a"] = np.where(d["st_ok"][a], d["cnt_st"][d["st1"][a]], -1).astype(np.float32)
    t = time.time()
    ty = pairtype.pair_types(s1, s23, a, c)
    df["pt_name"], df["pt_num"], df["pt_st"] = ty[:, 0], ty[:, 1], ty[:, 2]
    print(f"  [stage2] density + pair-type features done ({time.time() - t:.0f}s)", flush=True)
    return df


def attach_twin(df: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    from twin import TWIN_NAMES, twin_features
    t = time.time()
    tw = twin_features(s1, s23, df.s1_id.values, df.cand_id.values)
    for j, n in enumerate(TWIN_NAMES):
        df["tw_" + n] = tw[:, j]
    print(f"  [stage2] twin features done ({time.time() - t:.0f}s)", flush=True)
    return df


def attach_ce(df: pd.DataFrame, ce: pd.DataFrame, name: str = "ce") -> pd.DataFrame:
    """Add the cross-encoder score of each (s1_id, cand_id) pair, and its rank / gap within the S1."""
    k = ce.s1_id.values.astype(np.int64) * (1 << 32) + ce.cand_id.values
    order = np.argsort(k)
    k = k[order]
    kd = df.s1_id.values.astype(np.int64) * (1 << 32) + df.cand_id.values
    pos = np.minimum(np.searchsorted(k, kd), len(k) - 1)
    hit = k[pos] == kd
    assert hit.mean() > 0.999, f"cross-encoder scores missing for {1 - hit.mean():.2%} of pairs"
    df[name] = np.where(hit, ce.ce.values[order[pos]], np.nan).astype(np.float32)
    g = df.groupby("s1_id")[name]
    df[name + "_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df[name + "_gap"] = (df[name] - g.transform("max")).astype(np.float32)
    return df


CE_FEATS = ["ce", "ce_rank", "ce_gap"]
CE2_FEATS = ["ce2", "ce2_rank", "ce2_gap"]
NAME_FEATS = [f"{f}{col}{g}" for col in ("name", "core") for f, g in
              (("", "_eq"), ("n_", "_c"), ("n_", "_a"), ("", "_eq_uniq"))]
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


def model_tag_of(tag, use_ce, ce2, extra, dens=False, noghost=False):
    return tag + ("ce" if use_ce else "") + (ce2 or "") + ("x" if extra else "") + ("d" if dens else "") + \
        ("g" if noghost else "")


def fit(tag, rounds, use_ce=False, ce2=None, extra=False, dens=False, noghost=False):
    from twin import TWIN_NAMES
    feats = FEATS + (CE_FEATS if use_ce else []) + (CE2_FEATS if ce2 else []) +         (NAME_FEATS + ["tw_" + n for n in TWIN_NAMES] if extra else []) + (DENS_FEATS if dens else [])
    out_tag = model_tag_of(tag, use_ce, ce2, extra, dens, noghost)
    s1, s23 = load_split("train")
    role = s1.entity_id.map(config.s1_role).values
    from train import truth_rows
    truth = truth_rows(s1, s23)
    truth_va = truth[role[truth.s1_id.values] == "valid"]
    va_ids = np.flatnonzero(role == "valid")
    sc = pd.read_parquet(config.FEAT_DIR / f"val_scores_{tag}.parquet")
    if noghost:
        # D-020: records owned by ghost S1s are true variants of an S1 absent from the pool. The test has none
        # (its no-address rates match address-bearing distractors only), so as negatives they only teach stage 2
        # to distrust pairs that look exactly like true matches.
        owner_role = np.full(len(s23), "", dtype=object)
        owner_role[truth.cand_id.values] = role[truth.s1_id.values]
        drop = owner_role[sc.cand_id.values] == "ghost"
        print(f"  [stage2] noghost: dropping {int(drop.sum()):,} of {len(sc):,} val rows (ghost-owned records)", flush=True)
        sc = sc[~drop].reset_index(drop=True)
    emb = np.load(config.ART_DIR / "train_emb_s23.npy", mmap_mode="r")
    df = context_features(sc, s23, emb)
    if use_ce:
        df = attach_ce(df, pd.read_parquet(config.FEAT_DIR / "ce_scores_val.parquet", columns=["s1_id", "cand_id", "ce"]))
    if ce2:
        df = attach_ce(df, pd.read_parquet(config.FEAT_DIR / f"ce_scores_val_{ce2}.parquet",
                                           columns=["s1_id", "cand_id", "ce"]), "ce2")
    if extra:
        df = name_features(df, name_codes(s1, s23, role != "ghost"))
        df = attach_twin(df, s1, s23)
    if dens:
        df = density_features(df, density_codes("train", s1, s23, role != "ghost"), s1, s23)
    y = df.label.values
    fold = (pd.util.hash_array(df.s1_id.values.astype(np.int64)) % 2).astype(int)
    oof = np.empty(len(df), dtype=np.float32)
    iters = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        dtr = lgb.Dataset(df.loc[tr, feats], y[tr])
        dte = lgb.Dataset(df.loc[te, feats], y[te], reference=dtr)
        mdl = lgb.train(PARAMS, dtr, rounds, valid_sets=[dte], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        oof[te] = mdl.predict(df.loc[te, feats], num_iteration=mdl.best_iteration)
        iters.append(mdl.best_iteration)
    s2 = rescored(sc, df, oof)
    s2.assign(label=sc.label.values).to_parquet(config.FEAT_DIR / f"val_scores2_{out_tag}.parquet", index=False)
    res = {}
    for name, table in (("stage1", sc), ("stage2", s2)):
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
            r = macro_f05(expected_f_select(table, mode="thr", thr=thr), truth_va, va_ids)
            res[f"{name} thr{thr}"] = r
            print(f"{name} thr {thr}: F0.5={r['f05']:.4f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f}", flush=True)
    n_iter = int(np.mean(iters) * 1.1)
    mdl = lgb.train(PARAMS, lgb.Dataset(df[feats], y), n_iter)
    mdl.save_model(str(config.ART_DIR / f"lgb2_{out_tag}.txt"))
    imp = pd.Series(mdl.feature_importance("gain"), index=feats)
    print("importance:\n", (imp / imp.sum()).sort_values(ascending=False).round(4).to_string())
    json.dump({"results": res, "iters": iters, "final_iter": n_iter}, open(config.ART_DIR / f"stage2_report_{out_tag}.json", "w"), indent=1)


def apply(tag, use_ce=False, ce2=None, extra=False, n_chunks=12, split="test", ce_sfx="", dens=False,
          noghost=False):
    """Test re-scoring in chunks of whole S1s (all context is within-S1, so chunking is exact).

    Each chunk's stage-2 probabilities are checkpointed to FEAT_DIR/stage2_<split>_parts/, so a machine reset
    (D-012) only loses the chunk in progress, and the peak memory is ~1/n_chunks of a single pass.
    split: 'test' or a test subset such as 'testfr' (France only, D-018).
    """
    import os
    model_tag = model_tag_of(tag, use_ce, ce2, extra, dens, noghost)
    t = time.time()
    s1, s23 = load_split(split)
    codes = name_codes(s1, s23, np.ones(len(s1), dtype=bool)) if extra else None
    dcodes = density_codes(split, s1, s23, np.ones(len(s1), dtype=bool)) if dens else None
    s1 = s1[["name", "core", "raw_addr", "toks"]]
    s23 = s23[["name", "core", "addr", "raw_addr", "toks"]]
    sc = pd.read_parquet(config.FEAT_DIR / f"{split}_scores_lgb_{tag}.parquet")
    pos = np.flatnonzero(sc.p.values >= MIN_P)                       # rows stage 2 re-scores
    sub = sc.iloc[pos].reset_index(drop=True)
    emb = np.load(config.ART_DIR / f"{split}_emb_s23.npy", mmap_mode="r")
    mdl = lgb.Booster(model_file=str(config.ART_DIR / f"lgb2_{model_tag}.txt"))
    feats = mdl.feature_name()
    ce = (pd.read_parquet(config.FEAT_DIR / f"ce_scores_{split}{ce_sfx}.parquet", columns=["s1_id", "cand_id", "ce"])
          if use_ce else None)
    ce2_df = (pd.read_parquet(config.FEAT_DIR / f"ce_scores_{split}_{ce2}.parquet", columns=["s1_id", "cand_id", "ce"])
              if ce2 else None)
    split = split + ce_sfx                    # output names carry the cross-encoder variant
    d = config.FEAT_DIR / f"stage2_{split}_parts_{model_tag}"
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
            if use_ce:
                df = attach_ce(df, ce)
            if ce2:
                df = attach_ce(df, ce2_df, "ce2")
            if extra:
                df = attach_twin(name_features(df, codes), s1, s23)
            if dens:
                df = density_features(df, dcodes, s1, s23)
            p2 = mdl.predict(df[feats], num_threads=config.N_JOBS).astype(np.float32)
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
    out.to_parquet(config.FEAT_DIR / f"{split}_scores2_{model_tag}.parquet", index=False)
    print("saved", config.FEAT_DIR / f"{split}_scores2_{model_tag}.parquet", f"({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "apply"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--ce", action="store_true", help="add cross-encoder score features")
    ap.add_argument("--ce2", default=None, help="tag of a second cross-encoder (ce_scores_<split>_<tag>.parquet)")
    ap.add_argument("--x", action="store_true", help="add name-ambiguity and twin features")
    ap.add_argument("--split", default="test", help="apply: test split name (test, or a subset like testfr)")
    ap.add_argument("--ce-sfx", default="", help="apply: read ce_scores_<split><sfx>.parquet (e.g. _fr)")
    ap.add_argument("--dens", action="store_true", help="add neighbourhood-density + pair-type features (D-018)")
    ap.add_argument("--noghost", action="store_true", help="fit without ghost-owned val records (D-020)")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.tag, args.rounds, args.ce, args.ce2, args.x, args.dens, args.noghost)
    else:
        apply(args.tag, args.ce, args.ce2, args.x, split=args.split, ce_sfx=args.ce_sfx, dens=args.dens,
              noghost=args.noghost)


if __name__ == "__main__":
    main()
