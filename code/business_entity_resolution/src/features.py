"""Step 3 - pairwise features for (S1, candidate) pairs.

All features are language-agnostic similarity scores (no country one-hot), so the model can be
applied to France although it is trained on US + India only.
"""
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

import config

LEGAL_SET = {"private", "limited", "inc", "corp", "co", "llc", "llp", "lp", "plc", "pllc", "pc", "pa", "sarl",
             "sas", "sasu", "eurl", "sci", "sa", "snc", "ei", "selarl", "etablissements"}
FIELDS = ["core", "concat", "alt", "name", "toks", "nums", "state", "addr", "indic", "raw_addr"]

_IDF = {}


def build_idf(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """IDF of core-name tokens (over S1+S2+S3 of the split). Used for weighted token overlap."""
    from collections import Counter
    c = Counter()
    for col in (s1.core, s23.core):
        for s in col:
            c.update(set(s.split()))
    n = len(s1) + len(s23)
    return {t: math.log(n / v) for t, v in c.items() if v >= 2}


def _init(idf):
    _IDF.clear(); _IDF.update(idf)


def _idf(t):
    return _IDF.get(t, 12.0)


def _pair_feats(a, b):
    """a, b: tuples of FIELDS for the S1 record and the candidate."""
    core1, cc1, alt1, nm1, tk1, nu1, st1, ad1, _, _ = a
    core2, cc2, alt2, nm2, tk2, nu2, st2, ad2, ind2, raw2 = b
    f = []
    # ---- name ----------------------------------------------------------------------------
    f.append(fuzz.ratio(core1, core2))
    f.append(fuzz.token_sort_ratio(core1, core2))
    f.append(fuzz.token_set_ratio(core1, core2))
    f.append(fuzz.partial_ratio(core1, core2) if core1 and core2 else 0)
    f.append(fuzz.WRatio(core1, core2) if core1 and core2 else 0)
    f.append(JaroWinkler.similarity(cc1, cc2) * 100)
    f.append(fuzz.partial_ratio(cc1, cc2) if cc1 and cc2 else 0)
    f.append(Levenshtein.distance(cc1, cc2))
    alt_best = 0
    for x in (alt1, core1):
        for y in (alt2, core2):
            if x and y and (x is alt1 or y is alt2):
                alt_best = max(alt_best, fuzz.token_set_ratio(x, y))
    f.append(alt_best)
    t1, t2 = set(core1.split()), set(core2.split())
    inter = t1 & t2
    f.append(len(inter) / max(1, len(t1 | t2)))
    w1 = sum(_idf(t) for t in t1); w2 = sum(_idf(t) for t in t2); wi = sum(_idf(t) for t in inter)
    f.append(wi / w1 if w1 else 0)
    f.append(wi / w2 if w2 else 0)
    f.append(max((_idf(t) for t in inter), default=0))
    # soft idf: tokens of s1 matched fuzzily (>=85) in cand
    soft = 0.0
    for t in t1 - inter:
        best = max((fuzz.ratio(t, u) for u in t2 - inter), default=0)
        if best >= 80:
            soft += _idf(t) * best / 100
    f.append((wi + soft) / w1 if w1 else 0)
    f.append(len(t1)); f.append(len(t2))
    f.append(int(bool(t1) and bool(t2) and core1.split()[0] == core2.split()[0]))
    f.append(fuzz.token_set_ratio(nm1, nm2))
    l1 = set(nm1.split()) & LEGAL_SET; l2 = set(nm2.split()) & LEGAL_SET
    f.append(int(bool(l1 & l2)))
    f.append(int(bool(l1) and bool(l2) and not (l1 & l2)))
    f.append(int(ind2))
    f.append(int(cc1 in cc2 or cc2 in cc1) if cc1 and cc2 else 0)
    # ---- address ------------------------------------------------------------------------
    has1, has2 = bool(tk1 or nu1), bool(tk2 or nu2)
    f.append(int(has2)); f.append(len(raw2))
    f.append(fuzz.token_set_ratio(tk1, tk2) if tk1 and tk2 else -1)
    f.append(fuzz.token_sort_ratio(tk1, tk2) if tk1 and tk2 else -1)
    f.append(fuzz.partial_ratio(tk1, tk2) if tk1 and tk2 else -1)
    a1, a2 = set(tk1.split()), set(tk2.split())
    f.append(len(a1 & a2) / max(1, len(a1 | a2)) if a1 and a2 else -1)
    f.append(len(a1 & a2) / max(1, len(a1)) if a1 and a2 else -1)
    f.append(fuzz.token_set_ratio(ad1, ad2) if ad1 and ad2 else -1)
    n1, n2 = nu1.split(), nu2.split()
    s1n, s2n = set(n1), set(n2)
    if s1n and s2n:
        ni = s1n & s2n
        f.append(len(ni) / len(s1n | s2n)); f.append(len(ni))
        f.append(max((len(x) for x in ni), default=0))
        f.append(int(n1[0] in s2n))
        big1 = max(n1, key=len); f.append(int(big1 in s2n))
        # fuzzy number agreement (typos like 4809 vs 4808, 43 vs 431)
        f.append(max(fuzz.ratio(x, y) for x in s1n for y in s2n))
        f.append(len(s1n - s2n)); f.append(len(s2n - s1n))
    else:
        f.extend([-1, -1, -1, -1, -1, -1, -1, -1])
    f.append(1 if (st1 and st2 and st1 == st2) else (0 if (st1 and st2) else -1))
    return f


FEAT_NAMES = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_wratio", "cc_jw", "cc_partial", "cc_lev", "n_alt_best",
              "n_jacc", "n_idf_cov1", "n_idf_cov2", "n_idf_max", "n_soft_idf", "n_len1", "n_len2", "n_first_eq",
              "nm_tset_full", "legal_agree", "legal_conflict", "c_indic", "cc_contain",
              "a_has2", "a_rawlen2", "a_tset", "a_tsort", "a_partial", "a_jacc", "a_cov1", "a_full_tset",
              "num_jacc", "num_n_shared", "num_maxlen_shared", "num_first_in", "num_big_in", "num_fuzzy",
              "num_miss1", "num_miss2", "state_eq"]


def _chunk(args):
    A, B = args
    return np.array([_pair_feats(a, b) for a, b in zip(A, B)], dtype=np.float32)


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, idf: dict) -> pd.DataFrame:
    """pairs must have s1_id, cand_id (+ blocking columns). Returns pairs with feature columns added."""
    import time
    ia = pd.Index(s1.entity_id).get_indexer(pairs.s1_id.values)
    ib = pd.Index(s23.entity_id).get_indexer(pairs.cand_id.values)
    assert (ia >= 0).all() and (ib >= 0).all()
    colsA = [s1[c].values for c in FIELDS]
    colsB = [s23[c].values for c in FIELDS]
    X = np.empty((len(pairs), len(FEAT_NAMES)), dtype=np.float32)
    block, step = 4_000_000, 20000
    t = time.time()
    with Pool(config.N_JOBS, initializer=_init, initargs=(idf,)) as pool:
        for s in range(0, len(pairs), block):
            a_idx, b_idx = ia[s:s + block], ib[s:s + block]
            A = list(zip(*[c[a_idx] for c in colsA]))
            B = list(zip(*[c[b_idx] for c in colsB]))
            jobs = [(A[i:i + step], B[i:i + step]) for i in range(0, len(A), step)]
            X[s:s + len(A)] = np.vstack(pool.map(_chunk, jobs, chunksize=1))
            print(f"    features {s + len(A):,}/{len(pairs):,} ({time.time() - t:.0f}s)", flush=True)
    out = pairs.reset_index(drop=True).copy()
    for j, n in enumerate(FEAT_NAMES):
        out[n] = X[:, j]
    del X
    out["is_s3"] = out.cand_id.str.startswith("S3").astype(np.int8)
    add_group_features(out)
    return out


def add_group_features(df: pd.DataFrame) -> None:
    """Competition features: how this pair compares with other pairs of the same S1 / same candidate."""
    df["pair_sim"] = (df.n_tset + df.cc_jw) / 2 + np.where(df.a_tset >= 0, df.a_tset, 50) + \
        np.where(df.num_jacc >= 0, df.num_jacc * 100, 30)
    df["_k1"] = pd.factorize(df.s1_id)[0]
    df["_k2"] = pd.factorize(df.cand_id)[0]
    extra = [c for c in ("emb_cos",) if c in df]
    for key, tag in (("_k1", "s1"), ("_k2", "cd")):
        g = df.groupby(key, sort=False)
        df[f"{tag}_n"] = g[key].transform("size").astype(np.int16)
        for col in ["pair_sim", "blk_score", "n_tset", "a_tset"] + extra:
            mx = g[col].transform("max")
            df[f"{tag}_{col}_gap"] = df[col] - mx
            df[f"{tag}_{col}_rank"] = g[col].rank(ascending=False, method="min").astype(np.float32)
    # second-best gap for candidate: margin of this pair over the best *other* S1
    df["cd_pair_sim_2nd_gap"] = _gap_to_best_other(df, "_k2", "pair_sim")
    df["s1_pair_sim_2nd_gap"] = _gap_to_best_other(df, "_k1", "pair_sim")
    for c in extra:
        df[f"cd_{c}_2nd_gap"] = _gap_to_best_other(df, "_k2", c)
    df.drop(columns=["_k1", "_k2"], inplace=True)


def _gap_to_best_other(df, key, col):
    g = df.groupby(key, sort=False)[col]
    top1 = g.transform("max")
    # second max per group
    tmp = df[[key, col]].copy()
    tmp["r"] = g.rank(ascending=False, method="first")
    second = tmp[tmp.r == 2].set_index(key)[col]
    sec = df[key].map(second).fillna(-1e3).values
    return np.where(df[col].values >= top1.values, df[col].values - sec, df[col].values - top1.values)
