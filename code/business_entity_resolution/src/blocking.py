"""Step 2: candidate generation (blocking).

Pass A (this file): IDF-weighted sparse token retrieval, per country, in both directions.
Each record becomes a bag of typed tokens:
    n:<core name token>   p:<first 4 chars of name token>   c:<first 7 chars of concatenated name>
    a:<address word>      #:<address number>
Tokens with document frequency above MAX_DF are dropped (too common to discriminate). Rows are
IDF-weighted and L2-normalized; cosine top-K is taken S1->S2/S3 (K_FWD) and S2/S3->S1 (K_REV).

Output: a DataFrame of (s1_id, cand_id, blk_score, blk_rank_fwd, blk_rank_rev) pairs.
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

import config

MAX_DF = 5000
K_FWD = 30
K_REV = 8
N_FEAT = 2 ** 24


def record_tokens(core, concat, alt, toks, nums):
    """Typed token list of one record (see the module docstring; `k:` is the whole concatenated name)."""
    out = []
    for t in (core + " " + alt).split():
        out.append("n:" + t)
        if len(t) >= 4:
            out.append("p:" + t[:4])
    if len(concat) >= 5:
        out.append("c:" + concat[:7])
    if concat:
        out.append("k:" + concat)                     # whole-name exact key (helps empty addresses)
    out.extend("a:" + t for t in toks.split())
    out.extend("#:" + t for t in set(nums.split()))
    return out


def _analyzer(row):
    return record_tokens(*row)


def vectorize(df: pd.DataFrame, pool: Pool) -> sp.csr_matrix:
    """Binary hashed token matrix (rows = records of df), built in 50k-row chunks on the worker pool."""
    hv = HashingVectorizer(analyzer=_analyzer, n_features=N_FEAT, alternate_sign=False, norm=None, binary=True,
                           dtype=np.float32)
    rows = list(zip(df.core, df.concat, df.alt, df.toks, df.nums))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    mats = pool.map(hv.transform, chunks)
    return sp.vstack(mats).tocsr()


def idf_weight(A: sp.csr_matrix, B: sp.csr_matrix):
    """IDF over the union of both sides; drop too-frequent features; L2-normalize rows."""
    df_ = np.asarray((A > 0).sum(0)).ravel() + np.asarray((B > 0).sum(0)).ravel()
    n = A.shape[0] + B.shape[0]
    idf = np.log((n + 1) / (df_ + 1)).astype(np.float32)
    idf[df_ > MAX_DF] = 0.0
    D = sp.diags(idf)
    out = []
    for M in (A, B):
        M = (M @ D).tocsr()
        M.eliminate_zeros()
        norm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
        norm[norm == 0] = 1
        out.append(sp.diags(1 / norm).astype(np.float32) @ M)
    return out[0].tocsr(), out[1].tocsr()


# ---- worker state for parallel top-k ---------------------------------------------------------
_W = {}


def _init(path_t):
    _W["BT"] = sp.load_npz(path_t).tocsr()


def _topk_chunk(args):
    Q, k, row_offset = args
    C = (Q @ _W["BT"]).tocsr()
    qi, ci, sc = [], [], []
    for r in range(C.shape[0]):
        s, e = C.indptr[r], C.indptr[r + 1]
        if s == e:
            continue
        d = C.data[s:e]; idx = C.indices[s:e]
        if e - s > k:
            top = np.argpartition(-d, k)[:k]
            d, idx = d[top], idx[top]
        qi.append(np.full(len(d), r + row_offset, dtype=np.int32)); ci.append(idx.astype(np.int32)); sc.append(d)
    if not qi:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return np.concatenate(qi), np.concatenate(ci), np.concatenate(sc)


def topk(Q: sp.csr_matrix, B: sp.csr_matrix, k: int, tag: str, chunk: int = 1000):
    """Top-k rows of B by dot product for every row of Q, in parallel over row chunks of Q.

    B^T is written to a temporary .npz (named by `tag`) that each worker loads once. Returns (query row, B row, score)
    arrays; rows of Q with no overlapping token are absent.
    """
    path_t = config.ART_DIR / f"_tmp_BT_{tag}.npz"
    sp.save_npz(path_t, B.T.tocsr(), compressed=False)
    jobs = [(Q[i:i + chunk], k, i) for i in range(0, Q.shape[0], chunk)]
    with Pool(min(16, config.N_JOBS), initializer=_init, initargs=(str(path_t),)) as pool:
        res = pool.map(_topk_chunk, jobs, chunksize=1)
    path_t.unlink(missing_ok=True)
    q = np.concatenate([r[0] for r in res]); c = np.concatenate([r[1] for r in res]); s = np.concatenate([r[2] for r in res])
    return q, c, s


def _rank_within(groups: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rank (0 = best) of each score within its group."""
    order = np.lexsort((-scores, groups))
    g = groups[order]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    ranks_sorted = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def block_country(s1: pd.DataFrame, s23: pd.DataFrame, tag: str, pool: Pool) -> pd.DataFrame:
    """Pass A for one country: union of forward (S1->S2/S3) and reverse top-k pairs.

    A rank of 99 means the pair was not found in that direction. Returns
    (s1_id, cand_id, blk_score, blk_rank_fwd, blk_rank_rev).
    """
    t = time.time()
    A = vectorize(s1, pool); B = vectorize(s23, pool)
    A, B = idf_weight(A, B)
    print(f"  [{tag}] vectorized {A.shape[0]:,} x {B.shape[0]:,}, nnz {A.nnz:,}/{B.nnz:,} ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    q, c, s = topk(A, B, K_FWD, tag + "f")
    print(f"  [{tag}] forward top-{K_FWD}: {len(q):,} pairs ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    q2, c2, s2 = topk(B, A, K_REV, tag + "r")
    print(f"  [{tag}] reverse top-{K_REV}: {len(q2):,} pairs ({time.time()-t:.0f}s)", flush=True)
    fwd = pd.DataFrame({"i1": q, "i2": c, "blk_score": s})
    fwd["blk_rank_fwd"] = _rank_within(q, s)
    rev = pd.DataFrame({"i1": c2, "i2": q2, "blk_score": s2})
    rev["blk_rank_rev"] = _rank_within(q2, s2)
    m = fwd.merge(rev.drop(columns="blk_score"), on=["i1", "i2"], how="outer")
    m = m.merge(rev[["i1", "i2", "blk_score"]].rename(columns={"blk_score": "bs2"}), on=["i1", "i2"], how="left")
    m["blk_score"] = m.blk_score.fillna(m.bs2)
    m = m.drop(columns="bs2")
    m["blk_rank_fwd"] = m.blk_rank_fwd.fillna(99).astype(np.int16)
    m["blk_rank_rev"] = m.blk_rank_rev.fillna(99).astype(np.int16)
    m["s1_id"] = s1.entity_id.values[m.i1.values]
    m["cand_id"] = s23.entity_id.values[m.i2.values]
    return m[["s1_id", "cand_id", "blk_score", "blk_rank_fwd", "blk_rank_rev"]]


def run_blocking(s1n: pd.DataFrame, s23n: pd.DataFrame) -> pd.DataFrame:
    """Pass A over all countries present on both sides; returns the concatenated block_country outputs."""
    parts = []
    with Pool(config.N_JOBS) as pool:
        for country in sorted(set(s1n.country) | set(s23n.country)):
            a = s1n[s1n.country == country].reset_index(drop=True)
            b = s23n[s23n.country == country].reset_index(drop=True)
            if len(a) == 0 or len(b) == 0:
                continue
            print(f"country {country}: S1 {len(a):,}  S2+S3 {len(b):,}", flush=True)
            parts.append(block_country(a, b, country, pool))
    return pd.concat(parts, ignore_index=True)
