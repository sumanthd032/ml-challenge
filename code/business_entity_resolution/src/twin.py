"""Twin features: how a candidate *differs* from the S1 record, not how similar it is.

Error analysis of stage 1 (decision D-013) showed that most confident false merges are "twins":
a single S2/S3 record of another business copied from the S1 with a few edits, typically
  * another legal form   ("Private Limited" -> "Limited", "SA" -> "SARL", none -> "Inc"),
  * a swapped name word  ("Bordeaux Union SASU" -> "Bordeaux Club SASU"),
  * a nearby house number (1002 -> 1005, 12 -> 14, 9-7 -> 9-12),
while true records carry noise of another kind: typos, digit substitutions (157 -> 757), truncations,
dropped legal forms, appended generic words ("Services", "Center"). Similarity scores mix these up; the
features below describe the edit itself. All are language-agnostic, so they transfer to unseen countries.
"""
from multiprocessing import Pool

import re

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import config
from features import LEGAL_SET

TWIN_NAMES = ["lg_rel", "lg_n1", "lg_n2", "lg_seq_eq",
              "nt_miss", "nt_extra", "nt_sub", "nt_miss_first", "nt_order_eq",
              "hn_has1", "hn_has2", "hn_eq", "hn_absdiff", "hn_reldiff", "hn_digit_ed", "hn_prefix", "hn_samelen",
              "hn_best_absdiff", "hn_n_extra", "hn_frac_in",
              "st_miss", "st_extra"]

# legal relation codes
_NONE, _SAME, _DROP, _ADD, _SUBSET, _SUPERSET, _DIFF = range(7)


def _legal(name):
    return [t for t in name.split() if t in LEGAL_SET]


def _fuzzy_in(t, pool):
    if t in pool:
        return True
    for u in pool:
        if abs(len(u) - len(t)) <= 2 and fuzz.ratio(t, u) >= 80:
            return True
        if len(t) >= 3 and len(u) >= 3 and (u.startswith(t) or t.startswith(u)):
            return True
    return False


_NUM = re.compile(r"(?<![\w])\d+(?:\s?[-/]\s?\d+)*[a-z]?(?!\w)|(?<=[a-z])\d+(?:[-/]\d+)*(?!\w)")


def _numbers(raw):
    """Composite numbers of a raw address, in order: '9-7', '137/17', '4809b'. Ordinals (1st, 2nd) are skipped."""
    s = raw.lower()
    return [re.sub(r"\s", "", m.group()) for m in _NUM.finditer(s)
            if not re.match(r"(st|nd|rd|th)", s[m.end():m.end() + 3])]


def _as_int(x):
    """Value of the last digit group ('9-12' -> 12), so composite door numbers compare by their unit part."""
    d = re.findall(r"\d+", x)
    return int(d[-1][:9]) if d else None


def _one(a, b):
    name1, core1, raw1, toks1 = a
    name2, core2, raw2, toks2 = b
    f = []
    # ---- legal form ------------------------------------------------------------------------------
    l1, l2 = _legal(name1), _legal(name2)
    s1, s2 = set(l1), set(l2)
    if not s1 and not s2:
        rel = _NONE
    elif s1 == s2:
        rel = _SAME
    elif not s2:
        rel = _DROP
    elif not s1:
        rel = _ADD
    elif s2 < s1:
        rel = _SUBSET
    elif s2 > s1:
        rel = _SUPERSET
    else:
        rel = _DIFF
    f += [rel, len(s1), len(s2), int(l1 == l2)]
    # ---- name tokens (legal forms removed) ----------------------------------------------------------
    t1, t2 = core1.split(), core2.split()
    miss = [t for t in t1 if not _fuzzy_in(t, t2)]
    extra = [t for t in t2 if not _fuzzy_in(t, t1)]
    f += [len(miss), len(extra), min(len(miss), len(extra)),
          int(bool(t1) and bool(miss) and miss[0] == t1[0]),
          int([t for t in t1 if t in t2] == [t for t in t2 if t in t1])]
    # ---- house number (composite numbers from the raw address) ----------------------------------------
    c1, c2 = _numbers(raw1), _numbers(raw2)
    h1 = c1[0] if c1 else ""
    f += [int(bool(c1)), int(bool(c2))]
    if h1 and c2:
        h2 = min(c2, key=lambda x: (Levenshtein.distance(h1, x), abs(len(x) - len(h1))))   # closest cand number
        v1, v2 = _as_int(h1), _as_int(h2)
        ad = abs(v1 - v2) if v1 is not None and v2 is not None else -1
        rd = ad / max(v1, v2, 1) if ad >= 0 else -1
        f += [int(h1 in c2), ad, rd, Levenshtein.distance(h1, h2),
              int(h1 != h2 and (h1.startswith(h2) or h2.startswith(h1))), int(len(h1) == len(h2))]
        v1s = [_as_int(x) for x in c1]
        diffs = [abs(a - b) for a in v1s for b in map(_as_int, c2) if a is not None and b is not None]
        f += [min(diffs) if diffs else -1, len(set(c2) - set(c1)), sum(x in c2 for x in c1) / len(c1)]
    else:
        f += [-1] * 9
    # ---- street / locality words -----------------------------------------------------------------------
    w1, w2 = toks1.split(), toks2.split()
    if w1 and w2:
        f += [sum(not _fuzzy_in(t, w2) for t in w1), sum(not _fuzzy_in(t, w1) for t in w2)]
    else:
        f += [-1, -1]
    return f


def _chunk(args):
    A, B = args
    return np.array([_one(a, b) for a, b in zip(A, B)], dtype=np.float32)


FIELDS = ["name", "core", "raw_addr", "toks"]


def twin_features(s1, s23, i1, i2, step=20000) -> np.ndarray:
    """float matrix (len(i1), len(TWIN_NAMES)) of twin features for the pairs (s1 row i1[k], s23 row i2[k])."""
    A =list(zip(*[s1[c].values[i1] for c in FIELDS]))
    B = list(zip(*[s23[c].values[i2] for c in FIELDS]))
    jobs = [(A[i:i + step], B[i:i + step]) for i in range(0, len(A), step)]
    with Pool(config.N_JOBS) as pool:
        return np.vstack(pool.map(_chunk, jobs, chunksize=1))
