"""Discrete pair types from the generator's noise rules (D-018).

A pair type is (name relation, house-number relation, street relation). The types follow what train labels show the
generator does to a true record (typos, accents, legal-form edits, truncated / zero-padded / suffixed house numbers,
street abbreviations) versus what a twin of another business looks like (a swapped name word, another house number
of the same length, another street). Types are language-agnostic, so per-type true rates learned on India/US
validation carry over to unseen countries, unlike similarity scores whose scale shifts with the vocabulary.
"""
import re
from multiprocessing import Pool

import numpy as np
from rapidfuzz import fuzz

import config

NAME_TYPES = ["eq", "fuzzy_eq", "extra", "missing", "swap1", "other", "empty"]
NUM_TYPES = ["eq", "noise", "diff", "c_none", "s1_none", "none"]
ST_TYPES = ["eq", "c_sub", "overlap", "diff", "c_empty", "s1_empty"]
_NUM = re.compile(r"\d+[a-z]?")


def _num(raw: str) -> str:
    m = _NUM.search(raw.lower())
    return m.group() if m else ""


def _name_rel(c1: str, c2: str) -> int:
    if not c2:
        return 6
    if c1 == c2:
        return 0
    if fuzz.ratio(c1, c2) >= 90 or c1.replace(" ", "") == c2.replace(" ", ""):
        return 1
    t1, t2 = c1.split(), c2.split()
    s1, s2 = set(t1), set(t2)
    if s1 < s2:
        return 2
    if s2 < s1:
        return 3
    if len(t1) == len(t2) and len(t1) > 1:
        diff = [(a, b) for a, b in zip(t1, t2) if a != b]
        if len(diff) == 1 and fuzz.ratio(*diff[0]) < 75:
            return 4
    return 5


def _num_rel(h1: str, h2: str) -> int:
    if not h1 and not h2:
        return 5
    if not h2:
        return 3
    if not h1:
        return 4
    if h1 == h2:
        return 0
    d1, d2 = re.sub(r"\D", "", h1), re.sub(r"\D", "", h2)
    a, b = d1.lstrip("0"), d2.lstrip("0")
    # generator noise on a true record: zero padding, letter suffix, a dropped leading/trailing digit
    if a == b or (a and b and (a.startswith(b) or a.endswith(b) or b.startswith(a) or b.endswith(a))
                  and abs(len(a) - len(b)) <= 1):
        return 1
    return 2


def _st_rel(w1: str, w2: str) -> int:
    if not w2:
        return 4
    if not w1:
        return 5
    s1, s2 = set(w1.split()), set(w2.split())
    if s1 == s2:
        return 0
    if s2 <= s1:
        return 1
    inter = len(s1 & s2) / max(1, min(len(s1), len(s2)))
    return 2 if inter >= 0.5 else 3


def _chunk(args):
    A, B = args
    out = np.empty((len(A), 3), dtype=np.int8)
    for i, ((c1, r1, w1), (c2, r2, w2)) in enumerate(zip(A, B)):
        out[i] = (_name_rel(c1, c2), _num_rel(_num(r1), _num(r2)), _st_rel(w1, w2))
    return out


def pair_types(s1, s23, i1, i2, step=50000) -> np.ndarray:
    """(n, 3) int8: name, number, street relation codes (see NAME_TYPES, NUM_TYPES, ST_TYPES)."""
    A = list(zip(s1.core.values[i1], s1.raw_addr.values[i1], s1.toks.values[i1]))
    B = list(zip(s23.core.values[i2], s23.raw_addr.values[i2], s23.toks.values[i2]))
    jobs = [(A[i:i + step], B[i:i + step]) for i in range(0, len(A), step)]
    with Pool(config.N_JOBS) as pool:
        return np.vstack(pool.map(_chunk, jobs, chunksize=1))


def type_key(t: np.ndarray) -> np.ndarray:
    return t[:, 0].astype(np.int16) * 100 + t[:, 1] * 10 + t[:, 2]


def type_label(k: int) -> str:
    return f"{NAME_TYPES[k // 100]}/{NUM_TYPES[(k // 10) % 10]}/{ST_TYPES[k % 10]}"
