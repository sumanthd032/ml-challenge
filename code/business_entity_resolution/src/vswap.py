"""France 'vocabulary swap' filter (D-021).

A selected pair is a vocabulary swap when the candidate's core name differs from the S1's core in exactly one token
(same token count, the two tokens not a typo of each other: similarity < 75) AND the swapped-in token w is ordinary
S1-name vocabulary rather than generator noise. Noise ratio R(w) = (share of the country's S2/S3 cores containing w)
/ (share of its S1 cores containing w), over words in >= 200 S2/S3 cores: generator noise words are added to
S2/S3 names only (fils 1.54, services 2.38, developpement 10.7, associes 77), vocabulary words occur at the S1 rate
(club 0.84, amicale 0.86, comite 0.87, ecole 0.83). R in [0.5, 1.1) = vocabulary.

Evidence that these France pairs are false (docs/DECISIONS.md D-021): they explain the drop from submission 8 to 9;
their per-source count signature (an S1's other same-source matches / its other-source matches) is 0.934 / 1.078,
exactly what records unrelated to the S1 give, vs 0.761 / 0.876 for true pairs, at every score level incl. p>=0.999;
they are 38-85% of France one-word swaps vs 1-2% in India/US, where one-word swaps are generator noise (98.9% true).
Found by an analysis agent (C:/Users/PC-07/Downloads/ber_scratch/agent_fr), rule unchanged.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def noise_ratio(s1: pd.DataFrame, s23: pd.DataFrame, min_df: int = 200) -> pd.Series:
    def df(cores):
        return pd.Series([t for c in cores for t in set(c.split())]).value_counts()
    d1, d23 = df(s1.core.values), df(s23.core.values)
    r = (d23 / len(s23)) / (d1.reindex(d23.index) / len(s1))
    return r[(d23 >= min_df) & r.notna()]


def vocab_swap_mask(i1, i2, s1, s23, R, lo=0.5, hi=1.1) -> np.ndarray:
    a, b = s1.core.values[i1], s23.core.values[i2]
    out = np.zeros(len(a), dtype=bool)
    for k, (x, y) in enumerate(zip(a, b)):
        t1, t2 = x.split(), y.split()
        if len(t1) != len(t2) or len(t1) < 2 or x == y or sorted(t1) == sorted(t2):
            continue
        if fuzz.ratio(x, y) >= 90 or x.replace(" ", "") == y.replace(" ", ""):
            continue                                   # a typo, not a swap
        s_1, s_2 = set(t1), set(t2)
        if s_1 < s_2 or s_2 < s_1:
            continue
        diff = [(p, q) for p, q in zip(t1, t2) if p != q]
        if len(diff) != 1 or fuzz.ratio(*diff[0]) >= 75:
            continue
        r = R.get(diff[0][1], np.nan)
        out[k] = lo <= r < hi
    return out
