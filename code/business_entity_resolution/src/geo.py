"""City extraction for density features (D-018).

A city vocabulary per country is learned from Source-1 addresses, whose layout is regular ("street, city, state" with
occasional reordering): a comma component without digits that is not a state/region and occurs often enough is a city.
Any record's city is then its first comma component (normalized) found in its country's vocabulary; '' when none is.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

MIN_CITY_COUNT = 20


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bst\.", "st", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _components(addr: pd.Series) -> pd.Series:
    return addr.fillna("").map(lambda a: [_norm(x) for x in a.split(",")])


def city_vocab(s1: pd.DataFrame) -> dict:
    """country -> set of normalized city names, from S1 raw addresses.

    A state/region is a component that is the last one in most of its occurrences; a city is a digit-free component
    that directly precedes a state (the regular "street, city, state" layout) at least MIN_CITY_COUNT times."""
    out = {}
    for c, g in s1.groupby("country"):
        comps = [p for p in _components(g.raw_addr) if p]
        allc = pd.Series([x for parts in comps for x in parts if x]).value_counts()
        last = pd.Series([parts[-1] for parts in comps]).value_counts()
        ratio = (last / allc.reindex(last.index)).fillna(0)
        states = set(ratio[(ratio >= 0.5) & (last >= 50)].index)
        before = pd.Series([parts[i - 1] for parts in comps for i in range(1, len(parts))
                            if parts[i] in states and parts[i - 1] and not re.search(r"\d", parts[i - 1])])
        vc = before.value_counts()
        out[c] = set(vc[vc >= MIN_CITY_COUNT].index) - states
    return out


def cities(df: pd.DataFrame, vocab: dict) -> np.ndarray:
    """First comma component of each record's raw address that is a known city of its country ('' if none)."""
    addr = df.raw_addr.fillna("")
    comps = addr.str.split(",").explode()
    row = np.repeat(np.arange(len(df)), addr.str.count(",").values + 1)
    normed = pd.Series(comps.values).map(_norm).values
    keys = pd.Series(df.country.values[row]) + "|" + pd.Series(normed)
    keyset = {f"{c}|{v}" for c, vs in vocab.items() for v in vs}
    ok = keys.isin(keyset).values
    hit = pd.DataFrame({"row": row[ok], "city": normed[ok]}).drop_duplicates("row")
    out = np.full(len(df), "", dtype=object)
    out[hit.row.values] = hit.city.values
    return out


def city_arrays(split: str, s1: pd.DataFrame, s23: pd.DataFrame, cache_dir):
    """(S1 cities, S2/S3 cities) with the vocabulary learned from this split's S1; cached as .npy."""
    from pathlib import Path
    f = Path(cache_dir) / f"{split}_cities.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["c1"], z["c23"]
    voc = city_vocab(s1)
    c1, c23 = cities(s1, voc), cities(s23, voc)
    np.savez(f, c1=c1, c23=c23)
    return c1, c23
