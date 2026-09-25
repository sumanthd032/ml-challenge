"""Learn a native-script-token -> English-token dictionary from TRAIN ground-truth pairs.

For every true pair (S1 English name, S2/S3 name containing Indic script) with the same number of
tokens, tokens are aligned by position (the data keeps word order for translated names). We count
(native_token, english_token) co-occurrences and keep the majority translation when it is supported
by >= MIN_COUNT pairs and >= MIN_SHARE of the native token's occurrences.

Output: ARTIFACTS/indic_dict.json. Used by normalize.py (falls back to rule transliteration).
Only training labels are used; no external resource.
"""
import json
import re
from collections import Counter, defaultdict

import pandas as pd

import config
from data import load_gt_pairs
from preprocess import read_source
from translit import has_indic

MIN_COUNT, MIN_SHARE = 2, 0.5
_TOK = re.compile(r"[^\wऀ-ൿ]+")


def toks(s):
    return [t for t in _TOK.sub(" ", s.lower()).split() if t]


def main():
    s1 = read_source("train", 1).set_index("entity_id").business_name
    s23 = pd.concat([read_source("train", 2), read_source("train", 3)]).set_index("entity_id").business_name
    ind = s23[s23.map(has_indic)]
    gt = load_gt_pairs()
    gt = gt[gt.cand_id.isin(ind.index)]
    print("indic positive pairs:", len(gt))
    co = defaultdict(Counter)
    for a, b in zip(s1.loc[gt.s1_id].values, ind.loc[gt.cand_id].values):
        ta, tb = toks(a), toks(b)
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if has_indic(x) and not has_indic(y):
                co[x][y] += 1
    d = {}
    for x, c in co.items():
        y, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            d[x] = y
    print("dictionary size:", len(d), "of", len(co), "native tokens seen")
    for k in list(d)[:30]:
        print(" ", k, "->", d[k])
    json.dump(d, open(config.ART_DIR / "indic_dict.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
