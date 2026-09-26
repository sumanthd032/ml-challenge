"""Submission from saved test scores with a score file and threshold chosen per S1 country (D-015, D-016).

Blocking is within country, so every candidate pair has both sides in one country and deciding each country
separately is identical to deciding over the whole test set.

    python country_mix.py --spec India=test_scores2_v1cel12x.parquet:0.7 US=test_scores2_v1cel12x.parquet:0.7 \
        France=test_scores2_v1cex.parquet:0.85 --out variants/matching_results_mix.tsv
    Score files are read from --scores-dir (default FEAT_DIR); --out is relative to OUT_DIR.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import config
from data import load_split
from decide import compete, expected_f_select
from predict import write_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True, help="Country=scores.parquet:thr")
    ap.add_argument("--scores-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compete", type=float, default=0.0, help="candidate competition weight (decide.compete, D-017)")
    a = ap.parse_args()
    sdir = Path(a.scores_dir) if a.scores_dir else config.FEAT_DIR

    s1, s23 = load_split("test")
    country = s1.country.values
    spec = {}
    for item in a.spec:
        c, rest = item.split("=")
        f, thr = rest.rsplit(":", 1)
        spec[c] = (f, float(thr))
    missing = set(s1.country.unique()) - set(spec)
    if missing:
        raise SystemExit(f"no spec for countries {missing}")

    parts, stats = [], {}
    for f in sorted({f for f, _ in spec.values()}):
        scored = pd.read_parquet(sdir / f, columns=["s1_id", "cand_id", "p"])
        cty = country[scored.s1_id.values]
        for c, (fc, thr) in spec.items():
            if fc != f:
                continue
            part = scored[cty == c]
            if a.compete:
                part = compete(part, a.compete)
            sel = expected_f_select(part, mode="thr", thr=thr)
            parts.append(sel)
            stats[c] = {"scores": f, "thr": thr, "pairs": len(sel), "pairs_per_s1": round(len(sel) / (country == c).sum(), 4)}
        del scored
    sel = pd.concat(parts, ignore_index=True)
    res = write_lists(sel.s1_id.values, sel.cand_id.values, s1, s23, "matched_entity_ids", config.OUT_DIR / a.out)
    stats["total"] = {"pairs": len(sel), "s1_with_match": int((res.matched_entity_ids != "").sum())}
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
