"""Blend two submission files per S1 country: take each S1's row from --base, except S1s whose country is in
--countries, which come from --alt. Used when a model helps some countries but not others (D-015).

    python blend_country.py --base variants/matching_results_cel12x_thr70.tsv \
        --alt variants/matching_results_cex_thr70.tsv --countries France \
        --out variants/matching_results_cel12x_cexFR_thr70.tsv        (paths relative to output/)
"""
import argparse

import pandas as pd

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    country = pd.read_csv(config.norm_path("test", 1), sep="\t", usecols=["entity_id", "country"],
                          dtype=str, keep_default_na=False).set_index("entity_id")["country"]
    unknown = set(a.countries) - set(country.unique())
    if unknown:
        raise SystemExit(f"unknown countries {unknown}; have {sorted(country.unique())}")

    read = lambda p: pd.read_csv(config.OUT_DIR / p, sep="\t", dtype=str, keep_default_na=False)  # noqa: E731
    base, alt = read(a.base), read(a.alt)
    col = base.columns[0]
    in_alt = lambda df: df[col].map(country).isin(a.countries)  # noqa: E731
    out = pd.concat([base[~in_alt(base)], alt[in_alt(alt)]], ignore_index=True)

    n_pairs = lambda df: df[df.columns[1]].str.count(",").add(df[df.columns[1]].ne("")).sum()  # noqa: E731
    print(f"base {len(base)} rows / {n_pairs(base)} pairs, alt {len(alt)} / {n_pairs(alt)}, "
          f"out {len(out)} / {n_pairs(out)} ({in_alt(out).sum()} rows from alt)")
    out.to_csv(config.OUT_DIR / a.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
