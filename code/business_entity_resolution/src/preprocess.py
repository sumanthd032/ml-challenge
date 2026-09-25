"""Step 1 - normalize every source file once and cache the result as TSV in ARTIFACTS.

Usage: python preprocess.py [--splits train test]
"""
import argparse
import csv
import time
from multiprocessing import Pool

import pandas as pd

import config
from normalize import norm_name, norm_address
from translit import has_indic

COLS = ["entity_id", "country", "raw_name", "raw_addr", "name", "core", "concat", "alt",
        "addr", "toks", "nums", "state", "indic"]


def read_source(split: str, s: int) -> pd.DataFrame:
    """Read a raw source TSV exactly as provided (strings only, no NA coercion)."""
    return pd.read_csv(config.source_path(split, s), sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE)


def _norm_chunk(rows):
    out = []
    for eid, name, addr, country in rows:
        n = norm_name(name)
        a = norm_address(addr)
        out.append((eid, country, name, addr, n["name"], n["core"], n["concat"], n["alt"],
                    a["addr"], a["toks"], a["nums"], a["state"], int(has_indic(name))))
    return out


def normalize_df(df: pd.DataFrame, pool: Pool) -> pd.DataFrame:
    rows = list(zip(df.entity_id, df.business_name, df.business_address, df.country))
    chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    res = [r for part in pool.imap(_norm_chunk, chunks) for r in part]
    return pd.DataFrame(res, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    with Pool(config.N_JOBS) as pool:
        for split in args.splits:
            for s in (1, 2, 3):
                t = time.time()
                df = normalize_df(read_source(split, s), pool)
                # raw text may contain tabs/quotes? write with QUOTE_NONE and escape-free (tabs never occur in values)
                df.to_csv(config.norm_path(split, s), sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
                print(f"{split} s{s}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
