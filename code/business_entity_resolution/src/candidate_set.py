"""Write candidate_pairs.tsv, the candidate set that every later stage works on.

Candidate generation has two steps:

1. Blocking (blocking.py) proposes Source-2/3 records for each Source-1 entity from IDF-weighted token overlap and
   bi-encoder nearest neighbours, all within the entity's country. This recalls almost every true pair but yields
   about 89 proposals per entity.
2. A LightGBM filter (stage 1, train.py) scores each proposal from cheap similarity features computed during
   blocking. Pairs below MIN_PROB are dropped and at most MAX_PER_S1 pairs are kept per entity.

The cross-encoders and the stage-2 model score exactly this pruned set, so the final matches are a subset of it.
French entities are scored by the adapted France pipeline (france_adapt.py, split "testfr"); their stage-1 scores
replace the full-test scores.

Usage:
    python candidate_set.py [--min-prob 0.002] [--max-per-s1 50] [--out candidate_pairs.tsv]
"""
import argparse

import numpy as np
import pandas as pd

import config
from data import load_split

MIN_PROB = 0.002
MAX_PER_S1 = 50
SCORE_FILES = {"test": "test_scores_lgb_v1.parquet", "testfr": "testfr_scores_lgb_v1.parquet"}


def pruned_pairs(split: str, min_prob: float, skip_country: str | None = None) -> pd.DataFrame:
    """Stage-1 pairs of one split with probability >= min_prob, as entity ids."""
    s1, s23 = load_split(split)
    sc = pd.read_parquet(config.FEAT_DIR / SCORE_FILES[split], columns=["s1_id", "cand_id", "p"])
    sc = sc[sc.p.values >= min_prob]
    pairs = pd.DataFrame({
        "s1": s1.entity_id.values[sc.s1_id.values],
        "cand": s23.entity_id.values[sc.cand_id.values],
        "p": sc.p.values,
    })
    if skip_country is not None:
        pairs = pairs[s1.country.values[sc.s1_id.values] != skip_country]
    return pairs


def build(min_prob: float = MIN_PROB, max_per_s1: int = MAX_PER_S1) -> dict[str, str]:
    """Candidate lists per Source-1 entity, best stage-1 score first."""
    pairs = pd.concat([pruned_pairs("test", min_prob, skip_country="France"),
                       pruned_pairs("testfr", min_prob)], ignore_index=True)
    pairs = pairs.sort_values(["s1", "p"], ascending=[True, False])
    pairs = pairs[pairs.groupby("s1").cumcount() < max_per_s1]
    return pairs.groupby("s1", sort=False).cand.agg(",".join).to_dict()


def write(lists: dict[str, str], out: str) -> None:
    """One row per Source-1 entity in test order; an empty list means blocking found nothing worth scoring."""
    s1, _ = load_split("test")
    path = config.OUT_DIR / out
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1.entity_id.values:
            f.write(f"{s1_id}\t{lists.get(s1_id, '')}\n")
    sizes = np.array([len(v.split(",")) for v in lists.values()])
    print(f"wrote {path}: {len(s1):,} entities, {sizes.sum():,} candidate pairs, "
          f"{sizes.sum() / len(s1):.2f} per entity (max {sizes.max()}), {len(s1) - len(lists):,} empty")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-prob", type=float, default=MIN_PROB)
    ap.add_argument("--max-per-s1", type=int, default=MAX_PER_S1)
    ap.add_argument("--out", default="candidate_pairs.tsv")
    args = ap.parse_args()
    write(build(args.min_prob, args.max_per_s1), args.out)
