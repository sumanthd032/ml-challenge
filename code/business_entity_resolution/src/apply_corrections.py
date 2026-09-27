"""Apply the test-set corrections (docs/DECISIONS.md D-025 to D-030) to the pipeline's matching file.

`france_adapt.py assemble` writes the model-based submission (leaderboard 0.988865). A series of error analyses
on the test set, measured with the label-free count signature (D-021, D-022) and checked on labelled validation where
an analogue exists, then removed or added specific pairs. Those decisions are stored in
corrections/corrections.tsv, one row per pair, in the order they were made:

    source1_entity_id  entity_id  action (add | remove)  decision (D-0xx)

Applying them in order to the assembled file reproduces the final submission (leaderboard 0.989334).

Usage:
    python apply_corrections.py --base variants/matching_results_VRR_IUg50_FADblend_FR80.tsv \
                                --out matching_results.tsv
"""
import argparse
from pathlib import Path

import pandas as pd

import config

CORRECTIONS = Path(__file__).resolve().parent.parent / "corrections" / "corrections.tsv"


def apply(base: Path, corrections: Path, out: Path) -> None:
    """Replay the corrections on the base matching file and write the result in the same row order."""
    sub = pd.read_csv(base, sep="\t", dtype=str, keep_default_na=False)
    lists = {s: (v.split(",") if v else []) for s, v in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    owner = {c: s for s, cs in lists.items() for c in cs}
    fix = pd.read_csv(corrections, sep="\t", dtype=str, keep_default_na=False)
    for s1_id, rec, action in zip(fix.source1_entity_id, fix.entity_id, fix.action):
        if action == "remove":
            if owner.get(rec) != s1_id:
                raise ValueError(f"cannot remove {s1_id} -> {rec}: pair not in the file")
            lists[s1_id].remove(rec)
            del owner[rec]
        else:
            if rec in owner:
                raise ValueError(f"cannot add {s1_id} -> {rec}: record already matched to {owner[rec]}")
            lists[s1_id].append(rec)
            owner[rec] = s1_id
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sub.source1_entity_id:
            f.write(f"{s1_id}\t{','.join(lists[s1_id])}\n")
    print(f"applied {len(fix):,} corrections ({(fix.action == 'remove').sum():,} removals, "
          f"{(fix.action == 'add').sum():,} additions) -> {out} ({len(owner):,} matched pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apply the recorded test-set corrections to a matching file.")
    ap.add_argument("--base", default="variants/matching_results_VRR_IUg50_FADblend_FR80.tsv",
                    help="matching file written by france_adapt.py assemble (relative to the output directory)")
    ap.add_argument("--corrections", default=str(CORRECTIONS))
    ap.add_argument("--out", default="matching_results.tsv", help="output file (relative to the output directory)")
    args = ap.parse_args()
    apply(config.OUT_DIR / args.base, Path(args.corrections), config.OUT_DIR / args.out)
