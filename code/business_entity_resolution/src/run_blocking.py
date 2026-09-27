"""Run blocking for a split and cache candidates. On train, report recall vs ground truth.

Usage: python run_blocking.py --split train [--sample-frac 0.2]
"""
import argparse
import time

import config
from blocking import run_blocking
from data import load_split, load_gt_pairs
from metrics import blocking_recall


def main():
    """Run pass A, write <split>_cands_A[_sNN].parquet to ART_DIR, and on train print recall by rank cutoff."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="subsample S1 (train dev runs)")
    args = ap.parse_args()
    t = time.time()
    s1, s23 = load_split(args.split)
    if args.sample_frac < 1:
        s1 = s1.sample(frac=args.sample_frac, random_state=config.SEED).reset_index(drop=True)
    print(f"loaded {len(s1):,} S1 / {len(s23):,} S2+S3 in {time.time()-t:.0f}s", flush=True)
    cands = run_blocking(s1, s23)
    suffix = "" if args.sample_frac >= 1 else f"_s{int(args.sample_frac * 100)}"
    out = config.ART_DIR / f"{args.split}_cands_A{suffix}.parquet"
    cands.to_parquet(out, index=False)
    print(f"saved {len(cands):,} candidate pairs -> {out} ({time.time()-t:.0f}s total)")
    if args.split == "train":
        truth = load_gt_pairs()
        print("recall all:", blocking_recall(cands, truth, set(s1.entity_id)))
        for k in (5, 10, 20, 40):
            sub = cands[(cands.blk_rank_fwd < k) | (cands.blk_rank_rev < 99)]
            print(f"  fwd<{k} + rev:", blocking_recall(sub, truth, set(s1.entity_id)))
        sub = cands[cands.blk_rank_fwd < 99]
        print("  fwd only:", blocking_recall(sub, truth, set(s1.entity_id)))


if __name__ == "__main__":
    main()
