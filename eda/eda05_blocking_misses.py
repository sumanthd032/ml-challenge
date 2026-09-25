"""EDA 05 - inspect true pairs missed by blocking (sample run)."""
import sys
import pandas as pd
sys.path.insert(0, "code/business_entity_resolution/src")
import config
from data import load_split, load_gt_pairs

c = pd.read_parquet(config.ART_DIR / "train_cands_A_s10.parquet")
s1, s23 = load_split("train")
truth = load_gt_pairs()
ids = set(c.s1_id)
t = truth[truth.s1_id.isin(ids)]
m = t.merge(c[["s1_id", "cand_id"]].assign(hit=1), how="left", on=["s1_id", "cand_id"])
miss = m[m.hit.isna()]
print("missed", len(miss), "of", len(t))
a = s1.set_index("entity_id"); b = s23.set_index("entity_id")
x = miss.sample(40, random_state=0)
for s, cid in zip(x.s1_id, x.cand_id):
    r1, r2 = a.loc[s], b.loc[cid]
    print(f"\nS1: {r1.raw_name} | {r1.raw_addr}\n    core={r1.core} toks={r1.toks} nums={r1.nums}")
    print(f"{cid[:2]}: {r2.raw_name} | {r2.raw_addr}\n    core={r2.core} toks={r2.toks} nums={r2.nums}")
mm = miss.merge(b[["indic", "toks", "nums"]], left_on="cand_id", right_index=True)
print("\nmissed: indic share", mm.indic.mean(), " empty-addr share", ((mm.toks == "") & (mm.nums == "")).mean())
allp = t.merge(b[["indic", "toks", "nums", "country"]], left_on="cand_id", right_index=True)
print("all:    indic share", allp.indic.mean(), " empty-addr share", ((allp.toks == "") & (allp.nums == "")).mean())
print("miss rate by country:", 1 - allp.assign(h=allp.set_index(["s1_id","cand_id"]).index.isin(c.set_index(["s1_id","cand_id"]).index)).groupby("country").h.mean())
