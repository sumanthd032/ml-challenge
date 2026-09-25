"""EDA 02 - inspect matched groups, unmatched S2/S3 records, and possible ordering leakage."""
import random
import pandas as pd

D = "student_resource/dataset"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = rd(f"{D}/train/train_source1.tsv")
s2 = rd(f"{D}/train/train_source2.tsv")
s3 = rd(f"{D}/train/train_source3.tsv")
gt = rd(f"{D}/train/train_ground_truth.tsv")
allr = pd.concat([s1, s2, s3]).set_index("entity_id")
random.seed(0)

for country in ["US", "India"]:
    ids = s1[s1.country == country].entity_id.sample(12, random_state=3)
    g = gt.set_index("source1_entity_id").loc[ids]
    print(f"\n==================== {country} groups")
    for sid, row in g.iterrows():
        r = allr.loc[sid]
        print(f"\n[{sid}] {r.business_name} | {r.business_address}")
        for m in [t for t in row.matched_entity_ids.split(",") if t]:
            r = allr.loc[m]
            print(f"   {m[:2]}: {r.business_name} | {r.business_address}")

# singletons
print("\n==================== singletons")
sing = gt[gt.matched_entity_ids == ""].source1_entity_id.sample(10, random_state=1)
for sid in sing:
    r = allr.loc[sid]
    print(f"[{r.country}] {r.business_name} | {r.business_address}")

# unmatched S2/S3
matched = set(",".join(gt.matched_entity_ids).split(","))
print("\n==================== unmatched S2 / S3 samples")
for df in [s2, s3]:
    um = df[~df.entity_id.isin(matched)].sample(12, random_state=2)
    print(um.to_string(index=False, max_colwidth=80))

# ordering leakage: row position of S1 vs its matches
pos2 = pd.Series(range(len(s2)), index=s2.entity_id)
pos1 = pd.Series(range(len(s1)), index=s1.entity_id)
ex = gt[gt.matched_entity_ids != ""].sample(2000, random_state=0)
import numpy as np
d = []
for sid, ms in zip(ex.source1_entity_id, ex.matched_entity_ids):
    for m in ms.split(","):
        if m.startswith("S2"):
            d.append((pos1[sid] / len(s1), pos2[m] / len(s2)))
d = np.array(d)
print("\nrow-position corr S1 vs S2 match:", np.corrcoef(d[:, 0], d[:, 1])[0, 1])
# id numeric leakage
num = lambda x: int(x.split("-")[1])
d = [(num(sid), num(m)) for sid, ms in zip(ex.source1_entity_id, ex.matched_entity_ids) for m in ms.split(",")]
d = np.array(d, dtype=float)
print("id-number corr:", np.corrcoef(d[:, 0], d[:, 1])[0, 1])
