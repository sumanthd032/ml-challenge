"""EDA 01 - overview of sources, ground truth structure, country mix, missingness.

Run: python eda/eda01_overview.py
"""
import collections
import pandas as pd

D = "student_resource/dataset"


def load(split, s):
    return pd.read_csv(f"{D}/{split}/{split}_source{s}.tsv", sep="\t", dtype=str,
                       keep_default_na=False, quoting=3, engine="c")


for split in ["train", "test"]:
    for s in [1, 2, 3]:
        df = load(split, s)
        print(f"\n### {split} source{s}: {len(df):,} rows, unique ids {df.entity_id.nunique():,}")
        print("country:", df.country.value_counts().to_dict())
        for c in ["business_name", "business_address"]:
            empty = (df[c].str.strip() == "").mean()
            nul = df[c].str.lower().isin(["null", "none", "nan", "n/a"]).mean()
            print(f"  {c}: empty={empty:.3%} null-literal={nul:.3%} mean_len={df[c].str.len().mean():.1f}")
        print("  sample:")
        print(df.sample(8, random_state=1).to_string(index=False, max_colwidth=90))

gt = pd.read_csv(f"{D}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
lst = gt.matched_entity_ids.map(lambda x: [t for t in x.split(",") if t])
n = lst.map(len)
print("\n### ground truth")
print("rows", len(gt), "singletons", (n == 0).mean())
print("matches per S1 distribution:", n.value_counts().sort_index().head(20).to_dict())
n2 = lst.map(lambda l: sum(t.startswith("S2") for t in l))
n3 = lst.map(lambda l: sum(t.startswith("S3") for t in l))
print("S2 per S1:", n2.value_counts().sort_index().head(15).to_dict())
print("S3 per S1:", n3.value_counts().sort_index().head(15).to_dict())
allm = [t for l in lst for t in l]
c = collections.Counter(allm)
print("total matched ids", len(allm), "unique", len(c), "ids matched to >1 S1:", sum(v > 1 for v in c.values()))
s2 = load("train", 2); s3 = load("train", 3)
print("S2 records matched:", len(set(allm) & set(s2.entity_id)) / len(s2))
print("S3 records matched:", len(set(allm) & set(s3.entity_id)) / len(s3))
