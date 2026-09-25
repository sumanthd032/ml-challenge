"""EDA 04 - token-level noise: which tokens S2/S3 add vs S1, special chars, address vocab, country agreement."""
import re
import collections
import pandas as pd

D = "student_resource/dataset"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = rd(f"{D}/train/train_source1.tsv").set_index("entity_id")
s2 = rd(f"{D}/train/train_source2.tsv"); s3 = rd(f"{D}/train/train_source3.tsv")
recs = pd.concat([s2, s3]).set_index("entity_id")
gt = rd(f"{D}/train/train_ground_truth.tsv")
pairs = gt[gt.matched_entity_ids != ""].sample(300000, random_state=0)
pairs = pairs.assign(m=pairs.matched_entity_ids.str.split(",")).explode("m")
a = s1.loc[pairs.source1_entity_id]; b = recs.loc[pairs.m]
print("country agreement:", (a.country.values == b.country.values).mean())

tok = lambda s: re.findall(r"[^\W_]+", s.lower())
added = collections.Counter(); removed = collections.Counter()
for n1, n2 in zip(a.business_name.values, b.business_name.values):
    t1, t2 = set(tok(n1)), set(tok(n2))
    added.update(t2 - t1); removed.update(t1 - t2)
print("\nTop tokens ADDED in S2/S3 names vs S1:", added.most_common(120))
print("\nTop tokens REMOVED (in S1 not in S2/S3):", removed.most_common(80))

chars = collections.Counter(ch for n in pd.concat([s2, s3]).business_name.sample(300000, random_state=0)
                            for ch in n if not ch.isalnum() and ch != " ")
print("\nnon-alnum chars in S2/S3 names:", chars.most_common(40))
chars = collections.Counter(ch for n in pd.concat([s2, s3]).business_address.sample(300000, random_state=0)
                            for ch in n if not ch.isalnum() and ch != " ")
print("non-alnum chars in S2/S3 addr:", chars.most_common(40))

# address token transformations
added = collections.Counter(); removed = collections.Counter()
for n1, n2 in zip(a.business_address.values, b.business_address.values):
    t1, t2 = set(tok(n1)), set(tok(n2))
    if not t2: continue
    added.update(t2 - t1); removed.update(t1 - t2)
print("\nTop addr tokens ADDED:", added.most_common(120))
print("\nTop addr tokens REMOVED:", removed.most_common(120))

# name prefix/suffix junk patterns
pat = collections.Counter()
for n in pd.concat([s2, s3]).business_name.sample(300000, random_state=1):
    m = re.match(r"^([^\w\s]+)", n)
    if m: pat["pre:" + m.group(1)] += 1
    m = re.search(r"([^\w\s]+)$", n)
    if m: pat["suf:" + m.group(1)] += 1
    for m in re.findall(r"#\d+|\[[^\]]*\]|\([^)]*\)|\bt/a\b|\bd/b/a\b|\bdba\b|www\.|\.com|@\w+", n, flags=re.I):
        pat["pat:" + re.sub(r"\d+", "9", m.lower())[:20]] += 1
print("\nname junk patterns:", pat.most_common(60))
