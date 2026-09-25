"""EDA 03 - quantify noise: scripts, exact-normalized-name agreement, number overlap, distractors, France."""
import re
import unicodedata
import collections
import numpy as np
import pandas as pd

D = "student_resource/dataset"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def script_of(s):
    c = collections.Counter()
    for ch in s:
        if ch.isalpha():
            n = unicodedata.name(ch, "X").split(" ")[0]
            c[n] += 1
    return c.most_common(1)[0][0] if c else "NONE"


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", s))


LEGAL = set("inc llc ltd limited pvt private corp corporation co company lp llp plc group holdings".split())
key = lambda s: " ".join(sorted(t for t in norm(s).split() if t not in LEGAL))
nums = lambda s: set(re.findall(r"\d+", s))

s1 = rd(f"{D}/train/train_source1.tsv"); s2 = rd(f"{D}/train/train_source2.tsv"); s3 = rd(f"{D}/train/train_source3.tsv")
gt = rd(f"{D}/train/train_ground_truth.tsv")

for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    smp = df.sample(200000, random_state=0)
    sc = smp.business_name.map(script_of)
    print(name, "name scripts:", (sc.value_counts(normalize=True).round(4)).head(10).to_dict())
    sa = smp.business_address.map(script_of)
    print(name, "addr scripts:", (sa.value_counts(normalize=True).round(4)).head(6).to_dict())

pairs = gt[gt.matched_entity_ids != ""].sample(100000, random_state=0)
pairs = pairs.assign(m=pairs.matched_entity_ids.str.split(",")).explode("m")
recs = pd.concat([s2, s3]).set_index("entity_id")
a = s1.set_index("entity_id").loc[pairs.source1_entity_id]
b = recs.loc[pairs.m]
P = pd.DataFrame({"n1": a.business_name.values, "a1": a.business_address.values, "c": a.country.values,
                  "n2": b.business_name.values, "a2": b.business_address.values, "src": pairs.m.str[:2].values})
P["exact_key"] = [key(x) == key(y) for x, y in zip(P.n1, P.n2)]
P["latin2"] = P.n2.map(lambda s: script_of(s) in ("LATIN", "NONE", "DIGIT"))
P["addr_empty2"] = P.a2.str.strip() == ""
P["num_overlap"] = [len(nums(x) & nums(y)) > 0 if nums(x) and nums(y) else np.nan for x, y in zip(P.a1, P.a2)]
tok_j = lambda x, y: len(set(norm(x).split()) & set(norm(y).split())) / max(1, len(set(norm(x).split()) | set(norm(y).split())))
P["name_jacc"] = [tok_j(x, y) for x, y in zip(P.n1, P.n2)]
print("\npositive-pair stats by country/source:")
print(P.groupby(["c", "src"])[["exact_key", "latin2", "addr_empty2", "num_overlap", "name_jacc"]].mean().round(3))
print("name_jacc quantiles (latin only):", P[P.latin2].name_jacc.quantile([.05, .1, .25, .5]).to_dict())
print("\nLow name-jaccard latin positives:")
print(P[P.latin2 & (P.name_jacc < 0.2)].sample(25, random_state=0)[["n1", "n2", "a1", "a2"]].to_string(index=False, max_colwidth=60))

# distractors: unmatched S2 records whose normalized key equals some S1 key
matched = set(",".join(gt.matched_entity_ids).split(","))
s1k = collections.Counter(s1.business_name.map(key))
um = s2[~s2.entity_id.isin(matched)].sample(50000, random_state=0)
mt = s2[s2.entity_id.isin(matched)].sample(50000, random_state=0)
print("\nS1 name-key frequency: share of S1 whose key is shared with another S1:",
      np.mean([s1k[k] > 1 for k in s1.business_name.sample(100000, random_state=0).map(key)]))
print("unmatched S2 whose key exists in S1:", np.mean([s1k[key(x)] > 0 for x in um.business_name]))
print("matched   S2 whose key exists in S1:", np.mean([s1k[key(x)] > 0 for x in mt.business_name]))

t1 = rd(f"{D}/test/test_source1.tsv"); t2 = rd(f"{D}/test/test_source2.tsv"); t3 = rd(f"{D}/test/test_source3.tsv")
for n, df in [("T1", t1), ("T2", t2), ("T3", t3)]:
    print(f"\n{n} France samples")
    print(df[df.country == "France"].sample(12, random_state=0).to_string(index=False, max_colwidth=80))
for n, df in [("T1", t1), ("T2", t2), ("T3", t3)]:
    print(n, "per-country counts", df.country.value_counts().to_dict())
