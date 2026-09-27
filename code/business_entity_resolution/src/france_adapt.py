"""France-only test pipeline with a France-adapted bi-encoder (D-018).

The stage-1 model takes 77% of its gain from two bi-encoder features (emb_rank_rev, cd_emb_cos_2nd_gap). The
bi-encoder is an English MiniLM fine-tuned on India/US pairs only; on French records it packs businesses close
together (obvious France pairs: 2nd-best gap 0.23 vs 0.37-0.43 India/US, 2.2 SD lower), so stage 1 cannot separate
true matches from twins there. This script reruns the test pipeline on France alone ('testfr' split) with the
bi-encoder adapted to French by contrastive fine-tuning on near-certain France matches (self-training; no labels,
no external data) and with the feature IDF put on the same scale as in training (BER_IDF_SCALE=2).

  python france_adapt.py split      # testfr_s{1,2,3}_norm.tsv, testfr_cands_A.parquet (France part of pass A)
  python france_adapt.py pseudo     # FEAT_DIR/testfr_pseudo_pairs.parquet from the current test scores
  python embed.py finetune_pseudo --split testfr --pairs <FEAT_DIR>/testfr_pseudo_pairs.parquet --model-dir biencoder_fr
  python embed.py knn --split testfr --model-dir biencoder_fr
  python france_adapt.py stage1     # feature store + stage-1 scores (BER_IDF_SCALE=2)
  python crossenc.py score --split testfr ; python crossenc.py score --split testfr --tag l12 --base ...
  python stage2.py apply --tag v1 --ce --x --split testfr
  python france_adapt.py assemble --fr-scores testfr_scores2_v1cex.parquet --fr-thr 0.85 --out ...
"""
import argparse
import csv
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from data import load_split

import os

COUNTRY = "France"
SPLIT = os.environ.get("BER_FR_SPLIT", "testfr")      # testfr2 = France rows re-normalized with fr_fix (D-019)


def _renorm_chunk(rows):
    from normalize import norm_address, norm_name
    from translit import has_indic
    out = []
    for eid, name, addr, country in rows:
        n = norm_name(name, fr_fix=True)
        a = norm_address(addr, fr_fix=True)
        out.append((eid, country, name, addr, n["name"], n["core"], n["concat"], n["alt"],
                    a["addr"], a["toks"], a["nums"], a["state"], int(has_indic(name))))
    return out


def renorm():
    """France rows of the raw test sources normalized with the France fixes, written as split SPLIT (same row
    order as 'testfr', so pseudo-pair positions carry over)."""
    from multiprocessing import Pool
    from preprocess import COLS, read_source
    with Pool(config.N_JOBS) as pool:
        for s in (1, 2, 3):
            df = read_source("test", s)
            df = df[df.country == COUNTRY]
            rows = list(zip(df.entity_id, df.business_name, df.business_address, df.country))
            res = [r for part in pool.imap(_renorm_chunk, [rows[i:i + 20000] for i in range(0, len(rows), 20000)])
                   for r in part]
            out = pd.DataFrame(res, columns=COLS)
            out.to_csv(config.norm_path(SPLIT, s), sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
            print(f"{SPLIT} s{s}: {len(out):,} rows (state set: {(out.state != '').mean():.3f})", flush=True)


def make_split():
    for s in (1, 2, 3):
        src = config.norm_path("test", s)
        df = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, escapechar="\\")
        df = df[df.country == COUNTRY]
        df.to_csv(config.norm_path(SPLIT, s), sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
        print(f"s{s}: {len(df):,} {COUNTRY} rows", flush=True)
    s1, _ = load_split(SPLIT)
    keep = set(s1.entity_id)
    parts = []
    for b in pq.ParquetFile(config.ART_DIR / "test_cands_A.parquet").iter_batches(batch_size=8_000_000):
        d = b.to_pandas()
        parts.append(d[d.s1_id.isin(keep)])
    a = pd.concat(parts, ignore_index=True)
    a.to_parquet(config.ART_DIR / f"{SPLIT}_cands_A.parquet", index=False)
    print(f"pass A pairs: {len(a):,}")


def pseudo(p2=0.995, p1=0.98):
    """Near-certain France matches: both stage-2 models >= p2 and stage 1 >= p1, one S1 per candidate."""
    t1, t23 = load_split("test")
    a = pd.read_parquet(config.FEAT_DIR / "test_scores2_v1cex.parquet", columns=["s1_id", "cand_id", "p"])
    b = pd.read_parquet(config.FEAT_DIR / "test_scores2_v1cel12x.parquet", columns=["p"]).p.values
    c = pd.read_parquet(config.FEAT_DIR / "test_scores_lgb_v1.parquet", columns=["p"]).p.values
    fr = t1.country.values[a.s1_id.values] == COUNTRY
    m = fr & (a.p.values >= p2) & (b >= p2) & (c >= p1)
    sel = a[m]
    sel = sel[~sel.cand_id.duplicated(keep=False)]                 # candidate claimed by one S1 only
    f1, f23 = load_split(SPLIT)
    r1 = pd.Index(f1.entity_id).get_indexer(t1.entity_id.values[sel.s1_id.values])
    r2 = pd.Index(f23.entity_id).get_indexer(t23.entity_id.values[sel.cand_id.values])
    assert (r1 >= 0).all() and (r2 >= 0).all()
    out = pd.DataFrame({"i1": r1.astype(np.int32), "i2": r2.astype(np.int32)})
    out.to_parquet(config.FEAT_DIR / f"{SPLIT}_pseudo_pairs.parquet", index=False)
    print(f"pseudo pairs: {len(out):,} over {out.i1.nunique():,} S1s ({len(out) / fr.sum():.3f} of France pairs)")


def pseudo2(files=("scores2_v1cexd.parquet", "fr3_scores2_v1cexd.parquet"), p2=0.99, p1=0.9):
    """Round-2 pseudo positives from this split's own adapted pipeline (D-019): both stage-2 variants >= p2 and the
    adapted stage 1 >= p1, one S1 per candidate. Richer than round 1 (which came from the English-only models):
    acronyms, domains, dotted legal forms and brand names at the same address are now confident."""
    a = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_{files[0]}", columns=["s1_id", "cand_id", "p"])
    b = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_{files[1]}", columns=["p"]).p.values
    c = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_scores_lgb_v1.parquet", columns=["p"]).p.values
    m = (a.p.values >= p2) & (b >= p2) & (c >= p1)
    sel = a[m]
    sel = sel[~sel.cand_id.duplicated(keep=False)]
    out = pd.DataFrame({"i1": sel.s1_id.values.astype(np.int32), "i2": sel.cand_id.values.astype(np.int32)})
    old = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_pseudo_pairs.parquet")
    n2 = np.int64(10 ** 9)
    new = ~np.isin(out.i1.values.astype(np.int64) * n2 + out.i2.values, old.i1.values.astype(np.int64) * n2 + old.i2.values)
    out.to_parquet(config.FEAT_DIR / f"{SPLIT}_pseudo2_pairs.parquet", index=False)
    print(f"round-2 pseudo pairs: {len(out):,} ({new.sum():,} not in round 1; round 1 had {len(old):,})")


def stage1(model="lgb_v1.txt"):
    import lightgbm as lgb
    import store
    t = time.time()
    s1, s23 = load_split(SPLIT)
    pairs = store.build(SPLIT, s1, s23)
    mdl = lgb.Booster(model_file=str(config.ART_DIR / model))
    cols = mdl.feature_name()
    s3 = store.s3_flags(s23)
    p = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), store.SHARD):
        X = store.load_rows(SPLIT, pairs, s3, np.arange(s, min(s + store.SHARD, len(pairs))))
        p[s:s + len(X)] = mdl.predict(X[cols], num_threads=config.N_JOBS)
    out = pd.DataFrame({"s1_id": pairs.i1.values, "cand_id": pairs.i2.values, "p": p})
    out.to_parquet(config.FEAT_DIR / f"{SPLIT}_scores_{model.removesuffix('.txt')}.parquet", index=False)
    print(f"stage 1: {len(out):,} pairs, {int((p >= 0.7).sum()):,} at p >= 0.7 ({time.time() - t:.0f}s)")


def ce_pairs(neg_p=0.02, max_neg_per_pos=2.0):
    """Cross-encoder self-training pairs for France: the pseudo positives, plus hard negatives that need no label:
    a candidate that is a near-certain match of S1 X is a negative for every other S1 listing it (each record
    belongs to at most one S1). Negatives are the other S1s' pairs of those candidates with stage-1 p >= neg_p."""
    pos = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_pseudo_pairs.parquet")
    sc = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_scores_lgb_v1.parquet")
    owner = pd.Series(pos.i1.values, index=pos.i2.values)
    m = sc.cand_id.isin(owner.index) & (sc.p.values >= neg_p)
    neg = sc[m]
    neg = neg[neg.s1_id.values != owner.reindex(neg.cand_id.values).values]
    rng = np.random.default_rng(config.SEED)
    cap = int(max_neg_per_pos * len(pos))
    if len(neg) > cap:                                   # keep the hardest (highest stage-1 p) first
        neg = neg.sort_values("p", ascending=False).iloc[:cap]
    out = pd.concat([pd.DataFrame({"i1": pos.i1.values, "i2": pos.i2.values, "label": 1}),
                     pd.DataFrame({"i1": neg.s1_id.values, "i2": neg.cand_id.values, "label": 0})], ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out.to_parquet(config.FEAT_DIR / f"{SPLIT}_ce_pairs.parquet", index=False)
    print(f"ce pairs: {len(out):,} ({(out.label == 1).sum():,} pos, {(out.label == 0).sum():,} neg; "
          f"neg stage-1 p median {neg.p.median():.3f}, share >= 0.5: {(neg.p >= 0.5).mean():.3f})")


def ce_pairs_nn(n_neg=2, k_nn=5, p1_new=0.9, type_filter=False, pairs_name="pseudo_pairs"):
    """Cross-encoder self-training set with near-twin negatives (D-018).

    Positives: pseudo pairs (A, c) that the adapted stage 1 also accepts (p >= p1_new) and whose edit looks like
    generator noise (no swapped name word, no other house number), so the old models' twin errors stay out.
    Negatives: (B, c) for n_neg of A's k_nn nearest other S1s B in the adapted embedding space; c belongs to A,
    so (B, c) is certainly false, and B is A's closest look-alike, i.e. the twin the cross-encoder must reject."""
    import torch
    import pairtype as pt
    from embed import gpu_topk
    s1, s23 = load_split(SPLIT)
    pos = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_{pairs_name}.parquet")
    sc = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_scores_lgb_v1.parquet")
    n2 = np.int64(len(s23))
    k_sc = sc.s1_id.values.astype(np.int64) * n2 + sc.cand_id.values
    order = np.argsort(k_sc)
    k_pos = pos.i1.values.astype(np.int64) * n2 + pos.i2.values
    j = np.minimum(np.searchsorted(k_sc[order], k_pos), len(order) - 1)
    p_new = np.where(k_sc[order][j] == k_pos, sc.p.values[order][j], 0.0)
    keep = p_new >= p1_new
    if type_filter:
        # off by default: on labelled val, best-S1 pairs with one swapped name word at the same number + street are
        # 98.9% true, and same name with another house number on the same street 79.5% (generic-word and number
        # edits are generator noise), so filtering them taught the cross-encoder to reject true matches (D-018)
        ty = pt.pair_types(s1, s23, pos.i1.values, pos.i2.values)
        keep &= (ty[:, 0] != pt.NAME_TYPES.index("swap1")) & (ty[:, 1] != pt.NUM_TYPES.index("diff"))
    pos = pos[keep].reset_index(drop=True)
    E1 = np.load(config.ART_DIR / f"{SPLIT}_emb_s1.npy")
    q, c, _ = gpu_topk(E1, E1, k_nn + 1)
    nn = pd.DataFrame({"a": q, "b": c})
    nn = nn[nn.a != nn.b]
    nn["r"] = nn.groupby("a").cumcount()
    nn = nn[nn.r < k_nn]
    rng = np.random.default_rng(config.SEED)
    pick = rng.integers(0, k_nn, size=(len(pos), n_neg))
    table = np.full((len(s1), k_nn), -1, dtype=np.int64)
    table[nn.a.values, nn.r.values] = nn.b.values
    negB = table[np.repeat(pos.i1.values, n_neg), pick.ravel()]
    negC = np.repeat(pos.i2.values, n_neg)
    ok = negB >= 0
    neg = pd.DataFrame({"i1": negB[ok], "i2": negC[ok], "label": 0})
    out = pd.concat([pd.DataFrame({"i1": pos.i1.values, "i2": pos.i2.values, "label": 1}), neg], ignore_index=True)
    out = out.drop_duplicates(["i1", "i2"]).reset_index(drop=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out.to_parquet(config.FEAT_DIR / f"{SPLIT}_ce_pairs.parquet", index=False)
    print(f"ce pairs: {len(out):,} ({(out.label == 1).sum():,} pos kept of {len(keep):,}, {(out.label == 0).sum():,} neg)")
    del E1
    torch.cuda.empty_cache()


def blend(a_file, b_file, out_file):
    """Mean of two stage-2 score files of this split (D-018: variant A = original cross-encoder, variant D =
    self-trained cross-encoder). Both come from the same stage-1 file, so rows must align exactly."""
    a = pd.read_parquet(config.FEAT_DIR / a_file)
    b = pd.read_parquet(config.FEAT_DIR / b_file, columns=["s1_id", "cand_id", "p"])
    assert (a.s1_id.values == b.s1_id.values).all() and (a.cand_id.values == b.cand_id.values).all(), "rows differ"
    a["p"] = (a.p.values + b.p.values) / 2
    a.to_parquet(config.FEAT_DIR / out_file, index=False)
    print(f"blend {a_file} + {b_file} -> {out_file}: {len(a):,} rows")


def assemble(iu_scores, iu_thr, fr_scores, fr_thr, out, iu_compete=1.0, fr_compete=1.0, vswap_norm=None):
    """India/US from the full-test score file, France from the testfr one; ids mapped back to the test split.
    vswap_norm: if set (e.g. 'testfr3'), drop France vocabulary swaps (vswap.py, D-021), judged on that split's
    names (same row order as SPLIT)."""
    from decide import compete, expected_f_select
    from predict import write_lists
    t1, t23 = load_split("test")
    f1, f23 = load_split(SPLIT)
    sc = pd.read_parquet(config.FEAT_DIR / iu_scores, columns=["s1_id", "cand_id", "p"])
    sc = sc[t1.country.values[sc.s1_id.values] != COUNTRY]
    sel_iu = expected_f_select(compete(sc, iu_compete), mode="thr", thr=iu_thr)
    fr = pd.read_parquet(config.FEAT_DIR / fr_scores, columns=["s1_id", "cand_id", "p"])
    sel_fr = expected_f_select(compete(fr, fr_compete), mode="thr", thr=fr_thr)
    if vswap_norm:
        import vswap
        g1, g23 = load_split(vswap_norm)
        assert (g1.entity_id.values == f1.entity_id.values).all() and (g23.entity_id.values == f23.entity_id.values).all()
        m = vswap.vocab_swap_mask(sel_fr.s1_id.values, sel_fr.cand_id.values, g1, g23, vswap.noise_ratio(g1, g23))
        print(f"France vocabulary swaps dropped: {int(m.sum()):,} of {len(sel_fr):,} pairs "
              f"({sel_fr[m].s1_id.nunique():,} S1s)", flush=True)
        sel_fr = sel_fr[~m]
    i1 = pd.Index(t1.entity_id).get_indexer(f1.entity_id.values[sel_fr.s1_id.values])
    i2 = pd.Index(t23.entity_id).get_indexer(f23.entity_id.values[sel_fr.cand_id.values])
    sel = pd.DataFrame({"s1_id": np.concatenate([sel_iu.s1_id.values, i1]),
                        "cand_id": np.concatenate([sel_iu.cand_id.values, i2])})
    res = write_lists(sel.s1_id.values, sel.cand_id.values, t1, t23, "matched_entity_ids", config.OUT_DIR / out)
    n1 = pd.Series(t1.country.values).value_counts()
    print(json.dumps({"pairs_iu": len(sel_iu), "pairs_fr": len(sel_fr),
                      "fr_pairs_per_s1": round(len(sel_fr) / n1[COUNTRY], 4),
                      "s1_with_match": int((res.matched_entity_ids != "").sum())}, indent=1))


def candidates_file(src="candidate_pairs.tsv", out="candidate_pairs_fa.tsv"):
    """candidate_pairs.tsv for the final package: rows of India/US S1s streamed unchanged from the full-test file,
    France rows replaced by the testfr candidate set (the pairs the France models scored). Streaming keeps the
    peak memory at the France part only."""
    f1, f23 = load_split(SPLIT)
    b = pd.read_parquet(config.FEAT_DIR / f"{SPLIT}_store" / "pairs.parquet", columns=["i1", "i2"])
    ids = pd.DataFrame({"s1": f1.entity_id.values[b.i1.values], "c": f23.entity_id.values[b.i2.values]})
    fr = ids.groupby("s1").c.agg(",".join).to_dict()
    fr_all = set(f1.entity_id)
    n_fr = n = 0
    with open(config.OUT_DIR / src, encoding="utf-8") as fi, \
            open(config.OUT_DIR / out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(fi.readline())
        for line in fi:
            s1_id = line.split("\t", 1)[0]
            n += 1
            if s1_id in fr_all:
                fo.write(f"{s1_id}\t{fr.get(s1_id, '')}\n")
                n_fr += 1
            else:
                fo.write(line)
    print(f"wrote {out}: {n:,} rows, {n_fr:,} France rows replaced ({len(ids):,} France candidate pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "renorm", "pseudo", "pseudo2", "stage1", "ce_pairs", "ce_pairs_nn",
                                    "ce_pairs_nn2", "blend", "assemble", "candidates"])
    ap.add_argument("--blend", nargs=3, metavar=("A", "B", "OUT"), help="blend: two score files and the output")
    ap.add_argument("--iu-scores", default="test_scores2_v1cel12x.parquet")
    ap.add_argument("--iu-thr", type=float, default=0.7)
    ap.add_argument("--fr-scores", default="testfr_scores2_v1cex.parquet")
    ap.add_argument("--fr-thr", type=float, default=0.85)
    ap.add_argument("--out", default="variants/matching_results_fradapt.tsv")
    ap.add_argument("--iu-compete", type=float, default=1.0)
    ap.add_argument("--fr-compete", type=float, default=1.0)
    ap.add_argument("--cand-out", default="candidate_pairs_fa.tsv")
    ap.add_argument("--vswap-norm", default=None, help="drop France vocabulary swaps judged on this split's names")
    a = ap.parse_args()
    {"split": make_split, "renorm": renorm, "pseudo": pseudo, "pseudo2": pseudo2, "stage1": stage1,
     "ce_pairs": ce_pairs, "ce_pairs_nn": ce_pairs_nn, "ce_pairs_nn2": lambda: ce_pairs_nn(pairs_name="pseudo2_pairs"),
     "blend": lambda: blend(*a.blend),
     "assemble": lambda: assemble(a.iu_scores, a.iu_thr, a.fr_scores, a.fr_thr, a.out, a.iu_compete, a.fr_compete,
                                  a.vswap_norm),
     "candidates": lambda: candidates_file(out=a.cand_out)}[a.cmd]()
