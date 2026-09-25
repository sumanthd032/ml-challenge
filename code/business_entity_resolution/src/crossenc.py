"""Cross-encoder re-ranker for the pruned candidate set (input of stage 2).

Stage-1 LightGBM scores every blocked pair from similarity features. Most remaining errors are "twins":
a record of a *different* business that copies an S1 with one or two edits (a swapped name word, another
legal form, a nearby house number), versus a true record with ordinary noise (typos, dropped legal form,
extra digits). A cross-encoder reads both records token by token, so it can learn which edits mean a
different business.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0, 22M params), fine-tuned for binary match
classification. Input: "<normalized name> | <normalized address>" of the S1 record and of the candidate.

Training pairs come from the 'fit' S1s only (validation S1s stay out-of-sample for stage 2, ghost S1s trained
the bi-encoder): all of their true pairs in the candidate set plus their hardest negatives
(top ranks by pair_sim / bi-encoder cosine on the S1 side).

Training and scoring checkpoint to disk, so a machine reset (D-012) loses at most a few minutes.

Usage:
  python crossenc.py pairs                  # -> FEAT_DIR/ce_train_pairs.parquet
  python crossenc.py train [--max-pairs N]  # -> ARTIFACTS/crossenc/
  python crossenc.py score --split val|test # -> FEAT_DIR/ce_scores_<split>.parquet
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from data import load_split

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_DIR = config.ART_DIR / "crossenc"
CKPT = config.ART_DIR / "crossenc_ckpt.pt"
MAX_LEN = 96
P_MIN = 0.002          # stage-1 probability floor of the pruned candidate set (= candidate_pairs.tsv)
NEG_RANK = 6           # hard negatives: S1-side rank <= NEG_RANK by pair_sim or by bi-encoder cosine
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def record_text(df: pd.DataFrame) -> np.ndarray:
    addr = np.where(df.addr.values != "", df.addr.values, "no address")
    name = np.where(df.name.values != "", df.name.values, "no name")
    return np.array([f"{n} | {a}" for n, a in zip(name, addr)], dtype=object)


def build_train_pairs():
    """(i1, i2, label) for fit S1s: all positives in the candidate set + S1-side hard negatives."""
    from train import truth_rows
    s1, s23 = load_split("train")
    role = s1.entity_id.map(config.s1_role).values
    d = config.FEAT_DIR / "train_store"
    names = json.load(open(d / "group_names.json"))
    G = np.load(d / "group.npy", mmap_mode="r")
    j1, j2 = names.index("s1_pair_sim_rank"), names.index("s1_emb_cos_rank")
    truth = truth_rows(s1, s23)
    n2 = np.int64(len(s23))
    tk = np.sort(truth.s1_id.values.astype(np.int64) * n2 + truth.cand_id.values)
    parts, s = [], 0
    # stream the 175M pairs in slices: peak memory stays at one slice (D-012)
    for b in pq.ParquetFile(d / "pairs.parquet").iter_batches(batch_size=4_000_000, columns=["i1", "i2"]):
        i1, i2 = b.column("i1").to_numpy(), b.column("i2").to_numpy()
        g = np.asarray(G[s:s + len(i1)])
        s += len(i1)
        hard = (g[:, j1] <= NEG_RANK) | (g[:, j2] <= NEG_RANK)
        pk = i1.astype(np.int64) * n2 + i2
        label = (tk[np.minimum(np.searchsorted(tk, pk), len(tk) - 1)] == pk).astype(np.int8)
        keep = (role[i1] == "fit") & (hard | (label == 1))
        parts.append(pd.DataFrame({"i1": i1[keep], "i2": i2[keep], "label": label[keep]}))
    assert s == len(G)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(config.FEAT_DIR / "ce_train_pairs.parquet", index=False)
    print(f"ce train pairs {len(out):,} pos rate {out.label.mean():.3f} "
          f"({out.i1.nunique():,} S1, {len(out) / out.i1.nunique():.1f}/S1)")


def _batches(A, B, idx, tok, bs):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, tok(list(A[j]), list(B[j]), padding=True, truncation="longest_first", max_length=MAX_LEN,
                     return_tensors="pt")


def train(max_pairs, bs=256, lr=4e-5, eval_every=1000):
    s1, s23 = load_split("train")
    t1, t2 = record_text(s1), record_text(s23)
    tr = pd.read_parquet(config.FEAT_DIR / "ce_train_pairs.parquet")
    rng = np.random.default_rng(config.SEED)
    if len(tr) > max_pairs:
        tr = tr.iloc[np.sort(rng.choice(len(tr), max_pairs, replace=False))].reset_index(drop=True)
    A, B, y = t1[tr.i1.values], t2[tr.i2.values], tr.label.values.astype(np.float32)
    # monitor: 40k validation pairs from the pruned set (out-of-sample S1s)
    va = pd.read_parquet(config.FEAT_DIR / "val_scores_v1.parquet")
    va = va[va.p >= P_MIN].sample(40_000, random_state=1)
    vA, vB, vy = t1[va.s1_id.values], t2[va.cand_id.values], va.label.values

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(A) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03,
                                                anneal_strategy="linear")
    perm = rng.permutation(len(A))
    step0 = 0
    if CKPT.exists():                       # resume after a crash
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
        print(f"resumed at step {step0}", flush=True)
    print(f"training on {len(A):,} pairs, pos rate {y.mean():.3f}, {steps} steps", flush=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    t = time.time()
    model.train()
    for step, (j, enc) in enumerate(_batches(A, B, perm, tok, bs)):
        if step < step0:
            continue
        enc = enc.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[j]).to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} ({time.time() - t:.0f}s)", flush=True)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            pv = predict(model, tok, vA, vB)
            ll = -np.mean(vy * np.log(np.clip(pv, 1e-6, 1)) + (1 - vy) * np.log(np.clip(1 - pv, 1e-6, 1)))
            acc = np.mean((pv >= 0.5) == vy)
            print(f"  [val] step {step + 1} logloss {ll:.4f} acc {acc:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1}, str(CKPT) + ".tmp")
            os.replace(str(CKPT) + ".tmp", CKPT)
            model.train()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CE_DIR); tok.save_pretrained(CE_DIR)
    print(f"saved {CE_DIR} ({time.time() - t:.0f}s)")


@torch.no_grad()
def predict(model, tok, A, B, bs=1024):
    model.eval()
    out = np.empty(len(A), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(A, B)], kind="stable")   # length bucketing
    for j, enc in _batches(A, B, order, tok, bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[j] = torch.sigmoid(model(**enc.to(DEV)).logits.squeeze(-1).float()).cpu().numpy()
    return out


def pruned_pairs(split: str) -> pd.DataFrame:
    """The pruned candidate set: stage-1 pairs with p >= P_MIN (s1_id / cand_id are row positions)."""
    f = {"val": "val_scores_v1.parquet", "test": "test_scores_lgb_v1.parquet"}[split]
    sc = pd.read_parquet(config.FEAT_DIR / f)
    return sc[sc.p >= P_MIN].reset_index(drop=True)


def score(split: str, chunk=1_000_000):
    s1, s23 = load_split("train" if split == "val" else "test")
    t1, t2 = record_text(s1), record_text(s23)
    pr = pruned_pairs(split)
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(DEV)
    d = config.FEAT_DIR / f"ce_{split}_parts"
    d.mkdir(exist_ok=True)
    t = time.time()
    n = -(-len(pr) // chunk)
    for k in range(n):
        p = d / f"part_{k:04d}.npy"
        if p.exists():
            continue
        sl = slice(k * chunk, (k + 1) * chunk)
        v = predict(model, tok, t1[pr.s1_id.values[sl]], t2[pr.cand_id.values[sl]])
        np.save(d / f"part_{k:04d}.tmp.npy", v); os.replace(d / f"part_{k:04d}.tmp.npy", p)
        print(f"  {split} chunk {k + 1}/{n} ({time.time() - t:.0f}s)", flush=True)
    pr["ce"] = np.concatenate([np.load(d / f"part_{k:04d}.npy") for k in range(n)])
    pr.to_parquet(config.FEAT_DIR / f"ce_scores_{split}.parquet", index=False)
    print(f"saved ce_scores_{split}.parquet: {len(pr):,} pairs ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "train", "score"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-pairs", type=int, default=6_000_000)
    args = ap.parse_args()
    {"pairs": build_train_pairs, "train": lambda: train(args.max_pairs),
     "score": lambda: score(args.split)}[args.cmd]()
