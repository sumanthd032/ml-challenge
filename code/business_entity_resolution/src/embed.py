"""Blocking pass B: fine-tuned bi-encoder + exact GPU kNN.

Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0, 22M params), fine-tuned with a symmetric
in-batch-negatives InfoNCE loss on TRAIN ground-truth pairs of the 'ghost' S1 entities only (they are
excluded from LightGBM training/validation, so the cosine feature is not overfit on validation).

Text of a record = "<core name> [alt name] | <canonical address>".

Usage:
  python embed.py finetune                 # -> ARTIFACTS/biencoder/
  python embed.py knn --split train|test   # -> ARTIFACTS/{split}_cands_B.parquet and {split}_emb_*.npy
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import config
from data import load_split, load_gt_pairs

BASE = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = config.ART_DIR / "biencoder"
MAX_LEN = 48
K_FWD, K_REV = 20, 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def record_text(df: pd.DataFrame) -> list:
    """Encoder input text of every row: "<core>[ <alt>] | <addr>"."""
    alt = np.where(df.alt.values != "", " " + df.alt.values, "")
    return [f"{c}{a} | {ad}" for c, a, ad in zip(df.core.values, alt, df.addr.values)]


def mean_pool(out, mask):
    """Mean of token states over non-padding positions."""
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Encoder(torch.nn.Module):
    """Transformer loaded from `path` with mean pooling and L2 normalization."""

    def __init__(self, path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(path)

    def forward(self, ids, mask):
        """Unit-norm embeddings of a tokenized batch."""
        return F.normalize(mean_pool(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state, mask), dim=-1)


def finetune(epochs=1, bs=512, lr=1e-4, max_pairs=1_500_000):
    """Fine-tune BASE on at most max_pairs ground-truth pairs of ghost S1s; saves to MODEL_DIR."""
    s1, s23 = load_split("train")
    gt = load_gt_pairs()
    gt = gt[gt.s1_id.map(config.s1_role) == "ghost"]
    gt = gt.sample(min(len(gt), max_pairs), random_state=config.SEED)
    t1 = dict(zip(s1.entity_id, record_text(s1)))
    sub = s23[s23.entity_id.isin(set(gt.cand_id))]
    t2 = dict(zip(sub.entity_id, record_text(sub)))
    A = [t1[x] for x in gt.s1_id]; B = [t2[x] for x in gt.cand_id]
    _train(A, B, gt.s1_id.values, BASE, MODEL_DIR, epochs, bs, lr)


def finetune_pseudo(split, pairs_file, init_dir, out_dir, epochs=1, bs=512, lr=5e-5):
    """Continue fine-tuning on pseudo-labelled pairs of a test split (D-018): `pairs_file` holds row positions
    (i1, i2) of near-certain matches. Same loss as `finetune`; starts from the already fine-tuned model."""
    s1, s23 = load_split(split)
    pp = pd.read_parquet(pairs_file)
    t1, t2 = record_text(s1), record_text(s23)
    A = [t1[i] for i in pp.i1.values]; B = [t2[i] for i in pp.i2.values]
    # hard in-batch negatives: batches of consecutive pairs in (address locality, name) order hold near-twin
    # businesses (same city, similar names), which random batches almost never do
    key = pd.DataFrame({"loc": s1.toks.values[pp.i1.values], "core": s1.core.values[pp.i1.values]})
    key["loc"] = key["loc"].str.split().str[-1].fillna("")
    order = key.sort_values(["loc", "core"], kind="stable").index.values
    _train(A, B, pp.i1.values, init_dir, out_dir, epochs, bs, lr, order=order)


def _train(A, B, s1_of, init, out_dir, epochs, bs, lr, order=None):
    """order: if given, batches are consecutive runs of `order` (shuffled as whole batches each epoch)."""
    print(f"fine-tuning on {len(A):,} pairs from {init}", flush=True)
    tok = AutoTokenizer.from_pretrained(init)
    model = Encoder(init).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(A) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    scale = 20.0  # 1 / temperature
    model.train()
    step = 0
    for ep in range(epochs):
        if order is None:
            perm = np.random.default_rng(ep).permutation(len(A))
        else:
            blocks = [order[i:i + bs] for i in range(0, len(order), bs)]
            perm = np.concatenate([blocks[j] for j in np.random.default_rng(ep).permutation(len(blocks))])
        t = time.time()
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            ea = tok([A[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEV)
            eb = tok([B[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                za = model(ea.input_ids, ea.attention_mask); zb = model(eb.input_ids, eb.attention_mask)
                sim = (za @ zb.T).float() * scale
            # pairs sharing the same S1 are not negatives of each other
            same = torch.as_tensor(s1_of[idx][:, None] == s1_of[idx][None, :], device=DEV)
            eye = torch.eye(len(idx), device=DEV, dtype=torch.bool)
            sim = sim.masked_fill(same & ~eye, -1e4)
            lab = torch.arange(len(idx), device=DEV)
            loss = (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t:.0f}s)", flush=True)
    out_dir.mkdir(exist_ok=True, parents=True)
    model.bert.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print("saved", out_dir)


@torch.no_grad()
def encode(texts, model, tok, bs=2048):
    """float16 embeddings of `texts`, in input order."""
    model.eval()
    out = np.empty((len(texts), model.bert.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])        # length bucketing for speed
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        e = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(e.input_ids, e.attention_mask)
        out[idx] = z.float().cpu().numpy().astype(np.float16)
    return out


@torch.no_grad()
def gpu_topk(Q: np.ndarray, K: np.ndarray, k: int, max_bytes=2e9):
    """Exact top-k inner product of each Q row against all K rows (score matrix chunk <= max_bytes)."""
    Kt = torch.from_numpy(np.ascontiguousarray(K)).to(DEV)
    chunk = max(64, int(max_bytes / (2 * Kt.shape[0])))
    qi, ci, sc = [], [], []
    for i in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[i:i + chunk]).to(DEV)
        s = q @ Kt.T
        v, j = torch.topk(s, min(k, Kt.shape[0]), dim=1)
        qi.append(np.repeat(np.arange(i, i + len(q)), v.shape[1])); ci.append(j.cpu().numpy().ravel())
        sc.append(v.float().cpu().numpy().ravel())
    del Kt
    torch.cuda.empty_cache()
    return np.concatenate(qi), np.concatenate(ci), np.concatenate(sc)


def knn(split: str):
    """Pass B for one split: forward top-K_FWD and reverse top-K_REV neighbours per country.

    Embeddings are cached as <split>_emb_s1.npy / <split>_emb_s23.npy (reused when present). Writes
    <split>_cands_B.parquet with (s1_id, cand_id, emb_score, emb_rank_fwd, emb_rank_rev); rank 99 = not found in
    that direction.
    """
    s1, s23 = load_split(split)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Encoder(MODEL_DIR).to(DEV).half()
    t = time.time()
    p1, p2 = config.ART_DIR / f"{split}_emb_s1.npy", config.ART_DIR / f"{split}_emb_s23.npy"
    if p1.exists() and p2.exists():
        E1, E2 = np.load(p1), np.load(p2)
    else:
        E1 = encode(record_text(s1), model, tok); E2 = encode(record_text(s23), model, tok)
        np.save(p1, E1); np.save(p2, E2)
    print(f"encoded {len(E1):,}+{len(E2):,} in {time.time() - t:.0f}s", flush=True)
    parts = []
    for country in sorted(set(s1.country) | set(s23.country)):
        a = np.flatnonzero(s1.country.values == country); b = np.flatnonzero(s23.country.values == country)
        if len(a) == 0 or len(b) == 0:
            continue
        q, c, s = gpu_topk(E1[a], E2[b], K_FWD)
        f = pd.DataFrame({"i1": a[q], "i2": b[c], "emb_score": s})
        f["emb_rank_fwd"] = f.groupby("i1").emb_score.rank(ascending=False, method="first").astype(np.int16) - 1
        q, c, s = gpu_topk(E2[b], E1[a], K_REV)
        r = pd.DataFrame({"i1": a[c], "i2": b[q], "emb_score_r": s})
        r["emb_rank_rev"] = r.groupby("i2").emb_score_r.rank(ascending=False, method="first").astype(np.int16) - 1
        m = f.merge(r, on=["i1", "i2"], how="outer")
        m["emb_score"] = m.emb_score.fillna(m.emb_score_r)
        m = m.drop(columns="emb_score_r")
        parts.append(m)
        print(f"  {country}: {len(m):,} pairs", flush=True)
    m = pd.concat(parts, ignore_index=True)
    m["emb_rank_fwd"] = m.emb_rank_fwd.fillna(99).astype(np.int16)
    m["emb_rank_rev"] = m.emb_rank_rev.fillna(99).astype(np.int16)
    m["s1_id"] = s1.entity_id.values[m.i1.values]; m["cand_id"] = s23.entity_id.values[m.i2.values]
    m[["s1_id", "cand_id", "emb_score", "emb_rank_fwd", "emb_rank_rev"]].to_parquet(
        config.ART_DIR / f"{split}_cands_B.parquet", index=False)
    print(f"saved {len(m):,} pairs ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "finetune_pseudo", "knn"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--model-dir", default=None, help="bi-encoder directory (default ARTIFACTS/biencoder)")
    ap.add_argument("--pairs", default=None, help="finetune_pseudo: parquet of (i1, i2) pseudo-positive pairs")
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    if args.model_dir:
        MODEL_DIR = config.ART_DIR / args.model_dir
    if args.cmd == "finetune":
        finetune(args.epochs)
    elif args.cmd == "finetune_pseudo":
        finetune_pseudo(args.split, args.pairs, config.ART_DIR / "biencoder", MODEL_DIR, args.epochs, lr=args.lr)
    else:
        knn(args.split)
