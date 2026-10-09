"""cm01b: per-residue ESMC-6B states for every campaign protein, for attention-pooling models (cm06).

Stores, for every residue once (windows of 2,000 with stride 1,800; overlaps skipped, as cm01/pd12):
  res_L80.npy        (R, 2560) fp16   hidden state 80 (final layer), memory-mapped
  res_sae_idx.npy    (R, 64)  int16   layer-60 SAE top-64 feature indices per residue
  res_sae_val.npy    (R, 64)  fp16    their activations
  res_index.npz      accession, start offset and length per protein
R is the total number of residues (7.36 M for the union, 37 GB for the dense layer). Same batching and the same
layer-60 SAE forward as pd12/cm01. One RTX card.
"""
import os, sys, time, argparse
import numpy as np, torch
from safetensors.torch import load_file

ROOT = os.environ["ESM_PROAE_ROOT"]
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{ROOT}/data/campaign/union.fasta")
ap.add_argument("--out", default=f"{ROOT}/data/campaign/residues")
ap.add_argument("--esm", default=f"{ROOT}/data/esmc6b")
ap.add_argument("--sae", default=f"{ROOT}/data/esmc6b_sae/layer_60.safetensors")
ap.add_argument("--budget", type=int, default=8000)
ap.add_argument("--window", type=int, default=2000)
ap.add_argument("--stride", type=int, default=1800)
ap.add_argument("--check-batches", type=int, default=30)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
LAYER, K = 60, 64

seqs = []
for line in open(a.fasta):
    if line.startswith(">"):
        seqs.append([line[1:].strip().split()[0].split("|")[0], ""])
    else:
        seqs[-1][1] += line.strip()
if a.limit:
    seqs = seqs[: a.limit]
print(f"{len(seqs)} sequences", flush=True)

pieces = []                     # (protein index, subsequence, residues to skip at the start)
for i, (_acc, s) in enumerate(seqs):
    if len(s) <= a.window:
        pieces.append((i, s, 0, 0))
    else:
        for st in range(0, len(s), a.stride):
            sub = s[st:st + a.window]
            if len(sub) <= a.window - a.stride and st > 0:
                break
            pieces.append((i, sub, 0 if st == 0 else a.window - a.stride, st))
order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))
batches, cur, curmax = [], [], 0
for k in order:
    L = len(pieces[k][1]); m = max(curmax, L)
    if cur and (len(cur) + 1) * ((m + 7) // 8 * 8) > a.budget:
        batches.append(cur); cur, curmax = [k], L
    else:
        cur.append(k); curmax = m
if cur:
    batches.append(cur)
real = sum(len(p[1]) for p in pieces)
print(f"  {len(pieces)} windows in {len(batches)} batches, {real/1e6:.2f}M residues", flush=True)

from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(a.esm)
net = AutoModel.from_pretrained(a.esm, dtype=torch.bfloat16).eval().to(dev)
for p in net.parameters():
    p.requires_grad = False
W = load_file(a.sae)
W_enc, b_dec, W_dec = W["W_enc"].to(dev), W["b_dec"].to(dev), W["W_dec"].to(dev)
F_ = W_enc.shape[1]
print(f"  ESMC loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB); SAE {tuple(W_enc.shape)} k={K}", flush=True)


def sae(x):
    """x: (R, 2560) float32 residue states -> (R, F) sparse-in-content activations, and FVU."""
    x = x - x.mean(-1, keepdim=True)
    x = x / (x.std(-1, keepdim=True) + 1e-5)
    pre = torch.relu((x - b_dec) @ W_enc)
    top = torch.topk(pre, K, dim=-1)
    f = torch.zeros_like(pre).scatter_(-1, top.indices, top.values)
    rec = f @ W_dec + b_dec
    fvu = ((rec - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    return f, float(fvu)


n = len(seqs)
os.makedirs(a.out, exist_ok=True)
plen = np.array([len(s_) for _, s_ in seqs]); starts = np.concatenate([[0], np.cumsum(plen)[:-1]]); Rtot = int(plen.sum())   # not "lens": the batch loop reuses that name
RL = np.lib.format.open_memmap(f"{a.out}/res_L80.npy", mode="w+", dtype=np.float16, shape=(Rtot, 2560))
RI = np.lib.format.open_memmap(f"{a.out}/res_sae_idx.npy", mode="w+", dtype=np.int16, shape=(Rtot, K))
RV = np.lib.format.open_memmap(f"{a.out}/res_sae_val.npy", mode="w+", dtype=np.float16, shape=(Rtot, K))
filled = np.zeros(n, np.int64)                       # residues written so far per protein (windows arrive in any order)
win_off = {}                                         # (protein, window start in residues) bookkeeping
print(f"  {Rtot/1e6:.2f}M residues -> {a.out}", flush=True)
NL = net.config.num_hidden_layers + 1
SEL = np.unique(np.linspace(0, NL - 1, 9).astype(int))            # 0, 10, ..., 80 as in pd03b
print(f"  pooling layers {list(SEL)}", flush=True)
fmax = torch.zeros((n, F_), dtype=torch.float16)
fsum = torch.zeros((n, F_), dtype=torch.float32)
fcnt = torch.zeros((n, F_), dtype=torch.float32)
nres = np.zeros(n)
chk = {59: [], 60: [], 61: []}
t0 = time.perf_counter()
for bi, b in enumerate(batches):
    strs = [pieces[k][1] for k in b]
    Lb = (max(len(s) for s in strs) + 7) // 8 * 8
    enc = tok(strs, return_tensors="pt", padding="max_length", max_length=Lb + 2, truncation=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                  output_hidden_states=True)
    hs = out.hidden_states
    lens = [len(s) for s in strs]
    keep = torch.zeros((len(b), Lb), dtype=torch.bool, device=dev)
    for j, (k, L) in enumerate(zip(b, lens)):
        keep[j, pieces[k][2]:L] = True                 # residues 1..L after BOS, minus overlap
    with torch.no_grad():
        H80 = hs[80][:, 1:Lb + 1]                                            # (B, Lb, 2560) bf16
        for j, k in enumerate(b):
            pi, _, skip, st = pieces[k]; v = H80[j][keep[j]].to(torch.float16).cpu().numpy()
            o = starts[pi] + st + skip; RL[o:o + len(v)] = v
        layers = (59, 60, 61) if bi < a.check_batches else (60,)
        for lay in layers:
            H = hs[lay][:, 1:Lb + 1].float()[keep]          # (R, 2560)
            f, fvu = sae(H)
            if bi < a.check_batches:
                chk[lay].append(fvu)
            if lay != LAYER:
                continue
            seg = torch.repeat_interleave(torch.arange(len(b), device=dev), keep.sum(1))
            top = torch.topk(f, K, dim=-1)
            for j, k in enumerate(b):
                pi, _, skip, st = pieces[k]; sel = seg == j; o = starts[pi] + st + skip; m_ = int(sel.sum())
                RI[o:o + m_] = top.indices[sel].to(torch.int16).cpu().numpy(); RV[o:o + m_] = top.values[sel].to(torch.float16).cpu().numpy()
                filled[pi] += m_
                fj = f[sel]
                pi = pieces[k][0]
                fmax[pi] = torch.maximum(fmax[pi].float(), fj.max(0).values.cpu()).half()
                fsum[pi] += fj.sum(0).cpu()
                fcnt[pi] += (fj > 0).sum(0).float().cpu()
                nres[pi] += fj.shape[0]
    del out, hs
    if bi == a.check_batches - 1:
        print("  reconstruction check, fraction of variance unexplained by the SAE:", flush=True)
        for lay in (59, 60, 61):
            print(f"    hidden_states[{lay}]  FVU {np.mean(chk[lay]):.4f}", flush=True)
        # The convention itself is established from the code: transformers' capture hook stores the
        # input to layer 0 as hidden_states[0] and appends each layer's output, so hidden_states[60] is
        # the input to block 60, which is what esm/models/esmc/layers.py hands the layer-60 SAE.
        # Adjacent residual-stream states differ too little for reconstruction error to rank 59 against
        # 60 (they tie within 0.01 here); the check only guards against a gross mismatch.
        f59, f60, f61 = (np.mean(chk[l]) for l in (59, 60, 61))
        assert f60 < f61 and f60 < f59 + 0.02, f"layer-60 SAE fits hidden_states[60] poorly: {f59:.3f} {f60:.3f} {f61:.3f}"
    if (bi + 1) % 100 == 0 or bi == len(batches) - 1:
        el = time.perf_counter() - t0
        done = sum(len(pieces[k][1]) for bb in batches[:bi + 1] for k in bb)
        print(f"  batch {bi+1}/{len(batches)}  {done/el/1e3:.1f}k residues/s  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
              f"  eta {(len(batches)-bi-1)*el/(bi+1)/60:.1f} min", flush=True)

mean = (fsum / torch.as_tensor(np.maximum(nres, 1))[:, None]).half()
freq = (fcnt / torch.as_tensor(np.maximum(nres, 1))[:, None]).half()
alive = (fcnt.sum(0) > 0).numpy()
print(f"\n  features ever active: {alive.sum()} of {F_};  active features per protein (max>0): "
      f"median {np.median((fmax > 0).sum(1).numpy()):.0f}", flush=True)
RL.flush(); RI.flush(); RV.flush()
assert (filled == plen).all(), f"{int((filled != plen).sum())} proteins not fully written"
np.savez(f"{a.out}/res_index.npz", accession=np.array([s_[0] for s_ in seqs]), start=starts, length=plen)
el = time.perf_counter() - t0
print(f"wrote residue arrays to {a.out}\n  {el/60:.1f} min, {real/el/1e3:.1f}k residues/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB\nCM01B_DONE", flush=True)
