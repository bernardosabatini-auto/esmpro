"""pd12: ESMC-6B SAE features (layer 60, k = 64, 16,384 features) for every protein in both
experiments, pooled per protein.

The SAE (biohub/ESMC-6B-sae-layer60-k64-codebook16384) reconstructs one residue at a time, and top-k
is nonlinear, so it must see each residue's state before any pooling: the mean-pooled embeddings
already on disk cannot be reused. Forward pass, exactly as esm/models/esmc/sae.py:
    x  = z-score of the residue's 2560-d state across its dimensions
    a  = ReLU((x - b_dec) @ W_enc);  keep the 64 largest, zero the rest
"Layer 60" is the input to block 60 (esm/models/esmc/layers.py collects x before block i), which is
hidden_states[60] in the Hugging Face convention where hidden_states[0] is the embedding output.
That convention is read from transformers' output-capture code, not assumed; on the first batches
the SAE is also applied to hidden_states[59] and [61] as a guard against a gross mismatch.

Per protein, over residues (BOS/EOS excluded): max activation (the Biohub convention), mean
activation, and the fraction of residues on which the feature is active. Proteins longer than the
2,000-residue window are embedded in windows with stride 1,800, and the 200 residues each later
window shares with the one before are skipped, so every residue is counted once.

Same length-sorted token-budget batching as pd03b; budget lowered because output_hidden_states keeps
all 81 states resident.
"""
import os, sys, time, argparse
import numpy as np, torch
from safetensors.torch import load_file

ROOT = os.environ["ESM_PROAE_ROOT"]
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{ROOT}/data/ga_data/sequences.fasta")
ap.add_argument("--out", default=f"{ROOT}/data/sae/sae_l60.npz")
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
        pieces.append((i, s, 0))
    else:
        for st in range(0, len(s), a.stride):
            sub = s[st:st + a.window]
            if len(sub) <= a.window - a.stride and st > 0:
                break
            pieces.append((i, sub, 0 if st == 0 else a.window - a.stride))
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
        layers = (59, 60, 61) if bi < a.check_batches else (60,)
        for lay in layers:
            H = hs[lay][:, 1:Lb + 1].float()[keep]          # (R, 2560)
            f, fvu = sae(H)
            if bi < a.check_batches:
                chk[lay].append(fvu)
            if lay != LAYER:
                continue
            seg = torch.repeat_interleave(torch.arange(len(b), device=dev), keep.sum(1))
            for j, k in enumerate(b):
                fj = f[seg == j]
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
np.savez(a.out, accession=np.array([s[0] for s in seqs]), max=fmax.numpy(), mean=mean.numpy(), freq=freq.numpy(),
         n_residues=nres, fvu_check=np.array([np.mean(chk[l]) for l in (59, 60, 61)]))
el = time.perf_counter() - t0
print(f"wrote {a.out}\n  {el/60:.1f} min, {real/el/1e3:.1f}k residues/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB\nPD12_DONE", flush=True)
