"""pd03b: pool several ESMC hidden states in ONE forward pass, to find which layer to read.

pd03 read the last hidden state. That is the wrong default to leave untested: the final layer
of a language model is specialised for token prediction, and representations of physical
protein properties usually peak somewhere in the middle. With 81 hidden states available this
is the largest untested lever on the sequence signal, and it costs one pass -- the model emits
every hidden state anyway, so pooling nine of them is free apart from activation memory.

Nine evenly spaced layers, mean-pooled per protein, same length-sorted token-budget batching
and same overlapping-window handling as pd03. Output is (n_proteins, n_layers, 2560), which
pd07_layers.py then scores one layer at a time.

The token budget is lower than pd03's because keeping all 81 hidden states resident multiplies
activation memory by ~81: at 8,000 padded residues that is 81 x 8000 x 2560 x 2 B = 3.3 GB on
top of the 12-13 GB of weights.
"""
import os, sys, time, argparse
import numpy as np, torch

ROOT = os.environ["ESM_PROAE_ROOT"]
sys.path.insert(0, ROOT + "/code")

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{ROOT}/data/pd_data/sequences.fasta")
ap.add_argument("--out", default=f"{ROOT}/data/pd_data/embeddings_layers.npz")
ap.add_argument("--esm", default=f"{ROOT}/data/esmc6b")
ap.add_argument("--budget", type=int, default=8000)
ap.add_argument("--window", type=int, default=2000)
ap.add_argument("--stride", type=int, default=1800)
ap.add_argument("--n-layers", type=int, default=9, help="how many evenly spaced hidden states")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = "cuda"

seqs, cur = [], None
for line in open(a.fasta):
    if line.startswith(">"):
        cur = line[1:].strip().split()[0].split("|")[0]
        seqs.append((cur, ""))
    else:
        seqs[-1] = (seqs[-1][0], seqs[-1][1] + line.strip())
if a.limit:
    seqs = seqs[: a.limit]
print(f"{len(seqs)} sequences from {a.fasta}", flush=True)

pieces, n_windowed = [], 0
for i, (_acc, s) in enumerate(seqs):
    if len(s) <= a.window:
        pieces.append((i, s, len(s)))
    else:
        n_windowed += 1
        for st in range(0, len(s), a.stride):
            sub = s[st:st + a.window]
            if len(sub) < 50 and st > 0:
                break
            pieces.append((i, sub, len(sub)))
print(f"  {n_windowed} proteins exceed {a.window} residues -> {len(pieces)} windows", flush=True)

order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))
batches, curb, curmax = [], [], 0
for k in order:
    L = len(pieces[k][1]); m = max(curmax, L)
    if curb and (len(curb) + 1) * ((m + 7) // 8 * 8) > a.budget:
        batches.append(curb); curb, curmax = [k], L
    else:
        curb.append(k); curmax = m
if curb:
    batches.append(curb)
real = sum(p[2] for p in pieces)
padded = sum(len(b) * ((max(pieces[k][2] for k in b) + 7) // 8 * 8) for b in batches)
print(f"  {len(batches)} batches, padding efficiency {100*real/padded:.1f}%", flush=True)

from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(a.esm)
t0 = time.perf_counter()
net = AutoModel.from_pretrained(a.esm, dtype=torch.bfloat16).eval().to(dev)
for p in net.parameters():
    p.requires_grad = False
d = net.config.hidden_size
NL = net.config.num_hidden_layers + 1
sel = np.unique(np.linspace(0, NL - 1, a.n_layers).astype(int))
print(f"  loaded in {time.perf_counter()-t0:.0f}s, d={d}, {NL} hidden states, "
      f"weights {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
print(f"  pooling layers {list(sel)}", flush=True)

acc = np.zeros((len(seqs), len(sel), d), dtype=np.float64)
accw = np.zeros(len(seqs), dtype=np.float64)
t0, done = time.perf_counter(), 0
for bi, b in enumerate(batches):
    strs = [pieces[k][1] for k in b]
    Lb = (max(len(s) for s in strs) + 7) // 8 * 8
    enc = tok(strs, return_tensors="pt", padding="max_length", max_length=Lb + 2, truncation=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(input_ids=enc["input_ids"].to(dev),
                  attention_mask=enc["attention_mask"].to(dev), output_hidden_states=True)
    # stack only the selected states, strip BOS, and pool on the GPU before coming back
    hs = torch.stack([out.hidden_states[l] for l in sel], 1)[:, :, 1:Lb + 1].float()
    del out
    for j, k in enumerate(b):
        pi, sub, w = pieces[k]
        acc[pi] += hs[j, :, :len(sub)].mean(1).cpu().numpy() * w
        accw[pi] += w
    done += len(b)
    if (bi + 1) % 50 == 0 or bi == len(batches) - 1:
        el = time.perf_counter() - t0
        print(f"  batch {bi+1}/{len(batches)}  {real*done/len(pieces)/el/1e3:.1f}k residues/s  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB  "
              f"eta {(len(batches)-bi-1)*el/(bi+1)/60:.1f} min", flush=True)

pool = acc / np.maximum(accw, 1e-9)[:, None, None]
bad = ~np.isfinite(pool).all((1, 2))
print(f"\n  {int(bad.sum())} proteins with a non-finite pooled vector", flush=True)
np.savez_compressed(a.out,
                    accession=np.array([ac for ac, _s in seqs]),
                    pool=pool.astype(np.float32), layers=sel,
                    length=np.array([len(s) for _ac, s in seqs]),
                    windowed=np.array([len(s) > a.window for _ac, s in seqs]))
el = time.perf_counter() - t0
print(f"wrote {a.out}  ({len(seqs)} x {len(sel)} x {d})", flush=True)
print(f"  {el/60:.1f} min, {real/el/1e3:.1f}k residues/s, "
      f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
print("PD03B_DONE", flush=True)
