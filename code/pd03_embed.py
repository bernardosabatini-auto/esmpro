"""pd03: ESMC-6B embeddings, mean- and max-pooled to one vector per protein.

Runs on ONE RTX card (96 GB) -- ESMC-6B is 24.8 GB in bf16 and this needs no more than that
plus activations, so an H200 would be wasted fairshare.

Two efficiency choices, because a naive loop would waste most of the GPU:

  * **Token-budget batching.** Sequences are sorted by length and packed into batches whose
    padded token count stays under one budget, so 200-residue proteins go 60 at a time and
    1800-residue proteins go 6 at a time. Padding every batch to a fixed 2048 instead would
    put ~3x more padding than real residues through the model, since the median length here
    is 524.
  * **Overlapping windows above the model limit.** ESMC's `max_position_embeddings` is 2048;
    238 of these 7,184 proteins are longer, up to 34,350 residues. Those are embedded in
    overlapping windows and pooled with weights proportional to the residues each window
    contributes, rather than being truncated.

  python pd03_embed.py                       # writes data/pd_data/embeddings.npz
  python pd03_embed.py --budget 24000 --dcgm  # tune the budget, sample SM counters
"""
import os, sys, time, argparse, subprocess, threading
import numpy as np, torch

ROOT = os.environ["ESM_PROAE_ROOT"]
sys.path.insert(0, ROOT + "/code")

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{ROOT}/data/pd_data/sequences.fasta")
ap.add_argument("--out", default=f"{ROOT}/data/pd_data/embeddings.npz")
ap.add_argument("--esm", default=f"{ROOT}/data/esmc6b")
ap.add_argument("--budget", type=int, default=20000, help="padded residues per batch")
ap.add_argument("--window", type=int, default=2000, help="residues per window (model limit 2048 incl. BOS/EOS)")
ap.add_argument("--stride", type=int, default=1800)
ap.add_argument("--layer", type=int, default=None, help="hidden state to read; default is the last")
ap.add_argument("--dcgm", action="store_true", help="sample SM/occupancy/tensor counters while running")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")


class Counters(threading.Thread):
    """DCGM sampler: SM active, occupancy, tensor-core active, memory bandwidth."""
    def __init__(s):
        super().__init__(daemon=True); s.rows, s.stop = [], False
        s.gpu = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS", "0")).split(",")[0]

    def run(s):
        try:
            p = subprocess.Popen(["dcgmi", "dmon", "-e", "1002,1003,1004,1005", "-i", s.gpu,
                                  "-d", "500", "-c", "100000"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            return
        for line in p.stdout:
            if s.stop:
                break
            f = line.split()
            if len(f) >= 6 and f[0] == "GPU":
                try:
                    s.rows.append([float(x) for x in f[2:6]])
                except ValueError:
                    pass
        p.terminate()

    def summary(s):
        if not s.rows:
            return None
        r = np.array(s.rows)[2:]
        if not len(r):
            return None
        return dict(zip(("sm_active", "occupancy", "tensor_active", "dram_active"), r.mean(0).round(3)))


def read_fasta(path):
    out, acc, buf = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                out.append((acc, "".join(buf)))
            acc = line[1:].split("|")[0].strip()
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        out.append((acc, "".join(buf)))
    return out


seqs = read_fasta(a.fasta)
if a.limit:
    seqs = seqs[:a.limit]
print(f"{len(seqs)} sequences from {os.path.basename(a.fasta)}", flush=True)

# split anything over the model limit into overlapping windows; each piece carries its weight
pieces = []          # (protein index, sub-sequence, weight)
n_windowed = 0
for i, (acc, s) in enumerate(seqs):
    if len(s) <= a.window:
        pieces.append((i, s, len(s)))
    else:
        n_windowed += 1
        for st in range(0, len(s), a.stride):
            sub = s[st:st + a.window]
            if len(sub) < 50 and st > 0:
                break
            pieces.append((i, sub, len(sub)))
print(f"  {n_windowed} proteins exceed {a.window} residues -> {len(pieces)} windows total", flush=True)

order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))
batches, cur, curmax = [], [], 0
for k in order:
    L = len(pieces[k][1])
    m = max(curmax, L)
    if cur and (len(cur) + 1) * ((m + 7) // 8 * 8) > a.budget:
        batches.append(cur); cur, curmax = [k], L
    else:
        cur.append(k); curmax = m
if cur:
    batches.append(cur)
real = sum(len(p[1]) for p in pieces)
padded = sum(len(b) * ((max(len(pieces[k][1]) for k in b) + 7) // 8 * 8) for b in batches)
print(f"  {len(batches)} batches, {real/1e6:.2f}M real residues, {padded/1e6:.2f}M padded "
      f"-> padding efficiency {100*real/padded:.1f}%", flush=True)

import gate7_latent_flow as G
t0 = time.perf_counter()
emb = G.OnlineESM(a.esm, layer_mix=False, device=dev, kind="esmc", layer=a.layer)
print(f"  loaded ESMC in {time.perf_counter()-t0:.0f}s, d={emb.d_cond}, "
      f"{emb.n_layers} hidden states, weights {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

d = emb.d_cond
acc_mean = np.zeros((len(seqs), d), dtype=np.float64)
acc_max = np.full((len(seqs), d), -np.inf, dtype=np.float64)
acc_w = np.zeros(len(seqs), dtype=np.float64)

ctr = Counters() if a.dcgm else None
if ctr:
    ctr.start()
t0 = time.perf_counter()
done = 0
for bi, b in enumerate(batches):
    strs = [pieces[k][1] for k in b]
    Lb = (max(len(s) for s in strs) + 7) // 8 * 8
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = emb(strs, L=Lb)                                  # (B, Lb, d), BOS/EOS stripped
    h = h.float()
    for j, k in enumerate(b):
        pi, sub, w = pieces[k]
        v = h[j, :len(sub)]
        acc_mean[pi] += v.mean(0).cpu().numpy() * w
        acc_max[pi] = np.maximum(acc_max[pi], v.max(0).values.cpu().numpy())
        acc_w[pi] += w
    done += len(b)
    if (bi + 1) % 25 == 0 or bi == len(batches) - 1:
        el = time.perf_counter() - t0
        print(f"  batch {bi+1}/{len(batches)}  {done}/{len(pieces)} windows  "
              f"{real*done/len(pieces)/el/1e3:.1f}k residues/s  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB  "
              f"eta {(len(batches)-bi-1)*el/(bi+1)/60:.1f} min", flush=True)
if ctr:
    ctr.stop = True
    time.sleep(0.6)

mean_pool = acc_mean / np.maximum(acc_w, 1e-9)[:, None]
lengths = np.array([len(s) for _acc, s in seqs])
bad = ~np.isfinite(mean_pool).all(1)
print(f"\n  {int(bad.sum())} proteins with a non-finite pooled vector", flush=True)

np.savez_compressed(
    a.out,
    accession=np.array([acc for acc, _s in seqs]),
    mean_pool=mean_pool.astype(np.float32),
    max_pool=acc_max.astype(np.float32),
    length=lengths,
    windowed=np.array([len(s) > a.window for _acc, s in seqs]),
    layer=-1 if a.layer is None else a.layer,
)
el = time.perf_counter() - t0
print(f"\nwrote {a.out}  ({len(seqs)} x {d} mean + max)", flush=True)
print(f"  {el/60:.1f} min, {real/el/1e3:.1f}k residues/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB of 96", flush=True)
if ctr:
    s = ctr.summary()
    if s:
        print(f"  GPU: SM active {s['sm_active']}, occupancy {s['occupancy']}, "
              f"tensor {s['tensor_active']}, bandwidth {s['dram_active']}", flush=True)
print("PD03_DONE")
