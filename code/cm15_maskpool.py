"""cm15: pool the ESMC-6B residue states over structurally defined masks instead of over the whole protein.

Section 3 found that a learned attention pooling over all 7.4 M residue states equals plain mean pooling
(+0.000 [-0.003, +0.004]). That is not evidence that residue detail is useless. It is evidence that 11.5k
proteins with noisy labels are not enough for a network to *discover* which residues matter. Supplying the
mask directly is far more data-efficient than making it learn one.

The masks come from the AlphaFold model (cm13, per-residue relative SASA and pLDDT), used only for proteins whose
model sequence is byte-identical to the sequence that was embedded, so residue i means the same thing in both:

  all          every residue (reproduces the existing mean pooling, as an internal check)
  surface      relative SASA > 0.25          buried      relative SASA <= 0.25
  ordered      pLDDT >= 70                   disordered  pLDDT < 70
  surf_charged surface and D/E/K/R           surf_hphob  surface and A/V/L/I/M/F/W/Y
  core_hphob   buried and hydrophobic -- the part a partial unfolding would expose

Writes data/campaign/maskpool.npy (n_union, n_mask, 2560) fp16 and maskpool_meta.json. Masks with fewer than 5
residues fall back to the whole-protein mean, and the fallback rate is reported per mask.

cm03 accepts these through the spec key "maskpool": each named mask becomes an extra slot alongside "layers",
with its own 2560 -> p projection.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
R = f"{C}/residues"
MASKS = ["all", "surface", "buried", "ordered", "disordered", "surf_charged", "surf_hphob", "core_hphob"]
HYD, CHG = set("AVLIMFWY"), set("DEKR")

U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist(); upos = {a: i for i, a in enumerate(U)}
ix = np.load(f"{R}/res_index.npz", allow_pickle=True)
racc, rstart, rlen = ix["accession"].astype(str), ix["start"], ix["length"]
rpos = {a: i for i, a in enumerate(racc)}
S = np.load(f"{C}/res_struct.npz", allow_pickle=True)
sacc, sstart, slen = S["accession"].astype(str), S["start"], S["length"]
rsa_all, pl_all = S["rsa"], S["plddt"]
spos = {a: i for i, a in enumerate(sacc)}

seq = {}
a = None
for l in open(f"{C}/union.fasta"):
    if l.startswith(">"): a = l[1:].strip().split()[0]; seq[a] = []
    else: seq[a].append(l.strip())
seq = {k: "".join(v) for k, v in seq.items()}

L80 = np.load(f"{R}/res_L80.npy", mmap_mode="r")
out = np.zeros((len(U), len(MASKS), 2560), dtype=np.float16)
have = np.zeros(len(U), bool); fallback = np.zeros(len(MASKS), int); done = 0
for acc in U:
    if acc not in rpos: continue
    i = upos[acc]; a0, aL = rstart[rpos[acc]], rlen[rpos[acc]]
    X = np.asarray(L80[a0:a0 + aL], dtype=np.float32)
    mean_all = X.mean(0)
    if acc in spos:
        j = spos[acc]
        if slen[j] != aL:                                   # should not happen: cm13 only stores verified matches
            print(f"  {acc}: residue count {slen[j]} != embedded {aL}, skipped", flush=True); continue
        rsa = rsa_all[sstart[j]:sstart[j] + slen[j]]; pl = pl_all[sstart[j]:sstart[j] + slen[j]]
        s = np.frombuffer(seq[acc].encode(), dtype="S1").astype(str)
        surf = rsa > .25; ordd = pl >= 70
        hy = np.isin(s, list(HYD)); ch = np.isin(s, list(CHG))
        M = dict(all=np.ones(aL, bool), surface=surf, buried=~surf, ordered=ordd, disordered=~ordd,
                 surf_charged=surf & ch, surf_hphob=surf & hy, core_hphob=(~surf) & hy)
        have[i] = True
    else:
        M = {m: np.ones(aL, bool) for m in MASKS}            # no structure: every mask is the whole protein
    for k, m in enumerate(MASKS):
        sel = M[m]
        if sel.sum() < 5: out[i, k] = mean_all; fallback[k] += 1
        else: out[i, k] = X[sel].mean(0)
    done += 1
    if done % 2000 == 0: print(f"  {done} proteins", flush=True)

np.save(f"{C}/maskpool.npy", out)
json.dump(dict(masks=MASKS, n=len(U), n_with_structure=int(have.sum()),
               fallback_per_mask={m: int(f) for m, f in zip(MASKS, fallback)}),
          open(f"{C}/maskpool_meta.json", "w"), indent=1)
print(f"{done} proteins pooled, {have.sum()} with structural masks", flush=True)
print("fallback to whole-protein mean:", {m: int(f) for m, f in zip(MASKS, fallback)}, flush=True)
# internal check: the "all" mask must reproduce the stored mean-pooled layer 80
LY = np.load(f"{C}/layers.npy", mmap_mode="r")
chk = [i for i in range(len(U)) if have[i]][:300]
r = np.corrcoef(out[chk, 0].astype(np.float32).ravel(), np.asarray(LY[chk, 8], dtype=np.float32).ravel())[0, 1]
print(f"check: mask 'all' vs stored mean-pooled layer 80, r = {r:.4f} (must be ~1)", flush=True)
print("CM15_DONE")
