"""Prepare GA_33 inputs for the GPU readout (ga09), which runs in an env without scikit-learn.

Writes, all aligned to the rows of data/ga_data/blocks.npz:
  emb_L50.npy, emb_L80.npy   merged ESMC mean-pools (salt-pass proteins + new ones), NaN where none
  folds_ga05.npz             rows_<t>, fold_<t> for all 15 ga05 targets -- scikit-learn GroupKFold
                             over the SAME rows ga05 scored, so readout and ridge numbers compare
                             like for like; plus the cluster label of every row
"""
import os, csv, itertools
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D, P = f"{ROOT}/data/ga_data", f"{ROOT}/data/pd_data"
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
status = {r["sheet_accession"]: r for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
res = np.array([status[s]["resolved_accession"] if s in status else s for s in acc])
has = np.array([status.get(s, {}).get("status", "notfound") != "notfound" for s in acc])


def keys(fn):
    return [ln[1:].strip().split()[0].split("|")[0] for ln in open(fn) if ln[0] == ">"]


old = np.load(f"{P}/embeddings_layers.npz", allow_pickle=True)
new = np.load(f"{D}/embeddings_layers_new.npz", allow_pickle=True)
LAY = [int(x) for x in old["layers"]]
src = {k: ("o", i) for i, k in enumerate(keys(f"{P}/sequences.fasta"))}
src.update({k: ("n", i) for i, k in enumerate(keys(f"{D}/new_sequences.fasta"))})
po, pn = old["pool"], new["pool"]
for lay in (50, 80):
    E = np.full((len(acc), po.shape[2]), np.nan, np.float32)
    for i, r in enumerate(res):
        if has[i] and r in src:
            w, j = src[r]
            E[i] = (po if w == "o" else pn)[j, LAY.index(lay)]
    np.save(f"{D}/emb_L{lay}.npy", E)
    print(f"emb_L{lay}.npy: {np.isfinite(E).all(1).sum()} rows with an embedding", flush=True)
emb_ok = np.isfinite(np.load(f"{D}/emb_L50.npy")).all(1)

cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(r, r) for r in res])
names = ([f"stau_{c}_{t}" for c, t in itertools.product(("21A", "24"), ("35", "37", "43"))]
         + [f"stau_avg_{t}" for t in ("35", "37", "43")]
         + [f"temp_{c}_{t}v35" for c, t in itertools.product(("21A", "24"), ("37", "43"))]
         + [f"temp_avg_{t}v35" for t in ("37", "43")])
out = {"groups": groups}
for nm in names:                                     # identical to ga05's keep_for / yof
    if "_avg_" in nm:
        kind, _, t = nm.split("_", 2)
        k = np.all([blk[f"keep_{kind}_{c}_{t}"] for c in ("21A", "24")], 0)
        y = np.mean([blk[f"y_{kind}_{c}_{t}"] for c in ("21A", "24")], 0)
    else:
        k, y = blk[f"keep_{nm}"].astype(bool), blk[f"y_{nm}"].astype(float)
    k = k & np.isfinite(y) & emb_ok
    fid = np.full(k.sum(), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(k.sum()), None, groups[k])):
        fid[te] = f
    out[f"rows_{nm}"], out[f"fold_{nm}"], out[f"y_{nm}"] = np.nonzero(k)[0], fid, y[k]
    print(f"  {nm:<17} {k.sum():>6} rows", flush=True)
np.savez(f"{D}/folds_ga05.npz", **out)
print(f"wrote {D}/folds_ga05.npz")
