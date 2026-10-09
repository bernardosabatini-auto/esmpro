"""cm02b: campaign inputs in one place, row-aligned with data/campaign/union.tsv and targets.npz.

  layers.npy     (n, 9, 2560) fp16   ESMC-6B hidden states 0, 10, ..., 80, mean-pooled (pd03b, ga, cm01)
  sae_max.npy    (n, 16384)   fp16   layer-60 SAE, max over residues (pd12 / cm01)
  sae_freq.npy   (n, 16384)   fp16   fraction of residues on which the feature is active
  simple.tsv     sequence-only covariates: log10 length, net charge / residue at pH 7.4, pI, GRAVY,
                 fraction of disorder-promoting residues, 20 amino-acid fractions
  folds.npz      outer 5-fold assignment grouped by 30 % identity cluster (MMseqs2 easy-cluster, c 0.8),
                 balanced by protein count; inner 4-fold assignment within each outer training set, also grouped
  has_features   proteins with a sequence and all features

Before merging, the control re-embedding (control_features.npz, 60 proteins that already had features from
the earlier runs) must reproduce the stored vectors; the script stops if it does not.
"""
import os, json
import numpy as np, pandas as pd
from pdpipe import partition as pt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
T = pd.read_csv(f"{C}/union.tsv", sep="\t"); U = T["accession"].astype(str).tolist(); idx = {a: i for i, a in enumerate(U)}; n = len(U)
e2a = pd.read_csv(f"{ROOT}/data/pd_data/targets.tsv", sep="\t").set_index("entry")["accession"].astype(str).to_dict()

layers = np.zeros((n, 9, 2560), np.float16); have_l = np.zeros(n, bool)
sae_max = np.zeros((n, 16384), np.float16); sae_freq = np.zeros((n, 16384), np.float16); have_s = np.zeros(n, bool)
src = {}
for f, keymap in ((f"{ROOT}/data/pd_data/embeddings_layers.npz", lambda x: e2a.get(x, x)), (f"{ROOT}/data/ga_data/embeddings_layers_new.npz", lambda x: x)):
    d = np.load(f, allow_pickle=True); assert list(d["layers"]) == list(range(0, 81, 10)), d["layers"]
    pool = d["pool"]                                               # load once: NpzFile re-reads on every access
    for j, a in enumerate(d["accession"].astype(str)):
        a = keymap(a)
        if a in idx and not have_l[idx[a]]: layers[idx[a]] = pool[j]; have_l[idx[a]] = True
d = np.load(f"{ROOT}/data/sae/sae_l60.npz", allow_pickle=True); dmax, dfreq = d["max"], d["freq"]
for j, a in enumerate(d["accession"].astype(str)):
    if a in idx: sae_max[idx[a]] = dmax[j]; sae_freq[idx[a]] = dfreq[j]; have_s[idx[a]] = True
old_l, old_s = have_l.copy(), have_s.copy()

# control: recomputed features must match the stored ones
c = dict(np.load(f"{C}/control_features.npz", allow_pickle=True)); chk = {}
rl, rs = [], []
for j, a in enumerate(c["accession"].astype(str)):
    i = idx[a]
    x, y = layers[i].astype(np.float32), c["pool"][j]
    rl.append([np.corrcoef(x[k], y[k])[0, 1] for k in range(9)])
    rs.append(np.corrcoef(np.log1p(sae_max[i].astype(np.float32)), np.log1p(c["max"][j].astype(np.float32)))[0, 1])
rl = np.array(rl); chk = dict(layer_r_min=float(rl.min()), layer_r_median=float(np.median(rl)), sae_r_min=float(np.min(rs)), sae_r_median=float(np.median(rs)))
print("control re-embedding vs stored:", chk, flush=True)
# bf16 batches make a rare protein's features depend slightly on batch context: Q8IYI0 re-embedded alone sits
# midway between its two batched versions (layer r 0.997 to each, SAE r 0.95), so neither is wrong. Require
# agreement for nearly all controls rather than every one.
chk["frac_layer_r_above_0.99"] = float((rl.min(1) > .99).mean()); chk["frac_sae_r_above_0.99"] = float((np.array(rs) > .99).mean())
assert chk["layer_r_median"] > .999 and chk["sae_r_median"] > .999 and chk["frac_layer_r_above_0.99"] >= .95, "recomputed features do not match the stored ones"

m = dict(np.load(f"{C}/missing_features.npz", allow_pickle=True)); assert list(m["layers"]) == list(range(0, 81, 10))
for j, a in enumerate(m["accession"].astype(str)):
    i = idx[a]
    if old_s[i]:   # recomputed although stored: compare, keep the stored
        rs.append(np.corrcoef(np.log1p(sae_max[i].astype(np.float32)), np.log1p(m["max"][j].astype(np.float32)))[0, 1])
    else: sae_max[i] = m["max"][j]; sae_freq[i] = m["freq"][j]; have_s[i] = True
    if not old_l[i]: layers[i] = m["pool"][j]; have_l[i] = True
print(f"SAE agreement on proteins embedded twice: median r {np.median(rs):.4f}, min {np.min(rs):.4f}", flush=True)
has = have_l & have_s
print(f"features for {has.sum()} of {n} proteins", flush=True)

# simple sequence covariates
seq, a = {}, None
for l in open(f"{C}/union.fasta"):
    if l.startswith(">"): a = l[1:].strip(); seq[a] = []
    else: seq[a].append(l.strip())
seq = {k: "".join(v) for k, v in seq.items()}
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
AA = "ACDEFGHIKLMNPQRSTVWY"; rows = []
for a in U:
    s = seq.get(a, "")
    if not s: rows.append({}); continue
    L = len(s); q, pI = pt.charge(s)
    r = dict(log10_length=np.log10(L), charge_per_res=q / L, pI=pI, gravy=np.mean([KD.get(x, 0) for x in s]),
             disorder_promoting=sum(s.count(x) for x in "ARGQSPEK") / L)
    r.update({f"aa_{x}": s.count(x) / L for x in AA}); rows.append(r)
S = pd.DataFrame(rows, index=U); S.index.name = "accession"; S.to_csv(f"{C}/simple.tsv", sep="\t")

# folds: clusters assigned greedily (largest first) to the fold with the fewest proteins
cl = pd.read_csv(f"{C}/mm/clu_cluster.tsv", sep="\t", header=None, names=["rep", "acc"]).set_index("acc")["rep"]
grp = np.array([cl.get(a, a) for a in U])


def grouped(groups, k, seed):
    g = pd.Series(groups); sizes = g.value_counts()
    rng = np.random.default_rng(seed); order = sizes.index[np.lexsort((rng.random(len(sizes)), -sizes.values))]
    load = np.zeros(k, int); fold_of = {}
    for c_ in order: f = int(np.argmin(load)); fold_of[c_] = f; load[f] += sizes[c_]
    return np.array([fold_of[x] for x in groups])


outer = grouped(grp, 5, 0); inner = np.full((5, n), -1)
for f in range(5):
    tr = outer != f; inner[f, tr] = grouped(grp[tr], 4, f + 1)
np.save(f"{C}/layers.npy", layers); np.save(f"{C}/sae_max.npy", sae_max); np.save(f"{C}/sae_freq.npy", sae_freq)
np.savez(f"{C}/folds.npz", outer=outer, inner=inner, group=grp, has_features=has)
json.dump(dict(control=chk, n=n, n_features=int(has.sum()), n_clusters=int(len(set(grp))), fold_sizes=np.bincount(outer).tolist()),
          open(f"{C}/features_meta.json", "w"), indent=1)
print("fold sizes", np.bincount(outer), "\nCM02B_DONE", flush=True)
