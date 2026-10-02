"""pd07b: score the layer pd07 chose on the SAME target as the main table, so the numbers are
directly comparable -- same 7-of-10 rule, same cluster-grouped folds, r and predictive R2.
pd07 used a disjoint-replicate target to keep the assay's dynamic range out of the comparison,
which makes its absolute values slightly different from the headline table.
"""
import os, csv, json
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/pd_data"
rng = np.random.default_rng(0)
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
L = np.load(f"{D}/embeddings_layers.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
fa = [ln[1:].strip().split()[0].split("|")[0] for ln in open(f"{D}/sequences.fasta") if ln[0] == ">"]
pos = {s: i for i, s in enumerate(fa)}
idx = np.array([pos[resolved.get(s, s)] for s in acc])
POOL, LAY = L["pool"][idx], list(L["layers"])
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])


def oof(X, y, gr, al):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, gr):
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
    return p


def r2f(y, p, s):
    return 1 - np.sum((y[s] - p[s]) ** 2) / np.sum((y[s] - y[s].mean()) ** 2)


out = {}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    y = blk[f"y_{cond}"].astype(float)
    k = blk[f"keep_{cond}"].astype(bool) & np.isfinite(y)
    yk, gk = y[k], groups[k]
    print(f"\n{label}  n={k.sum()}", flush=True)
    res, preds = {}, {}
    for lay in (50, 80):
        li = LAY.index(lay)
        bb = (-9e9, None, None)
        for al in (1e3, 1e4, 1e5):
            p = oof(POOL[k, li], yk, gk, al)
            r2 = 1 - np.sum((yk - p) ** 2) / np.sum((yk - yk.mean()) ** 2)
            if r2 > bb[0]:
                bb = (r2, al, p)
        r2, al, p = bb
        preds[lay] = p
        res[lay] = dict(r=float(stats.pearsonr(yk, p)[0]), r2=float(r2), alpha=float(al))
        print(f"  layer {lay:>2}  r = {res[lay]['r']:+.3f}  R2 = {r2:+.3f}  "
              f"({100*r2:.1f}% of variance)  alpha {al:g}", flush=True)
    i = np.arange(len(yk))
    d = [r2f(yk, preds[50], s) - r2f(yk, preds[80], s)
         for s in (rng.choice(i, len(i), replace=True) for _ in range(2000))]
    print(f"  layer 50 over layer 80: {100*(res[50]['r2']-res[80]['r2']):+.1f} points of variance "
          f"[{100*np.percentile(d,2.5):+.1f},{100*np.percentile(d,97.5):+.1f}]", flush=True)
    out[cond] = dict(n=int(k.sum()), layers=res,
                     gain=float(res[50]["r2"] - res[80]["r2"]),
                     gain_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
json.dump(out, open(f"{D}/pd07b_results.json", "w"), indent=1)
print("\nPD07B_DONE", flush=True)
