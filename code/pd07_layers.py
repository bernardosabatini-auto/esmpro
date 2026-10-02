"""pd07: which ESMC hidden state predicts the salt-dependent fold change best?

pd03b pooled nine evenly spaced hidden states in one pass. This scores each one the same way
pd05 scored the last layer: ridge with the penalty chosen over a fixed grid, 5-fold
cross-validation with folds grouped by MMseqs2 cluster at 30% identity.

Three targets per layer, because baseline abundance is a confound and not a result:
  y            the log2 fold change itself
  y | abund    the fold change with the baseline-abundance dependence projected out, the
               baseline taken from DISJOINT biological replicates so the residual is not
               contaminated by the baseline's own noise (see pd05c)
  abundance    the baseline level itself, as a reference for how much of the layer's signal is
               just expression level
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
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
L = np.load(f"{D}/embeddings_layers.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
fa = [ln[1:].strip().split()[0].split("|")[0] for ln in open(f"{D}/sequences.fasta") if ln[0] == ">"]
assert len(fa) == L["pool"].shape[0], (len(fa), L["pool"].shape)
pos = {s: i for i, s in enumerate(fa)}
idx = np.array([pos[resolved.get(s, s)] for s in acc])
POOL, LAY = L["pool"][idx], L["layers"]
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])
npep = np.log10(np.maximum(blk["npeptides"], 1))
npep[~np.isfinite(npep)] = np.nanmedian(npep[np.isfinite(npep)])
print(f"{POOL.shape[0]} proteins x {len(LAY)} layers x {POOL.shape[2]}d | layers {list(LAY)}", flush=True)


def oof(X, y, gr, alpha, folds=5):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, gr):
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr]).predict(X[te])
    return p


def best(X, y, gr, alphas=(100, 1e3, 1e4, 1e5)):
    o = (-9.0, None, None)
    for al in alphas:
        p = oof(X, y, gr, al)
        r = stats.pearsonr(y, p)[0]
        if r > o[0]:
            o = (float(r), al, p)
    return o


out = {"layers": [int(x) for x in LAY]}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    bb, bc = blk["br_base"], blk[f"br_{cond}"]
    F, T = [0, 1], [2, 3, 4]                      # feature replicates | target replicates
    with np.errstate(invalid="ignore"):
        feat = np.nanmean(bb[:, F], 1)
        y = np.nanmean(bc[:, T], 1) - np.nanmean(bb[:, T], 1)
    k = (np.isfinite(feat) & np.isfinite(y)
         & (np.isfinite(bb[:, F]).sum(1) == 2) & (np.isfinite(bc[:, T]).sum(1) == 3)
         & (np.isfinite(bb[:, T]).sum(1) == 3))
    yk, fk, nk, gk = y[k], feat[k], npep[k], groups[k]
    A = np.column_stack([fk, nk])
    _, ala, pa = best(A, yk, gk, alphas=(1, 10, 100))
    resid = yk - pa
    print(f"\n{'='*74}\n{label}  ({k.sum()} proteins, y sd {yk.std():.3f}, "
          f"abundance alone r {stats.pearsonr(yk,pa)[0]:+.3f})", flush=True)
    print(f"  {'layer':>6} {'y':>22} {'y | abundance':>22} {'-> abundance':>16}", flush=True)
    rows = {}
    for li, lay in enumerate(LAY):
        X = POOL[k, li]
        ry, aly, _ = best(X, yk, gk)
        rr, alr, _ = best(X, resid, gk)
        ra, _, _ = best(X, fk, gk)
        rows[int(lay)] = dict(r_y=ry, alpha_y=float(aly), r_resid=rr, alpha_resid=float(alr),
                              r_abund=ra)
        print(f"  {lay:>6} {ry:>+14.3f} (a{aly:g})  {rr:>+14.3f} (a{alr:g})  {ra:>+14.3f}", flush=True)
    bl = max(rows, key=lambda l: rows[l]["r_resid"])
    print(f"  best on the abundance-free target: layer {bl} at r = {rows[bl]['r_resid']:+.3f} "
          f"(last layer {LAY[-1]}: {rows[int(LAY[-1])]['r_resid']:+.3f})", flush=True)
    out[cond] = dict(n=int(k.sum()), sd=float(yk.std()),
                     r_abundance=float(stats.pearsonr(yk, pa)[0]), layers=rows, best=int(bl))

json.dump(out, open(f"{D}/pd07_results.json", "w"), indent=1)
print(f"\nwrote {D}/pd07_results.json\nPD07_DONE", flush=True)
