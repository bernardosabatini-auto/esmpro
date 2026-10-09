"""cm05b: SAE features associated with each target family, with a stability check, for annotation.

Gradient attribution through the SAE-input MLP (cm05) was unstable across fold models (fold-to-fold r of the
attribution vectors 0.19-0.29): the 16,384 features are highly redundant, so each fold model spreads credit over
different members of correlated groups. A feature-level reading needs a stable statistic instead:

  family score    each target z-scored, then averaged within its family (pull-down baseline, abundance,
                  bound:free / enrichment, salt, temperature, Mg) over the targets a protein has
  association     Spearman rho of each feature (log1p max activation) with the family score
  stability       proteins split into two halves by sequence cluster (no homolog crosses), rho computed in each;
                  r between the two halves' rho vectors over all features, and the overlap of the top 50
  consistency     overlap of the top 50 associated features with the top 50 by cm05 attribution

Writes data/campaign/attrib/assoc.json and the feature list to annotate (top 15 each sign per family).
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"; OUT = f"{C}/attrib"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
F = np.load(f"{C}/folds.npz"); has, grp = F["has_features"], F["group"]
FAM = {"pull-down baseline": "baseline", "abundance": "abundance", "bound:free / enrichment": "enrichment", "salt": "salt", "temperature": "temperature", "Mg": "Mg"}
X = np.log1p(np.load(f"{C}/sae_max.npy").astype(np.float32))
alive = (X[has] > 0).mean(0) >= .01                                # features active in at least 1 % of proteins
print(f"{alive.sum()} features active in >= 1 % of proteins", flush=True)
rng = np.random.default_rng(0); ug = np.unique(grp); half = np.isin(grp, rng.choice(ug, len(ug) // 2, replace=False))


def spearman_all(Xs, y):
    """rho of every column with y (ranks; ties averaged)"""
    ry = rankdata(y); ry = (ry - ry.mean()) / ry.std()
    R = np.apply_along_axis(rankdata, 0, Xs); R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    return (R * ry[:, None]).mean(0)


attr = np.load(f"{OUT}/attrib.npz", allow_pickle=True); A = attr["attribution"]; an = list(attr["names"])
J = {}; want = set()
for fam, kind in FAM.items():
    ts = [t for t in names if meta[t]["kind"] == kind]; Z = np.stack([(Y[:, names.index(t)] - np.nanmean(Y[:, names.index(t)])) / np.nanstd(Y[:, names.index(t)]) for t in ts], 1)
    score = np.nanmean(Z, 1); ok = has & np.isfinite(score)
    rho = np.full(X.shape[1], np.nan); rho[alive] = spearman_all(X[ok][:, alive], score[ok])
    h1 = np.full(X.shape[1], np.nan); h2 = h1.copy()
    h1[alive] = spearman_all(X[ok & half][:, alive], score[ok & half]); h2[alive] = spearman_all(X[ok & ~half][:, alive], score[ok & ~half])
    m = np.isfinite(h1) & np.isfinite(h2)
    top = lambda v, k=50: set(np.argsort(-np.abs(np.nan_to_num(v)))[:k])
    att = A[[an.index(t) for t in ts]].mean(0)
    pos, neg = np.argsort(-np.nan_to_num(rho, nan=-9))[:15], np.argsort(np.nan_to_num(rho, nan=9))[:15]
    J[fam] = dict(n_proteins=int(ok.sum()), halves_r=float(np.corrcoef(h1[m], h2[m])[0, 1]), top50_overlap_halves=len(top(h1) & top(h2)) / 50,
                  top50_overlap_with_attribution=len(top(rho) & top(att)) / 50,
                  positive=[dict(feature=int(i), rho=float(rho[i])) for i in pos], negative=[dict(feature=int(i), rho=float(rho[i])) for i in neg])
    want |= set(pos.tolist()) | set(neg.tolist())
    print(f"{fam:26} halves r {J[fam]['halves_r']:.3f}  top-50 overlap between halves {J[fam]['top50_overlap_halves']:.2f}  "
          f"with attribution {J[fam]['top50_overlap_with_attribution']:.2f}  strongest rho {rho[pos[0]]:+.2f} / {rho[neg[0]]:+.2f}", flush=True)
json.dump(J, open(f"{OUT}/assoc.json", "w"), indent=1)
open(f"{OUT}/assoc_features.txt", "w").write("\n".join(map(str, sorted(want))))
print(f"{len(want)} features to annotate\nCM05B_DONE")
