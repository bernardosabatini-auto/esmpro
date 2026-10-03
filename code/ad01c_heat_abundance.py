"""ad01c: is the GA_33 heat response confounded with abundance, as the salt response was?

The 43 vs 35 C response correlates at -0.45 with each protein's level in the 35 C control
pull-downs. As for salt (pd05c), two questions:
  1. Is the dependence real, or manufactured by sharing the 35 C term between feature and target?
     Abundance from replicates BR1-3, response from BR4-6 (and the reverse).
  2. Is the sequence signal abundance in disguise? ESMC (final layer) ridge on the response, on
     abundance, and on the response with abundance projected out, on the ga05 cluster folds.
"""
import os, csv
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
E = {50: np.load(f"{GA}/emb_L50.npy"), 80: np.load(f"{GA}/emb_L80.npy")}
r2 = lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def mean_cols(T, cols):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][:, cols] for c in ("21A", "24")]), 1)


def cvr(X, y, fid, alphas=(1e3, 3e3, 1e4, 3e4, 1e5)):
    best = (-9, None)
    for al in alphas:
        p = np.zeros(len(y))
        for f in range(5):
            tr, te = fid != f, fid == f
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
        v = r2(y, p)
        if v > best[0]:
            best = (v, p)
    return best


for t, T in (("temp_avg_43v35", "43"), ("temp_avg_37v35", "37")):
    rows = FG[f"rows_{t}"]; fid = FG[f"fold_{t}"]
    print(f"\n{t}", flush=True)
    for nm, F, R in (("BR1-3 -> BR4-6", [0, 1, 2], [3, 4, 5]), ("BR4-6 -> BR1-3", [3, 4, 5], [0, 1, 2])):
        ab = mean_cols("35", F)[rows]
        yr = (mean_cols(T, R) - mean_cols("35", R))[rows]
        ab_same = mean_cols("35", R)[rows]
        ok = np.isfinite(ab) & np.isfinite(yr) & np.isfinite(ab_same)
        print(f"  {nm}: corr(response, abundance) same replicates {np.corrcoef(yr[ok], ab_same[ok])[0,1]:+.3f}, "
              f"disjoint {np.corrcoef(yr[ok], ab[ok])[0,1]:+.3f}", flush=True)
    # sequence vs abundance, on the full target with abundance from disjoint replicates
    ab = mean_cols("35", [0, 1, 2])[rows]
    y = (mean_cols(T, [3, 4, 5]) - mean_cols("35", [3, 4, 5]))[rows]
    ok = np.isfinite(ab) & np.isfinite(y)
    yk, ak, fk = y[ok], ab[ok], fid[ok]
    A = np.column_stack([ak, ak ** 2])
    pa = np.zeros(len(yk))
    for f in range(5):
        tr, te = fk != f, fk == f
        w = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), A[tr]]), yk[tr], rcond=None)[0]
        pa[te] = np.column_stack([np.ones(te.sum()), A[te]]) @ w
    resid = yk - pa
    print(f"  abundance alone explains {100*r2(yk, pa):.1f}% (disjoint replicates, cross-validated)", flush=True)
    for L in (50, 80):
        X = E[L][rows][ok]
        ry, _ = cvr(X, yk, fk); ra, _ = cvr(X, ak, fk); rr, _ = cvr(X, resid, fk)
        print(f"  ESMC L{L}: -> response {100*ry:.1f}%   -> abundance {100*ra:.1f}%   -> response with abundance projected out {100*rr:.1f}% "
              f"(residual keeps {100*resid.var()/yk.var():.0f}% of the variance)", flush=True)
print("\nAD01C_DONE")
