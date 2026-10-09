"""cm04: what the shared latent of the campaign MLP carries, and how conditions relate through it.

Inputs: a cm03 run directory with per-fold latents of the best member for every protein (--save-latent).
Every read-out below is a ridge from the latent z (k dims) to one target, fitted on the outer training rows
of the SAME fold model whose latent is used (fold models' latents are not aligned with each other), penalty
on inner fold 0, scored on the outer test rows. So every number is out-of-fold.

1. Transfer (leave-one-run-out). For a group trained WITHOUT run r's targets (spec "exclude_run"), how well
   does its latent predict run r's targets? Compared with the same read-out from the group trained on all
   targets, and with ridge on the ESMC layer directly (cm03b).
2. Regress out. Within each fold, the subspace of z that predicts the abundance and baseline targets
   (rows of the ridge coefficients, orthonormalised) is projected out; each effect target is read out again
   from what is left. The drop is the part of that effect's predictable signal that it shares with
   baseline / abundance; what survives is condition-specific.
3. Target geometry. Correlations between the targets' read-out directions in z (averaged over folds):
   which conditions the model represents along the same axes.

  python cm04_latent.py --run data/campaign/runs/<sweep> --full <group trained on all targets> [--loro <prefix>]
"""
import os, json, argparse
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--full", required=True)
ap.add_argument("--loro", default=None, help="prefix of groups trained with one run's targets left out: <prefix><RUN>")
ap.add_argument("--out", default=None); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
F = np.load(f"{C}/folds.npz"); outer, inner = F["outer"], F["inner"]
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
meta = json.load(open(f"{C}/targets_meta.json")); ceil = np.array([meta[t]["ceiling"] for t in names])
def group_rows(g, targets=None):
    """row index of a group: saved per group (cm03 after the fix); otherwise recomputed with cm03's rule"""
    fn = f"{a.run}/{g}_rows.npy"
    if os.path.exists(fn): return np.load(fn)
    tix = [names.index(t) for t in (targets or names)]
    return np.nonzero(F["has_features"] & np.isfinite(Y[:, tix]).any(1))[0]
rows = group_rows(a.full); Yr = Y[rows]; out_r, in_r = outer[rows], inner[:, rows]
print(f"{a.full}: {len(rows)} rows", flush=True)
LAM = np.logspace(-3, 4, 15)
RUNS = sorted({meta[t]["run"] for t in names})
BASE = [t for t in names if meta[t]["kind"] in ("baseline", "abundance", "enrichment")]
EFF = [t for t in names if t not in BASE]


def readout(Z, y, f, remove=None):
    """ridge z -> y on fold f; returns test predictions (indices) and the coefficient vector"""
    ok = np.isfinite(y); tr = np.nonzero(ok & (out_r != f))[0]; te = np.nonzero(ok & (out_r == f))[0]
    itr, iva = tr[in_r[f, tr] > 0], tr[in_r[f, tr] == 0]
    X = Z.copy()
    if remove is not None: X = X - (X @ remove.T) @ remove
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd

    def fit(a_, lam):
        A = X[a_]; ym = y[a_].mean(); w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[a_] - ym)); return w, ym
    errs = []
    for lam in LAM:
        w, ym = fit(itr, lam); errs.append(((X[iva] @ w + ym - y[iva]) ** 2).mean())
    w, ym = fit(tr, LAM[int(np.argmin(errs))])
    return te, X[te] @ w + ym, w / sd


def r2(p, y): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def oof_scores(Z5, targets, remove_fn=None):
    res, dirs = {}, {}
    for t in targets:
        y = Yr[:, names.index(t)]; pred = np.full(len(y), np.nan); W = []
        for f in range(Z5.shape[0]):
            rem = remove_fn(f) if remove_fn else None
            te, p, w = readout(Z5[f], y, f, rem); pred[te] = p; W.append(w)
        m = np.isfinite(pred) & np.isfinite(y); res[t] = r2(pred[m], y[m]); dirs[t] = np.array(W)
    return res, dirs


J = {}
Zfull = np.load(f"{a.run}/{a.full}_latent.npy")                     # (5, n, k)
full, dirs = oof_scores(Zfull, names)
J["full_readout"] = {t: dict(r2=full[t], r2_over_ceiling=full[t] / meta[t]["ceiling"]) for t in names}
print("full model read-out, mean R2/ceiling", np.mean([v["r2_over_ceiling"] for v in J["full_readout"].values()]).round(3), flush=True)

# 2. regress out the baseline / abundance subspace, per fold
def base_subspace(f):
    W = np.stack([dirs[t][f] for t in BASE]); W = W / np.linalg.norm(W, axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(W, full_matrices=False); keep = S > 1e-6 * S[0]
    return Vt[keep]                                                    # orthonormal rows spanning the baseline read-outs
def first_axis(f):
    return base_subspace(f)[:1]
ro_all, _ = oof_scores(Zfull, EFF, base_subspace)
ro_one, _ = oof_scores(Zfull, EFF, first_axis)
J["regress_out"] = {t: dict(full=full[t], minus_first_baseline_axis=ro_one[t], minus_baseline_subspace=ro_all[t],
                            shared_fraction=1 - max(ro_all[t], 0) / full[t] if full[t] > 0 else None) for t in EFF}
J["baseline_subspace_rank"] = int(base_subspace(0).shape[0])
for t in EFF: print(f"  {t:20} full {full[t]:.3f}  minus axis1 {ro_one[t]:.3f}  minus baseline subspace {ro_all[t]:.3f}", flush=True)

# 3. geometry of the read-out directions (cosine, averaged over folds)
D = np.array([[np.mean([dirs[s][f] @ dirs[t][f] / (np.linalg.norm(dirs[s][f]) * np.linalg.norm(dirs[t][f])) for f in range(Zfull.shape[0])])
               for t in names] for s in names])
J["direction_cosine"] = dict(names=names, matrix=D.round(3).tolist())

# 1. transfer
if a.loro:
    J["transfer"] = {}
    for run in RUNS:
        fn = f"{a.run}/{a.loro}{run}_latent.npy"
        if not os.path.exists(fn): continue
        Zr = np.load(fn); tr_ = [t for t in names if meta[t]["run"] == run]
        rr = group_rows(f"{a.loro}{run}", [t for t in names if meta[t]["run"] != run])
        Zfullrows = np.full((Zr.shape[0], len(rows), Zr.shape[2]), np.nan, np.float32)      # align the LORO latent to the full model's rows
        pos = {r: i for i, r in enumerate(rr)}; idx = np.array([pos.get(r, -1) for r in rows]); okr = idx >= 0
        Zfullrows[:, okr] = Zr[:, idx[okr]]; Zfullrows = np.nan_to_num(Zfullrows)
        sc, _ = oof_scores(Zfullrows, tr_)
        for t in tr_: J["transfer"][t] = dict(left_out=sc[t], trained_with=full[t], ceiling=meta[t]["ceiling"])
        print(f"  transfer {run}: " + ", ".join(f"{t} {sc[t]:.2f} (with {full[t]:.2f})" for t in tr_), flush=True)
# 4. measured cross-run prediction (no sequence): each run's targets from the MEASURED targets of every other run,
#    ridge with median fill + missingness flags, same folds. What the conditions share as measured biology.
J["measured_cross_run"] = {}
for run in RUNS:
    tr_ = [t for t in names if meta[t]["run"] == run]; src = [t for t in names if meta[t]["run"] != run]
    Xs = Yr[:, [names.index(t) for t in src]].copy(); miss = np.isnan(Xs)
    for t in tr_:
        y = Yr[:, names.index(t)]; pred = np.full(len(y), np.nan)
        for f in range(Zfull.shape[0]):
            ok = np.isfinite(y); tr = np.nonzero(ok & (out_r != f))[0]; te = np.nonzero(ok & (out_r == f))[0]
            med = np.nanmedian(Xs[tr], 0); X = np.hstack([np.where(miss, med, Xs), miss.astype(float)])
            X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9); itr, iva = tr[in_r[f, tr] > 0], tr[in_r[f, tr] == 0]
            def fit(idx, lam):
                A = X[idx]; ym = y[idx].mean(); return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[idx] - ym)), ym
            err = [((X[iva] @ w + m - y[iva]) ** 2).mean() for w, m in (fit(itr, l) for l in LAM)]
            w, m = fit(tr, LAM[int(np.argmin(err))]); pred[te] = X[te] @ w + m
        k = np.isfinite(pred); J["measured_cross_run"][t] = dict(r2=r2(pred[k], y[k]), ceiling=meta[t]["ceiling"])
    print(f"  measured cross-run {run}: " + ", ".join(f"{t} {J['measured_cross_run'][t]['r2']:.2f}" for t in tr_), flush=True)
json.dump(J, open(a.out or f"{a.run}/cm04_{a.full}.json", "w"), indent=1, default=float)
print("CM04_DONE")
