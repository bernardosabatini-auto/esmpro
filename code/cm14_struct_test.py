"""cm14: does AlphaFold-derived structure explain anything the sequence model misses?

The decisive test is not whether structural features predict the response -- they will, because they correlate
with composition, which the model already has. It is whether they predict the **residual** of the out-of-fold
sequence prediction. Everything here is therefore run twice: against the measured value, and against the residual.

For every target:
  univariate   Spearman of each cm13 structural feature with the measured value and with the residual;
  block ridge  cross-fitted ridge (alpha by inner 5-fold CV on the training folds only) of the residual on
               (a) the structural block, (b) the simple composition block as a negative control -- the charge
               probe showed composition is fully absorbed, so (b) must come out near zero or the method is wrong,
               (c) structure + the complex-mate term of cm12, to see whether they are redundant;
  gain         R2/ceiling of the sequence model alone and with each block added, cluster-bootstrapped over the
               MMseqs2 30 % groups.

Writes data/campaign/cm14_struct.json and reports/figures/campaign/structure.png.
"""
import os, re, csv, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pdpipe import plots as pl

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True)
names = list(Tz["names"].astype(str)); Y = Tz["Y"].astype(np.float64); n = len(Y)
F = np.load(f"{C}/folds.npz"); outer, grp = F["outer"], F["group"]
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist(); pos = {a: i for i, a in enumerate(U)}
FC = ["GA20_salt75", "GA20_salt150", "GA22_ip_salt60", "GA22_ip_salt120", "GA24_ip_salt60", "GA24_ip_salt120",
      "GA33_A1_heat37", "GA33_A1_heat43", "GA33_B11_heat37", "GA33_B11_heat43", "FS76_mg"]

P = np.full((n, len(names)), np.nan)
oof = np.load(f"{C}/runs/s2/full_oof.npy")
rows = (np.load(f"{C}/runs/s2/full_rows.npy") if os.path.exists(f"{C}/runs/s2/full_rows.npy")
        else np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0])
for j, t in enumerate(names): P[rows, names.index(t)] = oof[:, j]
o5 = np.load(f"{C}/runs/s5/ip_fold_changes_oof.npy"); r5 = np.load(f"{C}/runs/s5/ip_fold_changes_rows.npy")
for j, t in enumerate(FC): P[r5, names.index(t)] = o5[:, j]

ST = pd.read_csv(f"{C}/struct.tsv", sep="\t", index_col=0).reindex(U)
SF = [c for c in ST.columns if c != "n_res_model"]
X_struct = ST[SF].values.astype(np.float64); has_struct = np.isfinite(X_struct).all(1)
SIM = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).reindex(U)
X_simple = SIM.values.astype(np.float64)
# named properties (cm07): abundance, Tm, keywords, Complex Portal membership. The s2/s5 models were trained on
# sequence only (layers + simple), so this block is the cheap non-sequence alternative any new feature must beat.
NP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).reindex(U)
Xn = NP.values.astype(np.float64); X_named = np.column_stack([np.nan_to_num(Xn), np.isfinite(Xn).astype(float)])
print(f"{has_struct.sum()} of {n} proteins have AlphaFold structural features", flush=True)

# complex-mate term (cm12), train-side version so it is deployable
comp, pairs = [], np.zeros((n, n), dtype=bool)
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    idx = sorted({pos[m] for m in mem if m in pos})
    if len(idx) >= 2: pairs[np.ix_(idx, idx)] = True
np.fill_diagonal(pairs, False)


def cx_term(v, k):
    """mean residual of co-complex members drawn from the training folds only"""
    pm = np.full(n, np.nan)
    for f in np.unique(outer):
        i, jj = np.nonzero(k & (outer == f))[0], np.nonzero(k & (outer != f))[0]
        if len(i) < 3 or len(jj) < 10: continue
        A = pairs[np.ix_(i, jj)].astype(np.float64); c = A.sum(1); ok = c > 0
        m = np.zeros(len(i)); m[ok] = (A @ v[jj])[ok] / c[ok]; pm[i] = m   # 0 = "no partner", the block mean
    return pm


def cv_ridge(X, y, k):
    """cross-fitted ridge prediction of y from X; alpha chosen inside the training folds only"""
    z = np.full(n, np.nan); mu = X[k].mean(0); sd = X[k].std(0) + 1e-9
    Xs = (X - mu) / sd
    for f in np.unique(outer):
        te, tr = k & (outer == f), k & (outer != f)
        if te.sum() < 5 or tr.sum() < 50: continue
        A, b = Xs[tr], y[tr] - y[tr].mean()
        best, ba = None, None
        for al in 10.0 ** np.arange(-1, 5):                                # inner split for alpha
            err = 0.0
            for g in np.unique(outer[tr])[:3]:
                i2, o2 = outer[tr] == g, outer[tr] != g
                if i2.sum() < 5: continue
                w = np.linalg.solve(A[o2].T @ A[o2] + al * np.eye(A.shape[1]), A[o2].T @ b[o2])
                err += ((A[i2] @ w - b[i2]) ** 2).mean()
            if best is None or err < best: best, ba = err, al
        w = np.linalg.solve(A.T @ A + ba * np.eye(A.shape[1]), A.T @ b)
        z[te] = Xs[te] @ w + y[tr].mean()
    return z


def r2c(y, p, k, t):
    return float(1 - ((y - p) ** 2)[k].sum() / ((y[k] - y[k].mean()) ** 2).sum()) / meta[t]["ceiling"]


def boot_ci(y, pa, pb, k):
    """paired cluster bootstrap on the difference in R2 between two predictions"""
    ug, gi = np.unique(grp[k], return_inverse=True); yy = y[k]; a, b = pa[k], pb[k]; d = []
    for s in range(400):
        w = np.random.default_rng(s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        mu = (w * yy).sum() / w.sum(); den = (w * (yy - mu) ** 2).sum()
        d.append(((w * (yy - a) ** 2).sum() - (w * (yy - b) ** 2).sum()) / den)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


J, uni = {}, {}
for t in names:
    j = names.index(t); y, p = Y[:, j], P[:, j]
    k = np.isfinite(y) & np.isfinite(p) & has_struct
    if k.sum() < 300: continue
    res = y - p
    uni[t] = {f: dict(measured=float(spearmanr(X_struct[k, i], y[k]).correlation),
                      residual=float(spearmanr(X_struct[k, i], res[k]).correlation)) for i, f in enumerate(SF)}
    cxv = cx_term(res, k)
    gs = cv_ridge(X_struct, res, k)                                        # structure
    gc = cv_ridge(np.nan_to_num(X_simple), res, k)                         # composition: negative control
    gn = cv_ridge(X_named, res, k)                                         # named properties: the cheap rival
    gcx = cv_ridge(cxv[:, None], res, k)                                   # complex context
    gsn = cv_ridge(np.column_stack([X_struct, X_named]), res, k)           # structure beyond named?
    gall = cv_ridge(np.column_stack([X_struct, X_named, cxv]), res, k)
    base = r2c(y, p, k, t)
    J[t] = dict(kind=meta[t]["kind"], n=int(k.sum()), base=base,
                plus_struct=r2c(y, p + gs, k, t), plus_simple=r2c(y, p + gc, k, t),
                plus_named=r2c(y, p + gn, k, t), plus_complex=r2c(y, p + gcx, k, t),
                plus_struct_named=r2c(y, p + gsn, k, t), plus_all=r2c(y, p + gall, k, t),
                ci_struct=boot_ci(y, p, p + gs, k), ci_simple=boot_ci(y, p, p + gc, k),
                ci_struct_over_named=boot_ci(y, p + gn, p + gsn, k),
                ci_complex_over_named=boot_ci(y, p + gn, p + cv_ridge(np.column_stack([X_named, cxv]), res, k), k),
                top_struct=sorted(uni[t].items(), key=lambda kv: -abs(kv[1]["residual"]))[:3])
    d = J[t]
    print(f"{t:22} {d['kind']:11} base {base:6.3f} | +simple(ctrl) {d['plus_simple']:6.3f} | +named {d['plus_named']:6.3f} "
          f"| +struct {d['plus_struct']:6.3f} [{d['ci_struct'][0]:+.3f},{d['ci_struct'][1]:+.3f}] "
          f"| struct over named [{d['ci_struct_over_named'][0]:+.3f},{d['ci_struct_over_named'][1]:+.3f}] "
          f"| +complex {d['plus_complex']:6.3f} | all {d['plus_all']:6.3f}", flush=True)

m = lambda key: np.mean([J[t][key] for t in J])
print(f"\nmean over {len(J)} targets (R2/ceiling): base {m('base'):.3f} | +composition control {m('plus_simple'):.3f}"
      f" | +named {m('plus_named'):.3f} | +structure {m('plus_struct'):.3f} | +structure&named {m('plus_struct_named'):.3f}"
      f" | +complex {m('plus_complex'):.3f} | everything {m('plus_all'):.3f}", flush=True)
json.dump(dict(gain=J, univariate=uni, n_with_struct=int(has_struct.sum()), features=SF),
          open(f"{C}/cm14_struct.json", "w"), indent=1)

# ---- figures
ts = [t for t in names if t in J]
f, ax = plt.subplots(1, 2, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]; x = np.arange(len(ts)); w = .2
for i, (lab, key, col) in enumerate([("sequence model", "base", pl.BASE), ("+ composition (control)", "plus_simple", "#999999"),
                                     ("+ named properties", "plus_named", "#6aa84f"), ("+ structure", "plus_struct", pl.HIT),
                                     ("+ complex context", "plus_complex", "#e08e0b")]):
    a.bar(x + (i - 2) * w, [J[t][key] for t in ts], w, label=lab, color=col)
a.set_xticks(x); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6.5)
a.set_ylabel("R2 / ceiling"); a.legend(fontsize=8); a.set_title("What explains the sequence model's residual", fontsize=10)
# heat map of univariate residual correlations
a = ax[1]; Mx = np.array([[uni[t][fe]["residual"] for t in ts] for fe in SF])
im = a.imshow(Mx, cmap="RdBu_r", vmin=-.2, vmax=.2, aspect="auto")
a.set_yticks(range(len(SF))); a.set_yticklabels(SF, fontsize=6.5)
a.set_xticks(range(len(ts))); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6)
a.set_title("Spearman of each structural feature with the residual", fontsize=10); f.colorbar(im, ax=a, shrink=.8)
f.tight_layout(); f.savefig(f"{FIG}/structure.png", dpi=140); plt.close(f)
print("CM14_DONE")
