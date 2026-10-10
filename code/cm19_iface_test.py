"""cm19: does native-partner occlusion explain what the sequence model misses?

The competition hypothesis, tested on experimental structures only (cm17/cm18, no prediction anywhere).
HSPB1 binds exposed hydrophobic and aggregation-prone surface on non-native clients. If that surface is
already occupied by a native partner, the protein should be a poor client; if a perturbation loosens the
native interface, it should become a better one.

Three directional predictions, fixed before looking (they are the point of the analysis; the block ridge
below is the omnibus test and is much easier to pass by accident):
  D1  salt fold change  ~ + iface_saltbridge_per_1000A2   an ionically held native interface lets go in KCl,
                                                          freeing surface for the chaperone
  D2  heat fold change  ~ + apr_occluded_frac             heat unmasks the aggregation-prone surface a partner
                                                          was covering
  D3  baseline pull-down ~ + hphob_exposed_bound_per_res  hydrophobic surface still exposed in the native
                                                          complex is what HSPB1 can bind without competing

Confounds, which are the real risk here. A protein with many PDB structures is abundant, stable and
well-studied, and abundance already correlates with every baseline at rho 0.46-0.57. So the interface block is
always tested against a DEPTH block (n_entries, partner-chain count, chain size, SASA per residue) and against
the named-property block (abundance, Tm, keywords). An interface effect that does not survive those is a
restatement of "this protein is well studied".

Everything is run against the measured value AND against the residual of the out-of-fold sequence prediction,
exactly as cm14 did, so the numbers are directly comparable to the AlphaFold-monomer null.

Writes data/campaign/cm19_iface.json and reports/figures/campaign/interface.png.
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

# ---- interface features, indexed by union row
IF = pd.read_csv(f"{C}/interface.tsv", sep="\t", index_col=0)
X = np.full((n, IF.shape[1]), np.nan); X[IF.index.values] = IF.values.astype(np.float64)
DEPTH = ["n_entries", "n_partner_chains", "n_partner_chains_max", "n_res_model", "sasa_free_per_res",
         "frac_seq", "n_chains", "n_copies", "has_nucleic", "ident"]
IFEAT = [c for c in IF.columns if c not in DEPTH]
ii = [IF.columns.get_loc(c) for c in IFEAT]; di = [IF.columns.get_loc(c) for c in DEPTH if c in IF.columns]
X_iface, X_depth = X[:, ii], X[:, di]

# A protein is in the analysis if it has ANY experimental structure. Within that set, a NaN interface feature
# is a measurement, not a missing value: "fraction of the interface that is hydrophobic" is undefined precisely
# when there is no interface, and a protein that crystallises alone is the informative zero of this analysis.
# Dropping those rows would condition on having a native partner -- the variable under test. So the ratio
# features take their natural zero and one flag records that they were undefined.
has_if = np.isfinite(X_depth).all(1)
undef = (~np.isfinite(X_iface)).any(1) & has_if
X_iface = np.where(np.isfinite(X_iface), X_iface, 0.0)
X_iface = np.column_stack([X_iface, undef.astype(float)]); IFEAT = IFEAT + ["iface_undefined"]
print(f"{has_if.sum()} of {n} proteins have an experimental structure; "
      f"{int(undef.sum())} of them have no interface in any of it", flush=True)
print(f"interface features ({len(IFEAT)}): {IFEAT}", flush=True)

NP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).reindex(U)
Xn = NP.values.astype(np.float64); X_named = np.column_stack([np.nan_to_num(Xn), np.isfinite(Xn).astype(float)])
SIM = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).reindex(U)
X_simple = np.nan_to_num(SIM.values.astype(np.float64))

pairs = np.zeros((n, n), dtype=bool)
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    idx = sorted({pos[m] for m in mem if m in pos})
    if len(idx) >= 2: pairs[np.ix_(idx, idx)] = True
np.fill_diagonal(pairs, False)


def cx_term(v, k):
    pm = np.full(n, np.nan)
    for f in np.unique(outer):
        i, jj = np.nonzero(k & (outer == f))[0], np.nonzero(k & (outer != f))[0]
        if len(i) < 3 or len(jj) < 10: continue
        A = pairs[np.ix_(i, jj)].astype(np.float64); c = A.sum(1); ok = c > 0
        m = np.zeros(len(i)); m[ok] = (A @ v[jj])[ok] / c[ok]; pm[i] = m
    return pm


def cv_ridge(Xb, y, k):
    z = np.full(n, np.nan); mu = Xb[k].mean(0); sd = Xb[k].std(0) + 1e-9
    Xs = (Xb - mu) / sd
    for f in np.unique(outer):
        te, tr = k & (outer == f), k & (outer != f)
        if te.sum() < 5 or tr.sum() < 50: continue
        A, b = Xs[tr], y[tr] - y[tr].mean()
        best, ba = None, None
        for al in 10.0 ** np.arange(-1, 5):
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


def r2c(y, p, k, t): return float(1 - ((y - p) ** 2)[k].sum() / ((y[k] - y[k].mean()) ** 2).sum()) / meta[t]["ceiling"]


def boot_ci(y, pa, pb, k):
    ug, gi = np.unique(grp[k], return_inverse=True); yy = y[k]; a, b = pa[k], pb[k]; d = []
    for s in range(400):
        w = np.random.default_rng(s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        mu = (w * yy).sum() / w.sum(); den = (w * (yy - mu) ** 2).sum()
        d.append(((w * (yy - a) ** 2).sum() - (w * (yy - b) ** 2).sum()) / den)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_rho(x, y, g, seeds=400):
    """cluster bootstrap CI on a Spearman correlation, resampling MMseqs2 groups"""
    ug, gi = np.unique(g, return_inverse=True); out = []
    byg = [np.nonzero(gi == u)[0] for u in range(len(ug))]
    rng = np.random.default_rng(0)
    for _ in range(seeds):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([byg[p] for p in pick])
        out.append(spearmanr(x[idx], y[idx]).correlation)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


# ---- positive control: does the interface measurement move when fed something it must agree with?
# A feature set that fails here cannot be interpreted downstream, whatever the residual tests say. Two
# independent anchors: Complex Portal annotation (a curated complex member should bury more surface) and
# Meltome Tm (buried surface stabilises a fold).
npc = {c: i for i, c in enumerate(NP.columns)}
ic = IFEAT.index("iface_frac")
inc = np.isfinite(Xn[:, npc["in_complex"]]) & has_if
in_c = X_iface[inc & (Xn[:, npc["in_complex"]] > 0), ic]
out_c = X_iface[inc & (Xn[:, npc["in_complex"]] <= 0), ic]
kt = has_if & np.isfinite(Xn[:, npc["Tm"]])
rho_tm = float(spearmanr(X_iface[kt, ic], Xn[kt, npc["Tm"]]).correlation)
ctrl = dict(iface_frac_in_complex=float(in_c.mean()), iface_frac_not_in_complex=float(out_c.mean()),
            n_in=int(len(in_c)), n_out=int(len(out_c)), rho_iface_frac_Tm=rho_tm, n_Tm=int(kt.sum()))
print(f"\nPOSITIVE CONTROL  iface_frac: Complex Portal members {in_c.mean():.3f} (n={len(in_c)}) vs "
      f"non-members {out_c.mean():.3f} (n={len(out_c)});  rho(iface_frac, Meltome Tm) = {rho_tm:+.3f} "
      f"(n={kt.sum()})\n", flush=True)

D1, D2, D3 = "iface_saltbridge_per_1000A2", "apr_occluded_frac", "hphob_exposed_bound_per_res"
DIRECT = {"salt": (D1, [t for t in names if meta[t]["kind"] == "salt"]),
          "temperature": (D2, [t for t in names if meta[t]["kind"] == "temperature"]),
          "baseline": (D3, [t for t in names if meta[t]["kind"] == "baseline"])}

J, uni, direct = {}, {}, {}
for t in names:
    j = names.index(t); y, p = Y[:, j], P[:, j]
    k = np.isfinite(y) & np.isfinite(p) & has_if
    if k.sum() < 200: continue
    res = y - p
    uni[t] = {f: dict(measured=float(spearmanr(X_iface[k, i], y[k]).correlation),
                      residual=float(spearmanr(X_iface[k, i], res[k]).correlation)) for i, f in enumerate(IFEAT)}
    cxv = cx_term(res, k)
    gi_ = cv_ridge(X_iface, res, k)                                        # interface
    gd = cv_ridge(X_depth, res, k)                                         # study depth / size: the control
    gc = cv_ridge(X_simple, res, k)                                        # composition: the cm14 negative control
    gn = cv_ridge(X_named, res, k)                                         # named properties: the cheap rival
    gcx = cv_ridge(cxv[:, None], res, k)
    gnd = cv_ridge(np.column_stack([X_named, X_depth]), res, k)
    gind = cv_ridge(np.column_stack([X_iface, X_named, X_depth]), res, k)
    gall = cv_ridge(np.column_stack([X_iface, X_named, X_depth, cxv]), res, k)
    # The sharpest version of the question. cm12 showed the residual is shared within complexes; if that shared
    # part is an interface property, then interface features should predict the complex-mate mean residual
    # itself -- a smoothed, less noisy target than the raw residual, so this test has more power than the ones
    # above. A null here with a positive result above would mean the interface explains something OTHER than
    # the complex effect.
    kc = k & np.isfinite(cxv) & (cxv != 0)
    r_cx = float(np.corrcoef(cv_ridge(X_iface, cxv, kc)[kc], cxv[kc])[0, 1]) if kc.sum() > 200 else np.nan
    r_cx_depth = float(np.corrcoef(cv_ridge(X_depth, cxv, kc)[kc], cxv[kc])[0, 1]) if kc.sum() > 200 else np.nan
    base = r2c(y, p, k, t)
    J[t] = dict(kind=meta[t]["kind"], n=int(k.sum()), base=base,
                plus_iface=r2c(y, p + gi_, k, t), plus_depth=r2c(y, p + gd, k, t),
                plus_simple=r2c(y, p + gc, k, t), plus_named=r2c(y, p + gn, k, t),
                plus_complex=r2c(y, p + gcx, k, t), plus_named_depth=r2c(y, p + gnd, k, t),
                plus_iface_named_depth=r2c(y, p + gind, k, t), plus_all=r2c(y, p + gall, k, t),
                ci_iface=boot_ci(y, p, p + gi_, k),
                ci_iface_over_depth=boot_ci(y, p + gd, p + cv_ridge(np.column_stack([X_iface, X_depth]), res, k), k),
                ci_iface_over_named_depth=boot_ci(y, p + gnd, p + gind, k),
                r_iface_predicts_complex=r_cx, r_depth_predicts_complex=r_cx_depth, n_complex=int(kc.sum()),
                top_iface=sorted(uni[t].items(), key=lambda kv: -abs(kv[1]["residual"]))[:3])
    d = J[t]
    print(f"{t:22} {d['kind']:11} n={d['n']:5} base {base:6.3f} | +depth(ctrl) {d['plus_depth']:6.3f} "
          f"| +composition {d['plus_simple']:6.3f} | +named {d['plus_named']:6.3f} | +interface {d['plus_iface']:6.3f} "
          f"[{d['ci_iface'][0]:+.3f},{d['ci_iface'][1]:+.3f}] | iface over depth "
          f"[{d['ci_iface_over_depth'][0]:+.3f},{d['ci_iface_over_depth'][1]:+.3f}] | all {d['plus_all']:6.3f} "
          f"|| iface->complex-mate residual r {r_cx:+.3f} (depth ctrl {r_cx_depth:+.3f})", flush=True)

m = lambda key: np.mean([J[t][key] for t in J])
print(f"\nmean over {len(J)} targets (R2/ceiling): base {m('base'):.3f} | +depth control {m('plus_depth'):.3f}"
      f" | +composition control {m('plus_simple'):.3f} | +named {m('plus_named'):.3f}"
      f" | +interface {m('plus_iface'):.3f} | +interface&named&depth {m('plus_iface_named_depth'):.3f}"
      f" | +complex {m('plus_complex'):.3f} | everything {m('plus_all'):.3f}", flush=True)
print(f"mean r(interface -> complex-mate residual) {np.nanmean([J[t]['r_iface_predicts_complex'] for t in J]):+.3f}"
      f"   (depth-only control {np.nanmean([J[t]['r_depth_predicts_complex'] for t in J]):+.3f})", flush=True)

# ---- the three directional predictions, with cluster-bootstrap CIs
print("\ndirectional predictions (one-sided in the stated direction):", flush=True)
for lab, (feat, ts_) in DIRECT.items():
    fi = IFEAT.index(feat); direct[lab] = dict(feature=feat, targets={})
    for t in ts_:
        if t not in J: continue
        j = names.index(t); y, p = Y[:, j], P[:, j]
        k = np.isfinite(y) & np.isfinite(p) & has_if
        x, res = X_iface[k, fi], (y - p)[k]
        rm = float(spearmanr(x, y[k]).correlation); rr = float(spearmanr(x, res).correlation)
        ci = boot_rho(x, res, grp[k])
        direct[lab]["targets"][t] = dict(rho_measured=rm, rho_residual=rr, ci_residual=ci, n=int(k.sum()))
        print(f"  {lab:12} {feat:30} {t:22} rho(measured) {rm:+.3f}  rho(residual) {rr:+.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)

json.dump(dict(gain=J, univariate=uni, directional=direct, control=ctrl, features=IFEAT,
               depth_features=DEPTH, n_with_iface=int(has_if.sum())),
          open(f"{C}/cm19_iface.json", "w"), indent=1)

# ---- figures
ts = [t for t in names if t in J]
f, ax = plt.subplots(1, 2, figsize=(19, 6.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]; x_ = np.arange(len(ts)); w = .17
for i, (lab, key, col) in enumerate([("sequence model", "base", pl.BASE), ("+ depth (control)", "plus_depth", "#999999"),
                                     ("+ named properties", "plus_named", "#6aa84f"), ("+ interface", "plus_iface", pl.HIT),
                                     ("+ complex context", "plus_complex", "#e08e0b")]):
    a.bar(x_ + (i - 2) * w, [J[t][key] for t in ts], w, label=lab, color=col)
a.set_xticks(x_); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6.5)
a.set_ylabel("R2 / ceiling"); a.legend(fontsize=8)
a.set_title("Does native-partner occlusion explain the residual?", fontsize=10)
a = ax[1]; Mx = np.array([[uni[t][fe]["residual"] for t in ts] for fe in IFEAT])
im = a.imshow(Mx, cmap="RdBu_r", vmin=-.2, vmax=.2, aspect="auto")
a.set_yticks(range(len(IFEAT))); a.set_yticklabels(IFEAT, fontsize=6.5)
a.set_xticks(range(len(ts))); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6)
a.set_title("Spearman of each interface feature with the residual", fontsize=10); f.colorbar(im, ax=a, shrink=.8)
f.tight_layout(); f.savefig(f"{FIG}/interface.png", dpi=140); plt.close(f)
print("CM19_DONE")
