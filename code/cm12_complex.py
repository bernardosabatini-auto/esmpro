"""cm12: is the variance the sequence model misses a property of the protein, or of the complex it belongs to?

A pull-down measures a protein that may be held in a multi-subunit assembly, so its level and its response to a
perturbation can be a property of the assembly rather than of the molecule. Sequence can never predict that: the
outer folds are grouped by 30 % identity, and subunits of one complex are unrelated in sequence, so they are
scattered across folds. Any complex-level effect therefore lands entirely in the residual.

Test. For every target: residual = measured - out-of-fold prediction. For each protein, the mean residual of the
OTHER measured members of its Complex Portal complexes, computed **within the same outer fold**, so the protein and
its partners are both held out of the same model. Statistic = Pearson r between a protein's own residual and that
leave-one-out partner mean; r^2 is the share of the remaining variance that complex context could recover.

Controls, because residuals are not white:
 - permutation null: residuals shuffled within outer fold, preserving every complex's size and each protein's fold
   and residual distribution, 500 draws -> the null band for r (it is not 0: shared abundance alone would lift it);
 - abundance control: the same statistic after projecting abundance_here, abundance_paxdb and log10_length out of
   the residual, since complex members are co-abundant;
 - reference: the same statistic on the measured value itself, i.e. how much of the raw signal is complex-level
   at all, which upper-bounds what is worth chasing;
 - cluster bootstrap over the MMseqs2 30 % groups for the interval on r.

Writes data/campaign/cm12_complex.json and reports/figures/campaign/complex_residual.png.
"""
import os, re, csv, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pdpipe import plots as pl

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True)
names = list(Tz["names"].astype(str)); Y = Tz["Y"].astype(np.float64); n = len(Y)
F = np.load(f"{C}/folds.npz"); outer, grp = F["outer"], F["group"]
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist(); pos = {a: i for i, a in enumerate(U)}
FC = ["GA20_salt75", "GA20_salt150", "GA22_ip_salt60", "GA22_ip_salt120", "GA24_ip_salt60", "GA24_ip_salt120",
      "GA33_A1_heat37", "GA33_A1_heat43", "GA33_B11_heat37", "GA33_B11_heat43", "FS76_mg"]

# ---- predictions: fold-change model for the 11 perturbations, full multi-task model for everything else
P = np.full((n, len(names)), np.nan)
oof = np.load(f"{C}/runs/s2/full_oof.npy")
rows = (np.load(f"{C}/runs/s2/full_rows.npy") if os.path.exists(f"{C}/runs/s2/full_rows.npy")       # pre-dates <group>_rows
        else np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0])                              # the rule cm03 used
for j, t in enumerate(names): P[rows, names.index(t)] = oof[:, j]
o5 = np.load(f"{C}/runs/s5/ip_fold_changes_oof.npy"); r5 = np.load(f"{C}/runs/s5/ip_fold_changes_rows.npy")
for j, t in enumerate(FC): P[r5, names.index(t)] = o5[:, j]

# ---- Complex Portal co-membership, as a list of member index arrays
comp = []
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    idx = sorted({pos[m] for m in mem if m in pos})
    if len(idx) >= 2: comp.append(np.array(idx))
pairs = np.zeros((n, n), dtype=bool)                      # co-membership adjacency (11.5k^2 bool = 134 MB, fine)
for idx in comp: pairs[np.ix_(idx, idx)] = True
np.fill_diagonal(pairs, False)
in_cx = pairs.any(1)
print(f"{len(comp)} complexes with >=2 measured members; {in_cx.sum()} proteins in at least one; "
      f"median partners {np.median(pairs[in_cx].sum(1)):.0f}", flush=True)

# ---- abundance control: residual with abundance projected out
NP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).reindex(U)
AB = NP[["abundance_here", "abundance_paxdb", "log10_length"]].values.astype(np.float64)
ABm = np.isfinite(AB); AB = np.where(ABm, AB, 0.0)
AB = np.column_stack([AB, ABm.astype(float), np.ones(n)])


def deabundance(v, k):
    """residual of v (rows k) on the abundance block; returns v with the abundance-explained part removed"""
    X = AB[k]; y = v[k]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    out = np.full_like(v, np.nan); out[k] = y - X @ b; return out


def fold_blocks(k):
    """per outer fold: the member indices and the dense co-membership block, built once per target mask"""
    B = []
    for f in np.unique(outer):
        i = np.nonzero(k & (outer == f))[0]
        if len(i) < 3: continue
        A = pairs[np.ix_(i, i)].astype(np.float64); c = A.sum(1)
        if (c > 0).sum() >= 10: B.append((i, A, c))
    return B


def partner_mean(v, B):
    """within outer fold, mean of the OTHER co-complex members' values; nan where a protein has no partner in fold"""
    pm = np.full(n, np.nan)
    for i, A, c in B:
        ok = c > 0; m = np.full(len(i), np.nan); m[ok] = (A @ v[i])[ok] / c[ok]; pm[i] = m
    return pm


def cross_fold_partner_mean(v, k):
    """the deployable version: a test protein's partners taken only from the TRAINING folds, i.e. proteins whose
    measurement was already in hand when its own fold model was fitted. No held-out measurement is used."""
    pm = np.full(n, np.nan)
    for f in np.unique(outer):
        i = np.nonzero(k & (outer == f))[0]; jj = np.nonzero(k & (outer != f))[0]
        if len(i) < 3 or len(jj) < 10: continue
        A = pairs[np.ix_(i, jj)].astype(np.float64); c = A.sum(1); ok = c > 0
        m = np.full(len(i), np.nan); m[ok] = (A @ v[jj])[ok] / c[ok]; pm[i] = m
    return pm


def stat(v, k, B, nperm=500, seed=0):
    """r between own value and within-fold partner mean, its permutation null, and a cluster-bootstrap CI"""
    pm = partner_mean(v, B); kk = k & np.isfinite(pm)
    if kk.sum() < 50: return None
    a, b = v[kk], pm[kk]; r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed); null = []
    for _ in range(nperm):
        vp = v.copy()
        for i, _A, _c in B: vp[i] = v[rng.permutation(i)]          # shuffle values within fold only
        pmp = partner_mean(vp, B); q = k & np.isfinite(pmp)
        null.append(np.corrcoef(vp[q], pmp[q])[0, 1])
    null = np.array(null)
    ug, gi = np.unique(grp[kk], return_inverse=True); boot = []
    for s_ in range(300):
        w = np.random.default_rng(1000 + s_).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
        boot.append(float((w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())))
    return dict(r=r, n=int(kk.sum()), null_mean=float(null.mean()), null_p975=float(np.percentile(null, 97.5)),
                z=float((r - null.mean()) / null.std()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                var_share=float(max(0.0, r ** 2 - null.mean() ** 2)))


J = {}
for t in names:
    j = names.index(t); y, p = Y[:, j], P[:, j]
    k = np.isfinite(y) & np.isfinite(p) & in_cx
    if k.sum() < 200: continue
    res = y - p; B = fold_blocks(k)
    if not B: continue
    J[t] = dict(kind=meta[t]["kind"], ceiling=meta[t]["ceiling"], n_in_complex=int(k.sum()),
                residual=stat(res, k, B), residual_deabundance=stat(deabundance(res, k), k, B), measured=stat(y, k, B))
    s, m = J[t]["residual"], J[t]["measured"]
    print(f"{t:22} {meta[t]['kind']:11} residual r={s['r']:6.3f} (null {s['null_mean']:5.3f}, z={s['z']:6.1f}, "
          f"CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f})   de-abund r={J[t]['residual_deabundance']['r']:6.3f}   measured r={m['r']:6.3f}", flush=True)

# ---- what it would be worth: add a cross-fitted complex term to the sequence prediction.
# The coefficient on the complex-mate mean is fitted on the other outer folds, so nothing about a protein's own
# fold is used to set it. Scored among in-complex proteins only, which is the population the term can apply to.
print("\n  R2/ceiling among in-complex proteins, sequence alone vs sequence + complex-mate residual", flush=True)
for t in list(J):
    j = names.index(t); y, p = Y[:, j], P[:, j]
    k = np.isfinite(y) & np.isfinite(p) & in_cx; res = y - p
    pm = partner_mean(res, fold_blocks(k)); q = k & np.isfinite(pm)
    p2 = p.copy()
    for f in np.unique(outer):
        te, tr = q & (outer == f), q & (outer != f)
        if te.sum() < 5 or tr.sum() < 20: continue
        b = np.linalg.lstsq(np.column_stack([pm[tr], np.ones(int(tr.sum()))]), res[tr], rcond=None)[0]
        p2[te] = p[te] + pm[te] * b[0] + b[1]
    # deployable variant: partner mean from training-fold partners only, scored on the same proteins
    pmx = cross_fold_partner_mean(res, k); p3 = p.copy()
    for f in np.unique(outer):
        te, tr = q & np.isfinite(pmx) & (outer == f), q & np.isfinite(pmx) & (outer != f)
        if te.sum() < 5 or tr.sum() < 20: continue
        b = np.linalg.lstsq(np.column_stack([pmx[tr], np.ones(int(tr.sum()))]), res[tr], rcond=None)[0]
        p3[te] = p[te] + pmx[te] * b[0] + b[1]
    ss = ((y[q] - y[q].mean()) ** 2).sum(); cl = meta[t]["ceiling"]
    a_, b_ = (1 - ((y - p) ** 2)[q].sum() / ss) / cl, (1 - ((y - p2) ** 2)[q].sum() / ss) / cl
    c_ = (1 - ((y - p3) ** 2)[q].sum() / ss) / cl
    J[t]["r2_over_ceiling_in_complex"] = float(a_); J[t]["r2_over_ceiling_plus_complex"] = float(b_)
    J[t]["r2_over_ceiling_plus_complex_trainside"] = float(c_); J[t]["n_with_partner"] = int(q.sum())
    print(f"  {t:22} {a_:6.3f} -> {b_:6.3f} (+{b_ - a_:.3f})   train-side partners only {c_:6.3f} (+{c_ - a_:.3f})   n={int(q.sum())}", flush=True)
print(f"\n  mean over targets: {np.mean([J[t]['r2_over_ceiling_in_complex'] for t in J]):.3f} -> "
      f"{np.mean([J[t]['r2_over_ceiling_plus_complex'] for t in J]):.3f} (held-out partners) / "
      f"{np.mean([J[t]['r2_over_ceiling_plus_complex_trainside'] for t in J]):.3f} (train-side partners)", flush=True)
J["_coverage"] = dict(n_proteins=n, n_in_complex=int(in_cx.sum()), n_complexes=len(comp))
json.dump(J, open(f"{C}/cm12_complex.json", "w"), indent=1)
ts = [t for t in names if t in J]

# ---- figure
f, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[2.0, 1, 1]))
a = ax[0]; x = np.arange(len(ts))
a.bar(x - .22, [J[t]["measured"]["r"] for t in ts], .22, color=pl.BASE, label="measured value")
a.bar(x, [J[t]["residual"]["r"] for t in ts], .22, color=pl.HIT, label="residual (measured - predicted)")
a.bar(x + .22, [J[t]["residual_deabundance"]["r"] for t in ts], .22, color="#e08e0b", label="residual, abundance removed")
a.plot(x, [J[t]["residual"]["null_mean"] for t in ts], "k_", ms=9, label="permutation null")
a.set_xticks(x); a.set_xticklabels(ts, rotation=70, ha="right", fontsize=6.5)
a.set_ylabel("r(own value, complex-mate mean) within fold"); a.axhline(0, color="k", lw=.6)
a.set_title("How much is a property of the complex, not the protein", fontsize=10); a.legend(fontsize=7)
for a_, t in zip(ax[1:], ["GA22_sup_base", "GA24_ip_salt120"]):
    if t not in J: continue
    j = names.index(t); k = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]) & in_cx; res = Y[:, j] - P[:, j]
    pm = partner_mean(res, fold_blocks(k)); q = k & np.isfinite(pm)
    pl.scatter_reg(a_, pm[q], res[q], "mean residual of complex-mates (same fold)", "own residual", f"{t}\n{meta[t]['what']}", s=4)
f.tight_layout(); f.savefig(f"{FIG}/complex_residual.png", dpi=140); plt.close(f)
print("CM12_DONE")
