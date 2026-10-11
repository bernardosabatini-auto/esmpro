"""cm26: is the complex effect of 3c a capture nonlinearity, co-abundance, or co-capture?

cm12 established that what the sequence model misses is shared among members of a complex, and ruled out
abundance with a LINEAR projection of log abundance. That control is weaker than it looks, because capture
in these pull-downs is demonstrably nonlinear: pull-down level grows as input^0.35 (`ad15`), so IP
efficiency falls as input^-0.65. A protein's position on that curve is set by its abundance, and complex
members are co-abundant, so shared curve position is a complete alternative explanation of 3c with no
chaperone biology in it.

Four things are tested here, in increasing order of how much they discriminate.

1. THE MEASUREMENT-STAGE GRADIENT. The decisive observation is already in cm12's output and was not read.
   GA_22 and GA_24 measure the same proteins, in the same reaction, under the same salt perturbation, at
   three stages: the supernatant (the unbound pool, which never passes through capture), the pull-down
   (which does), and the partition (their ratio, in which the lysate contribution is divided out
   experimentally). If shared position on a nonlinear capture curve manufactured the complex effect, the
   captured measurement would show MORE sharing than the uncaptured one. The ordering runs the other way,
   and this section makes that a paired, matched comparison with intervals rather than an eyeball.

2. A FLEXIBLE ABUNDANCE CONTROL. cm12's projection is linear in log abundance. A pure power law is linear
   in log space, so it is removed, but curvature and any abundance x response interaction are not. Here the
   abundance block is replaced by decile indicators plus their interactions, which absorbs any monotone or
   non-monotone function of abundance, and the complex statistic is recomputed.

3. ABUNDANCE-DISCORDANT PARTNERS. If co-abundance mediates the effect, it must be carried by partners of
   SIMILAR abundance. Partner means are therefore recomputed using only partners in a different abundance
   quintile from the protein itself. Co-abundance predicts the effect collapses; co-capture does not.

4. THE CO-CAPTURE SIGNATURE. Stoichiometric co-capture -- pull one subunit, the assembly follows -- is a
   nonlinearity of a different kind: an OR/max over subunits rather than a saturating function of
   concentration. It predicts an ASYMMETRY that additive sharing does not. A weak binder in a complex with a
   strong binder is carried up; a strong binder in a complex of weak binders is not dragged down as much. So
   the partner effect should be stronger for proteins whose own predicted binding is low. Tested as a
   tertile split on the out-of-fold prediction, and as max-partner versus mean-partner.

   Note the asymmetry of evidence in the max-vs-mean comparison: the mean is a lower-variance estimator of a
   shared additive factor, so "mean wins" is weak evidence for additivity, while "max wins" is strong
   evidence for an OR. Reported, but the tertile test carries the weight.

Writes data/campaign/cm26_nonlinearity.json and reports/figures/campaign/nonlinearity.png.
"""
import os, re, csv, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
os.makedirs(FIG, exist_ok=True)

meta = json.load(open(f"{C}/targets_meta.json"))
Tz = np.load(f"{C}/targets.npz", allow_pickle=True)
names = list(Tz["names"].astype(str)); Y = Tz["Y"].astype(np.float64); n = len(Y)
F = np.load(f"{C}/folds.npz"); outer, grp = F["outer"], F["group"]
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist()
pos = {a: i for i, a in enumerate(U)}
FC = ["GA20_salt75", "GA20_salt150", "GA22_ip_salt60", "GA22_ip_salt120", "GA24_ip_salt60", "GA24_ip_salt120",
      "GA33_A1_heat37", "GA33_A1_heat43", "GA33_B11_heat37", "GA33_B11_heat43", "FS76_mg"]

# ---- predictions, exactly as cm12 assembles them (fold-change model for the 11 perturbations, full model else)
P = np.full((n, len(names)), np.nan)
oof = np.load(f"{C}/runs/s2/full_oof.npy")
rows = (np.load(f"{C}/runs/s2/full_rows.npy") if os.path.exists(f"{C}/runs/s2/full_rows.npy")
        else np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0])
for j, t in enumerate(names): P[rows, names.index(t)] = oof[:, j]
o5 = np.load(f"{C}/runs/s5/ip_fold_changes_oof.npy"); r5 = np.load(f"{C}/runs/s5/ip_fold_changes_rows.npy")
for j, t in enumerate(FC): P[r5, names.index(t)] = o5[:, j]

# ---- Complex Portal co-membership
comp = []
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    idx = sorted({pos[m] for m in mem if m in pos})
    if len(idx) >= 2: comp.append(np.array(idx))
pairs = np.zeros((n, n), dtype=bool)
for idx in comp: pairs[np.ix_(idx, idx)] = True
np.fill_diagonal(pairs, False)
in_cx = pairs.any(1)
print(f"{len(comp)} complexes, {in_cx.sum()} proteins in at least one", flush=True)

NP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).reindex(U)
ab_here = NP["abundance_here"].values.astype(np.float64)
ab_pax = NP["abundance_paxdb"].values.astype(np.float64)
loglen = NP["log10_length"].values.astype(np.float64)


def _block(cols):
    X = np.column_stack(cols); m = np.isfinite(X)
    return np.column_stack([np.where(m, X, 0.0), m.astype(float), np.ones(len(X))])


AB_LIN = _block([ab_here, ab_pax, loglen])


def _deciles(v, q=10):
    """indicator columns for within-finite deciles of v; an extra column for missing"""
    ok = np.isfinite(v)
    D = np.zeros((len(v), q + 1))
    if ok.sum() > q:
        e = np.quantile(v[ok], np.linspace(0, 1, q + 1)); e[0] -= 1e-9; e[-1] += 1e-9
        b = np.clip(np.digitize(v[ok], e[1:-1]), 0, q - 1)
        D[np.nonzero(ok)[0], b] = 1.0
    D[~ok, q] = 1.0
    return D


# flexible abundance block: deciles of each abundance source, their product (the interaction that a
# nonlinear capture curve would create), and length deciles
Dh, Dp, Dl = _deciles(ab_here), _deciles(ab_pax), _deciles(loglen)
AB_FLEX = np.column_stack([Dh, Dp, Dl, (Dh[:, :, None] * Dp[:, None, :]).reshape(n, -1), np.ones(n)])
print(f"abundance blocks: linear {AB_LIN.shape[1]} columns, flexible {AB_FLEX.shape[1]} columns", flush=True)


def project_out(v, k, X):
    b = np.linalg.lstsq(X[k], v[k], rcond=None)[0]
    out = np.full_like(v, np.nan); out[k] = v[k] - X[k] @ b
    return out


def fold_blocks(k, adj=None):
    A_ = pairs if adj is None else adj
    B = []
    for f in np.unique(outer):
        i = np.nonzero(k & (outer == f))[0]
        if len(i) < 3: continue
        A = A_[np.ix_(i, i)].astype(np.float64); c = A.sum(1)
        if (c > 0).sum() >= 10: B.append((i, A, c))
    return B


def partner_mean(v, B):
    pm = np.full(n, np.nan)
    for i, A, c in B:
        ok = c > 0; m = np.full(len(i), np.nan); m[ok] = (A @ v[i])[ok] / c[ok]; pm[i] = m
    return pm


def partner_max(v, B):
    """max over co-complex partners within fold; nan where fewer than 2 partners (max == mean otherwise)"""
    pmx = np.full(n, np.nan)
    for i, A, c in B:
        V = np.where(A > 0, v[i][None, :], -np.inf)
        m = V.max(1); m[c < 2] = np.nan
        pmx[i] = m
    return pmx


def cboot_r(a, b, g, nb=400, seed=0):
    """cluster bootstrap over sequence groups; returns the CI on r"""
    ug, gi = np.unique(g, return_inverse=True); out = []
    for s in range(nb):
        w = np.random.default_rng(seed + s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
        den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
        out.append(float((w * (a - ma) * (b - mb)).sum() / den) if den > 0 else np.nan)
    o = np.array(out); o = o[np.isfinite(o)]
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def cboot_diff(a1, b1, a2, b2, g, nb=400, seed=0):
    """paired cluster bootstrap on r1 - r2, both computed on the SAME resampled proteins"""
    ug, gi = np.unique(g, return_inverse=True); out = []
    for s in range(nb):
        w = np.random.default_rng(seed + s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        rs = []
        for a, b in ((a1, b1), (a2, b2)):
            ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
            den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
            rs.append((w * (a - ma) * (b - mb)).sum() / den if den > 0 else np.nan)
        out.append(rs[0] - rs[1])
    o = np.array(out); o = o[np.isfinite(o)]
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def cboot_diff_unpaired(a1, b1, g1, a2, b2, g2, nb=400, seed=0):
    """CI on r1 - r2 where the two correlations are computed on DISJOINT protein sets (e.g. tertiles),
    so each is resampled over its own sequence groups independently."""
    def draws(a, b, g):
        ug, gi = np.unique(g, return_inverse=True); o = []
        for s in range(nb):
            w = np.random.default_rng(seed + s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
            ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
            den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
            o.append((w * (a - ma) * (b - mb)).sum() / den if den > 0 else np.nan)
        return np.array(o)
    d = draws(a1, b1, g1) - draws(a2, b2, g2); d = d[np.isfinite(d)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def resid(t):
    j = names.index(t); return Y[:, j] - P[:, j], np.isfinite(Y[:, j]) & np.isfinite(P[:, j]) & in_cx


J = {}

# =====================================================================================================
# 1. The measurement-stage gradient, matched within run and perturbation
# =====================================================================================================
print("\n==== 1. complex sharing by measurement stage (same proteins, same reaction, same perturbation) ====",
      flush=True)
TRIPLETS = [("GA22", "salt60"), ("GA22", "salt120"), ("GA24", "salt60"), ("GA24", "salt120"),
            ("GA22", "base"), ("GA24", "base")]
STAGE = {"sup": "supernatant (never captured)", "ip": "pull-down (captured)", "part": "partition (ratio)"}
J["stage"] = {}
for run, pert in TRIPLETS:
    ts = {s: f"{run}_{s}_{pert}" for s in ("sup", "ip", "part")}
    if not all(t in names for t in ts.values()): continue
    R, K = {}, None
    for s, t in ts.items():
        R[s], k = resid(t)
        K = k if K is None else (K & k)
    B = fold_blocks(K)
    if not B: continue
    row = {}
    pm = {s: partner_mean(R[s], B) for s in R}
    kk = K & np.all([np.isfinite(pm[s]) for s in pm], 0)      # identical protein set across the three stages
    for s in R:
        a, b = R[s][kk], pm[s][kk]
        row[s] = dict(r=float(np.corrcoef(a, b)[0, 1]), ci=cboot_r(a, b, grp[kk]), n=int(kk.sum()))
    row["d_sup_minus_ip"] = cboot_diff(R["sup"][kk], pm["sup"][kk], R["ip"][kk], pm["ip"][kk], grp[kk])
    row["d_ip_minus_part"] = cboot_diff(R["ip"][kk], pm["ip"][kk], R["part"][kk], pm["part"][kk], grp[kk])
    # Reliability runs AGAINST this result and has to be shown, not assumed away. The supernatant salt
    # responses are the least reliable targets in the campaign (ceiling 0.76-0.91) and the pull-downs the
    # most (0.90-0.97), so attenuation shrinks the supernatant's r more than the pull-down's. Both own
    # residual and partner mean carry the same target's noise, so the attenuation factor is the
    # reliability itself, not its square root.
    for s, t in ts.items():
        row[s]["ceiling"] = float(meta[t]["ceiling"])
        row[s]["r_disattenuated"] = float(row[s]["r"] / meta[t]["ceiling"])
    J["stage"][f"{run}_{pert}"] = row
    print(f"  {run} {pert:7s} n={row['sup']['n']:5d}  "
          f"sup {row['sup']['r']:.3f} [{row['sup']['ci'][0]:.3f},{row['sup']['ci'][1]:.3f}]  "
          f"ip {row['ip']['r']:.3f} [{row['ip']['ci'][0]:.3f},{row['ip']['ci'][1]:.3f}]  "
          f"part {row['part']['r']:.3f} [{row['part']['ci'][0]:.3f},{row['part']['ci'][1]:.3f}]   "
          f"sup-ip {row['d_sup_minus_ip'][0]:+.3f},{row['d_sup_minus_ip'][1]:+.3f}  "
          f"ip-part {row['d_ip_minus_part'][0]:+.3f},{row['d_ip_minus_part'][1]:+.3f}", flush=True)
    print(f"      disattenuated (r / ceiling): sup {row['sup']['r_disattenuated']:.3f} "
          f"(ceil {row['sup']['ceiling']:.3f})  ip {row['ip']['r_disattenuated']:.3f} "
          f"(ceil {row['ip']['ceiling']:.3f})  part {row['part']['r_disattenuated']:.3f} "
          f"(ceil {row['part']['ceiling']:.3f})", flush=True)

# =====================================================================================================
# 2. Flexible abundance control
# =====================================================================================================
print("\n==== 2. linear vs flexible (decile + interaction) abundance control ====", flush=True)
J["abundance_control"] = {}
for t in names:
    res, k = resid(t)
    if k.sum() < 200: continue
    B = fold_blocks(k)
    if not B: continue
    out = {}
    for lab, X in (("raw", None), ("linear", AB_LIN), ("flexible", AB_FLEX)):
        v = res if X is None else project_out(res, k, X)
        pm = partner_mean(v, B); q = k & np.isfinite(pm) & np.isfinite(v)
        out[lab] = float(np.corrcoef(v[q], pm[q])[0, 1])
    J["abundance_control"][t] = out
    print(f"  {t:22s} raw {out['raw']:6.3f}   linear {out['linear']:6.3f}   flexible {out['flexible']:6.3f}",
          flush=True)
_ac = J["abundance_control"]
print(f"  MEAN over {len(_ac)} targets: raw {np.mean([v['raw'] for v in _ac.values()]):.3f}   "
      f"linear {np.mean([v['linear'] for v in _ac.values()]):.3f}   "
      f"flexible {np.mean([v['flexible'] for v in _ac.values()]):.3f}", flush=True)

# =====================================================================================================
# 3. Abundance-discordant partners
# =====================================================================================================
print("\n==== 3. partners matched vs discordant in abundance ====", flush=True)
ab = np.where(np.isfinite(ab_here), ab_here, np.where(np.isfinite(ab_pax), ab_pax, np.nan))
okq = np.isfinite(ab)
qn = np.full(n, -1)
if okq.sum() > 5:
    e = np.quantile(ab[okq], [0.2, 0.4, 0.6, 0.8])
    qn[okq] = np.digitize(ab[okq], e)
same_q = (qn[:, None] == qn[None, :]) & (qn[:, None] >= 0)
ADJ_SAME = pairs & same_q
ADJ_DIFF = pairs & ~same_q & (qn[:, None] >= 0) & (qn[None, :] >= 0)
J["partner_abundance"] = {}
for t in names:
    res, k = resid(t)
    if k.sum() < 200: continue
    row = {}
    for lab, adj in (("all", pairs), ("same_quintile", ADJ_SAME), ("diff_quintile", ADJ_DIFF)):
        B = fold_blocks(k, adj)
        if not B: continue
        pm = partner_mean(res, B); q = k & np.isfinite(pm)
        if q.sum() < 100: continue
        row[lab] = dict(r=float(np.corrcoef(res[q], pm[q])[0, 1]), n=int(q.sum()))
    if "all" in row and "diff_quintile" in row:
        J["partner_abundance"][t] = row
        print(f"  {t:22s} all {row['all']['r']:6.3f} (n={row['all']['n']:5d})   "
              f"same-quintile {row.get('same_quintile',{}).get('r',float('nan')):6.3f}   "
              f"diff-quintile {row['diff_quintile']['r']:6.3f} (n={row['diff_quintile']['n']:5d})", flush=True)
_pa = J["partner_abundance"]
if _pa:
    print(f"  MEAN: all {np.mean([v['all']['r'] for v in _pa.values()]):.3f}   "
          f"same {np.mean([v['same_quintile']['r'] for v in _pa.values() if 'same_quintile' in v]):.3f}   "
          f"diff {np.mean([v['diff_quintile']['r'] for v in _pa.values()]):.3f}", flush=True)

# =====================================================================================================
# 4. The co-capture signature
# =====================================================================================================
print("\n==== 4a. partner effect by the protein's OWN predicted binding (OR/co-capture asymmetry) ====",
      flush=True)
J["own_binding_tertile"] = {}
# Only abundance-free binding targets are usable here. Stratifying a BASELINE by its predicted level would
# stratify by predicted abundance, which is the confound this whole script exists to remove.
# The stratifier is, wherever one exists, the protein's measured binding in an INDEPENDENT run -- different
# reaction, different measurement noise, and it never enters this target's residual. Otherwise the
# out-of-fold prediction of the target itself, which is at least not the protein's own measurement.
INDEP = {"GA22_part_base": "GA24_part_base", "GA24_part_base": "GA22_part_base",
         "GA33_A1_enrich": "GA33_B11_enrich", "GA33_B11_enrich": "GA33_A1_enrich"}
CAPTURE_T = [t for t in names if meta[t]["kind"] == "enrichment"]
for t in CAPTURE_T:
    res, k = resid(t); j = names.index(t)
    B = fold_blocks(k)
    if not B: continue
    pm = partner_mean(res, B); q = k & np.isfinite(pm)
    if q.sum() < 300: continue
    if t in INDEP and INDEP[t] in names:
        strat, how = Y[:, names.index(INDEP[t])], f"measured {INDEP[t]}"
    else:
        strat, how = P[:, j], "out-of-fold prediction"
    q = q & np.isfinite(strat)
    if q.sum() < 300: continue
    e = np.quantile(strat[q], [1 / 3, 2 / 3])
    row = {"stratifier": how}
    sels = {"low": q & (strat < e[0]), "mid": q & (strat >= e[0]) & (strat < e[1]), "high": q & (strat >= e[1])}
    for lab, s in sels.items():
        if s.sum() < 80: continue
        row[lab] = dict(r=float(np.corrcoef(res[s], pm[s])[0, 1]), n=int(s.sum()))
    if {"low", "high"} <= set(row):
        lo, hi = sels["low"], sels["high"]
        row["d_low_minus_high"] = cboot_diff_unpaired(res[lo], pm[lo], grp[lo], res[hi], pm[hi], grp[hi])
        J["own_binding_tertile"][t] = row
        print(f"  {t:22s} [{how}]  low {row['low']['r']:6.3f}  "
              f"mid {row.get('mid',{}).get('r',float('nan')):6.3f}  high {row['high']['r']:6.3f}   "
              f"low-high CI [{row['d_low_minus_high'][0]:+.3f},{row['d_low_minus_high'][1]:+.3f}]", flush=True)
_ob = J["own_binding_tertile"]
if _ob:
    print(f"  MEAN: low {np.mean([v['low']['r'] for v in _ob.values()]):.3f}   "
          f"high {np.mean([v['high']['r'] for v in _ob.values()]):.3f}", flush=True)

print("\n==== 4b. max-partner vs mean-partner (>=2 partners in fold) ====", flush=True)
J["max_vs_mean"] = {}
for t in names:
    res, k = resid(t)
    if k.sum() < 200: continue
    B = fold_blocks(k)
    if not B: continue
    pmean, pmax = partner_mean(res, B), partner_max(res, B)
    q = k & np.isfinite(pmean) & np.isfinite(pmax)
    if q.sum() < 150: continue
    rm, rx = float(np.corrcoef(res[q], pmean[q])[0, 1]), float(np.corrcoef(res[q], pmax[q])[0, 1])
    J["max_vs_mean"][t] = dict(mean=rm, max=rx, n=int(q.sum()),
                               d=cboot_diff(res[q], pmax[q], res[q], pmean[q], grp[q]))
    print(f"  {t:22s} mean {rm:6.3f}   max {rx:6.3f}   max-mean CI "
          f"[{J['max_vs_mean'][t]['d'][0]:+.3f},{J['max_vs_mean'][t]['d'][1]:+.3f}]  n={int(q.sum())}", flush=True)
_mm = J["max_vs_mean"]
if _mm:
    print(f"  MEAN: mean-partner {np.mean([v['mean'] for v in _mm.values()]):.3f}   "
          f"max-partner {np.mean([v['max'] for v in _mm.values()]):.3f}", flush=True)

json.dump(J, open(f"{C}/cm26_nonlinearity.json", "w"), indent=1)
print(f"\nwrote {C}/cm26_nonlinearity.json")

# =====================================================================================================
# figure
# =====================================================================================================
fig, ax = plt.subplots(1, 4, figsize=(21, 5.0))
fig.subplots_adjust(wspace=0.32)

a = ax[0]
ks = list(J["stage"]); x = np.arange(len(ks)); w = 0.26
for off, s, col in ((-w, "sup", "#36c"), (0, "ip", "#c44"), (w, "part", "#2a7")):
    v = [J["stage"][k][s]["r"] for k in ks]
    lo = [J["stage"][k][s]["r"] - J["stage"][k][s]["ci"][0] for k in ks]
    hi = [J["stage"][k][s]["ci"][1] - J["stage"][k][s]["r"] for k in ks]
    a.bar(x + off, v, w, color=col, label=STAGE[s], yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
    a.plot(x + off, [J["stage"][k][s]["r_disattenuated"] for k in ks], "k_", ms=9, mew=1.4,
           label="divided by the target's own reliability" if s == "sup" else None)
a.set_xticks(x); a.set_xticklabels(ks, rotation=35, ha="right", fontsize=8)
a.set_ylabel("r(own residual, complex-mate mean)")
a.legend(fontsize=7.5); a.set_title("A  Complex sharing by measurement stage\n(same proteins, same reaction)",
                                    fontsize=10)

a = ax[1]
ts = list(J["abundance_control"])
a.scatter([J["abundance_control"][t]["raw"] for t in ts], [J["abundance_control"][t]["flexible"] for t in ts],
          s=26, c="#36c", edgecolor="none")
lim = [0, max(max(J["abundance_control"][t]["raw"] for t in ts),
              max(J["abundance_control"][t]["flexible"] for t in ts)) * 1.08]
a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("r, no abundance control"); a.set_ylabel("r, flexible abundance control")
a.set_title("B  A nonlinear abundance control\nremoves almost nothing", fontsize=10)

a = ax[2]
ts = list(J["partner_abundance"])
if ts:
    aa = np.array([J["partner_abundance"][t]["all"]["r"] for t in ts])
    dd = np.array([J["partner_abundance"][t]["diff_quintile"]["r"] for t in ts])
    ctl = np.array([t == "GA33_input" for t in ts])          # the pure-abundance target: must collapse
    a.scatter(aa[~ctl], dd[~ctl], s=28, c="#2a7", edgecolor="none", label="33 targets")
    a.scatter(aa[ctl], dd[ctl], s=70, c="#c44", marker="D", edgecolor="k", lw=0.6, zorder=5,
              label="GA33_input (pure abundance)")
    lim = [min(0, dd.min()) - 0.05, max(aa.max(), dd.max()) * 1.08]
    a.plot(lim, lim, "k--", lw=1); a.axhline(0, color="k", lw=0.6)
    a.set_xlim(lim); a.set_ylim(lim); a.legend(fontsize=7.5, loc="upper left")
    if ctl.any():
        a.annotate("collapses to zero,\nas it must", (aa[ctl][0], dd[ctl][0]), (0.30, -0.02),
                   fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
a.set_xlabel("r, all partners"); a.set_ylabel("r, partners in a DIFFERENT abundance quintile")
a.set_title(f"C  Two thirds survives abundance-discordant\npartners (mean {np.mean(aa):.3f} "
            f"-> {np.mean(dd):.3f})", fontsize=10)

a = ax[3]
ts = list(J["own_binding_tertile"])
if ts:
    lo = [J["own_binding_tertile"][t]["low"]["r"] for t in ts]
    hi = [J["own_binding_tertile"][t]["high"]["r"] for t in ts]
    sig = [J["own_binding_tertile"][t]["d_low_minus_high"][0] > 0 for t in ts]
    ndir, nsig = sum(l > h for l, h in zip(lo, hi)), sum(sig)
    yy = np.arange(len(ts))
    a.barh(yy - 0.2, lo, 0.38, color="#36c", label="weakest binders (carried)")
    a.barh(yy + 0.2, hi, 0.38, color="#c44", label="strongest binders")
    for i, s in enumerate(sig):
        if s: a.text(max(lo[i], hi[i]) + 0.008, i, "*", fontsize=13, va="center")
    a.set_yticks(yy); a.set_yticklabels(ts, fontsize=7.5); a.invert_yaxis()
    a.legend(fontsize=7.5, loc="lower right")
    a.set_title(f"D  Co-capture asymmetry: predicted direction\nin {ndir} of {len(ts)}, "
                f"interval excludes 0 in {nsig} (*)", fontsize=10)
a.set_xlabel("r(own residual, complex-mate mean)")

fig.suptitle("Is the complex effect a capture nonlinearity, co-abundance, or co-capture?", fontsize=12.5,
             y=1.02)
fig.savefig(f"{FIG}/nonlinearity.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIG}/nonlinearity.png")
print("CM26_DONE")
