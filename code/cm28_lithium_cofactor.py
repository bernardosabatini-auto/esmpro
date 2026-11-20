"""cm28: the lithium (FS_76) and the NAD / NADP (GA_24) arms, analysed as their own experiments.

The campaign used the drug-free arms of every run and left these two aside. They are read here on
their own terms, and the first question in each case is whether the perturbation is identifiable at
all against the way the run was acquired.

FS_76   HSPB1 pull-down, 37 C, Mg2+ 0.25 or 2.5 mM x Li 0 / 0.3 / 1 / 3 / 5 / 10 mM, 3-4 BR per cell.
        A crossed design, so the mechanistic prediction (Li competes with Mg2+) is testable as an
        interaction, and the Mg contrast is an internal positive control for the same divalent axis.
GA_24   HSPB1 pull-down, cofactor CTRL / NAD 11 / NAD 300 / NADP 0.6 / NADP 15 (units unrecorded)
        x KCl 0 / 60 / 120 mM, 5 BR per cell. Cofactor was acquired as a fixed within-block order,
        so the cofactor-vs-control contrast is collinear with acquisition drift. Two contrasts are
        not: the within-cofactor dose step, and the cofactor-by-salt interaction.

Writes data/campaign/cm28_lithium_cofactor.json and figures to reports/figures/arms/.
"""
import os, json, argparse, collections
import numpy as np, pandas as pd
from scipy import stats as sps

import sys
ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pdpipe import io, stats as PS

C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/arms"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{C}/cm28_lithium_cofactor.json")
ap.add_argument("--boot", type=int, default=2000); A = ap.parse_args()
os.makedirs(FIG, exist_ok=True)
J = {}

# ------------------------------------------------------------------ shared helpers
def cvec(names, w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] = v
    return c


def fit_contrast(ds, weights, block=True, covars=None, conds=None):
    """Moderated contrast over cell means, on the samples of the conditions involved."""
    S = ds.samples[ds.samples["cond"].isin(conds or list(weights))].reset_index(drop=True)
    X = ds.X[S["column"]]
    cv = covars(S) if callable(covars) else None
    D, names = PS.design(S, block=block, covars=cv)
    ci = [i for i, n in enumerate(names) if cv is not None and n in cv.columns]
    f = PS.ebayes(PS.lmfit(X, D, covar=ci))
    r = PS.contrast(f, cvec(names, weights))
    return r, S


def reliability(ds, weights, n_perm=200, adjust=None):
    S = ds.samples[ds.samples["cond"].isin(list(weights))].reset_index(drop=True)
    return PS.split_half(ds.X[S["column"]], S, weights, n_perm=n_perm, adjust=adjust)


GRP = None
def cluster_ci(v, m, boot=A.boot, seed=0):
    """Mean of v[m] with a cluster bootstrap over 30 % identity groups."""
    g = GRP[m]; v = np.asarray(v)[m]; gl = np.unique(g)
    where = {k: np.flatnonzero(g == k) for k in gl}
    rng = np.random.default_rng(seed)
    bs = [np.nanmean(v[np.concatenate([where[k] for k in rng.choice(gl, len(gl), True)])]) for _ in range(boot)]
    return float(np.nanmean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------ load
print("loading ...", flush=True)
def plain(ds):
    """pdpipe returns pandas 'str' columns; stats.design needs plain object arrays."""
    ds.samples = ds.samples.copy()
    for c in ds.samples.columns:
        if str(ds.samples[c].dtype) == "str": ds.samples[c] = ds.samples[c].astype(object)
    return ds

fs76 = plain(io.load(io.load_config("fs76")))
ga24 = plain(io.load(io.load_config("ga24")))
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).values
F = np.load(f"{C}/folds.npz", allow_pickle=True)
g_of = dict(zip(U, F["group"].astype(str)))
print(f"  FS76 {fs76.X.shape}   GA24 {ga24.X.shape}", flush=True)

# ================================================================== PART A: GA_24 design audit
print("\n--- GA_24: is the cofactor identifiable? ---", flush=True)
S24 = ga24.samples.copy()
ARM = ["CTRL|0", "NAD|11", "NAD|300", "NADP|0.6", "NADP|15"]
S24["arm"] = S24["cof"].astype(str) + "|" + S24["conc"].astype(str)
S24["arm_i"] = S24["arm"].map({a: i for i, a in enumerate(ARM)})
# correlation of arm order with acquisition position, within each BR x salt block
rs = []
for (br, kcl), g in S24.groupby(["BR", "KCl"]):
    if g["arm_i"].nunique() > 2: rs.append(np.corrcoef(g["arm_i"], g["inj_pos"])[0, 1])
arm_pos = S24.groupby("arm")["inj_pos"].mean().reindex(ARM)
J["ga24_design"] = dict(
    arm_inj_pos={k: float(v) for k, v in arm_pos.items()},
    r_arm_vs_inj_within_block=float(np.mean(rs)), n_blocks=len(rs),
    gap_ctrl_to_nadp=float(arm_pos["NADP|15"] - arm_pos["CTRL|0"]),
    gap_salt0_to_salt120=float(S24.groupby("KCl")["inj_pos"].mean()["120"] - S24.groupby("KCl")["inj_pos"].mean()["0"]),
    gap_nad11_to_nad300=float(arm_pos["NAD|300"] - arm_pos["NAD|11"]),
    gap_nadp06_to_nadp15=float(arm_pos["NADP|15"] - arm_pos["NADP|0.6"]))
print(f"  arm vs inj_pos within BR x salt: r = {np.mean(rs):+.4f} over {len(rs)} blocks", flush=True)
print(f"  inj_pos gap CTRL -> NADP 15 : {J['ga24_design']['gap_ctrl_to_nadp']:+.3f}", flush=True)
print(f"  inj_pos gap KCl 0 -> KCl 120: {J['ga24_design']['gap_salt0_to_salt120']:+.3f}", flush=True)

# ---- the three contrasts: confounded, and the two that are not
def w_arm(arm, kcls=(0, 60, 120), sign=1.0):
    cof, conc = arm.split("|")
    return {f"{cof}|{conc}|{k}": sign / len(kcls) for k in kcls}

def merge(*ws):
    out = collections.defaultdict(float)
    for w in ws:
        for k, v in w.items(): out[k] += v
    return dict(out)

CON = {
    # confounded with acquisition order
    "NAD_vs_CTRL":   merge(w_arm("NAD|11", sign=.5), w_arm("NAD|300", sign=.5), w_arm("CTRL|0", sign=-1)),
    "NADP_vs_CTRL":  merge(w_arm("NADP|0.6", sign=.5), w_arm("NADP|15", sign=.5), w_arm("CTRL|0", sign=-1)),
    # nearly drift-free: a dose step inside one cofactor
    "NAD_dose":      merge(w_arm("NAD|300"), w_arm("NAD|11", sign=-1)),
    "NADP_dose":     merge(w_arm("NADP|15"), w_arm("NADP|0.6", sign=-1)),
}
# drift-free: cofactor changes the salt response (both arms carry the same internal drift)
salt = lambda arm: merge(w_arm(arm, kcls=(120,)), w_arm(arm, kcls=(0,), sign=-1))
CON["salt120_x_NADP"] = merge(salt("NADP|15"), {k: -v for k, v in salt("CTRL|0").items()})
CON["salt120_x_NAD"] = merge(salt("NAD|300"), {k: -v for k, v in salt("CTRL|0").items()})
CON["salt120_CTRL"] = salt("CTRL|0")                                      # the known-real anchor
# The relabelling null shuffles the labels of every condition the contrast touches. For a contrast
# pooled over salt that means salt levels get shuffled in too, and salt is the largest effect in the
# run, so the null inherits its variance and comes out far too high. Repeating each cofactor contrast
# inside ONE salt level leaves the null shuffling cofactor and nothing else.
CON["NAD_vs_CTRL@0"] = merge(w_arm("NAD|11", (0,), .5), w_arm("NAD|300", (0,), .5), w_arm("CTRL|0", (0,), -1))
CON["NADP_vs_CTRL@0"] = merge(w_arm("NADP|0.6", (0,), .5), w_arm("NADP|15", (0,), .5), w_arm("CTRL|0", (0,), -1))
CON["NAD_dose@0"] = merge(w_arm("NAD|300", (0,)), w_arm("NAD|11", (0,), -1))
CON["NADP_dose@0"] = merge(w_arm("NADP|15", (0,)), w_arm("NADP|0.6", (0,), -1))

drift = lambda S: PS.drift_basis(S["inj_pos"], 2)
J["ga24"] = {}
for name, w in CON.items():
    r, _ = fit_contrast(ga24, w, block=True)
    rd, _ = fit_contrast(ga24, w, block=True, covars=drift)
    rel = reliability(ga24, w, n_perm=200)
    J["ga24"][name] = dict(
        n=int(r["logFC"].notna().sum()), sd=float(r["logFC"].std()), sd_driftadj=float(rd["logFC"].std()),
        n_q05=int((r["q"] < .05).sum()), n_q05_driftadj=int((rd["q"] < .05).sum()),
        reliability=rel["reliability"], null_rel95=rel.get("null_rel95"), p_perm=rel.get("p_perm"),
        r_raw_vs_driftadj=float(r["logFC"].corr(rd["logFC"])))
    print(f"  {name:16} sd {r['logFC'].std():.3f} -> {rd['logFC'].std():.3f} (drift-adj)   "
          f"rel {rel['reliability']:.3f} (null95 {rel.get('null_rel95', float('nan')):.3f})   "
          f"q<.05 {int((r['q'] < .05).sum()):>5} -> {int((rd['q'] < .05).sum()):>5}", flush=True)

# ---- the attribution test: does a cofactor-free drift slope predict the cofactor effect?
# The drift slope is the coefficient on acquisition position in a model that ALSO carries a free mean
# for all 15 conditions. Condition is therefore already absorbed, and what identifies the slope is the
# position difference between biological replicates measured in the SAME condition -- nothing to do
# with cofactor. It is learned on one set of BRs and tested on the disjoint rest, so the two estimates
# share no noise and the agreement cannot be manufactured by a common term.
print("\n  attribution: cofactor-free drift slope (held-out BRs) vs apparent cofactor effect", flush=True)

def contrast_gap(S, w):
    """The acquisition-position difference the contrast itself spans."""
    m = S.groupby("cond")["inj_pos"].mean()
    return float(sum(v * m[k] for k, v in w.items()))


def drift_slope(S_tr):
    cv = pd.DataFrame({"inj1": S_tr["inj_pos"].values - .5})
    D, names = PS.design(S_tr, block=False, covars=cv)
    f = PS.ebayes(PS.lmfit(ga24.X[S_tr["column"]], D, covar=[names.index("inj1")], min_df=4))
    return pd.Series(f["beta"][:, names.index("inj1")], index=ga24.X.index)


BRS = sorted(S24["BR"].unique())
J["ga24_attribution"] = {}
ATTR = {}
sl_rel = []                       # how reproducible is the drift slope itself? -> attenuation correction
for Ahalf in PS._splits(BRS):
    x = drift_slope(S24[S24["BR"].isin(Ahalf)].reset_index(drop=True))
    y = drift_slope(S24[~S24["BR"].isin(Ahalf)].reset_index(drop=True))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 100: sl_rel.append(np.corrcoef(x[ok], y[ok])[0, 1])
r_half = float(np.mean(sl_rel)); rel_slope = float(2 * r_half / (1 + r_half))
J["ga24_drift_slope"] = dict(split_half_r=r_half, reliability=rel_slope, n_splits=len(sl_rel))
print(f"    drift slope itself: split-half r {r_half:.3f}, reliability {rel_slope:.3f}", flush=True)

for cname in ("NADP_vs_CTRL", "NAD_vs_CTRL", "NAD_dose", "NADP_dose", "salt120_CTRL"):
    w = CON[cname]; res = []; P_, O_ = [], []
    for Ahalf in PS._splits(BRS):
        Bhalf = [x for x in BRS if x not in Ahalf]
        for tr, te in ((list(Ahalf), Bhalf), (Bhalf, list(Ahalf))):
            S_tr = S24[S24["BR"].isin(tr)].reset_index(drop=True)
            S_te = S24[S24["BR"].isin(te)].reset_index(drop=True)
            gap = contrast_gap(S_te, w)
            pred = drift_slope(S_tr) * gap
            obs = PS._effect(ga24.X, S_te, w, np.ones(len(S_te), bool))
            ok = (np.isfinite(obs) & np.isfinite(pred)).values
            if ok.sum() < 100: continue
            o = PS.ols(pred.values[ok], obs.values[ok])
            res.append(dict(r=o["r"], slope=o["slope"], n=o["n"], gap=gap))
            P_.append(pred.values); O_.append(obs.reindex(ga24.X.index).values)
    if not res:
        print(f"    {cname:16} no usable split"); continue
    r = float(np.mean([x["r"] for x in res])); sl = float(np.mean([x["slope"] for x in res]))
    J["ga24_attribution"][cname] = dict(
        gap_inj_pos=float(np.mean([x["gap"] for x in res])), n_splits=len(res), r=r,
        r_sd=float(np.std([x["r"] for x in res])), slope=sl,
        slope_disattenuated=float(sl / rel_slope), var_explained=float(r ** 2))
    ATTR[cname] = (np.nanmean(P_, 0), np.nanmean(O_, 0))
    print(f"    {cname:16} gap {np.mean([x['gap'] for x in res]):+.3f}   r {r:+.3f}   slope {sl:+.3f} "
          f"(disattenuated {sl / rel_slope:+.3f}; 1.0 = the apparent effect IS drift)   R2 {r ** 2:.3f}", flush=True)

# ---- position or cofactor? The salt contrast is drift-free INSIDE an arm (KCl 0 and 120 were
# acquired back to back). Measure it in all five arms. If the arms differ, the question is whether
# they order by acquisition slot -- which would make it position -- or step by cofactor identity.
print("\n  the salt response measured inside each arm", flush=True)
base = PS._effect(ga24.X, S24, salt("CTRL|0"), np.ones(len(S24), bool))
rows = []
for arm in ARM:
    e = PS._effect(ga24.X, S24, salt(arm), np.ones(len(S24), bool))
    ok = (np.isfinite(e) & np.isfinite(base)).values
    o = PS.ols(base.values[ok], e.values[ok])
    rows.append(dict(arm=arm, inj_pos=float(arm_pos[arm]), sd=float(e.std()),
                     r_vs_CTRL=o["r"], slope_vs_CTRL=o["slope"], n=o["n"]))
    print(f"    {arm:<10} inj_pos {arm_pos[arm]:.3f}   sd {e.std():.3f}   "
          f"vs CTRL: r {o['r']:+.3f} slope {o['slope']:+.3f}", flush=True)
# A regression slope below 1 means either a genuinely weaker salt response or simply a noisier
# measurement: pure attenuation predicts slope = r * sd(arm) / sd(CTRL), with no change in the
# underlying response. Comparing the two separates them.
sd0 = rows[0]["sd"]
for r_ in rows:
    r_["slope_if_only_noise"] = float(r_["r_vs_CTRL"] * r_["sd"] / sd0)
    r_["excess_over_noise"] = float(r_["slope_vs_CTRL"] - r_["slope_if_only_noise"])
print("    attenuation check (slope expected from noise alone):", flush=True)
for r_ in rows:
    print(f"      {r_['arm']:<10} observed {r_['slope_vs_CTRL']:+.3f}   noise-only "
          f"{r_['slope_if_only_noise']:+.3f}   excess {r_['excess_over_noise']:+.4f}", flush=True)
J["ga24_salt_by_arm"] = rows
# does the slope track acquisition position or cofactor class?
pos = np.array([r["inj_pos"] for r in rows]); slp = np.array([r["slope_vs_CTRL"] for r in rows])
cls = np.array([0, 0, 0, 1, 1], float)                      # NAD-family vs NADP-family
J["ga24_salt_by_arm_fit"] = dict(r_slope_vs_position=float(np.corrcoef(pos, slp)[0, 1]),
                                 r_slope_vs_cofactor_class=float(np.corrcoef(cls, slp)[0, 1]))
print(f"    slope vs acquisition position r = {np.corrcoef(pos, slp)[0, 1]:+.3f}   "
      f"vs NAD/NADP class r = {np.corrcoef(cls, slp)[0, 1]:+.3f}", flush=True)

# ================================================================== PART B: FS_76 lithium x Mg
print("\n--- FS_76: lithium ---", flush=True)
LI = ["0", "0.3", "1", "3", "5", "10"]
LIV = np.array([0., .3, 1., 3., 5., 10.])
MG = ["0.25", "2.5"]
cell = lambda mg, li: f"{mg}|{li}"

def w_mg(mg, w):            # a weight dict over one Mg background, lifted to full cond keys
    return {cell(mg, li): v for li, v in w.items()}

# the trend over Li doses, as the least-squares slope of the cell means on dose RANK (monotone,
# unit-free, and not committed to a guess about the shape of a dose-response)
tw = PS.trend_weights(LI, np.arange(len(LI)))
FCON = {
    "Mg_2.5_vs_0.25":  merge({cell("2.5", li): 1 / len(LI) for li in LI}, {cell("0.25", li): -1 / len(LI) for li in LI}),
    "Li_trend_lowMg":  w_mg("0.25", tw),
    "Li_trend_highMg": w_mg("2.5", tw),
    "Li_10_vs_0_lowMg":  {cell("0.25", "10"): 1., cell("0.25", "0"): -1.},
    "Li_10_vs_0_highMg": {cell("2.5", "10"): 1., cell("2.5", "0"): -1.},
}
FCON["Li_trend_pooled"] = merge({k: v / 2 for k, v in FCON["Li_trend_lowMg"].items()},
                                {k: v / 2 for k, v in FCON["Li_trend_highMg"].items()})
FCON["Li_x_Mg"] = merge(FCON["Li_trend_lowMg"], {k: -v for k, v in FCON["Li_trend_highMg"].items()})

J["fs76"] = {}
EFF = {}
for name, w in FCON.items():
    r, _ = fit_contrast(fs76, w, block=True)
    rel = reliability(fs76, w, n_perm=200)
    EFF[name] = r
    J["fs76"][name] = dict(n=int(r["logFC"].notna().sum()), sd=float(r["logFC"].std()),
                           n_q05=int((r["q"] < .05).sum()), n_q01=int((r["q"] < .01).sum()),
                           reliability=rel["reliability"], null_rel95=rel.get("null_rel95"),
                           p_perm=rel.get("p_perm"))
    print(f"  {name:20} sd {r['logFC'].std():.3f}   rel {rel['reliability']:+.3f} "
          f"(null95 {rel.get('null_rel95', float('nan')):+.3f}, p={rel.get('p_perm', float('nan')):.3f})   "
          f"q<.05 {int((r['q'] < .05).sum()):>5}", flush=True)

# how big a lithium effect would have been seen? The reproducible part of a contrast's spread is
# sd * sqrt(reliability); bounding reliability by the 95th percentile of its permutation null bounds
# the real between-protein spread that could be hiding under the noise.
J["fs76_detection_bound"] = {}
for name in ("Li_trend_lowMg", "Li_10_vs_0_lowMg", "Li_10_vs_0_highMg", "Mg_2.5_vs_0.25"):
    d = J["fs76"][name]; nb = d["null_rel95"] or 0.0
    med_se = float(EFF[name]["se"].median())
    J["fs76_detection_bound"][name] = dict(
        sd=d["sd"], null_rel95=nb, reproducible_sd_bound=float(d["sd"] * np.sqrt(max(nb, 0))),
        median_se=med_se, mde_80pc=float(2.8 * med_se))
    print(f"  {name:20} reproducible sd <= {d['sd'] * np.sqrt(max(nb, 0)):.3f} log2   "
          f"per-protein 80 % MDE {2.8 * med_se:.3f} log2", flush=True)

# per-dose spread, both Mg backgrounds: does ANY dose move the proteome more than the zero-dose noise?
J["fs76_dose"] = {}
for mg in MG:
    row = {}
    for li in LI[1:]:
        r, _ = fit_contrast(fs76, {cell(mg, li): 1., cell(mg, "0"): -1.}, block=True)
        row[li] = dict(sd=float(r["logFC"].std()), n_q05=int((r["q"] < .05).sum()))
    J["fs76_dose"][mg] = row
    print(f"  Mg {mg:>4}: sd by Li dose " + "  ".join(f"{li}={row[li]['sd']:.3f}" for li in LI[1:]), flush=True)

# the competition prediction: if Li acts by displacing Mg2+, adding Li should look like REMOVING Mg,
# so the Li trend should run opposite to the (2.5 - 0.25 mM Mg) effect.
GRP = np.array([g_of.get(a, f"_{a}") for a in fs76.X.index])
li_, mg_ = EFF["Li_trend_pooled"]["logFC"], EFF["Mg_2.5_vs_0.25"]["logFC"]
ok = (np.isfinite(li_) & np.isfinite(mg_)).values
o = PS.ols(mg_.values[ok], li_.values[ok])
# ... but both were measured on the same samples, so a shared normalisation or a shared replicate
# effect can manufacture this correlation. Repeat it with the two effects from DISJOINT replicates.
dis = []
FBRS = sorted(fs76.samples["BR"].unique())
for Ah in PS._splits(FBRS):
    Bh = [x for x in FBRS if x not in Ah]
    for tr, te in ((list(Ah), Bh), (Bh, list(Ah))):
        Sa = fs76.samples[fs76.samples["BR"].isin(tr)].reset_index(drop=True)
        Sb = fs76.samples[fs76.samples["BR"].isin(te)].reset_index(drop=True)
        a_ = PS._effect(fs76.X, Sa, FCON["Li_trend_pooled"], np.ones(len(Sa), bool))
        b_ = PS._effect(fs76.X, Sb, FCON["Mg_2.5_vs_0.25"], np.ones(len(Sb), bool))
        if a_ is None or b_ is None: continue
        k = (np.isfinite(a_) & np.isfinite(b_)).values
        if k.sum() > 100: dis.append(np.corrcoef(a_.values[k], b_.values[k])[0, 1])
J["fs76_competition"] = dict(r_same_samples=o["r"], slope=o["slope"], slope_ci=list(o["slope_ci"]),
                             n=o["n"], p=o["p"],
                             r_disjoint_BR=float(np.mean(dis)) if dis else None,
                             r_disjoint_sd=float(np.std(dis)) if dis else None, n_disjoint=len(dis))
print(f"  Li trend vs Mg effect: same samples r = {o['r']:+.3f}  slope {o['slope']:+.3f} "
      f"[{o['slope_ci'][0]:+.3f}, {o['slope_ci'][1]:+.3f}]  n={o['n']}", flush=True)
print(f"                         disjoint replicates r = {np.mean(dis):+.3f} "
      f"+- {np.std(dis):.3f} over {len(dis)} splits", flush=True)

# do the strongest Li hits hold up in held-out replicates?
for name in ("Li_trend_pooled", "Mg_2.5_vs_0.25"):
    rep = PS.top_hit_replication(fs76.X, fs76.samples, FCON[name], n_top=20, block=True)
    J["fs76"][name]["top20_train"] = float(np.mean([x["train_effect"] for x in rep]))
    J["fs76"][name]["top20_heldout"] = float(np.mean([x["heldout_effect"] for x in rep]))
    J["fs76"][name]["top20_same_sign"] = float(np.mean([x["heldout_same_sign"] for x in rep]))
    print(f"  {name:20} top-20 hits: train {J['fs76'][name]['top20_train']:+.3f} -> "
          f"held out {J['fs76'][name]['top20_heldout']:+.3f}  "
          f"(same sign {J['fs76'][name]['top20_same_sign']:.2f})", flush=True)

# ================================================================== PART C: which proteins
print("\n--- which proteins ---", flush=True)
ann = io.uniprot_annotation()
KWS = ["NAD", "NADP", "Oxidoreductase", "Magnesium", "Metal-binding", "Kinase", "ATP-binding",
       "Nucleotide-binding", "Chaperone", "Ribosomal protein", "RNA-binding", "Nucleus", "Cytoplasm",
       "Mitochondrion", "Transferase", "Hydrolase", "Lyase", "Glycolysis", "Flavoprotein", "Zinc"]
GENES = {"GSK3 / inositol (lithium's named targets)": ["GSK3A", "GSK3B", "IMPA1", "IMPA2", "INPP1", "BPNT1"],
         "NAD consumers": ["PARP1", "PARP2", "SIRT1", "SIRT2", "SIRT3", "SIRT5", "SIRT6", "SIRT7",
                           "NAMPT", "NMNAT1", "NMNAT2", "NMNAT3", "NADK", "NNT"]}

def enrich(eff, X_index, label):
    """Mean effect inside each keyword set minus outside, with a cluster bootstrap over 30 % groups."""
    global GRP
    GRP = np.array([g_of.get(a, f"_{a}") for a in X_index])
    kw = ann.reindex(X_index)["Keywords"].fillna("") if "Keywords" in ann else pd.Series("", index=X_index)
    v = np.asarray(eff, float); out = []
    for k in KWS:
        m = kw.str.contains(rf"(?:^|;\s*){k}(?:;|$)", regex=True).values & np.isfinite(v)
        rest = (~kw.str.contains(rf"(?:^|;\s*){k}(?:;|$)", regex=True).values) & np.isfinite(v)
        if m.sum() < 30: continue
        mi, lo, hi = cluster_ci(v, m)
        out.append(dict(keyword=k, n=int(m.sum()), mean=mi, lo=lo, hi=hi,
                        delta=float(mi - np.nanmean(v[rest])),
                        excludes_rest=bool(lo > np.nanmean(v[rest]) or hi < np.nanmean(v[rest]))))
    out.sort(key=lambda d: -abs(d["delta"]))
    print(f"  {label}: top keyword shifts", flush=True)
    for d in out[:6]:
        print(f"    {d['keyword']:<20} n={d['n']:>5}  mean {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
              f"delta vs rest {d['delta']:+.3f}{'  *' if d['excludes_rest'] else ''}", flush=True)
    return out

gene_of = {}
for ds in (fs76, ga24):
    if "gene" in ds.proteins:
        for a, g in zip(ds.proteins["uniprot_id"].astype(str), ds.proteins["gene"].astype(str)):
            gene_of.setdefault(a, g)

J["enrichment"] = {}
J["enrichment"]["Li_trend_pooled"] = enrich(EFF["Li_trend_pooled"]["logFC"].values, fs76.X.index, "Li trend")
J["enrichment"]["Mg_2.5_vs_0.25"] = enrich(EFF["Mg_2.5_vs_0.25"]["logFC"].values, fs76.X.index, "Mg 2.5 vs 0.25")

# named targets, one row each
J["named_targets"] = {}
for setname, genes in GENES.items():
    rows = []
    for nm, r in (("Li_trend_pooled", EFF["Li_trend_pooled"]), ("Mg_2.5_vs_0.25", EFF["Mg_2.5_vs_0.25"])):
        for a in r.index:
            g = str(gene_of.get(str(a), "") or "")
            if any(str(g) == x or str(g).startswith(x + ";") for x in genes):
                rows.append(dict(contrast=nm, gene=g.split(";")[0], acc=str(a),
                                 logFC=float(r.loc[a, "logFC"]), q=float(r.loc[a, "q"])))
    J["named_targets"][setname] = rows
    print(f"  {setname}: {len(rows)//2} proteins found; "
          f"min q = {min([x['q'] for x in rows], default=float('nan')):.3f}", flush=True)

# top hits by name, for the report
J["top_hits"] = {}
for nm in ("Li_trend_pooled", "Li_x_Mg", "Mg_2.5_vs_0.25"):
    r = EFF[nm].copy(); r["gene"] = [gene_of.get(str(a), "") for a in r.index]
    t = r.nsmallest(15, "q")[["gene", "logFC", "q"]]
    J["top_hits"][nm] = [dict(gene=str(x.gene).split(";")[0], acc=str(i), logFC=float(x.logFC), q=float(x.q))
                         for i, x in t.iterrows()]

# per-protein vectors the figures need
np.savez(f"{C}/cm28_vectors.npz",
         fs76_acc=np.array(fs76.X.index, dtype=object),
         li_trend=EFF["Li_trend_pooled"]["logFC"].values,
         li_x_mg=EFF["Li_x_Mg"]["logFC"].values,
         li_q=EFF["Li_trend_pooled"]["q"].values,
         mg=EFF["Mg_2.5_vs_0.25"]["logFC"].values,
         mg_q=EFF["Mg_2.5_vs_0.25"]["q"].values,
         ga24_acc=np.array(ga24.X.index, dtype=object),
         attr_pred=ATTR["NADP_vs_CTRL"][0], attr_obs=ATTR["NADP_vs_CTRL"][1],
         anchor_pred=ATTR["salt120_CTRL"][0], anchor_obs=ATTR["salt120_CTRL"][1])
json.dump(J, open(A.out, "w"), indent=1)
print("\nwrote " + A.out, flush=True)
print("CM28_DONE", flush=True)
