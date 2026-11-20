"""cm27_free_energy: can a free energy be put on each protein in these pull-downs?

A pull-down measures a partition. Where the perturbation is applied during the binding
incubation (which, for the salt series, it is), the log ratio of bound to free IS a
binding constant up to one additive constant per condition, so a contrast between two
conditions is a ddG. This script establishes what that identification can and cannot
support, from the exported tables only.

  reports/figures/dg/*.png    five figures
  data/campaign/cm27_free_energy.json   every number quoted in additional_text_10

Run:  python code/cm27_free_energy.py
"""
import json, collections, os, zipfile, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PD_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
E, C = f"{ROOT}/data/export", f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/dg"; os.makedirs(FIG, exist_ok=True)
R = 1.98720425e-3                                  # kcal / mol / K
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

OUT = {}
T = pd.read_csv(f"{E}/targets.tsv", sep="\t"); U = T["accession"].astype(str).values
M = pd.read_csv(f"{E}/targets_meta.tsv", sep="\t").set_index("target")
grp = np.load(f"{C}/folds.npz", allow_pickle=True)["group"].astype(str)
S = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).reindex(U)

def kT(TC): return R * (TC + 273.15) * np.log(2)       # kcal/mol per log2 unit
OUT["kcal_per_log2"] = {f"{t}C": round(kT(t), 4) for t in (35, 37, 43)}
dinvT = 1/(43+273.15) - 1/(35+273.15)
OUT["vanthoff_kcal_per_log2_35_43"] = round(-R*np.log(2)/dinvT, 2)

# ---------------------------------------------------------------- meltome Tm
HUM = ["HepG2","Jurkat","K562","U937","HAOEC","HL60","HEK293T","colon_cancer_spheroids","HaCaT","pTcells"]
mz = f"{ROOT}/data/external/meltome/full_dataset.json"
raw = json.load(open(mz)) if os.path.exists(mz) else json.loads(
    zipfile.ZipFile(f"{ROOT}/data/external/meltome/full_dataset.json.zip").read("full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in raw:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
acc = collections.defaultdict(list)
for r in HUM:
    for k, v in {k: np.median(v) for k, v in per[r].items()}.items(): acc[k].append(v)
Tm = np.array([np.median(acc[u]) if u in acc else np.nan for u in U])
del raw
ABUND = T["GA33_input"].values.astype(float); LEN = S["log10_length"].values.astype(float)
CHG = S["charge_per_res"].values.astype(float); CTRL = np.column_stack([ABUND, LEN])
OUT["n_with_Tm"] = int(np.isfinite(Tm).sum())

def cluster_boot(fn, m, boot=2000, seed=0):
    g = grp[m]; gl = np.unique(g); where = {k: np.flatnonzero(g == k) for k in gl}
    rng = np.random.default_rng(seed)
    return np.array([fn(np.concatenate([where[k] for k in rng.choice(gl, len(gl), True)]))
                     for _ in range(boot)])

def partial_r(y, x, ctrl=CTRL, boot=2000):
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(ctrl), axis=1)
    yy, xx, Z = y[m], x[m], np.column_stack([ctrl[m], np.ones(m.sum())])
    def r_of(i):
        b = np.linalg.lstsq(Z[i], np.column_stack([yy[i], xx[i]]), rcond=None)[0]
        e = np.column_stack([yy[i], xx[i]]) - Z[i] @ b
        return np.corrcoef(e[:,0], e[:,1])[0,1]
    obs = r_of(np.arange(m.sum()))
    lo, hi = np.percentile(cluster_boot(r_of, m, boot), [2.5, 97.5])
    return dict(r=round(float(obs),3), lo=round(float(lo),3), hi=round(float(hi),3), n=int(m.sum()))

def deming(x, y, lam):
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1); sxy = np.cov(x, y, ddof=1)[0,1]
    d = syy - lam*sxx
    return (d + np.sqrt(d*d + 4*lam*sxy*sxy)) / (2*sxy)

def noise_sd(t):
    y = T[t].values.astype(float); y = y[np.isfinite(y)]
    return y.std(ddof=1)*np.sqrt(1 - float(M.loc[t, "ceiling"]))

# ============================================================ 1. precision of a ddG
rows = []
for t in ["GA22_part_salt60","GA22_part_salt120","GA24_part_salt60","GA24_part_salt120"]:
    y = T[t].values.astype(float); y = y[np.isfinite(y)]
    c = float(M.loc[t,"ceiling"]); sd = y.std(ddof=1); k = kT(37)
    rows.append(dict(target=t, n=len(y), ceiling=c,
                     spread_5_95=round(float((np.percentile(y,95)-np.percentile(y,5))*k),3),
                     noise_kcal=round(float(sd*np.sqrt(1-c)*k),4),
                     signal_to_noise=round(float(np.sqrt(c/(1-c))),2)))
OUT["ddG_precision"] = rows

# ============================================================ 2. dose-response shape
shape = []
for run, lo, hi, clo, chi in [("GA22","part_salt60","part_salt120",60,120),
                              ("GA24","part_salt60","part_salt120",60,120),
                              ("GA20","salt75","salt150",75,150)]:
    a, b = f"{run}_{lo}", f"{run}_{hi}"
    x, y = T[a].values.astype(float), T[b].values.astype(float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    lam = (noise_sd(a)/noise_sd(b))**2
    sl = float(deming(x, y, lam))
    bs = []
    mm = np.isfinite(T[a].values.astype(float)) & np.isfinite(T[b].values.astype(float))
    xa, ya = T[a].values.astype(float)[mm], T[b].values.astype(float)[mm]
    bs = cluster_boot(lambda i: deming(xa[i], ya[i], lam), mm, 1000)
    shape.append(dict(run=run, steps=f"{clo}->{chi} mM", n=int(m.sum()), deming=round(sl,3),
                      lo=round(float(np.percentile(bs,2.5)),3), hi=round(float(np.percentile(bs,97.5)),3),
                      pred_linear=round(chi/clo,2),
                      pred_loglinear={f"floor_{f}": round(float(np.log((chi+f)/(clo+f))/np.log((clo+f)/f)*np.log((clo+f)/f)/np.log((clo+f)/f)),3) for f in (50,)},
                      pred_loglin_50=round(float(np.log((chi+50)/50)/np.log((clo+50)/50)),3)))
OUT["dose_response_shape"] = shape

# ============================================================ 3. does a ddG reproduce across experiments?
a, b = "GA22_part_salt120", "GA24_part_salt120"
x, y = T[a].values.astype(float), T[b].values.astype(float)
m = np.isfinite(x) & np.isfinite(y)
r_obs = float(np.corrcoef(x[m], y[m])[0,1])
c1, c2 = float(M.loc[a,"ceiling"]), float(M.loc[b,"ceiling"])
bs = cluster_boot(lambda i: np.corrcoef(x[m][i], y[m][i])[0,1], m, 2000)
OUT["cross_experiment"] = dict(pair=f"{a} vs {b}", n=int(m.sum()), r=round(r_obs,3),
    lo=round(float(np.percentile(bs,2.5)),3), hi=round(float(np.percentile(bs,97.5)),3),
    ceiling_a=c1, ceiling_b=c2, max_attainable=round(float(np.sqrt(c1*c2)),3),
    frac_of_ceiling=round(float(r_obs/np.sqrt(c1*c2)),3))

# ============================================================ 4. conservation / depletion
cons = []
for run in ("GA22","GA24"):
    p, q = T[f"{run}_ip_salt120"].values.astype(float), T[f"{run}_sup_salt120"].values.astype(float)
    m2 = np.isfinite(p) & np.isfinite(q)
    rr = float(np.corrcoef(p[m2], q[m2])[0,1])
    cons.append(dict(run=run, r=round(rr,3), slope=round(float(rr*q[m2].std()/p[m2].std()),3),
                     var_ratio=round(float(np.var(q[m2])/np.var(p[m2])),3)))
OUT["conservation"] = cons

# ============================================================ 5. Tm and charge, with specificity
RESP = {"GA33_A1_heat43":"heat 43 vs 35 C (DNAJA1)", "GA33_B11_heat43":"heat 43 vs 35 C (DNAJB11)",
        "GA22_part_salt120":"salt 120 vs 0 mM (GA_22)", "GA24_part_salt120":"salt 120 vs 0 mM (GA_24)"}
spec = []
for t, lab in RESP.items():
    for pn, px in (("Tm", Tm), ("charge_per_res", CHG)):
        d = partial_r(T[t].values.astype(float), px); d.update(response=t, label=lab, predictor=pn)
        d["significant"] = bool(d["lo"]*d["hi"] > 0)
        spec.append(d)
OUT["specificity"] = spec

# joint model: does Tm survive charge?
joint = []
for t in ("GA33_A1_heat43","GA33_B11_heat43"):
    y = T[t].values.astype(float)
    X = np.column_stack([Tm, CHG, ABUND, LEN])
    m3 = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Z = (X[m3]-X[m3].mean(0))/X[m3].std(0); Z = np.column_stack([Z, np.ones(m3.sum())]); yy = y[m3]
    beta = np.linalg.lstsq(Z, yy, rcond=None)[0]
    bs = cluster_boot(lambda i: np.linalg.lstsq(Z[i], yy[i], rcond=None)[0], m3, 1500)
    e = dict(target=t, n=int(m3.sum()))
    for i, nm in enumerate(["Tm","charge_per_res","abundance","log10_length"]):
        lo, hi = np.percentile(bs[:,i], [2.5, 97.5])
        e[nm] = dict(beta=round(float(beta[i]),3), lo=round(float(lo),3), hi=round(float(hi),3))
    joint.append(e)
OUT["joint_model"] = joint

# ============================================================ 6. the two-state test
def two_state_log2(TmC, dH=100.0, T1=308.15, T2=316.15):
    Tk = np.asarray(TmC, float) + 273.15
    f = lambda Tx: 1/(1+np.exp(dH*(1 - Tx/Tk)/(R*Tx)))
    return np.log2(f(T2)/f(T1))
ts = []
for dH in (50., 100., 200.):
    p = two_state_log2(Tm, dH)
    for t in ("GA33_A1_heat43","GA33_B11_heat43"):
        y = T[t].values.astype(float); m4 = np.isfinite(y)&np.isfinite(p)&np.isfinite(Tm)
        ts.append(dict(dH=dH, target=t, r_two_state=round(float(np.corrcoef(y[m4],p[m4])[0,1]),3),
                       r_linear_Tm=round(float(np.corrcoef(y[m4],Tm[m4])[0,1]),3),
                       predicted_spread_5_95=round(float(np.percentile(p[m4],95)-np.percentile(p[m4],5)),3),
                       observed_spread_5_95=round(float(np.percentile(y[m4],95)-np.percentile(y[m4],5)),3)))
OUT["two_state"] = ts
OUT["two_state_flatness"] = {f"Tm_{t}C": round(float(two_state_log2(np.array([t]), 100.)[0]),2)
                             for t in (45., 48.5, 52., 57.)}

# apparent van 't Hoff enthalpy per protein
conv = OUT["vanthoff_kcal_per_log2_35_43"]
dHapp = {}
for t in ("GA33_A1_heat43","GA33_B11_heat43"):
    y = T[t].values.astype(float); y = y[np.isfinite(y)]
    dHapp[t] = {f"p{p}": round(float(np.percentile(y,p)*conv),1) for p in (5,25,50,75,95,99)}
OUT["apparent_dH_kcal"] = dHapp
json.dump(OUT, open(f"{C}/cm27_free_energy.json","w"), indent=1)
print(json.dumps({k: OUT[k] for k in ("cross_experiment","two_state_flatness","apparent_dH_kcal")}, indent=1))

# ================================================================= FIGURES
def panel(ax, title):
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
    ax.grid(True, color=GRID, lw=.6, alpha=.9); ax.set_axisbelow(True)

# --- F1 dose-response shape -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
for ax, s in zip(axes, OUT["dose_response_shape"]):
    run = s["run"]
    a = f"{run}_part_salt60" if run != "GA20" else "GA20_salt75"
    b = f"{run}_part_salt120" if run != "GA20" else "GA20_salt150"
    x, y = T[a].values.astype(float), T[b].values.astype(float)
    m = np.isfinite(x)&np.isfinite(y)
    ax.hexbin(x[m], y[m], gridsize=55, cmap="Blues", mincnt=1, linewidths=0)
    lim = np.percentile(np.abs(np.r_[x[m], y[m]]), 99.3)
    g = np.linspace(-lim, lim, 10)
    ax.plot(g, 2.0*g, color=ORANGE, lw=2, label="linear in [salt]  (2.00)")
    ax.plot(g, s["pred_loglin_50"]*g, color=AQUA, lw=2, ls=(0,(5,2)),
            label=f"log-linear  ({s['pred_loglin_50']:.2f})")
    ax.plot(g, s["deming"]*g, color=VIOLET, lw=2, ls=(0,(1.5,1.5)),
            label=f"measured  ({s['deming']:.2f})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim*2, lim*2)
    panel(ax, f"{run}   {s['steps']}   n={s['n']}")
    ax.set_xlabel(f"smaller salt step (log2)"); ax.set_ylabel("larger salt step (log2)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK2)
fig.suptitle("The salt dose-response is log-linear in ionic strength, not linear in concentration",
             fontsize=9.5, x=.012, ha="left", color=INK)
fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f"{FIG}/f1_dose_response_shape.png", dpi=200); plt.close(fig)

# --- F2 cross-experiment reproducibility ------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 4.3))
x, y = T["GA22_part_salt120"].values.astype(float), T["GA24_part_salt120"].values.astype(float)
m = np.isfinite(x)&np.isfinite(y)
k = kT(37)
ax.hexbin(x[m]*k, y[m]*k, gridsize=58, cmap="Blues", mincnt=1, linewidths=0)
lim = np.percentile(np.abs(np.r_[x[m], y[m]]), 99.3)*k
ax.plot([-lim,lim], [-lim,lim], color=INK2, lw=1.2, ls=(0,(4,3)))
ce = OUT["cross_experiment"]
ax.text(.04, .96, f"r = {ce['r']:.3f}  [{ce['lo']:.3f}, {ce['hi']:.3f}]\n"
                  f"ceiling = {ce['max_attainable']:.3f}\n"
                  f"{ce['frac_of_ceiling']*100:.1f} % of attainable\nn = {ce['n']}",
        transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
ax.set_xlabel("GA_22  ddG at 120 mM KCl (kcal/mol)"); ax.set_ylabel("GA_24  ddG at 120 mM KCl (kcal/mol)")
panel(ax, "A per-protein ddG reproduces across two independent experiments")
ax.title.set_fontsize(8.0)
fig.tight_layout(); fig.savefig(f"{FIG}/f2_cross_experiment.png", dpi=200); plt.close(fig)

# --- F3 Tm: observed shape vs what two-state predicts ------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
dec = {}
for t, col, lab in (("GA33_A1_heat43", BLUE, "DNAJA1"), ("GA33_B11_heat43", ORANGE, "DNAJB11")):
    y = T[t].values.astype(float); m = np.isfinite(y)&np.isfinite(Tm)
    q = np.quantile(Tm[m], np.linspace(0,1,11)); b = np.digitize(Tm[m], q[1:-1])
    mid = [(q[i]+q[i+1])/2 for i in range(10)]
    mu  = [np.mean(y[m][b==i]) for i in range(10)]
    se  = [np.std(y[m][b==i], ddof=1)/np.sqrt((b==i).sum()) for i in range(10)]
    dec[t] = (mid, mu)
    for ax in axes:
        ax.errorbar(mid, mu, yerr=se, color=col, lw=2, marker="o", ms=4.5, capsize=2,
                    mec="white", mew=.8, label=lab)
    axes[0].annotate(lab, (mid[-1], mu[-1]), textcoords="offset points", xytext=(5,0),
                     color=col, fontsize=7.5, va="center")
gx = np.linspace(42, 63, 50)
axes[1].plot(gx, two_state_log2(gx, 100.), color=AQUA, lw=2.2, ls=(0,(5,2)))
axes[1].annotate("two-state, dH = 100 kcal/mol", (gx[8], two_state_log2(gx,100.)[8]),
                 textcoords="offset points", xytext=(4,-10), color=AQUA, fontsize=7.5)
for ax, ttl in zip(axes, ["Observed: an inverted U, peaking in the high 40s",
                          "Same axis, with what two-state unfolding predicts"]):
    panel(ax, ttl); ax.set_xlabel("meltome Tm decile (C)"); ax.set_ylabel("43 vs 35 C response (log2)")
axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
axes[1].set_ylim(-.4, 6.4)
fig.suptitle("The temperature response does not have the size or the shape of two-state unfolding",
             fontsize=9.5, x=.012, ha="left", color=INK)
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(f"{FIG}/f3_two_state.png", dpi=200); plt.close(fig)

# --- F4 specificity ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.4))
labs = ["heat 43 C\nDNAJA1","heat 43 C\nDNAJB11","salt 120 mM\nGA_22","salt 120 mM\nGA_24"]
keys = ["GA33_A1_heat43","GA33_B11_heat43","GA22_part_salt120","GA24_part_salt120"]
w, xs = .36, np.arange(4)
for off, pn, col in ((-w/2, "Tm", BLUE), (w/2, "charge_per_res", ORANGE)):
    d = [next(s for s in OUT["specificity"] if s["response"]==k and s["predictor"]==pn) for k in keys]
    ax.bar(xs+off, [e["r"] for e in d], w, color=col, edgecolor="white", lw=1.2,
           label="meltome Tm" if pn=="Tm" else "net charge / residue")
    ax.errorbar(xs+off, [e["r"] for e in d],
                yerr=[[e["r"]-e["lo"] for e in d], [e["hi"]-e["r"] for e in d]],
                fmt="none", ecolor=INK2, lw=1.1, capsize=2.5)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
ax.set_ylabel("partial r  (abundance and length controlled)")
panel(ax, "Tm predicts the temperature response and not the salt response; net charge predicts both")
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/f4_specificity.png", dpi=200); plt.close(fig)

# --- F5 can ANY enthalpy reproduce the observed between-protein spread? ----------
# Per-sample median normalisation forces the median response to zero, so the ABSOLUTE
# level of the response carries no information. Only the protein-to-protein SPREAD does,
# and that is what the two-state model has to reproduce.
grid = np.linspace(15, 300, 120)
obs_spread = {}
for t in ("GA33_A1_heat43","GA33_B11_heat43"):
    y = T[t].values.astype(float); m = np.isfinite(y)&np.isfinite(Tm)
    obs_spread[t] = float(np.percentile(y[m],95)-np.percentile(y[m],5))
mTm = Tm[np.isfinite(Tm)]
pred = np.array([np.percentile(two_state_log2(mTm, d),95)-np.percentile(two_state_log2(mTm, d),5)
                 for d in grid])
OUT["spread_vs_dH"] = dict(observed=round(obs_spread["GA33_A1_heat43"],2),
                           best_predicted=round(float(pred.max()),3),
                           at_dH=round(float(grid[pred.argmax()]),1),
                           shortfall_fold=round(float(obs_spread["GA33_A1_heat43"]/pred.max()),1))
json.dump(OUT, open(f"{C}/cm27_free_energy.json","w"), indent=1)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(grid, pred, color=AQUA, lw=2.2)
ax.annotate("spread two-state unfolding predicts", (grid[25], pred[25]),
            textcoords="offset points", xytext=(8, 12), color="#0f7a55", fontsize=7.5)
for (t, col, lab), xa, dy in ((("GA33_A1_heat43", BLUE, "DNAJA1"), 70, 6),
                              (("GA33_B11_heat43", ORANGE, "DNAJB11"), 210, -14)):
    ax.axhline(obs_spread[t], color=col, lw=2, ls=(0,(5,2)))
    ax.annotate(f"observed, {lab} ({obs_spread[t]:.2f})", (xa, obs_spread[t]),
                textcoords="offset points", xytext=(0, dy), color=col, fontsize=7.5)
ax.set_yscale("log"); ax.set_ylim(.02, 8)
ax.set_xlabel("two-state unfolding enthalpy assumed (kcal/mol)")
ax.set_ylabel("5-95 % spread of the 43 vs 35 C response (log2)")
panel(ax, "No unfolding enthalpy reproduces the spread actually observed")
fig.tight_layout(); fig.savefig(f"{FIG}/f5_spread_vs_enthalpy.png", dpi=200); plt.close(fig)

print(f"\nfigures -> {FIG}")
for f in sorted(os.listdir(FIG)): print("  ", f)
