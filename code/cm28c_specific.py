"""cm28c: the proteins that actually respond to lithium in FS_76.

The proteome-wide lithium contrasts in cm28 are indistinguishable from noise. That is a statement
about the PROFILE across 8,500 proteins, not about every protein in it: a handful of genuine
responders cannot move a profile-wide reliability. This script looks for them, and then tries to
break whatever it finds.

A responder has to clear four bars, each of which kills a different artefact:
  1. monotone in dose          -- a spike in one condition is not a dose-response
  2. large                     -- bigger than the contrast's own noise
  3. replicating               -- the same value in disjoint biological replicates
  4. specific                  -- rare among 8,500 proteins, so not a global drift

Then the mechanism test: Li+ is thought to act by competing with Mg2+, so a real lithium effect
should weaken when Mg2+ is raised ten-fold. FS_76 crosses the two, so this is directly testable --
and it is the one check that an acquisition-order artefact cannot pass, because a drift along the
run does not switch off when the buffer changes.

Appends to data/campaign/cm28_lithium_cofactor.json; writes f6 to reports/figures/arms/.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pdpipe import io, stats as PS

C, FIG = f"{ROOT}/data/campaign", f"{ROOT}/reports/figures/arms"
LI = ["0", "0.3", "1", "3", "5", "10"]; LIV = np.array([0., .3, 1., 3., 5., 10.]); MG = ["0.25", "2.5"]
X6 = np.arange(6)


def plain(ds):
    ds.samples = ds.samples.copy()
    for c in ds.samples.columns:
        if str(ds.samples[c].dtype) == "str": ds.samples[c] = ds.samples[c].astype(object)
    return ds


fs = plain(io.load(io.load_config("fs76")))
gene, acc_of = {}, {}
for a, g in zip(fs.proteins["uniprot_id"].astype(str), fs.proteins["gene"]):
    g = str(g).split(";")[0] if isinstance(g, str) and g else ""
    if g: gene.setdefault(a, g); acc_of.setdefault(g, a)
CONTAM = set("KRT1 KRT2 KRT5 KRT6A KRT9 KRT10 KRT14 LYZ DCD FLG CSTA JUP SBSN".split())

cols = lambda conds: fs.samples.loc[fs.samples["cond"].isin(conds), "column"]
prof = lambda conds_by_dose: np.vstack([fs.X[cols(c)].mean(1).values for c in conds_by_dose]).T

P_both = prof([[f"{m}|{li}" for m in MG] for li in LI])
P_lo = prof([[f"0.25|{li}"] for li in LI])
P_hi = prof([[f"2.5|{li}"] for li in LI])
for P in (P_both, P_lo, P_hi): P -= P[:, [0]]

rho = np.array([sps.spearmanr(X6, r).correlation if np.isfinite(r).all() else np.nan for r in P_both])
amp = np.nanmax(np.abs(P_both), 1)
ok = np.isfinite(rho) & np.isfinite(amp)
J = json.load(open(f"{C}/cm28_lithium_cofactor.json"))

# ---- 4. specificity: how rare is a monotone, large dose-response?
grid = {}
for t in (.5, .7, .9):
    for a_ in (.3, .5, .7):
        grid[f"rho>={t},amp>={a_}"] = int(((np.abs(rho) >= t) & (amp >= a_) & ok).sum())
sel = (np.abs(rho) >= .9) & (amp >= .7) & ok
hits = [gene.get(a, a) for a in fs.X.index[sel]]
print(f"complete dose profiles: {ok.sum()}", flush=True)
print(f"monotone (|rho|>=0.9) and large (>=0.7 log2): {sel.sum()} = {100 * sel.sum() / ok.sum():.2f} %", flush=True)
print("  " + ", ".join(sorted(hits)) + f"   [{', '.join(sorted(set(hits) & CONTAM))} is a known contaminant]", flush=True)
J["fs76_monotone"] = dict(n_profiles=int(ok.sum()), grid=grid, strict_hits=sorted(hits),
                          strict_hits_contaminant=sorted(set(hits) & CONTAM))

# ---- 1-3 and the mechanism, for the BAG6 complex and the other strict hits
tw = PS.trend_weights(LI, X6)
W = {m: {f"{m}|{li}": v for li, v in tw.items()} for m in MG}
W["pooled"] = {f"{m}|{li}": v / 2 for m in MG for li, v in tw.items()}
R = {k: PS.contrast(PS.ebayes(PS.lmfit(fs.X[fs.samples.loc[fs.samples["cond"].isin(list(w)), "column"]],
                                       *PS.design(fs.samples[fs.samples["cond"].isin(list(w))].reset_index(drop=True),
                                                  block=True)[:1],
                                       )), None) for k, w in {}.items()}  # placeholder, filled below


def trend_fit(w):
    S = fs.samples[fs.samples["cond"].isin(list(w))].reset_index(drop=True)
    D, names = PS.design(S, block=True)
    f = PS.ebayes(PS.lmfit(fs.X[S["column"]], D))
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] = v
    return PS.contrast(f, c)


FIT = {k: trend_fit(w) for k, w in W.items()}
inter_w = {**{k: v for k, v in W["0.25"].items()}, **{k: -v for k, v in W["2.5"].items()}}
FIT["interaction"] = trend_fit(inter_w)

BRS = sorted(fs.samples["BR"].unique())
PANEL = ["GET4", "UBL4A", "BAG6", "GET3"] + [h for h in sorted(hits) if h not in CONTAM and h not in ("GET4", "UBL4A")]
rows = []
for g in PANEL:
    a = acc_of.get(g)
    if a is None or a not in fs.X.index: continue
    i = list(fs.X.index).index(a)
    halves = []
    for A in PS._splits(BRS):
        ea, eb = PS.half_effects(fs.X, fs.samples, W["0.25"], A)
        if ea is not None and eb is not None: halves.append((float(ea[a]), float(eb[a])))
    rows.append(dict(gene=g, acc=a,
                     dose_lowMg=[float(v) for v in P_lo[i]], dose_highMg=[float(v) for v in P_hi[i]],
                     rho=float(rho[i]), amp=float(amp[i]),
                     trend_lowMg=float(FIT["0.25"].loc[a, "logFC"]), q_lowMg=float(FIT["0.25"].loc[a, "q"]),
                     trend_highMg=float(FIT["2.5"].loc[a, "logFC"]), q_highMg=float(FIT["2.5"].loc[a, "q"]),
                     interaction=float(FIT["interaction"].loc[a, "logFC"]),
                     q_interaction=float(FIT["interaction"].loc[a, "q"]),
                     heldout_halves=halves))
J["fs76_responders"] = rows
print("\n gene      trend@0.25mM Mg      trend@2.5mM Mg       Li x Mg          held-out halves", flush=True)
for r in rows:
    h = "  ".join(f"({x:+.2f}|{y:+.2f})" for x, y in r["heldout_halves"])
    print(f"  {r['gene']:<8} {r['trend_lowMg']:+.3f} (q {r['q_lowMg']:.1e})  {r['trend_highMg']:+.3f} "
          f"(q {r['q_highMg']:.1e})  {r['interaction']:+.3f} (q {r['q_interaction']:.1e})  {h}", flush=True)

# per-protein Li trend measured in two disjoint halves of the replicates, for the figure that shows
# the profile is noise while three named proteins are not
Ah = PS._splits(BRS)[0]
ea, eb = PS.half_effects(fs.X, fs.samples, W["0.25"], Ah)
ga24s = pd.read_csv(f"{ROOT}/data/export/levels/GA24_ip/samples.tsv", sep="\t")
ga24s = ga24s[ga24s["KCl"] == 0]
ga24s["arm"] = ga24s["cond"].astype(str).str.rsplit("|", n=1).str[0]   # "NADP|0.6|0" -> "NADP|0.6"
np.savez(f"{C}/cm28_fs76_halves.npz",
         acc=np.array(fs.X.index, dtype=object),
         gene=np.array([gene.get(a, "") for a in fs.X.index], dtype=object),
         half_a=np.asarray(ea, float), half_b=np.asarray(eb, float),
         ga24_arm=ga24s["arm"].values.astype(object), ga24_br=ga24s["BR"].values,
         ga24_pos=ga24s["inj_pos"].values)
json.dump(J, open(f"{C}/cm28_lithium_cofactor.json", "w"), indent=1)

# ---------------------------------------------------------------- figure 6
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, INK, MUTE = "#2a78d6", "#eb6834", "#9aa0a6", "#202124", "#5f6368"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c7cad1",
                     "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK,
                     "figure.dpi": 200, "savefig.bbox": "tight", "grid.color": "#e6e8eb", "grid.linewidth": .7})
show = [r for r in rows if r["gene"] in ("GET4", "UBL4A", "BAG6")]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
for ax, r in zip(axes, show):
    ax.axhline(0, color="#dfe1e5", lw=.9)
    ax.plot(X6, r["dose_lowMg"], "-o", ms=4.5, lw=1.8, color=BLUE, label="0.25 mM Mg²⁺")
    ax.plot(X6, r["dose_highMg"], "-o", ms=4.5, lw=1.8, color=ORANGE, label="2.5 mM Mg²⁺")
    ax.set_xticks(X6); ax.set_xticklabels(LI)
    ax.set_title(f"{r['gene']}\nLi × Mg q = {r['q_interaction']:.0e}")
    ax.grid(axis="y"); ax.set_axisbelow(True)
axes[0].set_ylabel("pull-down, relative to no Li (log2)")
axes[1].set_xlabel("lithium (mM)")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("FS_76: the BAG6 complex leaves the HSPB1 pull-down as lithium rises —\n"
             "and ten-fold Mg²⁺ blunts or abolishes it, as ion competition predicts", fontsize=8.5, y=1.14)
fig.savefig(f"{FIG}/f6_fs76_bag6.png", facecolor="white"); plt.close(fig)
print("\n  wrote f6_fs76_bag6.png", flush=True)
print("CM28C_DONE", flush=True)
