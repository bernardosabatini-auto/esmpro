"""fs02: FS76 lithium screen. HSPB1 pull-down from lysate at 37 C. LiCl replaces NaCl so ionic strength
is constant (Li 0, 0.3, 1, 3, 5, 10 mM), at two Mg2+ levels (0.25 and 2.5 mM), because Li+ competes with
Mg2+ at some metal sites. 4 BR per arm, 4 samples excluded upstream.

Questions
  1. Mg: what does raising Mg2+ from 0.25 to 2.5 mM do to the pull-down, and how reproducibly?
  2. Li: is there a dose-dependent effect, broad or confined to a few proteins? Are the hits robust?
  3. Competition: are Li responders enriched for Mg-binding proteins, and is the Li effect weaker at high Mg?
  4. Does Li look like Mg removal (Li effect anti-correlated with the Mg effect)?

Dose trend = least-squares slope of the six cell means on log10(Li), with 0 placed at 0.09 mM as in the
workbook's dose model: log2 change per 10-fold more lithium.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact
from pdpipe import run, stats as st, plots as pl, io

warnings.filterwarnings("ignore")
ds, fit, names, res, R = run.first_pass("fs76", n_perm=31)
S, X, P = ds.samples, ds.X, ds.proteins
FIG = f"{io.ROOT}/reports/figures/FS76"; OUT = f"{io.ROOT}/data/fs_data/FS76"
gene = P["gene"]; J = {}
DOSES = ["0", "0.3", "1", "3", "5", "10"]; LX = np.log10([0.09, .3, 1, 3, 5, 10])
GO = P["Gene Ontology (molecular function)"].fillna("").astype(str)
CC = P["Gene Ontology (cellular component)"].fillna("").astype(str)
# Mg2+-dependent: UniProt cofactor Mg(2+), keyword Magnesium, or GO magnesium ion binding (GO alone covers 2 %).
# Zinc-binding proteins are the negative control: a metal site, but not one Li+ is expected to compete for.
import gzip, csv
cof = {r["Entry"]: r for r in csv.DictReader(gzip.open(f"{io.ROOT}/data/annot/uniprot_cofactor.tsv.gz", "rt"), delimiter="\t")}
c_ = [cof.get(a, {}) for a in P.index]
mgbind = np.array([("Mg(2+)" in r.get("Cofactor", "")) or ("Magnesium" in r.get("Keywords", "")) for r in c_]) | GO.str.contains("magnesium ion binding").values
znbind = np.array([("Zn(2+)" in r.get("Cofactor", "")) or ("Zinc" in r.get("Keywords", "")) for r in c_]) & ~mgbind
J["n_mg_binding"] = int(mgbind.sum()); J["n_zn_binding"] = int(znbind.sum()); J["n_annotated"] = int(sum(bool(r) for r in c_))


def cvec(weights):
    c = np.zeros(len(names))
    for k, w in weights.items(): c[names.index(k)] += w
    return c


TR = {m: {f"{m}|{d}": w for d, w in zip(DOSES, st.trend_weights(DOSES, LX).values())} for m in ("0.25", "2.5")}
MG = {**{f"2.5|{d}": 1 / 6 for d in DOSES}, **{f"0.25|{d}": -1 / 6 for d in DOSES}}
INTER = {**{k: w for k, w in TR["0.25"].items()}, **{k: -w for k, w in TR["2.5"].items()}}
T = {"trend Mg0.25": st.contrast(fit, cvec(TR["0.25"])), "trend Mg2.5": st.contrast(fit, cvec(TR["2.5"])),
     "Mg effect": st.contrast(fit, cvec(MG)), "trend difference (0.25 - 2.5)": st.contrast(fit, cvec(INTER))}
for m in ("0.25", "2.5"):   # any dose effect: joint F over the five dose-vs-0 contrasts
    C = np.stack([cvec({f"{m}|{d}": 1, f"{m}|0": -1}) for d in DOSES[1:]], 1)
    T[f"any dose Mg{m}"] = st.ftest(fit, C)
pd.concat([v.add_prefix(f"{k}:") for k, v in T.items()], axis=1).assign(gene=gene).to_csv(f"{OUT}/fs02_tests.tsv", sep="\t")
J["hits"] = {k: int((v["q"] <= .05).sum()) for k, v in T.items()}
rel = {"Mg effect": st.split_half(X, S, MG, n_perm=31), "Li trend, Mg 0.25": st.split_half(X, S, TR["0.25"], n_perm=31),
       "Li trend, Mg 2.5": st.split_half(X, S, TR["2.5"], n_perm=31)}
J["reliability"] = {k: dict(reliability=v["reliability"], p_perm=v["p_perm"], n_null=v["n_null"]) for k, v in rel.items()}
print(J["hits"], {k: (round(v["reliability"], 3), v["p_perm"]) for k, v in rel.items()}, flush=True)

# ---- Figure 1: the Mg effect
mg = T["Mg effect"]
f, ax = plt.subplots(1, 4, figsize=(17, 4.1))
pl.volcano(ax[0], mg, "A  Mg 2.5 vs 0.25 mM (all Li doses)", genes=gene.values, label_top=10)
sp = {int(b) for b in st._splits(S["BR"].unique())[0]}; inA = S["BR"].isin(sp)
a = st._effect(X, S, MG, inA); b = st._effect(X, S, MG, ~inA)
pl.scatter_reg(ax[1], a.values, b.values, f"Mg effect, BR {sorted(sp)}", f"Mg effect, BR {sorted(int(x) for x in set(S['BR']) - sp)}",
               f"B  Reproducibility (reliability {rel['Mg effect']['reliability']:.2f})", identity=True)
ab = X[S.loc[S["Mg"] == .25, "column"]].mean(1)
pl.scatter_reg(ax[2], ab.values, mg["logFC"].values, "pull-down level at Mg 0.25 (log2)", "Mg effect (log2)", "C  Mg effect vs level")
pl.binned(ax[2], ab.values, mg["logFC"].values, nb=12)
classes = {"cytosolic ribosome": CC.str.contains("cytosolic (large|small) ribosomal subunit", regex=True),
           "mitochondrial ribosome": CC.str.contains("mitochondrial (large|small) ribosomal subunit", regex=True),
           "RNA binding (other)": GO.str.contains("RNA binding") & ~CC.str.contains("ribosom"),
           "Mg2+ binding": pd.Series(mgbind, index=P.index),
           "membrane (integral)": CC.str.contains("membrane") & P["Transmembrane"].fillna("").astype(str).str.len().gt(0)}
a3 = ax[3]; J["mg_classes"] = {}
base = ~np.logical_or.reduce([m.values for m in classes.values()])
groups = [("none of these", pd.Series(base, index=P.index))] + list(classes.items())
for i, (k, m) in enumerate(groups):
    y = mg["logFC"][m.values].dropna().values
    a3.scatter(np.full(len(y), i) + np.random.default_rng(i).uniform(-.28, .28, len(y)), y, s=3 if i == 0 else 6,
               color=pl.BASE if i == 0 else pl.PAL[i], alpha=.3 if i == 0 else .6, lw=0, rasterized=True)
    a3.plot([i - .32, i + .32], [np.median(y)] * 2, "k-", lw=1.4)
    J["mg_classes"][k] = dict(n=int(len(y)), median=float(np.median(y)))
a3.set_xticks(range(len(groups))); a3.set_xticklabels([k for k, _ in groups], rotation=30, ha="right", fontsize=7)
a3.axhline(0, color="k", lw=.5); a3.set_ylabel("Mg effect (log2)"); a3.set_title("D  Mg effect by protein class")
f.tight_layout(); f.savefig(f"{FIG}/mg_effect.png", dpi=150); plt.close(f)
J["mg_effect"] = dict(n_up=int(((mg["q"] <= .05) & (mg["logFC"] > 0)).sum()), n_down=int(((mg["q"] <= .05) & (mg["logFC"] < 0)).sum()),
                      sd=float(mg["logFC"].std()), bait=float(mg.loc["P04792", "logFC"]),
                      r_level=float(st.ols(ab.values, mg["logFC"].values)["r"]))

# ---- Figure 2: Li reproducibility and Mg dependence
f, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
pl.reliability_bar(ax[0], rel, "A  Proteome-wide reproducibility")
hitset = ((T["trend Mg0.25"]["q"] <= .05) | (T["trend Mg2.5"]["q"] <= .05) | (T["any dose Mg0.25"]["q"] <= .05) | (T["any dose Mg2.5"]["q"] <= .05))
J["li_hits"] = list(gene[hitset.values])
cc = st.cross_concordance(X, S, TR["0.25"], TR["2.5"]); ex, ey = cc["example"]
pl.scatter_reg(ax[1], ex.values, ey.values, "Li trend at Mg 0.25 (one BR half)", "Li trend at Mg 2.5 (other BR half)",
               "B  Li effect at low vs high Mg, all proteins", highlight=hitset.reindex(ex.index).values, labels=gene.reindex(ex.index).values)
ax[1].text(.97, .03, f"all splits: r = {cc['r']:.2f}", transform=ax[1].transAxes, ha="right", fontsize=7)
h = hitset.values
o = pl.scatter_reg(ax[2], T["trend Mg0.25"]["logFC"].values[h], T["trend Mg2.5"]["logFC"].values[h], "Li trend at Mg 0.25 (log2 per 10x Li)",
                   "Li trend at Mg 2.5", f"C  Li hits only (n = {h.sum()})", identity=True, alpha=.9, s=14, color=pl.HIT, robust_lim=False)
for i in np.nonzero(h)[0]:
    ax[2].annotate(gene.iloc[i], (T["trend Mg0.25"]["logFC"].iloc[i], T["trend Mg2.5"]["logFC"].iloc[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
J["li_low_vs_high_mg"] = dict(all_r_disjoint=cc["r"], hits_slope=o["slope"], hits_slope_ci=o["slope_ci"], hits_r=o["r"],
                              hits_weaker_at_high=int((np.abs(T["trend Mg2.5"]["logFC"].values[h]) < np.abs(T["trend Mg0.25"]["logFC"].values[h])).sum()))
f.tight_layout(); f.savefig(f"{FIG}/li_reproducibility.png", dpi=150); plt.close(f)

# ---- Figure 3: dose curves for every Li hit
acc_h = list(P.index[h]); order = np.argsort(T["trend Mg0.25"]["p"].values[h]); acc_h = [acc_h[i] for i in order]
nc = 6; nr = int(np.ceil(len(acc_h) / nc))
f, ax = plt.subplots(nr, nc, figsize=(2.65 * nc, 2.35 * nr), squeeze=False)
for a_, acc in zip(ax.flat, acc_h):
    pl.dose_curve(a_, ds, acc, "conc", "Mg", zero=0.09, title=f"{gene[acc]}{'  [Mg2+]' if mgbind[P.index.get_loc(acc)] else ''}")
    a_.set_xlabel("Li (mM)"); a_.legend().remove() if a_ is not ax.flat[0] else a_.legend(fontsize=6)
for a_ in ax.flat[len(acc_h):]: a_.axis("off")
f.tight_layout(); f.savefig(f"{FIG}/li_dose_curves.png", dpi=140); plt.close(f)

# ---- Figure 4: Mg-binding proteins and Li; Li vs Mg profile
f, ax = plt.subplots(1, 3, figsize=(14.5, 4.2)); J["mg_binding"] = {}
for i, m in enumerate(("0.25", "2.5")):
    t = T[f"trend Mg{m}"]["logFC"]; ok = t.notna().values
    y1, y0 = t.values[ok & mgbind], t.values[ok & ~mgbind]
    J["mg_binding"][f"Mg{m}"] = dict(n=int(len(y1)), median_mg=float(np.median(y1)), median_rest=float(np.median(y0)),
                                     p_mannwhitney=float(mannwhitneyu(y1, y0).pvalue))
# fraction Mg-binding among the strongest Li responders (ranked by trend p) vs background
ns = [10, 20, 30, 50, 100, 200]; tested = T["trend Mg0.25"]["p"].notna().values
for lab_, ann, ls in (("Mg2+-dependent", mgbind, "-"), ("Zn2+-binding (control)", znbind, "--")):
    bg = ann[tested].mean()
    for m, col in (("0.25", pl.PAL[0]), ("2.5", pl.PAL[1])):
        p = T[f"trend Mg{m}"]["p"].values; o = np.argsort(np.where(np.isfinite(p), p, 2))[:tested.sum()]
        fr = [ann[o[:n]].mean() for n in ns]
        pv = [fisher_exact([[ann[o[:n]].sum(), n - ann[o[:n]].sum()], [ann[o].sum() - ann[o[:n]].sum(), len(o) - n - (ann[o].sum() - ann[o[:n]].sum())]], alternative="greater")[1] for n in ns]
        ax[0].plot(ns, np.array(fr) / bg, "o" + ls, color=col, ms=4, label=f"{lab_}, Li at Mg {m}")
        J["mg_binding"][f"{lab_} top_fraction_Mg{m}"] = dict(background=float(bg), **dict(zip(map(str, ns), [dict(frac=float(a_), p=float(b_)) for a_, b_ in zip(fr, pv)])))
ax[0].axhline(1, color="k", ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel("top N proteins, ranked by Li dose-trend p"); ax[0].set_ylabel("fold over the fraction among all proteins")
ax[0].set_title("A  Metal-site proteins among Li responders"); ax[0].legend(fontsize=6.5)
# Mg effect of the Li hits against all proteins (the trend weights sum to zero within Mg 0.25, so the two contrasts share no noise)
me = T["Mg effect"]["logFC"]
J["li_hits_mg_effect"] = dict(median_hits=float(me[hitset.values].median()), median_all=float(me.median()),
                              p_mannwhitney=float(mannwhitneyu(me[hitset.values].dropna(), me[~hitset.values].dropna()).pvalue))
# B: all proteins, Li trend (Mg 0.25) vs Mg effect on disjoint BR halves
cc = st.cross_concordance(X, S, TR["0.25"], MG); ex, ey = cc["example"]
pl.scatter_reg(ax[1], ey.values, ex.values, "Mg effect, 2.5 vs 0.25 (one BR half)", "Li trend at Mg 0.25 (other BR half)",
               "B  Li vs Mg effect, all proteins", highlight=hitset.reindex(ex.index).values, labels=gene.reindex(ex.index).values)
ax[1].text(.97, .03, f"all splits: r = {cc['r']:.2f}", transform=ax[1].transAxes, ha="right", fontsize=7)
J["li_vs_mg_all"] = dict(r=cc["r"])
# C: Li hits. Does 10 mM Li at low Mg reproduce the Mg effect? Disjoint sample groups:
#    Li effect = (0.25|10) - (0.25|0);  Mg effect = mean(2.5|0, 2.5|0.3, 2.5|1) - (0.25|0.3)
cm = st.cond_means(X, S)
li10 = cm["0.25|10"] - cm["0.25|0"]; mgd = cm[["2.5|0", "2.5|0.3", "2.5|1"]].mean(1) - cm["0.25|0.3"]
hh = hitset.values
o = pl.scatter_reg(ax[2], mgd.values[hh], li10.values[hh], "Mg effect at low Li (log2)", "10 mM Li effect at Mg 0.25 (log2)",
                   f"C  Li hits: 10 mM Li reproduces the Mg effect", identity=True, alpha=.9, s=14, color=pl.HIT, robust_lim=False)
for i in np.nonzero(hh)[0]:
    ax[2].annotate(gene.iloc[i], (mgd.iloc[i], li10.iloc[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
same = np.sign(mgd.values[hh]) == np.sign(li10.values[hh])
J["li_reproduces_mg"] = dict(slope=o["slope"], slope_ci=o["slope_ci"], r=o["r"], n=o["n"], same_sign=int(same.sum()),
                             median_ratio=float(np.median((li10 / mgd).values[hh][np.abs(mgd.values[hh]) > .3])))
f.tight_layout(); f.savefig(f"{FIG}/li_mg.png", dpi=150); plt.close(f)

# hit table
rows = []
for acc in acc_h:
    lo = st.leave_one_br_out(X, S, TR["0.25"], [acc]).loc[acc]
    rows.append(dict(gene=gene[acc], accession=acc, mg_binding=bool(mgbind[P.index.get_loc(acc)]),
                     trend_025=float(T["trend Mg0.25"].loc[acc, "logFC"]), q_025=float(T["trend Mg0.25"].loc[acc, "q"]),
                     trend_25=float(T["trend Mg2.5"].loc[acc, "logFC"]), q_25=float(T["trend Mg2.5"].loc[acc, "q"]),
                     q_inter=float(T["trend difference (0.25 - 2.5)"].loc[acc, "q"]), mg_effect=float(T["Mg effect"].loc[acc, "logFC"]),
                     lobo_same_sign_025=bool((np.sign(lo) == np.sign(T["trend Mg0.25"].loc[acc, "logFC"])).all()),
                     q_anydose_025=float(T["any dose Mg0.25"].loc[acc, "q"]), q_anydose_25=float(T["any dose Mg2.5"].loc[acc, "q"])))
J["hit_table"] = rows
json.dump(J, open(f"{OUT}/fs02.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in J.items() if k != "hit_table"}, indent=1, default=float))
for r in rows: print(f"{r['gene']:<9} Mg{'+' if r['mg_binding'] else ' '}  trend0.25 {r['trend_025']:+.2f} (q {r['q_025']:.2g})  trend2.5 {r['trend_25']:+.2f} (q {r['q_25']:.2g})  inter q {r['q_inter']:.2g}  Mg eff {r['mg_effect']:+.2f}  lobo {r['lobo_same_sign_025']}")
print("FS02_DONE")
