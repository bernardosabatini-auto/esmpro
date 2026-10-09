"""fs04: FS76, lithium effects that the Mg2+ effect does not predict.

Most Li hits look like partial Mg2+ mimicry: at low Mg, Li moves a protein part of the way toward its
high-Mg level. A Li effect is called Li-specific when it cannot be explained that way even granting
full mimicry, i.e. 10 mM Li at 0.25 mM Mg doing everything 2.5 mM Mg does:

  Li effect     L = (0.25|10) - (0.25|0)                              must be significant (BH over all proteins)
  excess        E = L - M,  M = mean(2.5 arms, all Li) - (0.25|0)     must be significant (p <= 0.05) with the sign of L

E > 0 in the direction of L covers Li moving a protein the opposite way to Mg, moving it when Mg does
not, and overshooting the Mg level. A second, independent route: any Li effect at 2.5 mM Mg, where the
Mg2+ sites should already be occupied (Li trend or any-dose F test at Mg 2.5, BH q <= 0.05).
Li replaces Na, so "Li-specific" also includes effects of removing that Na.
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pdpipe import stats as st, plots as pl, io

warnings.filterwarnings("ignore")
cfg = io.load_config("fs76"); ds = io.load_proteoda(cfg)
S, X, P = ds.samples, ds.X, ds.proteins; gene = P["gene"]
D, names = st.design(S, block=True); fit = st.ebayes(st.lmfit(X, D))
FIG = f"{io.ROOT}/reports/figures/FS76"; OUT = f"{io.ROOT}/data/fs_data/FS76"
DOSES = ["0", "0.3", "1", "3", "5", "10"]


def cvec(w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] += v
    return c


L = {"0.25|10": 1, "0.25|0": -1}
M = {**{f"2.5|{d}": 1 / 6 for d in DOSES}, "0.25|0": -1}
E = {"0.25|10": 1, **{f"2.5|{d}": -1 / 6 for d in DOSES}}   # L - M: the 0.25|0 terms cancel
tL, tM, tE = (st.contrast(fit, cvec(w)) for w in (L, M, E))
tr = {m: st.trend_weights([f"{m}|{d}" for d in DOSES], np.log10([0.09, .3, 1, 3, 5, 10])) for m in ("0.25", "2.5")}
tT25 = st.contrast(fit, cvec(tr["2.5"])); tT025 = st.contrast(fit, cvec(tr["0.25"]))
F25 = st.ftest(fit, np.stack([cvec({f"2.5|{d}": 1, "2.5|0": -1}) for d in DOSES[1:]], 1))

sigL = (tL["q"] <= .05).values
excess = sigL & (tE["p"] <= .05).values & (np.sign(tE["logFC"]) == np.sign(tL["logFC"])).values
mglike = sigL & ~excess
highmg = ((tT25["q"] <= .05) | (F25["q"] <= .05)).values


def kind(i):
    l, m = tL["logFC"].iloc[i], tM["logFC"].iloc[i]
    if abs(m) < .15: return "Mg has no effect"
    if np.sign(l) != np.sign(m): return "opposite to Mg"
    return "beyond the Mg level"


J = dict(n_li_sig=int(sigL.sum()), n_excess=int(excess.sum()), n_mglike=int(mglike.sum()), n_highmg=int(highmg.sum()), rows=[])
for i in np.nonzero(sigL | highmg)[0]:
    J["rows"].append(dict(gene=gene.iloc[i], accession=P.index[i], li10=float(tL["logFC"].iloc[i]), q_li=float(tL["q"].iloc[i]),
                          mg=float(tM["logFC"].iloc[i]), excess=float(tE["logFC"].iloc[i]), p_excess=float(tE["p"].iloc[i]),
                          trend_25=float(tT25["logFC"].iloc[i]), q_trend_25=float(tT25["q"].iloc[i]), q_F25=float(F25["q"].iloc[i]),
                          trend_025=float(tT025["logFC"].iloc[i]),
                          call="Li-specific" if excess[i] else ("Mg-like" if mglike[i] else "") + (" + effect at high Mg" if highmg[i] else ""),
                          kind=kind(i) if excess[i] else ""))
pd.DataFrame(J["rows"]).to_csv(f"{OUT}/fs04_li_specific.tsv", sep="\t", index=False)

# Figure: Li effect against Mg effect; the band between 0 and the identity line is what Mg mimicry can explain
f, a = plt.subplots(figsize=(7.5, 5.6)); x, y = tM["logFC"].values, tL["logFC"].values
lim = (-3, 2.2)
xx = np.linspace(*lim, 2)
a.fill_between(xx, 0, xx, color="#3b6fb6", alpha=.08, lw=0, label="explainable by Mg mimicry (0 to full)")
a.scatter(x, y, s=3, color=pl.BASE, alpha=.35, lw=0, rasterized=True)
a.scatter(x[mglike], y[mglike], s=18, color="#3b6fb6", label=f"Li effect, Mg-like ({mglike.sum()})")
a.scatter(x[excess], y[excess], s=22, color=pl.HIT, label=f"Li effect beyond full Mg mimicry ({excess.sum()})")
hm = highmg & ~excess
a.scatter(x[hm], y[hm], s=40, facecolor="none", edgecolor="k", lw=.8, label=f"also changes with Li at 2.5 mM Mg ({highmg.sum()})")
for i in np.nonzero(excess | mglike | highmg)[0]:
    a.annotate(gene.iloc[i], (x[i], y[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
a.plot(xx, xx, "k:", lw=.8); a.axhline(0, color="k", lw=.4); a.axvline(0, color="k", lw=.4)
a.set_xlim(*lim); a.set_ylim(-1.8, 1.6)
a.set_xlabel("Mg effect: 2.5 mM arms minus 0.25 mM, no Li (log2)"); a.set_ylabel("10 mM Li at 0.25 mM Mg, minus no Li (log2)")
a.set_title("Which Li effects does Mg predict?"); a.legend(fontsize=6.5, loc="lower right")
o = sorted(np.nonzero(excess)[0], key=lambda i: tL["p"].iloc[i])
f.tight_layout(); f.savefig(f"{FIG}/li_specific_scatter.png", dpi=150); plt.close(f)

acc = [P.index[i] for i in o] + [P.index[i] for i in np.nonzero(highmg & ~excess)[0]]
if acc:
    nc = 5; nr = int(np.ceil(len(acc) / nc))
    f, ax = plt.subplots(nr, nc, figsize=(2.75 * nc, 2.4 * nr), squeeze=False)
    for a_, ac in zip(ax.flat, acc):
        i = P.index.get_loc(ac)
        pl.dose_curve(a_, ds, ac, "conc", "Mg", zero=0.09, title=f"{gene[ac]}  ({'Li-specific' if excess[i] else 'Mg-like, also at high Mg'})")
        a_.set_xlabel("Li (mM)"); a_.legend().remove() if a_ is not ax.flat[0] else a_.legend(fontsize=6)
    for a_ in ax.flat[len(acc):]: a_.axis("off")
    f.tight_layout(); f.savefig(f"{FIG}/li_specific_curves.png", dpi=140); plt.close(f)

json.dump(J, open(f"{OUT}/fs04.json", "w"), indent=1, default=float)
print({k: v for k, v in J.items() if k != "rows"})
for r in sorted(J["rows"], key=lambda r: r["q_li"]):
    print(f"{r['gene']:<9} Li10 {r['li10']:+.2f} (q {r['q_li']:.2g})  Mg {r['mg']:+.2f}  excess {r['excess']:+.2f} (p {r['p_excess']:.2g})  "
          f"trend@2.5 {r['trend_25']:+.2f} (q {r['q_trend_25']:.2g}, F q {r['q_F25']:.2g})  -> {r['call']} {r['kind']}")
