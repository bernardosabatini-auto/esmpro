"""fs01: FS73 anticonvulsant screen. HA-HSPB1 pull-down from HEK293 lysate at 37 C with carbamazepine,
lamotrigine or topiramate at 10 or 100 uM (control: no drug).

Questions
  1. Does any drug change the HSPB1 pull-down broadly (proteome-wide reproducible profile)?
  2. Are the called hits real: do they survive leaving out each biological replicate, do the strongest
     hits found in one half of the replicates reappear in the other half, do they follow dose?
  3. Are the hits drug-specific, and do they match known drug targets (lamotrigine: DHFR, a weak
     antifolate target; topiramate: carbonic anhydrases)?

Standard pass via pdpipe.run, then the screen-specific tests. Writes data/fs_data/FS73/fs01.json and
reports/figures/FS73/*.png.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pdpipe import run, stats as st, plots as pl, io

warnings.filterwarnings("ignore")
ds, fit, names, res, R = run.first_pass("fs73", n_perm=31)
S, X, P = ds.samples, ds.X, ds.proteins
FIG = f"{io.ROOT}/reports/figures/FS73"; OUT = f"{io.ROOT}/data/fs_data/FS73"
DRUG = {"car": "carbamazepine", "lam": "lamotrigene", "top": "topiramate"}
NICE = {"car": "carbamazepine", "lam": "lamotrigine", "top": "topiramate"}
W = {cn: {a: 1, b: -1} for cn, (a, b) in ds.meta["contrasts"].items()}
gene = P["gene"]; J = {}

# 1. proteome-wide: reliability bar + split-half scatter + cross-validated top-hit replication
rep = {cn: st.top_hit_replication(X, S, w, n_top=10) for cn, w in W.items()}
J["top_hit_replication"] = {cn: dict(heldout_mean=float(np.mean([r["heldout_effect"] for r in v])),
                                     heldout_same_sign=float(np.mean([r["heldout_same_sign"] for r in v])),
                                     train_mean=float(np.mean([r["train_effect"] for r in v]))) for cn, v in rep.items()}
f, ax = plt.subplots(1, 3, figsize=(15, 4.2))
pl.reliability_bar(ax[0], R["reliability"], "A  Proteome-wide reproducibility")
sp = {int(b) for b in st._splits(S["BR"].unique())[0]}
inA = S["BR"].isin(sp)
a = st._effect(X, S, W["lam_100"], inA); b = st._effect(X, S, W["lam_100"], ~inA)
hit = (res["lam_100"]["q"] <= .05).reindex(X.index).values
pl.scatter_reg(ax[1], a.values, b.values, f"lamotrigine 100 uM effect, BR {sorted(sp)}", f"same, BR {sorted(int(b) for b in set(S['BR']) - sp)}",
               "B  Lamotrigine 100: one half vs the other", highlight=hit, labels=gene.values)
a2 = ax[2]
for i, (cn, v) in enumerate(rep.items()):
    y = [r["heldout_effect"] for r in v]; yb = [r["train_effect"] for r in v]
    a2.scatter(np.full(len(y), i) + np.linspace(-.15, .15, len(y)), y, s=12, color=pl.HIT, label="held-out half" if i == 0 else None)
    a2.plot([i - .25, i + .25], [np.mean(yb)] * 2, color="k", lw=1, label="discovery half (mean)" if i == 0 else None)
a2.axhline(0, color="k", lw=.5); a2.set_xticks(range(len(rep))); a2.set_xticklabels(list(rep), rotation=40, ha="right")
a2.set_ylabel("mean |log2 change| of the top 10,\nsigned by the discovery half"); a2.set_title("C  Do the top 10 hits replicate?"); a2.legend(fontsize=7)
f.tight_layout(); f.savefig(f"{FIG}/overview.png", dpi=160); plt.close(f)

# 2. dose consistency and drug specificity on disjoint replicates
f, ax = plt.subplots(2, 3, figsize=(13.5, 8.4)); J["concordance"] = {}
pairs = [(f"{d}_10", f"{d}_100", f"{NICE[d]}: 10 vs 100 uM") for d in DRUG] + \
        [("lam_100", "top_100", "lamotrigine vs topiramate, 100 uM"), ("lam_100", "car_100", "lamotrigine vs carbamazepine, 100 uM"),
         ("car_100", "top_100", "carbamazepine vs topiramate, 100 uM")]
for a_, (x, y, t) in zip(ax.flat, pairs):
    cc = st.cross_concordance(X, S, W[x], W[y]); ex, ey = cc["example"]
    hl = ((res[x]["q"] <= .05) | (res[y]["q"] <= .05)).reindex(ex.index).values
    o = pl.scatter_reg(a_, ex.values, ey.values, f"{x} (one BR half)", f"{y} (other BR half)", t, highlight=hl, labels=gene.reindex(ex.index).values)
    a_.text(.97, .03, f"all splits: r = {cc['r']:.2f}", transform=a_.transAxes, ha="right", fontsize=7)
    J["concordance"][f"{x} vs {y}"] = dict(r=cc["r"], r_disattenuated=cc["r_disattenuated"], example_slope=o["slope"])
f.tight_layout(); f.savefig(f"{FIG}/concordance.png", dpi=150); plt.close(f)

# 3. every hit: per-sample values, and leave-one-BR-out stability
hits = []
for cn in W:
    for acc in res[cn].index[res[cn]["q"] <= .05]: hits.append((cn, acc))
J["hits"] = []
for cn, acc in hits:
    lo = st.leave_one_br_out(X, S, W[cn], [acc]).loc[acc]
    J["hits"].append(dict(contrast=cn, gene=gene[acc], accession=acc, logFC=float(res[cn].loc[acc, "logFC"]), q=float(res[cn].loc[acc, "q"]),
                          lobo_min=float(lo.abs().min()), lobo_same_sign=bool((np.sign(lo) == np.sign(res[cn].loc[acc, "logFC"])).all()),
                          other_contrasts={c: float(res[c].loc[acc, "logFC"]) for c in W}))
order = ["ctl|0", "carbamazepine|10", "carbamazepine|100", "lamotrigene|10", "lamotrigene|100", "topiramate|10", "topiramate|100"]
xt = ["ctl", "car 10", "car 100", "lam 10", "lam 100", "top 10", "top 100"]
uh = list(dict.fromkeys(acc for _, acc in hits)); nc = 6; nr = int(np.ceil(len(uh) / nc))
f, ax = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.5 * nr), squeeze=False)
for a_, acc in zip(ax.flat, uh):
    cs = ", ".join(sorted({c for c, x in hits if x == acc}))
    pl.cond_strip(a_, ds, acc, order, xt, title=f"{gene[acc]}  ({cs})")
for a_ in ax.flat[len(uh):]: a_.axis("off")
f.tight_layout(); f.savefig(f"{FIG}/hits.png", dpi=140); plt.close(f)

# 4. known targets and target classes
GO = P["Gene Ontology (molecular function)"].fillna("").astype(str)
cls = {"protein kinase / nucleoside kinase": GO.str.contains("kinase activity") & ~GO.str.contains("kinase regulator"),
       "carbonic anhydrase": GO.str.contains("carbonate dehydratase activity"),
       "folate binding": GO.str.contains("folic acid binding|tetrahydrofolate|dihydrofolate", regex=True)}
J["classes"] = {}
from scipy.stats import mannwhitneyu
for cn in ("lam_100", "top_10", "top_100"):
    v = res[cn]["logFC"]
    for k, m in cls.items():
        m = m.reindex(v.index).fillna(False).values; y = v[m].dropna().values; z = v[~m].dropna().values
        J["classes"][f"{cn}: {k}"] = dict(n=int(len(y)), median=float(np.median(y)), median_rest=float(np.median(z)),
                                          p_mannwhitney=float(mannwhitneyu(y, z).pvalue) if len(y) > 2 else None)

# 5. hits against abundance: are they confined to weak signals?
f, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ab = X[S.loc[S["cond"] == "ctl|0", "column"]].mean(1)
for a_, cn in zip(ax, ["car_100", "lam_100", "top_10"]):
    hl = (res[cn]["q"] <= .05).values
    pl.scatter_reg(a_, ab.values, res[cn]["logFC"].values, "control pull-down level (log2)", f"{cn} log2 change", cn, highlight=hl, labels=gene.values)
f.tight_layout(); f.savefig(f"{FIG}/abundance.png", dpi=150); plt.close(f)

json.dump(J, open(f"{OUT}/fs01.json", "w"), indent=1, default=float)
print(json.dumps({k: J[k] for k in ("top_hit_replication", "concordance", "classes")}, indent=1, default=float))
for h in J["hits"]: print(f"{h['contrast']:<8} {h['gene']:<9} {h['logFC']:+.2f} q {h['q']:.2g}  leave-one-BR-out min |effect| {h['lobo_min']:.2f} same sign {h['lobo_same_sign']}")
print("FS01_DONE")
