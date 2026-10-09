"""fs08: GA_22 staurosporine and GA_24 NAD / NADP in the HSPB1 pull-down, with the post-IP supernatants.

The ligand arms are partly confounded with MS injection order (GA_24: arm mean positions 0.30-0.70; GA_22:
staurosporine 0.1 and 10 uM later than 0 and 1 uM). All models carry a quadratic drift term (pdpipe
drift_basis, guarded lmfit) and split-half reliabilities use the cross-fitted drift_adjuster. Condition-level
bench batches cannot be corrected, so the decisive tests are biological specificity, which drift cannot fake:

  staurosporine  an ATP-site kinase inhibitor: do kinases respond more than other proteins, with dose?
  NAD / NADP     do NAD-binding proteins respond to NAD and NADP-binding proteins to NADP, more than the
                 other cofactor's binders and more than FAD-binding proteins (a cofactor-binding control)?

Effects are averaged over the three KCl levels (salt and ligand are crossed and balanced) and also shown per KCl.
Units of the NAD(P) arms as in the sample names (NADP 0.6, 15; NAD 11, 300).
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pdpipe import io, stats as st, plots as pl, partition as pt

warnings.filterwarnings("ignore")
FIG = f"{io.ROOT}/reports/figures/GA22_24"; OUT = f"{io.ROOT}/data/fs_data/GA22_24"
J = {}; KCL = ["0", "60", "120"]


def drift_tools(ds):
    H = st.drift_basis(ds.samples["inj_pos"], 2); H.index = ds.samples["column"].values
    return H.reset_index(drop=True), st.drift_adjuster(H)


def fit(ds):
    cov = drift_tools(ds)[0]; D, names = st.design(ds.samples, block=True, covars=cov)
    return st.ebayes(st.lmfit(ds.X, D, covar=[names.index(c) for c in cov])), names


def contrast(f, names, w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] += v
    return st.contrast(f, c)


def load(e):
    ip, sup = io.load(io.load_config(e.lower())), io.load(io.load_config(e.lower() + "s"))
    return dict(ip=ip, sup=sup, part=pt.pair(ip, sup))


U = io.uniprot_annotation(); KW = U["Keywords"].fillna(""); GOF = U["Gene Ontology (molecular function)"].fillna("")
kin = KW.str.contains("Kinase") & GOF.str.contains("kinase activity")
nad = KW.str.contains(r"(?:^|;)NAD(?:;|$)", regex=True); nadp = KW.str.contains(r"(?:^|;)NADP(?:;|$)", regex=True)
fad = KW.str.contains(r"(?:^|;)FAD(?:;|$)", regex=True)
CLASS = {"NAD only": nad & ~nadp, "NADP only": nadp & ~nad, "NAD and NADP": nad & nadp, "FAD (control)": fad & ~nad & ~nadp}
J["class_sizes"] = {k: int(v.sum()) for k, v in CLASS.items()} | {"kinase": int(kin.sum())}


def class_test(y, mask):
    m = mask.reindex(y.index).fillna(False).values & y.notna().values; r = ~mask.reindex(y.index).fillna(False).values & y.notna().values
    return dict(n=int(m.sum()), median=float(np.median(y[m])) if m.any() else None, median_rest=float(np.median(y[r])),
                p=float(mannwhitneyu(y[m], y[r]).pvalue) if m.sum() > 3 else None)


def ligand(e, arms, ref, cond, label):
    """arms: ligand levels; ref: reference level; cond(level, kcl) -> condition key."""
    D = load(e); R, REL = {}, {}
    for kind, ds in D.items():
        f, names = fit(ds); adj = drift_tools(ds)[1]
        for a in arms:
            w = {**{cond(a, k): 1 / 3 for k in KCL}, **{cond(ref, k): -1 / 3 for k in KCL}}
            R[(kind, a, "all")] = contrast(f, names, w)
            r = st.split_half(ds.X, ds.samples, w, n_perm=20, adjust=adj)
            REL[(kind, a)] = dict(reliability=r["reliability"], null95=r["null_rel95"], p_perm=r["p_perm"],
                                  hits=int((R[(kind, a, "all")]["q"] <= .05).sum()))
            for k in KCL: R[(kind, a, k)] = contrast(f, names, {cond(a, k): 1, cond(ref, k): -1})
            print(e, kind, a, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in REL[(kind, a)].items()}, flush=True)
    return D, R, REL


# ---------------------------------------------------------------- GA_22 staurosporine
STAU = ["0.1", "1", "10"]
D22, R22, REL22 = ligand("GA22", STAU, "0", lambda a, k: f"{k}|{a}", "staurosporine")
J["GA22"] = dict(reliability={f"{k[0]} {k[1]}": v for k, v in REL22.items()}, kinase={}, dose={})
for kind in ("ip", "sup", "part"):
    for a in STAU:
        for k in ["all"] + KCL: J["GA22"]["kinase"][f"{kind} {a} {k}"] = class_test(R22[(kind, a, k)]["logFC"], kin)
    ds = D22[kind]; adj = drift_tools(ds)[1]
    wa = {a: {**{f"{k}|{a}": 1 / 3 for k in KCL}, **{f"{k}|0": -1 / 3 for k in KCL}} for a in STAU}
    for a, b in (("0.1", "1"), ("1", "10"), ("0.1", "10")):
        cc = st.cross_dataset(ds, wa[a], ds, wa[b], adj, adj); J["GA22"]["dose"][f"{kind} {a} vs {b}"] = dict(r=cc["r"], r_dis=cc["r_disattenuated"])
print("GA22 kinase", {k: (v["n"], round(v["median"] - v["median_rest"], 3), f"{v['p']:.2g}") for k, v in J["GA22"]["kinase"].items() if k.endswith("all")}, flush=True)
print("GA22 dose", J["GA22"]["dose"], flush=True)

# ---------------------------------------------------------------- GA_24 NAD / NADP
ARMS = ["NADP|0.6", "NADP|15", "NAD|11", "NAD|300"]
D24, R24, REL24 = ligand("GA24", ARMS, "CTRL|0", lambda a, k: f"{a}|{k}", "cofactor")
J["GA24"] = dict(reliability={f"{k[0]} {k[1]}": v for k, v in REL24.items()}, classes={}, concordance={})
for kind in ("ip", "sup", "part"):
    for a in ARMS:
        for k in ["all"] + KCL:
            for cn, m in CLASS.items(): J["GA24"]["classes"][f"{kind} {a} {k} {cn}"] = class_test(R24[(kind, a, k)]["logFC"], m)
    ds = D24[kind]; adj = drift_tools(ds)[1]
    wa = {a: {**{f"{a}|{k}": 1 / 3 for k in KCL}, **{f"CTRL|0|{k}": -1 / 3 for k in KCL}} for a in ARMS}
    for a, b in (("NADP|0.6", "NADP|15"), ("NAD|11", "NAD|300"), ("NADP|15", "NAD|300"), ("NADP|0.6", "NAD|11")):
        cc = st.cross_dataset(ds, wa[a], ds, wa[b], adj, adj); J["GA24"]["concordance"][f"{kind} {a} vs {b}"] = dict(r=cc["r"], r_dis=cc["r_disattenuated"])
    # the same ligand at different salts: a real cofactor effect should not depend on which KCl level it is read at
    for a in ("NADP|15", "NAD|300"):
        for k1, k2 in (("0", "120"), ("0", "60"), ("60", "120")):
            cc = st.cross_dataset(ds, {f"{a}|{k1}": 1, f"CTRL|0|{k1}": -1}, ds, {f"{a}|{k2}": 1, f"CTRL|0|{k2}": -1}, adj, adj)
            J["GA24"]["concordance"][f"{kind} {a} KCl{k1} vs KCl{k2}"] = dict(r=cc["r"], r_dis=cc["r_disattenuated"])
print("GA24 classes (all KCl):", {k: (v["n"], round((v["median"] or 0) - v["median_rest"], 3), f"{v['p']:.2g}" if v["p"] else None) for k, v in J["GA24"]["classes"].items() if " all " in k}, flush=True)
print("GA24 concordance", {k: round(v["r"], 3) for k, v in J["GA24"]["concordance"].items()}, flush=True)

# ---------------------------------------------------------------- cross-controls and internal replication
# class x ligand matrix: each ligand's own class should move, the other classes should not
MAT = {"kinase": kin, **CLASS}
J["matrix"] = {}
for kind in ("ip", "part", "sup"):
    for lig, R, a in (("staurosporine 10", R22, "10"), ("NADP 15", R24, "NADP|15"), ("NAD 300", R24, "NAD|300"), ("NADP 0.6", R24, "NADP|0.6"), ("NAD 11", R24, "NAD|11")):
        for cn, m in MAT.items():
            for k in ["all"] + KCL:
                J["matrix"][f"{kind}|{lig}|{cn}|{k}"] = class_test(R[(kind, a, k)]["logFC"], m)
# the proteins behind each class effect: cognate binders ranked by effect in the pull-down, with dose and salt
def top(R, a_lo, a_hi, mask, kind="ip", n=12):
    y = R[(kind, a_hi, "all")]; m = mask.reindex(y.index).fillna(False)
    rows = []
    for acc in y.index[m.values & y["logFC"].notna().values]:
        rows.append(dict(accession=acc, hi=float(y.loc[acc, "logFC"]), q_hi=float(y.loc[acc, "q"]), lo=float(R[(kind, a_lo, "all")]["logFC"].get(acc, np.nan)),
                         **{f"hi_KCl{k}": float(R[(kind, a_hi, k)]["logFC"].get(acc, np.nan)) for k in KCL},
                         sup_hi=float(R[("sup", a_hi, "all")]["logFC"].get(acc, np.nan)), part_hi=float(R[("part", a_hi, "all")]["logFC"].get(acc, np.nan)) if acc in R[("part", a_hi, "all")].index else np.nan))
    return pd.DataFrame(rows).sort_values("hi")
G24 = D24["ip"].proteins["gene"]; G22 = D22["ip"].proteins["gene"]
for nm, R, lo, hi, m, G in (("NADP binders", R24, "NADP|0.6", "NADP|15", CLASS["NADP only"] | CLASS["NAD and NADP"], G24),
                             ("NAD binders", R24, "NAD|11", "NAD|300", CLASS["NAD only"] | CLASS["NAD and NADP"], G24),
                             ("kinases", R22, "1", "10", kin, G22)):
    T = top(R, lo, hi, m); T.insert(0, "gene", G.reindex(T["accession"]).values)
    T.to_csv(f"{OUT}/{nm.replace(' ', '_')}_responders.tsv", sep="\t", index=False); J[f"top {nm}"] = T.head(15).round(3).to_dict("records")
    hits = R[("ip", hi, "all")]; dn = (hits["q"] <= .05) & (hits["logFC"] < 0); mm = m.reindex(hits.index).fillna(False)
    J[f"enrichment {nm}"] = dict(down_hits=int(dn.sum()), class_in_down=int((dn & mm).sum()), class_total=int((mm & hits["logFC"].notna()).sum()),
                                 frac_down=float((dn & mm).sum() / max(dn.sum(), 1)), frac_background=float(mm[hits["logFC"].notna()].mean()))
    print(nm, J[f"enrichment {nm}"], flush=True); print(T.head(12).round(2).to_string(index=False), flush=True)

# tables
for e, D, R in (("GA22", D22, R22), ("GA24", D24, R24)):
    for kind in ("ip", "sup", "part"):
        cols = [v.add_prefix(f"{a}@{k}:") for (kk, a, k), v in R.items() if kk == kind]
        pd.concat(cols, axis=1).assign(gene=D[kind].proteins["gene"]).to_csv(f"{OUT}/{e}_{kind}_ligand.tsv", sep="\t")

# ---------------------------------------------------------------- figures
NICE = {"ip": "pull-down", "sup": "supernatant", "part": "partition"}
# A. reliability of every ligand contrast, with the permutation null
f, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[3, 4]))
for a_, (e, REL, arms) in zip(ax, (("GA22", REL22, STAU), ("GA24", REL24, ARMS))):
    keys = [(kind, a) for kind in ("ip", "sup", "part") for a in arms]
    v = [REL[k]["reliability"] for k in keys]; nl = [REL[k]["null95"] for k in keys]
    a_.bar(range(len(keys)), v, color=[pl.HIT if x > y else pl.BASE for x, y in zip(v, nl)]); a_.scatter(range(len(keys)), nl, marker="_", s=200, color="k")
    a_.set_xticks(range(len(keys))); a_.set_xticklabels([f"{NICE[k]}\n{a}" for k, a in keys], fontsize=6.5, rotation=60, ha="right")
    a_.axhline(0, color="k", lw=.4); a_.set_ylabel("split-half reliability\n(drift-corrected)")
    a_.set_title(f"{'AB'[e == 'GA24']}  {e}: {'staurosporine (uM)' if e == 'GA22' else 'cofactor arms vs no cofactor'}")
f.tight_layout(); f.savefig(f"{FIG}/ligand_reliability.png", dpi=150); plt.close(f)

# B. staurosporine: kinases vs the rest, by dose and fraction
f, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a_, kind in zip(ax, ("ip", "sup", "part")):
    for i, a in enumerate(STAU):
        y = R22[(kind, a, "all")]["logFC"]; m = kin.reindex(y.index).fillna(False).values & y.notna().values; r = ~m & y.notna().values
        parts = a_.violinplot([y[r].values, y[m].values], positions=[i * 3, i * 3 + 1], showmedians=True, showextrema=False, widths=.8); parts["cmedians"].set_color("k")
        for b, col in zip(parts["bodies"], [pl.BASE, pl.HIT]): b.set_facecolor(col); b.set_alpha(.5)
        t = J["GA22"]["kinase"][f"{kind} {a} all"]; a_.text(i * 3 + .5, a_.get_ylim()[1] if False else .9, f"p {t['p']:.2g}", ha="center", fontsize=7, transform=a_.get_xaxis_transform())
    a_.set_xticks([0, 1, 3, 4, 6, 7]); a_.set_xticklabels(["other\n0.1", "kinase\n0.1", "other\n1", "kinase\n1", "other\n10", "kinase\n10"], fontsize=7)
    a_.axhline(0, color="k", lw=.4); a_.set_ylabel("staurosporine effect (log2)"); a_.set_ylim(-1.2, 1.2)
    a_.set_title(f"{'ABC'[('ip', 'sup', 'part').index(kind)]}  {NICE[kind]}: kinases vs other proteins")
f.tight_layout(); f.savefig(f"{FIG}/stau_kinases.png", dpi=150); plt.close(f)

# C. NAD(P): class medians (minus the rest) for every arm, in pull-down and partition
f, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, kind in zip(ax, ("ip", "sup", "part")):
    xx = np.arange(len(ARMS)); wd = .2
    for i, cn in enumerate(CLASS):
        v = [(J["GA24"]["classes"][f"{kind} {a} all {cn}"]["median"] or 0) - J["GA24"]["classes"][f"{kind} {a} all {cn}"]["median_rest"] for a in ARMS]
        p = [J["GA24"]["classes"][f"{kind} {a} all {cn}"]["p"] for a in ARMS]
        a_.bar(xx + (i - 1.5) * wd, v, wd, color=[pl.HIT, pl.LINE, "#6a3d9a", pl.BASE][i], label=f"{cn} (n={J['GA24']['classes'][f'{kind} {ARMS[0]} all {cn}']['n']})")
        for x, vv, pp in zip(xx + (i - 1.5) * wd, v, p):
            if pp is not None and pp < .01: a_.text(x, vv, "*", ha="center", va="bottom" if vv >= 0 else "top", fontsize=9)
    a_.axhline(0, color="k", lw=.4); a_.set_xticks(xx); a_.set_xticklabels([a.replace("|", " ") for a in ARMS])
    a_.set_ylabel("median effect, class minus rest (log2)"); a_.set_title(f"{'ABC'[('ip', 'sup', 'part').index(kind)]}  {NICE[kind]} (* p < 0.01)")
ax[0].legend(fontsize=7)
f.tight_layout(); f.savefig(f"{FIG}/nadp_classes.png", dpi=150); plt.close(f)

# D. the ligand x protein-class matrix: median shift of each class (pull-down), and the class's share of the
#    proteins each ligand significantly removes (q <= 0.05, down), as fold over its share of all proteins
LIGS = ["staurosporine 10", "NADP 0.6", "NADP 15", "NAD 11", "NAD 300"]; CL = ["kinase", "NADP only", "NAD only", "NAD and NADP", "FAD (control)"]
LR = {"staurosporine 10": (R22, "10"), "NADP 0.6": (R24, "NADP|0.6"), "NADP 15": (R24, "NADP|15"), "NAD 11": (R24, "NAD|11"), "NAD 300": (R24, "NAD|300")}
MAT2 = {"kinase": kin, **CLASS}; J["fold"] = {}
F = np.zeros((len(LIGS), len(CL))); NN = np.zeros_like(F, dtype=int)
for i, l in enumerate(LIGS):
    R, a = LR[l]; y = R[("ip", a, "all")]; ok = y["logFC"].notna(); dn = ok & (y["q"] <= .05) & (y["logFC"] < 0)
    for j, c in enumerate(CL):
        m = MAT2[c].reindex(y.index).fillna(False); k = int((dn & m).sum()); NN[i, j] = k
        F[i, j] = (k / max(dn.sum(), 1)) / max(m[ok].mean(), 1e-9)
        J["fold"][f"{l}|{c}"] = dict(in_class=k, down=int(dn.sum()), fold=float(F[i, j]))
f, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
M = np.array([[(J["matrix"][f"ip|{l}|{c}|all"]["median"] or 0) - J["matrix"][f"ip|{l}|{c}|all"]["median_rest"] for c in CL] for l in LIGS])
P = np.array([[J["matrix"][f"ip|{l}|{c}|all"]["p"] or 1 for c in CL] for l in LIGS])
im = ax[0].imshow(M, cmap="RdBu_r", vmin=-.3, vmax=.3, aspect="auto")
for (i, j), v in np.ndenumerate(M): ax[0].text(j, i, f"{v:+.2f}" + ("*" if P[i, j] < 1e-3 else ""), ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax[0], shrink=.8, label="log2")
im = ax[1].imshow(np.log2(np.maximum(F, .25)), cmap="Purples", vmin=0, vmax=6, aspect="auto")
for (i, j), v in np.ndenumerate(F): ax[1].text(j, i, f"{v:.0f}x\n({NN[i, j]})" if NN[i, j] else "-", ha="center", va="center", fontsize=6.5, color="w" if v > 16 else "k")
plt.colorbar(im, ax=ax[1], shrink=.8, label="log2 fold enrichment")
for a_ in ax:
    a_.set_xticks(range(len(CL))); a_.set_xticklabels(CL, rotation=25, ha="right", fontsize=8); a_.set_yticks(range(len(LIGS))); a_.set_yticklabels(LIGS, fontsize=8)
ax[0].set_title("A  Median shift, class vs other proteins\n(pull-down, * p < 0.001)", fontsize=9)
ax[1].set_title("B  Class share of the proteins each ligand removes\n(fold over background; n in brackets)", fontsize=9)
f.tight_layout(); f.savefig(f"{FIG}/ligand_matrix.png", dpi=150); plt.close(f)
print("fold", {k: round(v["fold"], 1) for k, v in J["fold"].items()}, flush=True)

# E. each ligand moves its own class: scatter of two ligands' effects, proteins coloured by class
f, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ds = D24["ip"]; adj = drift_tools(ds)[1]
wN = {**{f"NADP|15|{k}": 1 / 3 for k in KCL}, **{f"CTRL|0|{k}": -1 / 3 for k in KCL}}; wD = {**{f"NAD|300|{k}": 1 / 3 for k in KCL}, **{f"CTRL|0|{k}": -1 / 3 for k in KCL}}
cc = st.cross_dataset(ds, wN, ds, wD, adj, adj); x, y = cc["example"]; j = x.index.intersection(y.index)
COL = {"NADP binders": (CLASS["NADP only"], pl.LINE), "NAD binders": (CLASS["NAD only"], pl.HIT), "kinases": (kin, "#e08e0b")}
def panel(a_, x, y, xl, yl, ttl, classes):
    a_.scatter(x, y, s=3, color=pl.BASE, alpha=.3, lw=0, rasterized=True)
    for nm in classes:
        m, col = COL[nm]; mm = m.reindex(x.index).fillna(False).values
        a_.scatter(x[mm], y[mm], s=14, color=col, label=f"{nm} (n={mm.sum()})", zorder=3)
    a_.axhline(0, color="k", lw=.4); a_.axvline(0, color="k", lw=.4); a_.set_xlabel(xl); a_.set_ylabel(yl); a_.set_title(ttl, fontsize=9); a_.legend(fontsize=7)
panel(ax[0], x[j], y[j], "NADP 15 effect, pull-down (one half of BRs)", "NAD 300 effect (other half)", "A  NADP and NAD remove different proteins", ["NADP binders", "NAD binders"])
ax[0].text(.97, .03, f"all proteins: r = {cc['r']:.2f}", transform=ax[0].transAxes, ha="right", fontsize=7)
s10 = R22[("ip", "10", "all")]["logFC"]; n15 = R24[("ip", "NADP|15", "all")]["logFC"]; j = s10.index.intersection(n15.index)
panel(ax[1], n15[j], s10[j], "NADP 15 effect, pull-down (GA_24)", "staurosporine 10 uM effect, pull-down (GA_22)", "B  Staurosporine removes kinases, NADP removes NADP binders", ["NADP binders", "kinases"])
for nm, R, lo, hi, m, col in (("NADP binders", R24, "NADP|0.6", "NADP|15", CLASS["NADP only"] | CLASS["NAD and NADP"], pl.LINE),
                              ("NAD binders", R24, "NAD|11", "NAD|300", CLASS["NAD only"] | CLASS["NAD and NADP"], pl.HIT),
                              ("kinases", R22, "1", "10", kin, "#e08e0b")):
    yh = R[("ip", hi, "all")]; resp = yh.index[(yh["q"] <= .05) & (yh["logFC"] < 0) & m.reindex(yh.index).fillna(False).values]
    xl = R[("ip", lo, "all")]["logFC"].reindex(resp); ax[2].scatter(xl, yh.loc[resp, "logFC"], s=16, color=col, label=f"{nm}: {lo.split('|')[-1]} vs {hi.split('|')[-1]} (n={len(resp)})")
lim = [-3.8, .5]; ax[2].plot(lim, lim, "k:", lw=.8); ax[2].set_xlim(lim); ax[2].set_ylim(lim); ax[2].axhline(0, color="k", lw=.4); ax[2].axvline(0, color="k", lw=.4)
ax[2].set_xlabel("effect at the low dose (log2)"); ax[2].set_ylabel("effect at the high dose (log2)"); ax[2].legend(fontsize=7)
ax[2].set_title("C  Responders at low vs high dose (cognate class, q <= 0.05 at high dose)", fontsize=9)
f.tight_layout(); f.savefig(f"{FIG}/ligand_specificity.png", dpi=150); plt.close(f)

# F. the second NAD 300 effect: ribonucleoproteins leave the bound fraction at high salt only
f, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a_, kind in zip(ax, ("ip", "sup", "part")):
    L = R24; xx = np.arange(3); wd = .25
    for i, (nm, m, col) in enumerate((("ribosomal proteins", KW.str.contains("Ribosomal protein"), pl.LINE), ("spliceosome", KW.str.contains("Spliceosome"), "#6a3d9a"), ("NAD binders", CLASS["NAD only"], pl.HIT))):
        v = []
        for k in KCL:
            y = L[(kind, "NAD|300", k)]["logFC"]; mm = m.reindex(y.index).fillna(False).values & y.notna().values; v.append(np.median(y[mm]) - np.median(y[~mm & y.notna().values]))
        a_.bar(xx + (i - 1) * wd, v, wd, color=col, label=nm)
    a_.axhline(0, color="k", lw=.4); a_.set_xticks(xx); a_.set_xticklabels([f"KCl {k}" for k in KCL]); a_.set_ylim(-.55, .15)
    a_.set_ylabel("NAD 300 effect, class minus rest (log2)"); a_.set_title(f"{'ABC'[('ip', 'sup', 'part').index(kind)]}  {NICE[kind]}", fontsize=9)
ax[0].legend(fontsize=7)
f.tight_layout(); f.savefig(f"{FIG}/nad300_rnp.png", dpi=150); plt.close(f)

json.dump(J, open(f"{OUT}/fs08.json", "w"), indent=1, default=float)
print("FS08_DONE")
