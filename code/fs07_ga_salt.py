"""fs07: GA_22 and GA_24, the salt (KCl) effect on the HSPB1 pull-down, read against each reaction's own
post-IP supernatant.

Both experiments are HSPB1 pull-downs from lysate at 37 C with KCl 0 / 60 / 120 mM crossed with a second
factor (GA_22: staurosporine 0-10 uM; GA_24: NAD / NADP). Salt is balanced across the MS run and across the
second factor, so here the salt main effect is averaged over the second factor (fs08 treats the ligands).

Three measurements per reaction (condition x BR):
  IP          the pull-down (3 MS injections, averaged)
  sup         the post-IP supernatant, i.e. what was left unbound (1 injection)
  partition   IP - sup, the bound : free ratio up to a constant (pdpipe.partition)

Questions
  1. How large and reproducible is the salt effect on each? Does salt change total capture?
  2. Does the pull-down change because the free pool changes (IP follows sup), or because binding changes?
     Estimated on disjoint replicates, since a reaction's IP and sup share its handling noise.
  3. What kind of binding does salt break? Electrostatic binding predicts that the loss tracks protein charge.
  4. Is the bound fraction large enough to deplete the supernatant?
  5. Does the salt effect replicate across GA_22, GA_24 and GA_20 (an earlier HSPB1 salt titration)?

Drift: the IP runs drift with injection order (fs07 QC; ~9-10 % of within-condition variance, sup runs none).
IP and partition models carry a quadratic injection-position term, and split-half reliabilities use the
cross-fitted drift_adjuster. Supernatant blocks that failed the leave-one-out sample check are excluded in
the configs (GA_22 BR5 at KCl 60; GA_24 BR5 at KCl 120).
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from pdpipe import io, stats as st, plots as pl, partition as pt

warnings.filterwarnings("ignore")
FIG = f"{io.ROOT}/reports/figures/GA22_24"; OUT = f"{io.ROOT}/data/fs_data/GA22_24"
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
J = {}
EXP = {"GA22": dict(ip="ga22", sup="ga22s", other="stau", levels=["0", "0.1", "1", "10"]),
       "GA24": dict(ip="ga24", sup="ga24s", other="arm", levels=["CTRL|0", "NAD|11", "NAD|300", "NADP|0.6", "NADP|15"])}


def cond(e, other, k):
    return f"{k}|{other}" if e == "GA22" else f"{other}|{k}"


def salt_w(e, k, levels):
    """KCl k vs 0, averaged over the second factor."""
    return {**{cond(e, o, k): 1 / len(levels) for o in levels}, **{cond(e, o, "0"): -1 / len(levels) for o in levels}}


def drift_tools(ds):
    H = st.drift_basis(ds.samples["inj_pos"], 2); H.index = ds.samples["column"].values
    return H.reset_index(drop=True), st.drift_adjuster(H)


def fit(ds, drift):
    cov = drift_tools(ds)[0] if drift else None
    D, names = st.design(ds.samples, block=True, covars=cov)
    cv = [names.index(c) for c in cov] if cov is not None else None
    return st.ebayes(st.lmfit(ds.X, D, covar=cv)), names


def contrast(f, names, w):
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] += v
    return st.contrast(f, c)


def drift_share(ds, n=50, seed=0):
    """Share of residual variance (after condition and BR effects) explained by a quadratic in injection position,
    projected orthogonal to condition and BR, against positions shuffled within condition."""
    X = ds.X.dropna(); S = ds.samples
    D, _ = st.design(S, block=True); W = X.values - (D @ np.linalg.lstsq(D, X.values.T, rcond=None)[0]).T
    h = S["inj_pos"].values; Hd = np.c_[h - .5, (h - .5) ** 2]; P = np.eye(len(h)) - D @ np.linalg.pinv(D)

    def share(Hr):
        F = (Hr @ np.linalg.lstsq(Hr, W.T, rcond=None)[0]).T; return float(((F ** 2).sum(1) / (W ** 2).sum(1)).mean())
    rng = np.random.default_rng(seed); null = []
    for _ in range(n):
        idx = np.arange(len(h))
        for cc in S["cond"].unique():
            i = np.nonzero(S["cond"].values == cc)[0]; idx[i] = rng.permutation(i)
        null.append(share(P @ Hd[idx]))
    return dict(share=share(P @ Hd), null95=float(np.percentile(null, 95)))


DS, FITS, RES, DRIFT = {}, {}, {}, {}
for e, c in EXP.items():
    ip, sup = io.load(io.load_config(c["ip"])), io.load(io.load_config(c["sup"]))
    part = pt.pair(ip, sup, f"{e}_partition")
    DS[e] = dict(ip=ip, sup=sup, part=part)
    for kind, ds in DS[e].items(): io.save(ds, f"{OUT}/{e}_{kind}")
    J[e] = dict(n=dict(ip=ip.X.shape, sup=sup.X.shape, part=part.X.shape), offsets={}, drift={}, sample_fit={})
    # global capture: the raw per-sample median shift removed by normalisation, by KCl
    for kind in ("ip", "sup"):
        S = DS[e][kind].samples.copy(); S["off"] = S["column"].map(DS[e][kind].meta["offsets"])
        J[e]["offsets"][kind] = S.groupby("KCl")["off"].agg(["mean", "std"]).round(3).to_dict("index")
        J[e]["sample_fit"][kind] = st.sample_fit(DS[e][kind].X, DS[e][kind].samples).round(3).to_dict()
    # drift term wherever injection position explains more within-condition variance than shuffled positions
    for kind in ("ip", "sup"):
        J[e]["drift"][kind] = drift_share(DS[e][kind]); DRIFT[(e, kind)] = J[e]["drift"][kind]["share"] > J[e]["drift"][kind]["null95"]
    DRIFT[(e, "part")] = DRIFT[(e, "ip")] or DRIFT[(e, "sup")]
    for kind in ("ip", "sup", "part"):
        drift = DRIFT[(e, kind)]
        FITS[(e, kind)] = fit(DS[e][kind], drift)
        RES[(e, kind)] = {f"KCl{k}": contrast(*FITS[(e, kind)], salt_w(e, k, c["levels"])) for k in ("60", "120")}
    print(e, {k: v for k, v in J[e]["n"].items()}, "drift", J[e]["drift"], flush=True)



# ---------------------------------------------------------------- 1. size and reproducibility
J["reliability"] = {}
for e, c in EXP.items():
    for kind in ("ip", "sup", "part"):
        ds = DS[e][kind]; adj = drift_tools(ds)[1] if DRIFT[(e, kind)] else None
        for k in ("60", "120"):
            r = st.split_half(ds.X, ds.samples, salt_w(e, k, c["levels"]), n_perm=20, adjust=adj)
            t = RES[(e, kind)][f"KCl{k}"]
            J["reliability"][f"{e} {kind} KCl{k}"] = dict(reliability=r["reliability"], null95=r["null_rel95"], p_perm=r["p_perm"],
                                                          n_up=int(((t["q"] <= .05) & (t["logFC"] > 0)).sum()), n_down=int(((t["q"] <= .05) & (t["logFC"] < 0)).sum()),
                                                          sd=float(t["logFC"].std()), n=int(t["logFC"].notna().sum()))
            print(e, kind, k, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in J["reliability"][f"{e} {kind} KCl{k}"].items()}, flush=True)
        # how much of the 120 mM effect is already there at 60 mM (disjoint halves)
        cc = st.cross_dataset(ds, salt_w(e, "120", c["levels"]), ds, salt_w(e, "60", c["levels"]), adj, adj)
        J["reliability"][f"{e} {kind} 60 vs 120"] = dict(r=cc["r"], r_dis=cc["r_disattenuated"], slope_corrected=cc["slope_corrected"])

# ---------------------------------------------------------------- 2. IP change vs supernatant change (disjoint replicates)
J["ip_vs_sup"] = {}; EXAMPLE = {}
for e, c in EXP.items():
    w = salt_w(e, "120", c["levels"]); adj = drift_tools(DS[e]["ip"])[1]
    adj_s = drift_tools(DS[e]["sup"])[1] if DRIFT[(e, "sup")] else None
    cc = st.cross_dataset(DS[e]["sup"], w, DS[e]["ip"], w, adj_s, adj)
    EXAMPLE[e] = cc.pop("example"); J["ip_vs_sup"][e] = cc
    print(e, "IP vs sup", {k: round(v, 3) for k, v in cc.items() if isinstance(v, float)}, flush=True)

# ---------------------------------------------------------------- 3. what binding salt breaks: charge and protein classes
U = io.uniprot_annotation()
CH = pd.DataFrame([pt.charge(s) for s in U["sequence"]], index=U.index, columns=["net_charge", "pI"])
CH["charge_density"] = CH["net_charge"] / U["length"]
KW = U["Keywords"].fillna(""); GOC = U["Gene Ontology (cellular component)"].fillna(""); GOF = U["Gene Ontology (molecular function)"].fillna("")
CLS = {"cytosolic ribosome": GOC.str.contains("cytosolic ribosome"), "RNA-binding (not ribosome)": KW.str.contains("RNA-binding") & ~GOC.str.contains("cytosolic ribosome"),
       "DNA-binding": KW.str.contains("DNA-binding"), "membrane (transmembrane)": KW.str.contains("Transmembrane"),
       "mitochondrion": KW.str.contains("Mitochondrion"), "kinase": KW.str.contains("Kinase")}
J["charge"], J["classes"] = {}, {}
for e in EXP:
    for kind in ("ip", "sup", "part"):
        y = RES[(e, kind)]["KCl120"]["logFC"]; ch = CH.reindex(y.index)
        ok = y.notna() & ch["charge_density"].notna()
        J["charge"][f"{e} {kind}"] = dict(rho_density=float(spearmanr(ch.loc[ok, "charge_density"], y[ok]).correlation),
                                          rho_pI=float(spearmanr(ch.loc[ok, "pI"], y[ok]).correlation), n=int(ok.sum()),
                                          r2_density=float(st.ols(ch.loc[ok, "charge_density"].values, y[ok].values)["r2"]))
        for k, m in CLS.items():
            mm = m.reindex(y.index).fillna(False).values & y.notna().values; rest = ~m.reindex(y.index).fillna(False).values & y.notna().values
            J["classes"][f"{e} {kind} {k}"] = dict(n=int(mm.sum()), median=float(np.median(y[mm])), median_rest=float(np.median(y[rest])),
                                                   p=float(mannwhitneyu(y[mm], y[rest]).pvalue) if mm.sum() > 3 else None)
print("charge", J["charge"], flush=True)

# ---------------------------------------------------------------- 4. depletion: does losing binding raise the free pool?
# Among proteins already strongly bound at KCl 0 (high baseline partition), a salt-driven loss of binding should
# raise their supernatant level if a noticeable fraction of the protein was on the beads. Slope of the sup change
# on the IP change, per decile of baseline partition, with IP and sup from disjoint replicates.
J["depletion"] = {}
for e, c in EXP.items():
    pa = DS[e]["part"]; base = pa.X[pa.samples.loc[pa.samples["KCl"] == "0", "column"]].mean(1)
    w = salt_w(e, "120", c["levels"]); adj = drift_tools(DS[e]["ip"])[1]
    brs = sorted(set(DS[e]["ip"].samples["BR"]) & set(DS[e]["sup"].samples["BR"])); rows = []
    for A in st._splits(brs):
        ia, ib = st.half_effects(DS[e]["ip"].X, DS[e]["ip"].samples, w, A, adj)
        sa, sb = st.half_effects(DS[e]["sup"].X, DS[e]["sup"].samples, w, A, drift_tools(DS[e]["sup"])[1] if DRIFT[(e, "sup")] else None)
        for x, y in ((ia, sb), (ib, sa)):
            d = pd.DataFrame({"ip": x, "sup": y, "base": base}).dropna(); d["dec"] = pd.qcut(d["base"], 10, labels=False)
            rows.append([st.ols(g["ip"].values, g["sup"].values)["slope"] for _, g in d.groupby("dec")])
    rows = np.array(rows); J["depletion"][e] = dict(slope_by_decile=rows.mean(0).tolist(), sd=rows.std(0).tolist())
print("depletion", J["depletion"], flush=True)

# ---------------------------------------------------------------- 5. replication across experiments
J["replication"] = {}
g20 = pd.read_csv(f"{io.ROOT}/data/pd_data/targets.tsv", sep="\t", index_col=0)
for kind in ("ip", "sup", "part"):
    a, b = RES[("GA22", kind)]["KCl120"]["logFC"], RES[("GA24", kind)]["KCl120"]["logFC"]
    j = a.index.intersection(b.index); ok = a[j].notna() & b[j].notna()
    o = st.ols(a[j][ok].values, b[j][ok].values)
    rel = np.sqrt(J["reliability"][f"GA22 {kind} KCl120"]["reliability"] * J["reliability"][f"GA24 {kind} KCl120"]["reliability"])
    J["replication"][f"GA22 vs GA24 {kind}"] = dict(r=o["r"], slope=o["slope"], n=o["n"], ceiling=float(rel), r_over_ceiling=float(o["r"] / rel))
for e in EXP:
    for k20, k in (("y_s150", "120"), ("y_s75", "60")):
        a = RES[(e, "ip")][f"KCl{k}"]["logFC"]; b = g20[k20].where(g20[f"keep_{k20[2:]}"] == 1)
        j = a.index.intersection(b.index); ok = a[j].notna() & b[j].notna(); o = st.ols(b[j][ok].values, a[j][ok].values)
        J["replication"][f"GA20 {k20[2:]} vs {e} IP KCl{k}"] = dict(r=o["r"], slope=o["slope"], n=o["n"])
print("replication", J["replication"], flush=True)

# tables
for (e, kind), r in RES.items():
    pd.concat([v.add_prefix(f"{k}:") for k, v in r.items()], axis=1).assign(gene=DS[e][kind].proteins["gene"]).to_csv(f"{OUT}/{e}_{kind}_salt.tsv", sep="\t")

# ---------------------------------------------------------------- figures
NICE = {"ip": "pull-down", "sup": "supernatant", "part": "partition (IP - sup)"}
# A. QC: raw capture shift with salt, drift, sample fit
f, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for i, e in enumerate(EXP):
    for j, kind in enumerate(("ip", "sup")):
        o = J[e]["offsets"][kind]; ks = ["0", "60", "120"]
        ax[0].errorbar(np.arange(3) + (i * 2 + j - 1.5) * .06, [o[k]["mean"] for k in ks], [o[k]["std"] for k in ks], fmt="o-" if kind == "ip" else "s--",
                       color=[pl.HIT, pl.LINE][i], alpha=1 if kind == "ip" else .6, capsize=2, label=f"{e} {NICE[kind]}")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["0", "60", "120"]); ax[0].set_xlabel("KCl (mM)"); ax[0].set_ylabel("raw median log2 shift of the sample")
ax[0].axhline(0, color="k", lw=.4); ax[0].legend(fontsize=7); ax[0].set_title("A  Global level before normalisation")
lab, v, nl = [], [], []
for e in EXP:
    for kind in ("ip", "sup"): lab.append(f"{e}\n{NICE[kind]}"); v.append(J[e]["drift"][kind]["share"]); nl.append(J[e]["drift"][kind]["null95"])
ax[1].bar(range(4), v, color=[pl.HIT if a > b else pl.BASE for a, b in zip(v, nl)]); ax[1].scatter(range(4), nl, marker="_", s=300, color="k", label="95% of shuffled positions")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_ylabel("within-condition variance\nexplained by injection position")
ax[1].legend(fontsize=7); ax[1].set_title("B  MS drift")
for i, e in enumerate(EXP):
    for j, kind in enumerate(("ip", "sup")):
        sf = pd.Series(J[e]["sample_fit"][kind]); x = i * 2 + j
        ax[2].scatter(np.full(len(sf), x) + np.random.default_rng(0).uniform(-.2, .2, len(sf)), sf.values, s=8, color=pl.BASE)
ax[2].axhline(0, color="k", lw=.4); ax[2].set_xticks(range(4)); ax[2].set_xticklabels(lab, fontsize=7)
ax[2].set_ylabel("leave-one-out sample fit"); ax[2].set_title("C  Sample quality (after dropping the two failed supernatant blocks)")
f.tight_layout(); f.savefig(f"{FIG}/qc.png", dpi=150); plt.close(f)

# B. reliability bars
f, ax = plt.subplots(figsize=(9, 3.6)); keys = [f"{e} {kind} KCl{k}" for e in EXP for kind in ("ip", "sup", "part") for k in ("60", "120")]
v = [J["reliability"][k]["reliability"] for k in keys]; nl = [J["reliability"][k]["null95"] for k in keys]
ax.bar(range(len(keys)), v, color=[pl.HIT if a > b else pl.BASE for a, b in zip(v, nl)]); ax.scatter(range(len(keys)), nl, marker="_", s=200, color="k")
ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace(" ip ", " pull-down ").replace(" sup ", " supernatant ").replace(" part ", " partition ") for k in keys], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("split-half reliability"); ax.set_title("Salt effect: reproducibility (black ticks: 95% of label permutations)")
f.tight_layout(); f.savefig(f"{FIG}/salt_reliability.png", dpi=150); plt.close(f)

# C. IP vs sup
f, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, e in zip(ax, EXP):
    x, y = EXAMPLE[e]; j = x.index.intersection(y.index)
    pl.scatter_reg(a_, x[j].values, y[j].values, "supernatant change, KCl 120 vs 0 (one half of BRs)", "pull-down change, KCl 120 vs 0 (other half)",
                   f"{'AB'[e == 'GA24']}  {e}: does the pull-down follow the free pool?", identity=True)
    cc = J["ip_vs_sup"][e]
    a_.text(.97, .03, f"all splits: r = {cc['r']:.2f}, ceiling-corrected r = {cc['r_disattenuated']:.2f}\nnoise-corrected slope = {cc['slope_corrected']:.2f}",
            transform=a_.transAxes, ha="right", fontsize=7, bbox=dict(fc="white", ec="none", alpha=.8))
f.tight_layout(); f.savefig(f"{FIG}/salt_ip_vs_sup.png", dpi=150); plt.close(f)

# D. charge and classes
f, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a_, kind in zip(ax[:2], ("ip", "part")):
    for i, e in enumerate(EXP):
        y = RES[(e, kind)]["KCl120"]["logFC"]; ch = CH["charge_density"].reindex(y.index)
        pl.binned(a_, ch.values * 100, y.values, nb=12, color=[pl.HIT, pl.LINE][i], label=f"{e} (rho {J['charge'][f'{e} {kind}']['rho_density']:+.2f})")
    a_.axhline(0, color="k", lw=.4); a_.axvline(0, color="k", lw=.4); a_.legend(fontsize=7)
    a_.set_xlabel("net charge at pH 7.4 per 100 residues"); a_.set_ylabel(f"{NICE[kind]} change, KCl 120 vs 0 (log2)")
    a_.set_title(f"{'AB'[kind == 'part']}  Salt effect by protein charge: {NICE[kind]}")
cl = list(CLS); xx = np.arange(len(cl)); wd = .13
for i, (e, kind) in enumerate([(e, k) for e in EXP for k in ("ip", "sup", "part")]):
    v = [J["classes"][f"{e} {kind} {k}"]["median"] - J["classes"][f"{e} {kind} {k}"]["median_rest"] for k in cl]
    ax[2].bar(xx + (i - 2.5) * wd, v, wd, color=[pl.HIT, "#e08e0b", "#6a3d9a", pl.LINE, "#5fa2dd", "#b0b0b0"][i], label=f"{e} {NICE[kind]}")
ax[2].axhline(0, color="k", lw=.4); ax[2].set_xticks(xx); ax[2].set_xticklabels(cl, rotation=30, ha="right", fontsize=7)
ax[2].set_ylabel("median salt effect, class minus rest (log2)"); ax[2].legend(fontsize=6); ax[2].set_title("C  Salt effect by protein class")
f.tight_layout(); f.savefig(f"{FIG}/salt_charge_classes.png", dpi=150); plt.close(f)

# E. replication and depletion
f, ax = plt.subplots(1, 4, figsize=(18, 4.2))
for a_, kind in zip(ax[:3], ("ip", "sup", "part")):
    a, b = RES[("GA22", kind)]["KCl120"]["logFC"], RES[("GA24", kind)]["KCl120"]["logFC"]; j = a.index.intersection(b.index)
    pl.scatter_reg(a_, a[j].values, b[j].values, "GA_22, KCl 120 vs 0 (log2)", "GA_24, KCl 120 vs 0 (log2)", f"{'ABC'[('ip', 'sup', 'part').index(kind)]}  Replication: {NICE[kind]}", identity=True)
    rr = J["replication"][f"GA22 vs GA24 {kind}"]; a_.text(.97, .03, f"r / ceiling = {rr['r_over_ceiling']:.2f}", transform=a_.transAxes, ha="right", fontsize=7)
for i, e in enumerate(EXP):
    d = J["depletion"][e]; ax[3].errorbar(np.arange(10) + 1 + i * .1, d["slope_by_decile"], d["sd"], fmt="o-", color=[pl.HIT, pl.LINE][i], capsize=2, label=e)
ax[3].axhline(0, color="k", lw=.4); ax[3].set_xlabel("decile of baseline partition (bound : free at KCl 0)"); ax[3].set_ylabel("slope of supernatant change\non pull-down change")
ax[3].legend(fontsize=7); ax[3].set_title("D  Does losing binding refill the supernatant?")
f.tight_layout(); f.savefig(f"{FIG}/salt_replication_depletion.png", dpi=150); plt.close(f)

J["drift_used"] = {f"{e} {k}": bool(v) for (e, k), v in DRIFT.items()}
json.dump(J, open(f"{OUT}/fs07.json", "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
print("FS07_DONE")
