"""cm29: every small-molecule arm of the FS pull-downs, asked the same question in the same way.

Runs covered
  FS_73  HSPB1, 37 C, carbamazepine / lamotrigine / topiramate at 10 and 100 uM, control, 4-6 BR
  FS_76  HSPB1, 37 C, Li+ 0-10 mM crossed with Mg2+ 0.25 or 2.5 mM, 3-4 BR

Every contrast gets the same battery, which is the accumulated list of ways this project has previously
been fooled:

  A. agreement with the fold changes shipped in the original proteoDA workbook -- an independent
     implementation, so a disagreement means one of us is wrong
  B. spread across proteins, never absolute level: every run is median-normalised
  C. split-half reliability against a permutation null computed INSIDE the pair of conditions being
     contrasted. Relabelling a contrast that pools over another factor shuffles that factor in too and
     inflates the null (GA_24, where pooling over salt pushed nulls to 0.77-0.89)
  D. top-hit replication in replicates held out from the selection
  E. leave-one-BR-out, so no hit rests on one sample
  F. a per-protein hunt that does NOT rely on the profile reliability. A reliability near zero means the
     8,500-protein profile is noise; it does not mean no protein responded. FS_76 had three real
     responders inside a null profile
  G. cross-drug specificity, estimated from DISJOINT replicates, because every drug contrast here shares
     the same control samples and would otherwise correlate by construction
  H. for anything that survives, the bound:free partition, to ask whether the protein is an enriched
     bait partner or background riding along
  I. contaminant flags, and named pharmacological targets checked explicitly

Not available here: FS_73 and FS_76 arrive as proteoDA workbooks with no MS acquisition order, so the
drift attribution test used on GA_24 cannot be run. Said plainly rather than quietly skipped.

Writes data/campaign/cm29_fs_drugs.json and data/campaign/cm29_fs_vectors.npz.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pdpipe import io, stats as PS

C = f"{ROOT}/data/campaign"
OUT = f"{C}/cm29_fs_drugs.json"
J, VEC = {}, {}


def plain(ds):
    ds.samples = ds.samples.copy()
    for c in ds.samples.columns:
        if str(ds.samples[c].dtype) == "str": ds.samples[c] = ds.samples[c].astype(object)
    return ds


fs73 = plain(io.load(io.load_config("fs73")))
fs76 = plain(io.load(io.load_config("fs76")))
CONTAM = set("KRT1 KRT2 KRT5 KRT6A KRT9 KRT10 KRT14 KRT16 KRT17 LYZ DCD FLG CSTA JUP SBSN HRNR".split())


def genes(ds):
    g = {}
    for a, x in zip(ds.proteins["uniprot_id"].astype(str), ds.proteins["gene"]):
        x = str(x).split(";")[0] if isinstance(x, str) and x else ""
        if x: g.setdefault(a, x)
    return g


G73, G76 = genes(fs73), genes(fs76)
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).values
F = np.load(f"{C}/folds.npz", allow_pickle=True)
g_of = dict(zip(U, F["group"].astype(str)))


def fit(ds, w, block=True):
    S = ds.samples[ds.samples["cond"].isin(list(w))].reset_index(drop=True)
    D, names = PS.design(S, block=block)
    f = PS.ebayes(PS.lmfit(ds.X[S["column"]], D))
    c = np.zeros(len(names))
    for k, v in w.items(): c[names.index(k)] = v
    return PS.contrast(f, c)


def battery(ds, name, w, published=None, n_perm=300):
    """A, B, C, D for one contrast."""
    r = fit(ds, w)
    rel = PS.split_half(ds.X, ds.samples[ds.samples["cond"].isin(list(w))].reset_index(drop=True),
                        w, n_perm=n_perm)
    rep = PS.top_hit_replication(ds.X, ds.samples, w, n_top=20, block=True)
    v = r["logFC"].dropna().values
    d = dict(n=int(r["logFC"].notna().sum()), spread=float(r["logFC"].std()),
             spread_robust=float(1.4826 * np.median(np.abs(v - np.median(v)))),
             n_q05=int((r["q"] < .05).sum()), n_q01=int((r["q"] < .01).sum()),
             max_abs=float(r["logFC"].abs().max()),
             reliability=rel["reliability"], null_rel95=rel.get("null_rel95"), p_perm=rel.get("p_perm"),
             top20_train=float(np.mean([x["train_effect"] for x in rep])),
             top20_heldout=float(np.mean([x["heldout_effect"] for x in rep])),
             top20_same_sign=float(np.mean([x["heldout_same_sign"] for x in rep])))
    if published is not None and ds.published is not None and published in ds.published:
        p = ds.published[published].reindex(r.index).astype(float)
        k = np.isfinite(p) & np.isfinite(r["logFC"])
        o = PS.ols(p[k].values, r.loc[k, "logFC"].values)
        d["vs_published"] = dict(r=o["r"], slope=o["slope"], n=o["n"])
    return r, d


# ================================================================ FS_73
print("=== FS_73: anticonvulsants ===", flush=True)
DRUG = {"carbamazepine": "car", "lamotrigene": "lam", "topiramate": "top"}
J["fs73"] = {}; R73 = {}
for drug, tag in DRUG.items():
    for conc in (10, 100):
        key = f"{drug}_{conc}"
        w = {f"{drug}|{conc}": 1.0, "ctl|0": -1.0}
        r, d = battery(fs73, key, w, published=f"{tag}_{conc}_logFC")
        R73[key] = r; J["fs73"][key] = d
        vp = d.get("vs_published", {})
        print(f"  {key:<22} spread {d['spread']:.3f}  rel {d['reliability']:+.3f} "
              f"(null95 {d['null_rel95']:+.3f}, p {d['p_perm']:.3f})  q<.05 {d['n_q05']:>4}  "
              f"top20 {d['top20_train']:+.3f}->{d['top20_heldout']:+.3f}  "
              f"vs published r {vp.get('r', float('nan')):.4f}", flush=True)

# omnibus: does ANY drug cell differ from control?
S = fs73.samples.reset_index(drop=True)
D, names = PS.design(S, block=True)
f73 = PS.ebayes(PS.lmfit(fs73.X[S["column"]], D))
Cm = np.zeros((len(names), 6)); j = 0
for drug in DRUG:
    for conc in (10, 100):
        Cm[names.index(f"{drug}|{conc}"), j] = 1; Cm[names.index("ctl|0"), j] = -1; j += 1
ft = PS.ftest(f73, Cm)
J["fs73_omnibus"] = dict(n=int(ft["F"].notna().sum()), n_q05=int((ft["q"] < .05).sum()),
                         n_q01=int((ft["q"] < .01).sum()), median_F=float(ft["F"].median()))
print(f"  omnibus (any of 6 drug cells vs control): {int((ft['q'] < .05).sum())} proteins at q<.05, "
      f"median F {ft['F'].median():.2f}", flush=True)

# ---- F: per-protein hunt. consistent sign across the two doses, large at 100, and replicating
print("\n  per-protein hunt (sign-consistent, large, replicating):", flush=True)
BR73 = sorted(fs73.samples["BR"].unique())
J["fs73_candidates"] = {}
for drug, tag in DRUG.items():
    lo, hi = R73[f"{drug}_10"], R73[f"{drug}_100"]
    same = np.sign(lo["logFC"]) == np.sign(hi["logFC"])
    cand = hi.index[(hi["logFC"].abs() >= .5) & same & (hi["q"] < .05)]
    w = {f"{drug}|100": 1.0, "ctl|0": -1.0}
    rows = []
    for a in cand:
        halves = []
        for A in PS._splits(sorted(fs73.samples.loc[fs73.samples["cond"].isin(list(w)), "BR"].unique())):
            ea, eb = PS.half_effects(fs73.X, fs73.samples, w, A)
            if ea is not None and eb is not None and np.isfinite(ea[a]) and np.isfinite(eb[a]):
                halves.append((float(ea[a]), float(eb[a])))
        lob = PS.leave_one_br_out(fs73.X, fs73.samples, w, [a]).loc[a]
        g = G73.get(str(a), "")
        rows.append(dict(acc=str(a), gene=g, contaminant=g in CONTAM,
                         logFC_10=float(lo.loc[a, "logFC"]), logFC_100=float(hi.loc[a, "logFC"]),
                         q_100=float(hi.loc[a, "q"]), halves=halves,
                         lobo_min=float(lob.min()), lobo_max=float(lob.max()),
                         lobo_sign_stable=bool((lob > 0).all() or (lob < 0).all())))
    rows.sort(key=lambda d: -abs(d["logFC_100"]))
    J["fs73_candidates"][drug] = rows
    print(f"    {drug:<15} {len(rows)} candidate(s): " +
          (", ".join(f"{r['gene'] or r['acc']} {r['logFC_100']:+.2f}" + ("*" if r["contaminant"] else "")
                     for r in rows[:8]) if rows else "none"), flush=True)

# ---- the hit call, done properly. Fitting each contrast on only its own two conditions throws away
# the residual degrees of freedom the other arms carry, and on this design that is the difference between
# finding nothing and finding the real effects. One joint model over all seven conditions with a replicate
# block is the standard choice and is what is reported; the two-condition fits above are kept as the
# conservative sensitivity check.
print("\n  joint-model hit call (all 7 conditions, replicate block):", flush=True)
Sj = fs73.samples.reset_index(drop=True)
Dj, nj = PS.design(Sj, block=True)
fj = PS.ebayes(PS.lmfit(fs73.X[Sj["column"]], Dj))


def jcon(cond):
    c = np.zeros(len(nj)); c[nj.index(cond)] = 1; c[nj.index("ctl|0")] = -1
    return PS.contrast(fj, c)


RJ = {f"{d}|{c}": jcon(f"{d}|{c}") for d in DRUG for c in (10, 100)}
J["fs73_joint"] = dict(resid_df=float(np.nanmax(fj["df"])), prior_df=float(fj["d0"]), hits={})
for d in DRUG:
    r = RJ[f"{d}|100"]
    others = [f"{o}|100" for o in DRUG if o != d]
    hi = r[(r["q"] < .05) & (r["logFC"].abs() >= .5)]
    rows = []
    for a in hi.index:
        oth = {o: float(RJ[o].loc[a, "logFC"]) for o in others}
        g = G73.get(str(a), "")
        rows.append(dict(acc=str(a), gene=g, contaminant=g in CONTAM,
                         logFC_10=float(RJ[f"{d}|10"].loc[a, "logFC"]),
                         logFC_100=float(r.loc[a, "logFC"]), q_100=float(r.loc[a, "q"]),
                         other_arms=oth, arm_specific=bool(all(abs(v) < .3 for v in oth.values()))))
    rows.sort(key=lambda x: -abs(x["logFC_100"]))
    n_br = int((fs73.samples["cond"] == f"{d}|100").sum())
    J["fs73_joint"]["hits"][d] = dict(n_br=n_br, n_at_bar=len(rows),
                                      n_arm_specific=sum(x["arm_specific"] for x in rows), rows=rows)
    print(f"    {d:<15} ({n_br} BR)  {len(rows)} at q<.05 and |lfc|>=0.5, "
          f"{sum(x['arm_specific'] for x in rows)} of them flat in both other drugs: " +
          ", ".join(f"{x['gene']} {x['logFC_100']:+.2f}" for x in rows if x["arm_specific"]), flush=True)

# ---- how much of the hit call is the moderation? an ordinary paired t-test, no variance shrinkage
def paired_t(drug, conc):
    """Within-replicate differences, ordinary one-sample t, BH. No variance shrinkage anywhere."""
    S = fs73.samples
    cols = []
    for br in sorted(S["BR"].unique()):
        a = S.loc[(S["cond"] == f"{drug}|{conc}") & (S["BR"] == br), "column"]
        b = S.loc[(S["cond"] == "ctl|0") & (S["BR"] == br), "column"]
        if len(a) and len(b):
            cols.append(fs73.X[a.values[0]].values - fs73.X[b.values[0]].values)
    Dm = np.vstack(cols).T                                   # proteins x replicates
    n = np.isfinite(Dm).sum(1)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(Dm, 1); sd = np.nanstd(Dm, 1, ddof=1)
    ok = (n >= 4) & np.isfinite(sd) & (sd > 0)
    t = np.full(len(Dm), np.nan); p = np.full(len(Dm), np.nan)
    t[ok] = mu[ok] / (sd[ok] / np.sqrt(n[ok]))
    p[ok] = 2 * sps.t.sf(np.abs(t[ok]), n[ok] - 1)
    return mu, PS.bh(p), ok


J["fs73_unmoderated"] = {}
idx73 = list(fs73.X.index)
for d in DRUG:
    mu, q, ok = paired_t(d, 100)
    J["fs73_unmoderated"][d] = dict(n=int(ok.sum()), min_q=float(np.nanmin(q[ok])),
                                    n_q05=int((q[ok] < .05).sum()),
                                    hits_of_joint={r["gene"]: float(q[idx73.index(r["acc"])])
                                                   for r in J["fs73_joint"]["hits"][d]["rows"] if r["arm_specific"]})
    print(f"    unmoderated paired t, {d:<15} min BH q {np.nanmin(q[ok]):.3f}, "
          f"{int((q[ok] < .05).sum())} at q<.05", flush=True)

# ---- G: cross-drug specificity from DISJOINT replicates (the controls are shared, so a same-sample
# correlation is manufactured)
print("\n  cross-drug agreement (same samples vs disjoint replicates):", flush=True)
J["fs73_cross_drug"] = {}
pairs = list(itertools.combinations(DRUG, 2))
for d1, d2 in pairs:
    w1 = {f"{d1}|100": 1.0, "ctl|0": -1.0}; w2 = {f"{d2}|100": 1.0, "ctl|0": -1.0}
    a_, b_ = R73[f"{d1}_100"]["logFC"], R73[f"{d2}_100"]["logFC"]
    k = (np.isfinite(a_) & np.isfinite(b_)).values
    same = float(np.corrcoef(a_.values[k], b_.values[k])[0, 1])
    shared = sorted(set(fs73.samples.loc[fs73.samples["cond"].isin(list(w1)), "BR"]) &
                    set(fs73.samples.loc[fs73.samples["cond"].isin(list(w2)), "BR"]))
    dis = []
    for A in PS._splits(shared):
        B = [x for x in shared if x not in A]
        for tr, te in ((list(A), B), (B, list(A))):
            S1 = fs73.samples[fs73.samples["BR"].isin(tr) & fs73.samples["cond"].isin(list(w1))].reset_index(drop=True)
            S2 = fs73.samples[fs73.samples["BR"].isin(te) & fs73.samples["cond"].isin(list(w2))].reset_index(drop=True)
            e1 = PS._effect(fs73.X, S1, w1, np.ones(len(S1), bool))
            e2 = PS._effect(fs73.X, S2, w2, np.ones(len(S2), bool))
            if e1 is None or e2 is None: continue
            kk = (np.isfinite(e1) & np.isfinite(e2)).values
            if kk.sum() > 100: dis.append(np.corrcoef(e1.values[kk], e2.values[kk])[0, 1])
    # a drug's own profile, split across disjoint replicates: the ceiling the cross-drug r is measured
    # against. If a drug does not replicate against ITSELF, a null cross-drug r says nothing.
    selfr = {}
    for dd, ww in ((d1, w1), (d2, w2)):
        v = []
        brs = sorted(fs73.samples.loc[fs73.samples["cond"].isin(list(ww)), "BR"].unique())
        for A in PS._splits(brs):
            ea, eb = PS.half_effects(fs73.X, fs73.samples, ww, A)
            if ea is None or eb is None: continue
            kk = (np.isfinite(ea) & np.isfinite(eb)).values
            if kk.sum() > 100: v.append(np.corrcoef(ea.values[kk], eb.values[kk])[0, 1])
        selfr[dd] = float(np.mean(v)) if v else None
    J["fs73_cross_drug"][f"{d1}|{d2}"] = dict(self_replication=selfr, r_same_samples=same,
                                              r_disjoint=float(np.mean(dis)) if dis else None,
                                              r_disjoint_sd=float(np.std(dis)) if dis else None,
                                              n_splits=len(dis))
    print(f"    {d1[:4]}/{d2[:4]:<6} same samples r {same:+.3f}   disjoint r {np.mean(dis):+.3f} "
          f"+- {np.std(dis):.3f}   (each drug vs ITSELF across disjoint BRs: "
          + ", ".join(f"{k[:4]} {v:+.3f}" for k, v in selfr.items() if v is not None) + ")", flush=True)

# ---- I: named pharmacological targets
TARGETS = {"topiramate -> carbonic anhydrase": ["CA1", "CA2", "CA3", "CA4", "CA12", "CA13"],
           "carbamazepine / lamotrigine -> Na_v channels": ["SCN1A", "SCN2A", "SCN3A", "SCN8A", "SCN9A", "SCN1B", "SCN2B"],
           "topiramate -> GABA-A / AMPA": ["GABRA1", "GABRB2", "GABRG2", "GRIA1", "GRIA2", "GRIK1"]}
J["fs73_named_targets"] = {}
acc73 = {v: k for k, v in G73.items()}
print("\n  named pharmacological targets:", flush=True)
for label, gl in TARGETS.items():
    found = []
    for g in gl:
        a = acc73.get(g)
        if a is None or a not in fs73.X.index: continue
        found.append(dict(gene=g, **{f"{t}_{c}": float(R73[f'{d}_{c}'].loc[a, 'logFC'])
                                     for d, t in DRUG.items() for c in (10, 100)},
                          min_q=float(min(R73[f'{d}_{c}'].loc[a, 'q'] for d in DRUG for c in (10, 100)))))
    J["fs73_named_targets"][label] = found
    print(f"    {label:<45} detected: {len(found)} of {len(gl)}"
          + (f"   min q {min(x['min_q'] for x in found):.2f}" if found else ""), flush=True)

VEC["fs73_acc"] = np.array(R73["topiramate_100"].index, dtype=object)
for k, r in R73.items(): VEC[f"fs73_{k}"] = r["logFC"].values

# ================================================================ FS_76, for one table with FS_73
print("\n=== FS_76: lithium and magnesium (from cm28, recomputed here on the same footing) ===", flush=True)
LI = ["0", "0.3", "1", "3", "5", "10"]
J["fs76"] = {}
W76 = {"Li_10_vs_0_lowMg": {"0.25|10": 1.0, "0.25|0": -1.0},
       "Li_10_vs_0_highMg": {"2.5|10": 1.0, "2.5|0": -1.0},
       "Mg_2.5_vs_0.25": {**{f"2.5|{l}": 1 / 6 for l in LI}, **{f"0.25|{l}": -1 / 6 for l in LI}}}
for k, w in W76.items():
    pub = {"Li_10_vs_0_lowMg": "conc_10_Mg0.25_logFC", "Li_10_vs_0_highMg": "conc_10_Mg2.5_logFC"}.get(k)
    r, d = battery(fs76, k, w, published=pub)
    J["fs76"][k] = d
    VEC[f"fs76_{k}"] = r["logFC"].values
    vp = d.get("vs_published", {})
    print(f"  {k:<22} spread {d['spread']:.3f}  rel {d['reliability']:+.3f} "
          f"(null95 {d['null_rel95']:+.3f}, p {d['p_perm']:.3f})  q<.05 {d['n_q05']:>4}  "
          f"top20 {d['top20_train']:+.3f}->{d['top20_heldout']:+.3f}"
          + (f"  vs published r {vp['r']:.4f}" if vp else ""), flush=True)
VEC["fs76_acc"] = np.array(fs76.X.index, dtype=object)

# ================================================================ H: partition of any survivor
T = pd.read_csv(f"{ROOT}/data/export/targets.tsv", sep="\t")
PC = ["GA22_part_base", "GA24_part_base", "GA33_A1_enrich", "GA33_B11_enrich"]
RK = T[PC].rank(pct=True) * 100
RK.index = T["accession"].astype(str)
J["partition_percentile"] = {}
survivors = sorted({r["gene"] for v in J["fs73_candidates"].values() for r in v if r["gene"]} |
                   {r["gene"] for v in J["fs73_joint"]["hits"].values() for r in v["rows"] if r["gene"]} |
                   {"GET4", "UBL4A", "BAG6", "HSPB1", "PPP4C"})
for g in survivors:
    a = acc73.get(g) or {v: k for k, v in G76.items()}.get(g)
    if a is None or a not in RK.index: continue
    J["partition_percentile"][g] = {c: (float(RK.loc[a, c]) if np.isfinite(RK.loc[a, c]) else None) for c in PC}

np.savez(f"{C}/cm29_fs_vectors.npz", **VEC)
json.dump(J, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}", flush=True)
print("CM29_DONE", flush=True)
