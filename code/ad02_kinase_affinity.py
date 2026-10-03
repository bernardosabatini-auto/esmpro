"""ad02: does staurosporine deplete kinases from the co-chaperone pull-downs in proportion to how
tightly it binds them?

Staurosporine Kd values from ChEMBL (CHEMBL388978): Davis et al. 2011 (CHEMBL1908390), Karaman et al.
2008 (CHEMBL1150977), Fabian et al. 2005 (CHEMBL1144455), all competition binding assays against
kinase panels. Records are mapped to UniProt through ChEMBL target components; records stated as
">" (no binding at the top concentration) are kept as censored values at their stated bound.
Per kinase: median log10 Kd over the wild-type-construct records of all three studies; agreement
between the studies is checked on the kinases they share.

Prediction of the release hypothesis: tighter binding (lower Kd) -> stronger depletion, so log10 Kd
and the staurosporine response correlate positively; and kinases staurosporine does not bind should
not be depleted.
"""
import os, csv, json, re, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, K = f"{ROOT}/data/ga_data", f"{ROOT}/data/external/kinase"
rng = np.random.default_rng(0)
acts = json.load(open(f"{K}/staurosporine_kd_chembl.json"))
tmap = json.load(open(f"{K}/chembl_target_map.json"))
STUDY = {"CHEMBL1908390": "Davis 2011", "CHEMBL1150977": "Karaman 2008", "CHEMBL1144455": "Fabian 2005"}
MUT = re.compile(r"\([A-Z]\d+[A-Z]|\bmutant|\bdel\b|-?(?:JH2|JH1)|domain 2|phosphorylated|nonphosphorylated|autoinhibited|\bITD\b", re.I)
per = collections.defaultdict(lambda: collections.defaultdict(list))
n_mut = 0
for a in acts:
    if a["document_chembl_id"] not in STUDY or a["standard_value"] is None or a["standard_units"] != "nM":
        continue
    t = tmap.get(a["target_chembl_id"], {})
    if t.get("type") != "SINGLE PROTEIN" or len(t.get("accessions", [])) != 1 or t.get("organism") != "Homo sapiens":
        continue
    if MUT.search(a.get("assay_description") or ""):
        n_mut += 1; continue
    v = float(a["standard_value"])
    per[STUDY[a["document_chembl_id"]]][t["accessions"][0]].append(np.log10(max(v, 1e-3)))
print(f"records kept per study: " + ", ".join(f"{s} {sum(len(v) for v in d.values())} ({len(d)} kinases)" for s, d in per.items()) + f"; {n_mut} mutant/variant-construct records excluded", flush=True)
lk = {s: {u: float(np.median(v)) for u, v in d.items()} for s, d in per.items()}
for s1, s2 in (("Davis 2011", "Karaman 2008"), ("Davis 2011", "Fabian 2005"), ("Karaman 2008", "Fabian 2005")):
    common = sorted(set(lk[s1]) & set(lk[s2]))
    if len(common) > 5:
        r = stats.spearmanr([lk[s1][u] for u in common], [lk[s2][u] for u in common])[0]
        print(f"  agreement {s1} vs {s2}: Spearman {r:.2f} on {len(common)} kinases", flush=True)
allk = collections.defaultdict(list)
for s, d in lk.items():
    for u, v in d.items(): allk[u].append(v)
LKD = {u: float(np.median(v)) for u, v in allk.items()}
print(f"consensus log10 Kd for {len(LKD)} kinases; median Kd {10**np.median(list(LKD.values())):.0f} nM; "
      f"non-binders (Kd >= 10 uM): {sum(v >= 4 for v in LKD.values())}", flush=True)

blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]]); gene = np.array([str(g) for g in blk["gene"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
fam = {r["Entry"]: r["Protein families"] for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}


def kgroup(a):
    f = fam.get(a, "")
    for g in ("AGC", "CAMK", "CMGC", "STE", "TKL", "CK1", "Tyr protein kinase", "NEK"):
        if g in f: return g.split()[0]
    return "other"


out = {"n_kinases_kd": len(LKD), "study_agreement": {}}
print(f"\n{'target':<15}{'kinases with Kd':>16}{'Spearman(logKd, y)':>20}{'p':>9}{'binders (<1uM) mean y':>23}{'non-binders (>=10uM)':>22}{'all non-kinases':>17}", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43", "stau_21A_37", "stau_24_37"):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]
    kin = np.array([a in KIN for a in a_]); kd = np.array([LKD.get(a, np.nan) for a in a_])
    m = kin & np.isfinite(kd)
    rho, p = stats.spearmanr(kd[m], y[m])
    b = m & (kd < 3); nb = m & (kd >= 4)
    rec = dict(n=int(m.sum()), rho=float(rho), p=float(p), mean_binders=float(y[b].mean()), n_binders=int(b.sum()),
               mean_nonbinders=float(y[nb].mean()) if nb.any() else None, n_nonbinders=int(nb.sum()),
               p_nonbinders_vs_nonkinase=float(stats.mannwhitneyu(y[nb], y[~kin]).pvalue) if nb.sum() > 3 else None,
               mean_nonkinase=float(y[~kin].mean()))
    out[t] = rec
    print(f"{t:<15}{m.sum():>16}{rho:>+20.3f}{p:>9.0e}{y[b].mean():>+16.3f} (n {b.sum():>3}){(y[nb].mean() if nb.any() else float('nan')):>+15.3f} (n {nb.sum():>2}){y[~kin].mean():>+17.3f}", flush=True)
# Kd bins and kinase groups for the clearest target
t = "stau_avg_37"
rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]; g_ = gene[rows]
kin = np.array([a in KIN for a in a_]); kd = np.array([LKD.get(a, np.nan) for a in a_]); m = kin & np.isfinite(kd)
print(f"\n{t}: response by staurosporine Kd", flush=True)
bins = [(-9, 0, "Kd < 1 nM"), (0, 1, "1-10 nM"), (1, 2, "10-100 nM"), (2, 3, "100 nM-1 uM"), (3, 4, "1-10 uM"), (4, 9, ">= 10 uM (non-binder)")]
binrows = []
for lo, hi, nm in bins:
    s = m & (kd >= lo) & (kd < hi)
    if s.sum():
        binrows.append(dict(bin=nm, n=int(s.sum()), mean=float(y[s].mean()), sem=float(y[s].std() / np.sqrt(max(s.sum(), 1)))))
        print(f"  {nm:<24} n {s.sum():>3}  mean {y[s].mean():+.3f} +/- {y[s].std()/np.sqrt(max(s.sum(),1)):.3f}", flush=True)
out["kd_bins_stau37"] = binrows
grp = np.array([kgroup(a) for a in a_])
print(f"\n  by kinase group (stau_avg_37): mean response and median Kd", flush=True)
grows = []
for gname in sorted(set(grp[m]), key=lambda z: y[m & (grp == z)].mean()):
    s = m & (grp == gname)
    grows.append(dict(group=gname, n=int(s.sum()), mean=float(y[s].mean()), median_kd_nM=float(10 ** np.median(kd[s]))))
    print(f"    {gname:<6} n {s.sum():>3}  mean {y[s].mean():+.3f}  median Kd {10**np.median(kd[s]):8.1f} nM  "
          f"rho within group {stats.spearmanr(kd[s], y[s])[0] if s.sum() > 5 else float('nan'):+.2f}", flush=True)
out["groups_stau37"] = grows
agc = m & (grp == "AGC")
print(f"\n  AGC kinases (stau_avg_37), Kd and response: " + ", ".join(f"{g_[i]} {10**kd[i]:.1f}nM {y[i]:+.2f}" for i in np.argsort(-y * agc)[:agc.sum()] if agc[i]), flush=True)
print(f"  most depleted kinases with Kd: " + ", ".join(f"{g_[i]} {10**kd[i]:.0f}nM {y[i]:+.2f}" for i in np.argsort(y + 99 * ~m)[:12]), flush=True)
# partial: does Kd add to kinase group?
from numpy.linalg import lstsq
G = np.array([[1.0 if grp[i] == gg else 0.0 for gg in sorted(set(grp[m]))] for i in np.nonzero(m)[0]])
yy, kk = y[m], kd[m]
r2 = lambda yv, pv: 1 - np.sum((yv - pv) ** 2) / np.sum((yv - yv.mean()) ** 2)
p_g = G @ lstsq(G, yy, rcond=None)[0]
X = np.column_stack([G, kk]); p_gk = X @ lstsq(X, yy, rcond=None)[0]
X2 = np.column_stack([np.ones(len(kk)), kk]); p_k = X2 @ lstsq(X2, yy, rcond=None)[0]
print(f"\n  in-sample R2 among {m.sum()} kinases: Kd {100*r2(yy,p_k):.1f}%, kinase group {100*r2(yy,p_g):.1f}%, group + Kd {100*r2(yy,p_gk):.1f}%", flush=True)
out["r2_insample"] = dict(kd=r2(yy, p_k), group=r2(yy, p_g), group_kd=r2(yy, p_gk))
json.dump(out, open(f"{K}/ad02_results.json", "w"), indent=1)
print("\nAD02_DONE")

# ---- checks requested by the critical review (appended)
print("\nCHECKS", flush=True)
chk = {}
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]
    kin = np.array([a in KIN for a in a_]); kd = np.array([LKD.get(a, np.nan) for a in a_]); m = kin & np.isfinite(kd)
    b, nb = m & (kd < 3), m & (kd >= 4)
    p_bnb = stats.mannwhitneyu(y[b], y[nb]).pvalue
    p_nbk = stats.mannwhitneyu(y[nb], y[~kin]).pvalue
    # occupancy at 10 uM staurosporine with apparent Kd shifted by s (cellular ATP competition)
    occ = {}
    for s_ in (1, 10, 100, 1000):
        o = 1e4 / (1e4 + (10 ** kd[m]) * s_)
        occ[s_] = float(stats.spearmanr(o, y[m])[0])
    # partial rank correlation of Kd with y, within kinase groups
    grp = np.array([kgroup(a) for a in a_])[m]
    ry, rk = stats.rankdata(y[m]), stats.rankdata(kd[m])
    for gname in set(grp):
        s = grp == gname
        ry[s] -= ry[s].mean(); rk[s] -= rk[s].mean()
    pr = stats.pearsonr(ry, rk)
    # bootstrap over kinases for the partial correlation
    idx = np.arange(m.sum()); bs = []
    for _ in range(2000):
        s = rng.choice(idx, len(idx), replace=True); bs.append(np.corrcoef(ry[s], rk[s])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ab = np.nanmean(np.hstack([blk[f"br_{c}_0_{t.split('_')[-1]}"][rows] for c in ("21A", "24")]), 1)
    ok = m & np.isfinite(ab)
    chk[t] = dict(p_binders_vs_nonbinders=float(p_bnb), p_nonbinders_vs_nonkinases=float(p_nbk), occupancy_rho=occ,
                  partial_rho_within_group=[float(pr[0]), float(lo), float(hi)], p_partial=float(pr[1]),
                  rho_abundance=float(stats.spearmanr(ab[ok], y[ok])[0]),
                  rho_kd_abundance=float(stats.spearmanr(ab[ok], kd[ok])[0]))
    print(f"  {t}: binders vs non-binders p {p_bnb:.0e}; non-binders vs non-kinases p {p_nbk:.2f}; "
          f"occupancy rho at ATP shift 1/10/100/1000x: " + "/".join(f"{occ[s_]:+.3f}" for s_ in (1, 10, 100, 1000))
          + f"; Kd within kinase groups: partial rho {pr[0]:+.3f} [{lo:+.3f},{hi:+.3f}]; "
          f"rho(y, abundance) {chk[t]['rho_abundance']:+.2f}, rho(Kd, abundance) {chk[t]['rho_kd_abundance']:+.2f}", flush=True)
out["checks"] = chk
json.dump(out, open(f"{K}/ad02_results.json", "w"), indent=1)

# ---- second round of checks (appended): occupancy shape, and abundance with disjoint replicates
print("\nCHECKS 2", flush=True)
chk2 = {}
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T = t.split("_")[-1]
    rows = FG[f"rows_{t}"]; a_ = acc[rows]
    kin = np.array([a in KIN for a in a_]); kd = np.array([LKD.get(a, np.nan) for a in a_]); m = kin & np.isfinite(kd)
    y = FG[f"y_{t}"].astype(float)
    pear = {"log Kd": float(stats.pearsonr(kd[m], y[m])[0])}
    for s_ in (1, 10, 100, 1000):
        pear[f"occupancy, shift {s_}x"] = float(stats.pearsonr(1e4 / (1e4 + (10 ** kd[m]) * s_), y[m])[0])
    # abundance from replicates 1-3 of the controls; response from replicates 4-6
    with np.errstate(invalid="ignore"):
        ab = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][rows][:, :3] for c in ("21A", "24")]), 1)
        yd = np.nanmean(np.hstack([blk[f"br_{c}_10_{T}"][rows][:, 3:] for c in ("21A", "24")]), 1) - \
             np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][rows][:, 3:] for c in ("21A", "24")]), 1)
        ab_same = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][rows][:, 3:] for c in ("21A", "24")]), 1)
    ok = np.isfinite(ab) & np.isfinite(yd) & np.isfinite(ab_same)
    r_same = stats.spearmanr(ab_same[ok], yd[ok])[0]; r_dis = stats.spearmanr(ab[ok], yd[ok])[0]
    A = np.column_stack([np.ones(ok.sum()), ab[ok], ab[ok] ** 2])
    res = yd[ok] - A @ np.linalg.lstsq(A, yd[ok], rcond=None)[0]
    kk = kin[ok]
    shift_raw = (yd[ok][kk].mean() - yd[ok][~kk].mean()) / yd[ok].std()
    shift_adj = (res[kk].mean() - res[~kk].mean()) / yd[ok].std()
    chk2[t] = dict(pearson=pear, rho_abund_same=float(r_same), rho_abund_disjoint=float(r_dis),
                   kinase_shift_raw_sd=float(shift_raw), kinase_shift_abundance_adjusted_sd=float(shift_adj),
                   p_adj=float(stats.mannwhitneyu(res[kk], res[~kk]).pvalue))
    print(f"  {t}: Pearson with y: " + ", ".join(f"{k} {v:+.3f}" for k, v in pear.items()), flush=True)
    print(f"      abundance-response rho same replicates {r_same:+.2f}, disjoint {r_dis:+.2f}; kinase shift {shift_raw:+.2f} sd raw, "
          f"{shift_adj:+.2f} sd after abundance (p {chk2[t]['p_adj']:.0e})", flush=True)
out["checks2"] = chk2
json.dump(out, open(f"{K}/ad02_results.json", "w"), indent=1)
