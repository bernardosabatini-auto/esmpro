"""ad09: GA_33 pull-down enrichment over the input lysate, and its sanity checks.

enrichment = log2(pull-down) - log2(input), per protein and condition. Pull-down levels are the
biological-replicate means of the median-normalised runs (ga01, blocks.npz); the input is the ad08
reference (10 clean input samples). The two are separate runs on separate scales, so each enrichment
profile is centred on its median and only differences between proteins carry meaning.

Checks:
  1. How pull-down level scales with input (slope 1 = pull-down proportional to lysate).
  2. Each bait's rank in its own pull-down, absolute and as enrichment.
  3. Abundant housekeeping classes (ribosome, glycolysis, tubulin, histones, ...) relative to the bulk.
  4. Enrichment by cellular compartment (UniProt), all proteins.
  5. Matched input (each reaction's own input, where clean) versus the reference.
  6. Is there a separate enriched (client) mode? One- vs two-component Gaussian mixture, BIC.
  7. Proteins detected in the pull-down but not in the input.
Saves data/ga_data/enrichment.npz.
"""
import os, csv, json, re, itertools
import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
R = np.load(f"{GA}/input_reference.npz", allow_pickle=True)
inp, con = R["level"], R["contaminant"].astype(bool)
gene = blk["gene"].astype(str)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
CH, ST, TE = ("21A", "24"), ("0", "10"), ("35", "37", "43")
NAME = {"21A": "DNAJA1", "24": "DNAJB11"}
mean = lambda A: np.nanmean(A, 1)
res = {}

PD = {k: mean(blk[f"br_{k[0]}_{k[1]}_{k[2]}"]) for k in itertools.product(CH, ST, TE)}
E = {}
for k, p in PD.items():
    e = p - inp
    ok = np.isfinite(e) & ~con
    E[k] = np.where(ok, e - np.median(e[ok]), np.nan)

# 1. scaling of pull-down with input
print("1. pull-down level against input level (control, 35 C; contaminants excluded)", flush=True)
for c in CH:
    p = PD[(c, "0", "35")]; ok = np.isfinite(p) & np.isfinite(inp) & ~con
    sl, ic = np.polyfit(inp[ok], p[ok], 1)
    r = np.corrcoef(inp[ok], p[ok])[0, 1]
    q = np.percentile(inp[ok], [10, 90])
    ce = np.corrcoef(inp[ok], E[(c, "0", "35")][ok])[0, 1]
    print(f"  {NAME[c]}: n {ok.sum()}, r {r:.3f}, slope {sl:.3f}; input spans {q[1]-q[0]:.1f} log2 (10-90th pct), "
          f"pull-down {np.diff(np.percentile(p[ok], [10, 90]))[0]:.1f}; corr(enrichment, input) {ce:+.3f}", flush=True)
    res[f"scaling_{c}"] = dict(n=int(ok.sum()), r=float(r), slope=float(sl), corr_enrich_input=float(ce))

# 2. baits
print("\n2. bait ranks in the control pull-downs (rank 1 = highest)", flush=True)
for c in CH:
    for T in TE:
        p = PD[(c, "0", T)]; e = E[(c, "0", T)]
        line = []
        for bt in ("DNAJA1", "DNAJB11"):
            i = int(np.where(gene == bt)[0][0])
            ra = int((np.nan_to_num(p, nan=-99) > p[i]).sum()) + 1
            re_ = int((np.nan_to_num(e, nan=-99) > e[i]).sum()) + 1
            line.append(f"{bt} level rank {ra} of {np.isfinite(p).sum()}, enrichment {e[i]:+.2f} rank {re_}")
        print(f"  {NAME[c]} pull-down {T} C: " + "; ".join(line), flush=True)
for c, bt in (("21A", "DNAJA1"), ("24", "DNAJB11")):
    i = int(np.where(gene == bt)[0][0]); o = "24" if c == "21A" else "21A"
    d_pd = np.mean([PD[(c, s, T)][i] - PD[(o, s, T)][i] for s in ST for T in TE])
    print(f"  {bt}: own minus other pull-down {d_pd:+.2f} log2 (normalised levels, mean over 6 conditions)", flush=True)
    res[f"bait_{bt}_own_minus_other"] = float(d_pd)

# 3. housekeeping classes
CLS = {"cytosolic ribosome": r"^RP[LS]\d", "glycolysis": r"^(GAPDH|ENO1|PKM|ALDOA|TPI1|PGK1|LDHA|LDHB|PGAM1|GPI|HK1|PFKP)$",
       "tubulins": r"^TUB[AB]", "actin": r"^ACT[BG]1?$", "histones": r"^H[1-4]|^H2[AB]", "proteasome": r"^PSM[ABCD]\d",
       "HSP70/HSP90": r"^(HSPA[1-9]|HSPA1[0-9]|HSP90A[AB]1|HSP90B1)", "mito ATP synthase": r"^ATP5",
       "translation factors": r"^EEF|^EIF[2-5]"}
print("\n3. housekeeping classes: median enrichment (control, 35 C) and median input level; bulk median enrichment = 0", flush=True)
mid = np.nanmedian(inp[~con])
res["classes"] = {}
for nm, pat in CLS.items():
    m = np.array([re.match(pat, g) is not None for g in gene])
    r = {NAME[c]: float(np.nanmedian(E[(c, "0", "35")][m])) for c in CH}
    print(f"  {nm:<22} n {m.sum():>4}  input {np.nanmedian(inp[m]) - mid:+.2f} vs bulk   enrichment "
          + "  ".join(f"{k} {v:+.2f}" for k, v in r.items()), flush=True)
    res["classes"][nm] = dict(n=int(m.sum()), input=float(np.nanmedian(inp[m]) - mid), **r)

# 4. compartments
loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in acc]
kw = [ann.get(a, {}).get("Keywords", "") for a in acc]
COMP = {"transmembrane": np.array(["Transmembrane" in k for k in kw]),
        "mitochondrion": np.array(["Mitochondrion" in l for l in loc]),
        "ER": np.array(["Endoplasmic reticulum" in l for l in loc]),
        "nucleus only": np.array([("Nucleus" in l) and ("Cytoplasm" not in l) for l in loc]),
        "cytoplasm, not membrane": np.array([("Cytoplasm" in l) and ("Nucleus" not in l) and ("Transmembrane" not in k) for l, k in zip(loc, kw)]),
        "secreted": np.array(["Secreted" in l for l in loc])}
print("\n4. compartments: median enrichment (control, 35 C), and the same after removing the input-level trend", flush=True)
res["compartments"] = {}
for c in CH:
    e = E[(c, "0", "35")]; ok = np.isfinite(e)
    sl, ic = np.polyfit(inp[ok], e[ok], 1)
    rr = np.where(ok, e - (ic + sl * inp), np.nan)
    for nm, m in COMP.items():
        mm = m & ok
        res["compartments"][f"{c}_{nm}"] = dict(n=int(mm.sum()), enrichment=float(np.median(e[mm])), detrended=float(np.median(rr[mm])),
                                                input=float(np.median(inp[mm]) - mid))
        print(f"  {NAME[c]:<8}{nm:<26} n {mm.sum():>5}  input {np.median(inp[mm]) - mid:+.2f}  enrichment {np.median(e[mm]):+.2f}  detrended {np.median(rr[mm]):+.2f}", flush=True)

# 5. matched vs reference input
print("\n5. matched input (each reaction's own) against the reference", flush=True)
samp = [tuple(s.split("/")) for s in R["samples"].astype(str)]
Mt = R["matched"]
rs = []
for j, k in enumerate(samp):
    em = PD[k] - Mt[:, j]; ok = np.isfinite(em) & np.isfinite(E[k]) & ~con
    r = np.corrcoef(em[ok], E[k][ok])[0, 1]; rs.append(r)
    # the 43 vs 35 heat response on the enrichment scale is unchanged by the reference (same input on both sides)
print(f"  enrichment from matched vs reference input: r {min(rs):.3f} .. {max(rs):.3f} over the {len(rs)} clean reactions", flush=True)
res["matched_vs_reference_r"] = [float(x) for x in rs]
for c in CH:
    k35, k43 = (c, "0", "35"), (c, "0", "43")
    if k35 in samp and k43 in samp:
        j35, j43 = samp.index(k35), samp.index(k43)
        dm = (PD[k43] - Mt[:, j43]) - (PD[k35] - Mt[:, j35]); dr = PD[k43] - PD[k35]
        ok = np.isfinite(dm) & np.isfinite(dr) & ~con
        r = np.corrcoef(dm[ok], dr[ok])[0, 1]
        print(f"  {NAME[c]} 43 vs 35 C response, with matched inputs subtracted vs the pull-down-only response: r {r:.3f}", flush=True)
        res[f"heat_matched_{c}"] = float(r)

# 6. a separate client mode?
print("\n6. is there a separate enriched mode? Gaussian mixtures on the enrichment (control, 35 C)", flush=True)
for c in CH:
    e = E[(c, "0", "35")]; x = e[np.isfinite(e)].reshape(-1, 1)
    bic = {}
    for k in (1, 2, 3):
        gm = GaussianMixture(k, random_state=0, n_init=5).fit(x); bic[k] = gm.bic(x)
        if k == 2:
            o = np.argsort(gm.means_[:, 0])
            desc = ", ".join(f"mean {gm.means_[i,0]:+.2f} sd {np.sqrt(gm.covariances_[i,0,0]):.2f} weight {gm.weights_[i]:.2f}" for i in o)
    sk = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    print(f"  {NAME[c]}: BIC 1/2/3 components {bic[1]:.0f} / {bic[2]:.0f} / {bic[3]:.0f}; skew {sk:+.2f}; two components: {desc}", flush=True)
    for thr in (1, 2, 3):
        print(f"    proteins > {thr} log2 above the median: {(x[:,0] > thr).sum()} ({100*(x[:,0] > thr).mean():.1f}%)", flush=True)
    res[f"mixture_{c}"] = dict(bic={str(k): float(v) for k, v in bic.items()}, skew=sk, two=desc)

# 7. pull-down only
print("\n7. detected in the pull-down (>= 4 of 6 replicates, control 35 C) but not in the input", flush=True)
for c in CH:
    det = np.isfinite(blk[f"br_{c}_0_35"]).sum(1) >= 4
    no = det & ~np.isfinite(inp) & ~con
    p = PD[(c, "0", "35")]
    m = {nm: float(v[no].mean()) for nm, v in COMP.items()}
    mall = {nm: float(v[det & ~con].mean()) for nm, v in COMP.items()}
    print(f"  {NAME[c]}: {no.sum()} of {det.sum()} detected; pull-down level median {np.median(p[no]) - np.nanmedian(p[det]):+.2f} vs all detected; "
          + ", ".join(f"{nm} {100*m[nm]:.0f}% (all {100*mall[nm]:.0f}%)" for nm in ("transmembrane", "mitochondrion", "nucleus only")), flush=True)
    res[f"pd_only_{c}"] = dict(n=int(no.sum()), of=int(det.sum()), frac=m, frac_all=mall)

np.savez(f"{GA}/enrichment.npz", **{f"e_{c}_{s}_{t}": E[(c, s, t)] for c, s, t in E}, input=inp, contaminant=con)
json.dump(res, open(f"{GA}/ad09_results.json", "w"), indent=1)
print("\nAD09_DONE", flush=True)
