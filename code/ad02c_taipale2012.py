"""ad02c: HSP90 client strength across the kinome (Taipale et al. 2012, Cell 150:987, Table S1,
NIHMS509007-supplement-mmc1, sheet "Kinases") against kinase depletion by staurosporine.

Per kinase: LUMIER HSP90 interaction score; dissociation from HSP90 after 1 h ganetespib (500 nM);
protein level after 20 h ganetespib (100 nM; low = stability depends on HSP90); client class
(strong / weak / non); CDC37 luminescence where given. Matched by HGNC symbol.

Release from the HSP90-CDC37 machinery predicts that strong, HSP90-dependent clients lose more
co-chaperone association under staurosporine (negative correlation with the interaction score,
positive with the protein level after ganetespib).
"""
import os, csv, json, re, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, K = f"{ROOT}/data/ga_data", f"{ROOT}/data/external/kinase"
rng = np.random.default_rng(0)
T12 = list(csv.reader(open(f"{K}/taipale2012_kinases.tsv"), delimiter="\t")); hdr = T12[0]
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
sym2acc = collections.defaultdict(set)
for a, r in ann.items():
    for g in (r.get("Gene Names (primary)") or "").split(";"):
        if g.strip(): sym2acc[g.strip().upper()].add(a)


def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


COLS = {"HSP90 score": "Hsp90 interaction score", "dissociation (ganetespib)": "Dissociation after ganetespib treatment",
        "protein level (ganetespib)": "Protein level after ganetespib treatment", "CDC37 luminescence": "Cdc37 luminescence",
        "HSP90 luminescence": "Hsp90 luminescence"}
D = {}
for r in T12[1:]:
    r = r + [""] * (len(hdr) - len(r))
    for a in sym2acc.get(r[0].upper(), ()):
        D[a] = {k: num(r[hdr.index(c)]) for k, c in COLS.items()}
        D[a]["class"] = r[hdr.index("Client class")]; D[a]["family"] = r[hdr.index("Family")]
print(f"Taipale 2012: {len(T12)-1} kinases; {len(D)} matched to UniProt accessions", flush=True)

blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]]); gene = np.array([str(g) for g in blk["gene"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
fam = {a: r["Protein families"] for a, r in ann.items()}


def kgroup(a):
    f = fam.get(a, "")
    for g in ("AGC", "CAMK", "CMGC", "STE", "TKL", "CK1", "Tyr protein kinase", "NEK"):
        if g in f: return g.split()[0]
    return "other"


acts = json.load(open(f"{K}/staurosporine_kd_chembl.json")); tmap = json.load(open(f"{K}/chembl_target_map.json"))
MUT = re.compile(r"\([A-Z]\d+[A-Z]|\bmutant|\bdel\b|-?(?:JH2|JH1)|domain 2|phosphorylated|nonphosphorylated|autoinhibited|\bITD\b", re.I)
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in acts:
    t = tmap.get(x["target_chembl_id"], {})
    if x["document_chembl_id"] in {"CHEMBL1908390", "CHEMBL1150977", "CHEMBL1144455"} and x["standard_value"] and x["standard_units"] == "nM" \
            and t.get("type") == "SINGLE PROTEIN" and len(t.get("accessions", [])) == 1 and t.get("organism") == "Homo sapiens" and not MUT.search(x.get("assay_description") or ""):
        per[x["document_chembl_id"]][t["accessions"][0]].append(np.log10(max(float(x["standard_value"]), 1e-3)))
allk = collections.defaultdict(list)
for d in per.values():
    for u, v in d.items(): allk[u].append(float(np.median(v)))
LKD = {u: float(np.median(v)) for u, v in allk.items()}

out = {"n_matched": len(D)}
print(f"\n1. CLIENT MEASURES vs STAUROSPORINE RESPONSE among kinases (Spearman rho, p; release predicts HSP90 score < 0, protein level > 0)", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T = t.split("_")[-1]; r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][r_] for c in ("21A", "24")]), 1)
    m = np.array([a in KIN and a in D for a in a_])
    rec = {"n": int(m.sum())}
    parts = []
    for k in COLS:
        s = np.array([D.get(a, {}).get(k, np.nan) for a in a_]); mm = m & np.isfinite(s)
        if mm.sum() > 10:
            rho, p = stats.spearmanr(s[mm], y[mm]); rec[k] = [float(rho), float(p), int(mm.sum())]
            parts.append(f"{k} {rho:+.2f} (p {p:.0e}, n {mm.sum()})")
    s = np.array([D.get(a, {}).get("HSP90 score", np.nan) for a in a_]); mm = m & np.isfinite(s) & np.isfinite(lev)
    rec["hsp90_vs_level"] = [float(x) for x in stats.spearmanr(s[mm], lev[mm])]
    cls = np.array([D.get(a, {}).get("class", "") for a in a_])
    means = {c: float(y[m & (cls == c)].mean()) for c in ("Strong", "Weak", "Non") if (m & (cls == c)).sum()}
    kw = stats.kruskal(*[y[m & (cls == c)] for c in ("Strong", "Weak", "Non")]).pvalue
    rec["class_means"] = means; rec["class_counts"] = {c: int((m & (cls == c)).sum()) for c in ("Strong", "Weak", "Non")}; rec["kruskal_p"] = float(kw)
    out[t] = rec
    print(f"  {t} ({m.sum()} kinases): " + "; ".join(parts), flush=True)
    print(f"     client class mean response: " + ", ".join(f"{c} {v:+.3f} (n {rec['class_counts'][c]})" for c, v in means.items())
          + f"  Kruskal p {kw:.0e};   HSP90 score vs pull-down level rho {rec['hsp90_vs_level'][0]:+.2f} (p {rec['hsp90_vs_level'][1]:.0e})", flush=True)

print(f"\n2. JOINT RANK MODEL (kinases with Kd, pull-down level and HSP90 score; 95 % bootstrap)", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T = t.split("_")[-1]; r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][r_] for c in ("21A", "24")]), 1)
    kd = np.array([LKD.get(a, np.nan) for a in a_]); h = np.array([D.get(a, {}).get("HSP90 score", np.nan) for a in a_])
    m = np.array([a in KIN for a in a_]) & np.isfinite(kd) & np.isfinite(h) & np.isfinite(lev)
    g = np.array([kgroup(a) for a in a_])[m]; G = np.column_stack([(g == x).astype(float) for x in sorted(set(g))])
    z = lambda v: (stats.rankdata(v) - stats.rankdata(v).mean()) / stats.rankdata(v).std()
    X = np.column_stack([G, z(kd[m]), z(lev[m]), z(h[m])]); yy = z(y[m])
    b = np.linalg.lstsq(X, yy, rcond=None)[0][-3:]
    bs = [np.linalg.lstsq(X[s], yy[s], rcond=None)[0][-3:] for s in (rng.choice(m.sum(), m.sum()) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    r2 = lambda Xc: 1 - np.sum((yy - Xc @ np.linalg.lstsq(Xc, yy, rcond=None)[0]) ** 2) / np.sum((yy - yy.mean()) ** 2)
    out[t]["joint"] = dict(n=int(m.sum()), kd=[float(b[0]), float(lo[0]), float(hi[0])], level=[float(b[1]), float(lo[1]), float(hi[1])],
                           hsp90=[float(b[2]), float(lo[2]), float(hi[2])], r2=[r2(G), r2(X[:, :-1]), r2(X)])
    print(f"  {t} ({m.sum()} kinases): Kd {b[0]:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]  level {b[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]  "
          f"HSP90 score {b[2]:+.2f} [{lo[2]:+.2f},{hi[2]:+.2f}]  | R2 group {100*r2(G):.1f}% -> +Kd+level {100*r2(X[:,:-1]):.1f}% -> +HSP90 {100*r2(X):.1f}%", flush=True)

print(f"\n3. KINASE GROUPS AND THE AGC EXCEPTION (staurosporine 37 C)", flush=True)
t = "stau_avg_37"; r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]; g_ = gene[r_]
m = np.array([a in KIN and a in D for a in a_]); grp = np.array([kgroup(a) for a in a_])
h = np.array([D.get(a, {}).get("HSP90 score", np.nan) for a in a_]); cls = np.array([D.get(a, {}).get("class", "") for a in a_])
grows = []
for gname in sorted(set(grp[m]), key=lambda x: np.nanmedian(h[m & (grp == x)])):
    s = m & (grp == gname)
    frac_strong = float(np.mean(cls[s] == "Strong"))
    grows.append(dict(group=gname, n=int(s.sum()), hsp90_median=float(np.nanmedian(h[s])), frac_strong=frac_strong, response=float(y[s].mean())))
    print(f"   {gname:<6} n {s.sum():>3}  median HSP90 score {np.nanmedian(h[s]):+.2f}  strong clients {100*frac_strong:3.0f}%  mean response {y[s].mean():+.3f}", flush=True)
out["groups"] = grows
agc = [(g_[i], h[i], cls[i], y[i]) for i in np.nonzero(m & (grp == "AGC"))[0]]
print("   AGC kinases (HSP90 score, class, response): " + ", ".join(f"{g} {s:+.1f} {c} {v:+.2f}" for g, s, c, v in sorted(agc, key=lambda z: -z[3])), flush=True)
json.dump(out, open(f"{K}/ad02c_results.json", "w"), indent=1)
print("\nAD02C_DONE")
