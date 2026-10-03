"""ad02b: do kinases that depend more on the HSP90-CDC37 system lose more co-chaperone association
under staurosporine?

Client strength from Taipale et al. 2014 (Cell 158:434, "A quantitative chaperone interaction
network reveals the architecture of cellular protein homeostasis pathways"), supplementary table
NIHMS605110-supplement-02, sheet "LUMIER scores": 713 FLAG-tagged client proteins scored by LUMIER
against 60 chaperones and co-chaperones; scores > 7 are significant interactions. Used here: CDC37,
HSP90AB1, HSP90AA1, and for comparison HSPA8 (HSC70) and HSPB1. Clients are matched to GA_33
proteins by gene symbol (isoform suffixes removed; non-human constructs dropped).

Tests among the kinases with a staurosporine response:
  1. Spearman correlation of each chaperone score with the response (release predicts strong
     CDC37/HSP90 clients lose more: negative correlation);
  2. whether CDC37 client strength tracks each kinase's level in the control pull-downs, the proxy
     used in additional text 2;
  3. a joint rank model with staurosporine Kd, pull-down level and kinase group;
  4. the AGC kinases that rise despite tight binding: are they weak CDC37 clients?
"""
import os, csv, json, re, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, K = f"{ROOT}/data/ga_data", f"{ROOT}/data/external/kinase"
rng = np.random.default_rng(0)
rows = list(csv.reader(open(f"{K}/taipale2014_lumier.tsv"), delimiter="\t"))
hdr = rows[0]; CH = ["CDC37", "HSP90AB1", "HSP90AA1", "HSPA8", "HSPB1"]
ci = {c: hdr.index(c) for c in CH}
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
sym2acc = collections.defaultdict(set)
for a, r in ann.items():
    for g in (r.get("Gene Names (primary)") or "").split(";"):
        if g.strip(): sym2acc[g.strip().upper()].add(a)
score = collections.defaultdict(lambda: collections.defaultdict(list))
n_rows, n_mapped, nonhuman = 0, 0, 0
for r in rows[1:]:
    if not r or not r[0]: continue
    n_rows += 1
    if r[2] and not (r[2].endswith("_HUMAN") or re.match(r"^[OPQ]\d|^[A-NR-Z]\d", r[2])):
        nonhuman += 1; continue
    sym = re.sub(r"-[A-Z0-9]{1,2}$", "", r[0].upper())
    accs = sym2acc.get(sym) or sym2acc.get(r[0].upper())
    if not accs: continue
    n_mapped += 1
    for c in CH:
        try: v = float(r[ci[c]])
        except (ValueError, IndexError): continue
        for a in accs: score[a][c].append(v)
SC = {a: {c: float(np.mean(v)) for c, v in d.items()} for a, d in score.items()}
print(f"Taipale 2014 LUMIER: {n_rows} clients, {nonhuman} non-human dropped, {n_mapped} matched to a UniProt accession by gene symbol", flush=True)
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
fam = {a: r["Protein families"] for a, r in ann.items()}


def kgroup(a):
    f = fam.get(a, "")
    for g in ("AGC", "CAMK", "CMGC", "STE", "TKL", "CK1", "Tyr protein kinase", "NEK"):
        if g in f: return g.split()[0]
    return "other"


ad2 = json.load(open(f"{K}/ad02_results.json"))
# consensus Kd, rebuilt as in ad02
acts = json.load(open(f"{K}/staurosporine_kd_chembl.json")); tmap = json.load(open(f"{K}/chembl_target_map.json"))
MUT = re.compile(r"\([A-Z]\d+[A-Z]|\bmutant|\bdel\b|-?(?:JH2|JH1)|domain 2|phosphorylated|nonphosphorylated|autoinhibited|\bITD\b", re.I)
per = collections.defaultdict(lambda: collections.defaultdict(list))
for a in acts:
    t = tmap.get(a["target_chembl_id"], {})
    if a["document_chembl_id"] in {"CHEMBL1908390", "CHEMBL1150977", "CHEMBL1144455"} and a["standard_value"] and a["standard_units"] == "nM" \
            and t.get("type") == "SINGLE PROTEIN" and len(t.get("accessions", [])) == 1 and t.get("organism") == "Homo sapiens" \
            and not MUT.search(a.get("assay_description") or ""):
        per[a["document_chembl_id"]][t["accessions"][0]].append(np.log10(max(float(a["standard_value"]), 1e-3)))
allk = collections.defaultdict(list)
for d in per.values():
    for u, v in d.items(): allk[u].append(float(np.median(v)))
LKD = {u: float(np.median(v)) for u, v in allk.items()}

out = {"n_clients": n_rows, "n_mapped": n_mapped}
print(f"\n1. CHAPERONE CLIENT STRENGTH vs STAUROSPORINE RESPONSE among kinases (Spearman; negative = strong clients lose more)", flush=True)
print(f"  {'target':<13}{'n':>5}" + "".join(f"{c:>12}" for c in CH) + f"{'CDC37 vs pull-down level':>27}", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T = t.split("_")[-1]
    r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][r_] for c in ("21A", "24")]), 1)
    m = np.array([a in KIN and a in SC and "CDC37" in SC[a] for a in a_])
    res = {"n": int(m.sum())}
    line = f"  {t:<13}{m.sum():>5}"
    for c in CH:
        s = np.array([SC.get(a, {}).get(c, np.nan) for a in a_])
        mm = m & np.isfinite(s)
        rho, p = stats.spearmanr(s[mm], y[mm]); res[c] = [float(rho), float(p)]
        line += f"{rho:>+8.2f} ({p:.0e})"[:12].rjust(12)
    s = np.array([SC.get(a, {}).get("CDC37", np.nan) for a in a_]); mm = m & np.isfinite(lev)
    rl, pl = stats.spearmanr(s[mm], lev[mm]); res["cdc37_vs_level"] = [float(rl), float(pl)]
    print(line + f"{rl:>+17.2f} (p {pl:.0e})", flush=True)
    out[t] = res
# detailed lines
print("\n  (rho, p) per chaperone:", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    print(f"   {t}: " + "; ".join(f"{c} {out[t][c][0]:+.3f} (p {out[t][c][1]:.1e})" for c in CH), flush=True)

print(f"\n2. JOINT RANK MODEL among kinases with Kd, pull-down level and CDC37 score (standardised rank coefficients, 95 % bootstrap)", flush=True)
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T = t.split("_")[-1]; r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][r_] for c in ("21A", "24")]), 1)
    kd = np.array([LKD.get(a, np.nan) for a in a_]); cd = np.array([SC.get(a, {}).get("CDC37", np.nan) for a in a_])
    m = np.array([a in KIN for a in a_]) & np.isfinite(kd) & np.isfinite(cd) & np.isfinite(lev)
    g = np.array([kgroup(a) for a in a_])[m]
    G = np.column_stack([(g == x).astype(float) for x in sorted(set(g))])
    z = lambda v: (stats.rankdata(v) - stats.rankdata(v).mean()) / stats.rankdata(v).std()
    X = np.column_stack([G, z(kd[m]), z(lev[m]), z(cd[m])]); yy = z(y[m])
    b = np.linalg.lstsq(X, yy, rcond=None)[0][-3:]
    bs = [np.linalg.lstsq(X[s], yy[s], rcond=None)[0][-3:] for s in (rng.choice(m.sum(), m.sum()) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    r2 = lambda Xc: 1 - np.sum((yy - Xc @ np.linalg.lstsq(Xc, yy, rcond=None)[0]) ** 2) / np.sum((yy - yy.mean()) ** 2)
    out[t]["joint"] = dict(n=int(m.sum()), kd=[float(b[0]), float(lo[0]), float(hi[0])], level=[float(b[1]), float(lo[1]), float(hi[1])],
                           cdc37=[float(b[2]), float(lo[2]), float(hi[2])], r2_group=r2(G), r2_group_kd_level=r2(X[:, :-1]), r2_all=r2(X))
    print(f"  {t} ({m.sum()} kinases): Kd {b[0]:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]  level {b[1]:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]  "
          f"CDC37 {b[2]:+.2f} [{lo[2]:+.2f},{hi[2]:+.2f}]  | R2 group {100*r2(G):.1f}% -> +Kd+level {100*r2(X[:,:-1]):.1f}% -> +CDC37 {100*r2(X):.1f}%", flush=True)

print(f"\n3. CDC37 AND HSP90 SCORES BY KINASE GROUP (kinases in GA_33 with a score)", flush=True)
t = "stau_avg_37"; r_ = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[r_]
gene = np.array([str(x) for x in blk["gene"]])[r_]
m = np.array([a in KIN and a in SC and "CDC37" in SC[a] for a in a_])
grp = np.array([kgroup(a) for a in a_])
cd = np.array([SC.get(a, {}).get("CDC37", np.nan) for a in a_]); h9 = np.array([SC.get(a, {}).get("HSP90AB1", np.nan) for a in a_])
gres = []
for gname in sorted(set(grp[m]), key=lambda x: np.nanmedian(cd[m & (grp == x)])):
    s = m & (grp == gname)
    gres.append(dict(group=gname, n=int(s.sum()), cdc37=float(np.nanmedian(cd[s])), hsp90=float(np.nanmedian(h9[s])), stau37=float(y[s].mean())))
    print(f"   {gname:<6} n {s.sum():>3}  median CDC37 {np.nanmedian(cd[s]):+.2f}  HSP90AB1 {np.nanmedian(h9[s]):+.2f}  mean staurosporine response {y[s].mean():+.2f}", flush=True)
out["groups"] = gres
agcl = [(gene[i], cd[i], h9[i], y[i]) for i in np.nonzero(m & (grp == "AGC"))[0]]
print("   AGC kinases (CDC37, HSP90AB1, response): " + ", ".join(f"{g} {c:+.1f}/{h:+.1f}/{v:+.2f}" for g, c, h, v in sorted(agcl, key=lambda z: -z[3])), flush=True)
json.dump(out, open(f"{K}/ad02b_results.json", "w"), indent=1)
print("\nAD02B_DONE")
