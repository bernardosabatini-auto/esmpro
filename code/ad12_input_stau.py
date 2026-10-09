"""ad12: the staurosporine response read against the input lysate.

In ad02c the kinase joint model used each kinase's level in the control pull-down (more abundant in the
pull-down -> more depleted by staurosporine, rank slope -0.41 at 35 C). That level is input abundance +
control enrichment. Here the two are separated, with the control level/enrichment taken from biological
replicates disjoint from those of the response (BR1-3 vs BR4-6 and the reverse; both splits shown).

  1. all proteins: does staurosporine make the pull-down more or less lysate-like (mixing fit as ad10)?
  2. kinases: joint rank model (kinase group + Kd + input + control enrichment + HSP90 score)
  3. Taipale 2012 client classes: input abundance and control enrichment of strong / weak / non-clients
"""
import os, csv, json, re, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, K = f"{ROOT}/data/ga_data", f"{ROOT}/data/external/kinase"
rng = np.random.default_rng(0)
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
CH = ("21A", "24")
SPL = (("BR1-3 control, BR4-6 response", [0, 1, 2], [3, 4, 5]), ("BR4-6 control, BR1-3 response", [3, 4, 5], [0, 1, 2]))
res = {}


def bm(s, T, cols, chs=CH):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.hstack([blk[f"br_{c}_{s}_{T}"][:, cols] for c in chs]), 1)


# ---------------- 1. mixing
print("1. staurosporine pull-down from the control pull-down and the input (level_stau = a + beta level_ctrl + gamma input)", flush=True)
for T in ("35", "37", "43"):
    for c in CH:
        a = bm("0", T, [0, 1, 2], (c,)); b = bm("0", T, [3, 4, 5], (c,)); p = bm("10", T, [3, 4, 5], (c,))
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(p) & np.isfinite(inp) & ~con
        rel = np.corrcoef(a[ok], b[ok])[0, 1]
        S = np.cov(np.vstack([a[ok], inp[ok]])); S[0, 0] *= rel
        bc = np.linalg.solve(S, np.cov(np.vstack([a[ok], inp[ok], p[ok]]))[:2, 2])
        s0 = np.polyfit(inp[ok], bm("0", T, list(range(6)), (c,))[ok], 1)[0]; s1 = np.polyfit(inp[ok], bm("10", T, list(range(6)), (c,))[ok], 1)[0]
        res[f"mix_{c}_{T}"] = dict(beta=float(bc[0]), gamma=float(bc[1]), slope_ctrl=float(s0), slope_stau=float(s1))
        print(f"  {T} C {c}: beta {bc[0]:.3f} gamma {bc[1]:+.3f}; slope on input control {s0:.3f} -> staurosporine {s1:.3f}", flush=True)
    for nm, F, R in SPL:
        e = bm("0", T, F) - inp; y = bm("10", T, R) - bm("0", T, R)
        ok = np.isfinite(e) & np.isfinite(y) & np.isfinite(inp) & ~con
        print(f"  {T} C [{nm}] response vs input r {np.corrcoef(y[ok], inp[ok])[0,1]:+.3f}, vs control enrichment r {np.corrcoef(y[ok], e[ok])[0,1]:+.3f}", flush=True)

# ---------------- 2. kinases
T12 = list(csv.reader(open(f"{K}/taipale2012_kinases.tsv"), delimiter="\t")); hdr = T12[0]
sym2acc = collections.defaultdict(set)
for a_, r in ann.items():
    for g in (r.get("Gene Names (primary)") or "").split(";"):
        if g.strip(): sym2acc[g.strip().upper()].add(a_)
num = lambda x: float(x) if re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", str(x).strip(), re.I) else np.nan
D = {}
for r in T12[1:]:
    r = r + [""] * (len(hdr) - len(r))
    for a_ in sym2acc.get(r[0].upper(), ()):
        D[a_] = dict(h=num(r[hdr.index("Hsp90 interaction score")]), cls=r[hdr.index("Client class")])
fam = {a_: r["Protein families"] for a_, r in ann.items()}
def kgroup(a_):
    f = fam.get(a_, "")
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
kd = np.array([LKD.get(a_, np.nan) for a_ in acc]); h = np.array([D.get(a_, {}).get("h", np.nan) for a_ in acc])
cls = np.array([D.get(a_, {}).get("cls", "") for a_ in acc]); grp = np.array([kgroup(a_) for a_ in acc])
isk = np.array([a_ in KIN for a_ in acc])
z = lambda v: (stats.rankdata(v) - stats.rankdata(v).mean()) / stats.rankdata(v).std()

print("\n2. kinases, joint rank model: group + Kd + input + control enrichment + HSP90 score (95 % bootstrap over kinases)", flush=True)
for T in ("35", "37", "43"):
    for nm, F, R in SPL:
        e = bm("0", T, F) - inp; lev = bm("0", T, F); y = bm("10", T, R) - bm("0", T, R)
        m = isk & np.isfinite(kd) & np.isfinite(h) & np.isfinite(e) & np.isfinite(y) & ~con
        g = grp[m]; Gd = np.column_stack([(g == x).astype(float) for x in sorted(set(g))])
        yy = z(y[m])
        out = {}
        for mn, cols in (("level", [z(kd[m]), z(lev[m]), z(h[m])]), ("split", [z(kd[m]), z(inp[m]), z(e[m]), z(h[m])])):
            X = np.column_stack([Gd] + cols); k = len(cols)
            b = np.linalg.lstsq(X, yy, rcond=None)[0][-k:]
            bs = [np.linalg.lstsq(X[s], yy[s], rcond=None)[0][-k:] for s in (rng.choice(m.sum(), m.sum()) for _ in range(2000))]
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            out[mn] = [(float(b[i]), float(lo[i]), float(hi[i])) for i in range(k)]
        res[f"kin_{T}_{nm}"] = dict(n=int(m.sum()), **out)
        f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
        print(f"  {T} C [{nm}] n {m.sum()}: with level: Kd {f(out['level'][0])} level {f(out['level'][1])} HSP90 {f(out['level'][2])}", flush=True)
        print(f"  {'':<38} split:      Kd {f(out['split'][0])} input {f(out['split'][1])} enrichment {f(out['split'][2])} HSP90 {f(out['split'][3])}", flush=True)

# ---------------- 3. client classes
print("\n3. Taipale 2012 client classes: median input (vs all kinases) and control 35 C enrichment", flush=True)
e35 = bm("0", "35", list(range(6))) - inp
mk = isk & np.isfinite(inp) & np.isfinite(e35) & np.isin(cls, ["Strong", "Weak", "Non"])
for c in ("Strong", "Weak", "Non"):
    s = mk & (cls == c)
    print(f"  {c:<7} n {s.sum():>3}: input {np.median(inp[s]) - np.median(inp[mk]):+.2f}  enrichment {np.median(e35[s]) - np.median(e35[mk]):+.2f}", flush=True)
    res[f"class_{c}"] = dict(n=int(s.sum()), input=float(np.median(inp[s]) - np.median(inp[mk])), enrichment=float(np.median(e35[s]) - np.median(e35[mk])))
print(f"  kinases vs all proteins: input {np.nanmedian(inp[isk & ~con]) - np.nanmedian(inp[~con]):+.2f}, enrichment {np.nanmedian(e35[isk]) - np.nanmedian(e35):+.2f}", flush=True)
json.dump(res, open(f"{K}/ad12_results.json", "w"), indent=1)
print("\nAD12_DONE", flush=True)
