"""ad11: the heat response with the lysate-like component removed.

ad10 found the 43 C pull-downs to be about half "35 C pull-down" and half "lysate"
(level_43 ~ 0.42 level_35 + 0.46 input), so a large part of the 43 vs 35 C response is predictable from
two numbers per protein: its lysate abundance and its 35 C enrichment. Here that part is removed and the
earlier heat results are re-tested on what is left (the "selective" heat response).

Noise: the predictors (input, 35 C enrichment) and the response come from disjoint biological
replicates. Residual A = response(BR4-6) - fit(input, enrichment35(BR1-3)); residual B the reverse;
the selective response is their mean, so every replicate contributes and none is used on both sides.

  1. how much of the response the two numbers explain; what is left
  2. Tm deciles before and after; curvature and slope with cluster-bootstrap intervals
  3. the 35-37 and 37-43 C steps by Tm decile, adjusted the same way
  4. complex subunits, adjusted
  5. agreement between the two co-chaperones' heat responses, before and after
Sequence models of the three parts are fitted on the GPU by ad13 (from selective_heat.npz).
"""
import os, re, csv, json, collections
import numpy as np

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
rng = np.random.default_rng(0)
CH = ("21A", "24")
SPL = (([0, 1, 2], [3, 4, 5]), ([3, 4, 5], [0, 1, 2]))
res = {}


def bm(T, cols, chs=CH):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.hstack([blk[f"br_{c}_0_{T}"][:, cols] for c in chs]), 1)


def selective(T, chs=CH):
    """Selective response (mean of the two disjoint-split residuals), raw response, lysate-like part."""
    resid, raw, fitp = [], [], []
    for F, R in SPL:
        e35 = bm("35", F, chs) - inp
        y = bm(T, R, chs) - bm("35", R, chs)
        ok = np.isfinite(e35) & np.isfinite(y) & np.isfinite(inp) & ~con
        X = np.column_stack([np.ones(ok.sum()), inp[ok], e35[ok]])
        w = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        f = np.full(len(y), np.nan); f[ok] = X @ w
        resid.append(y - f); raw.append(y); fitp.append(f)
    return np.mean(resid, 0), np.mean(raw, 0), np.mean(fitp, 0)


sel43, raw43, fit43 = selective("43")
sel37, raw37, fit37 = selective("37")
G = FG["groups"]

# ---------------- 1
print("1. variance of the heat response explained by lysate abundance + 35 C enrichment (disjoint replicates)", flush=True)
for nm, s, r, f in (("43 vs 35", sel43, raw43, fit43), ("37 vs 35", sel37, raw37, fit37)):
    ok = np.isfinite(s)
    sh = 1 - np.var(s[ok]) / np.var(r[ok])
    print(f"  {nm} C: {ok.sum()} proteins; explained {100*sh:.1f}%; sd raw {r[ok].std():.3f} -> selective {s[ok].std():.3f}", flush=True)
    res[f"explained_{nm}"] = float(sh)

# ---------------- Tm
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
dm = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in dm:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
rm = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
cen = {r: float(np.median(list(rm[r].values()))) for r in HUM}; grand = float(np.median(list(cen.values())))
LL = collections.defaultdict(list)
for r in HUM:
    for a, tm_ in rm[r].items(): LL[a].append(tm_ - cen[r])
TM = {a: float(np.median(v)) + grand for a, v in LL.items()}
del dm, per
tm = np.array([TM.get(a, np.nan) for a in acc])

# restrict to the proteins of the published heat target, detected at all three temperatures for the step table
rows = FG["rows_temp_avg_43v35"]
inrows = np.zeros(len(acc), bool); inrows[rows] = True
okT = inrows & np.isfinite(tm) & np.isfinite(sel43)
edges = np.percentile(tm[okT], np.linspace(0, 100, 11))
bins = np.clip(np.digitize(tm, edges[1:-1]), 0, 9)
ug, inv = np.unique(G[okT], return_inverse=True); mem_cl = [np.nonzero(inv == k)[0] for k in range(len(ug))]


def shape(y, m):
    zt = (tm[m] - tm[m].mean()) / tm[m].std()
    X2 = np.column_stack([np.ones(m.sum()), zt, zt ** 2]); X1 = X2[:, :2]
    b2 = np.linalg.lstsq(X2, y[m], rcond=None)[0]; b1 = np.linalg.lstsq(X1, y[m], rcond=None)[0]
    ug_, inv_ = np.unique(G[m], return_inverse=True); mm = [np.nonzero(inv_ == k)[0] for k in range(len(ug_))]
    bs2, bs1 = [], []
    yy = y[m]
    for _ in range(1000):
        s = np.concatenate([mm[k] for k in rng.integers(0, len(ug_), len(ug_))])
        bs2.append(np.linalg.lstsq(X2[s], yy[s], rcond=None)[0][2]); bs1.append(np.linalg.lstsq(X1[s], yy[s], rcond=None)[0][1])
    return (float(b2[2]), *np.percentile(bs2, [2.5, 97.5]).tolist()), (float(b1[1]), *np.percentile(bs1, [2.5, 97.5]).tolist())


print(f"\n2. by Tm decile ({okT.sum()} proteins): mean 43 vs 35 C response", flush=True)
print("  Tm upper edge     " + " ".join(f"{e:>6.1f}" for e in edges[1:]), flush=True)
tab = {}
for nm, y in (("raw", raw43), ("lysate-like part", fit43), ("selective", sel43)):
    v = [float(y[okT & (bins == i)].mean()) for i in range(10)]
    tab[nm] = v
    print(f"  {nm:<18}" + " ".join(f"{x:>+6.2f}" for x in v), flush=True)
for nm, y in (("raw", raw43), ("selective", sel43)):
    q, l = shape(y, okT)
    print(f"  {nm}: curvature (per sd^2) {q[0]:+.3f} [{q[1]:+.3f}, {q[2]:+.3f}]; linear slope (per sd) {l[0]:+.3f} [{l[1]:+.3f}, {l[2]:+.3f}]", flush=True)
    res[f"shape_43_{nm}"] = dict(curv=q, slope=l)
res["tm_table_43"] = dict(edges=edges.tolist(), **tab)

# ---------------- 3 steps (proteins with a value at all three temperatures)
ok3 = okT & np.isfinite(sel37)
print(f"\n3. steps by Tm decile ({ok3.sum()} proteins with 35, 37 and 43 C)", flush=True)
steps = {}
for nm, a, b in (("raw", raw37, raw43), ("selective", sel37, sel43)):
    s1 = [float(a[ok3 & (bins == i)].mean()) for i in range(10)]
    s2 = [float((b - a)[ok3 & (bins == i)].mean()) for i in range(10)]
    steps[nm] = dict(step_35_37=s1, step_37_43=s2)
    print(f"  {nm:<10} 35->37 " + " ".join(f"{x:>+6.2f}" for x in s1), flush=True)
    print(f"  {nm:<10} 37->43 " + " ".join(f"{x:>+6.2f}" for x in s2), flush=True)
res["steps"] = steps
q, l = shape(sel37, ok3)
print(f"  selective 37 vs 35 C: curvature {q[0]:+.3f} [{q[1]:+.3f}, {q[2]:+.3f}], slope {l[0]:+.3f} [{l[1]:+.3f}, {l[2]:+.3f}]", flush=True)
res["shape_37_selective"] = dict(curv=q, slope=l)

# ---------------- 4 complexes
member = set()
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    m_ = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    if len(m_) >= 2: member |= m_
mem = np.array([a in member for a in acc])
loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in acc]
nuc = np.array(["Nucleus" in l for l in loc], float); cyt = np.array(["Cytoplasm" in l for l in loc], float)
tmem = np.array(["Transmembrane" in ann.get(a, {}).get("Keywords", "") for a in acc], float)
print("\n4. complex subunits vs non-members, adjusted for Tm (quadratic) and location", flush=True)
for nm, y, base in (("43 raw", raw43, okT), ("43 selective", sel43, okT), ("37 selective", sel37, ok3)):
    m = base
    zt = (tm[m] - tm[m].mean()) / tm[m].std()
    X = np.column_stack([np.ones(m.sum()), mem[m], zt, zt ** 2, nuc[m], cyt[m], tmem[m]])
    yy = y[m]; b = np.linalg.lstsq(X, yy, rcond=None)[0][1]
    ug_, inv_ = np.unique(G[m], return_inverse=True); mm = [np.nonzero(inv_ == k)[0] for k in range(len(ug_))]
    bs = [np.linalg.lstsq(X[s], yy[s], rcond=None)[0][1] for s in (np.concatenate([mm[k] for k in rng.integers(0, len(ug_), len(ug_))]) for _ in range(1000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {nm:<14} subunits {b:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n members {int(mem[m].sum())})", flush=True)
    res[f"complex_{nm}"] = [float(b), float(lo), float(hi)]

# ---------------- 5 agreement between co-chaperones
print("\n5. agreement between the DNAJA1 and DNAJB11 43 vs 35 C responses", flush=True)
s1, r1, f1 = selective("43", ("21A",)); s2, r2_, f2 = selective("43", ("24",))
ok = np.isfinite(s1) & np.isfinite(s2)
for nm, a, b in (("raw", r1, r2_), ("lysate-like part", f1, f2), ("selective", s1, s2)):
    print(f"  {nm:<18} r {np.corrcoef(a[ok], b[ok])[0,1]:.3f}", flush=True)
    res[f"agree_{nm}"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
# reliability of each co-chaperone's selective response: the two disjoint-split residuals share no response
# replicates but each half's predictor shares replicates with the other half's response, so use the raw
# split-half reliabilities as the reference and report the agreement between baits relative to them.

json.dump(res, open(f"{GA}/ad11_results.json", "w"), indent=1)
np.savez(f"{GA}/selective_heat.npz", sel43=sel43, raw43=raw43, fit43=fit43, sel37=sel37, raw37=raw37, fit37=fit37)
print("\nAD11_DONE", flush=True)
