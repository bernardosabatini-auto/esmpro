"""ad15: why does the pull-down rise less than proportionally with lysate abundance?

At 35 C the control pull-down level rises with input abundance at a slope of only ~0.35 (ad09/ad10), so
enrichment (pull-down minus input) falls with abundance. Three families of explanation:
  H1 intrinsic   abundant proteins are, as proteins, poorer clients (stability / solubility selected with
                 expression); capture depends on what the protein is, not on how much of it is in the tube
  H2 in the tube the amount captured grows less than proportionally with the amount present (a captured
                 sub-population not proportional to the total, or saturation)
  H3 measurement the low-load pull-down sample is quantified on a compressed scale
Tests:
  1. the slope, its curvature, by bait and temperature (cluster bootstrap)
  2. matched properties: does the slope move toward 1 at fixed stability (Meltome Tm), disorder, membrane
     helices, location, enzyme class and length?  (H1 predicts yes, H2/H3 no)
  3. typical vs lysate-specific abundance: input split into the part predicted by abundance elsewhere (PaxDb
     tissues and cell lines, and ESMC sequence) and the residual specific to this lysate. H1 predicts
     proportional capture of the lysate-specific part (slope toward 1); H2/H3 predict the same slope for both
  4. load: per run, raw total intensity and the slope on input; within and between conditions (H3 predicts
     the slope follows load)
Contaminants excluded; proteins without an input level are not imputed.
"""
import os, re, csv, json, glob, gzip, collections
import numpy as np, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True); G = FG["groups"]
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
rng = np.random.default_rng(0)
CH = ("21A", "24"); NAME = {"21A": "DNAJA1", "24": "DNAJB11"}
res = {}


def lev(c, T, s="0"):
    L = blk[f"br_{c}_{s}_{T}"]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(L).sum(1) >= 4, np.nanmean(L, 1), np.nan)


ip = {(c, T): lev(c, T) for c in CH for T in ("35", "37", "43")}
for T in ("35", "37", "43"):
    ip[("avg", T)] = (ip[("21A", T)] + ip[("24", T)]) / 2


def clusters(m):
    ug, inv = np.unique(G[m], return_inverse=True)
    return [np.nonzero(inv == k)[0] for k in range(len(ug))]


def fit_ci(X, y, m, cols, B=1000):
    """OLS on rows m; coefficients `cols` with 95 % intervals resampling 30 % sequence clusters."""
    Xm, ym = X[m], y[m]
    b = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    mem = clusters(m); bs = []
    for _ in range(B):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])
        bs.append(np.linalg.lstsq(Xm[s], ym[s], rcond=None)[0][cols])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return [(float(b[c]), float(l), float(h)) for c, l, h in zip(cols, lo, hi)]


f3 = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
one = lambda n: np.ones(n)

# ---------------- 1. slope and curvature
print("1. pull-down level on input abundance (control, >= 4 of 6 replicates)", flush=True)
curve = {}
for c in ("21A", "24", "avg"):
    for T in ("35", "37", "43"):
        y = ip[(c, T)]; m = np.isfinite(y) & np.isfinite(inp) & ~con
        z = (inp - np.nanmean(inp[m])) / np.nanstd(inp[m])
        X = np.column_stack([one(len(y)), inp, z ** 2])
        sl = fit_ci(np.column_stack([one(len(y)), inp]), y, m, [1])[0]
        cu = fit_ci(X, y, m, [2], B=500)[0]
        r = float(np.corrcoef(inp[m], y[m])[0, 1])
        res[f"slope_{c}_{T}"] = dict(n=int(m.sum()), slope=sl, curvature=cu, r=r, sd_ip=float(y[m].std()), sd_input=float(inp[m].std()))
        print(f"  {NAME.get(c, 'average'):<8} {T} C n {m.sum()}: slope {f3(sl)}, curvature (per sd^2) {f3(cu)}, r {r:.2f}", flush=True)
        if c == "avg" or T == "35":
            ed = np.percentile(inp[m], np.linspace(0, 100, 21)); b = np.clip(np.digitize(inp, ed[1:-1]), 0, 19)
            curve[f"{c}_{T}"] = np.array([[inp[m & (b == i)].mean(), y[m & (b == i)].mean()] for i in range(20)])

# ---------------- annotations: Tm, SAE, UniProt
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
dm = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in dm:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
del dm
rm = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
cen = {r: float(np.median(list(rm[r].values()))) for r in HUM}; grand = float(np.median(list(cen.values())))
LL = collections.defaultdict(list)
for r in HUM:
    for a, t_ in rm[r].items(): LL[a].append(t_ - cen[r])
tm = np.array([float(np.median(LL[a])) + grand if a in LL else np.nan for a in acc])
S = np.load(f"{ROOT}/data/sae/sae_l60.npz", allow_pickle=True); SMAX = S["max"]; spos = {str(a): i for i, a in enumerate(S["accession"])}
sae = lambda f: np.array([np.log1p(float(SMAX[spos[a], f])) if a in spos else np.nan for a in acc])
dis, mhx = sae(654), sae(10715)
loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in acc]; kw = [ann.get(a, {}).get("Keywords", "") for a in acc]
has = lambda words, src: np.array([any(w in s for w in words) for s in src], float)
LOC = {"nucleus": has(["Nucleus"], loc), "cytoplasm": has(["Cytoplasm"], loc), "membrane": has(["Membrane"], loc),
       "ER/Golgi": has(["Endoplasmic reticulum", "Golgi"], loc), "mitochondrion": has(["Mitochondrion"], loc), "secreted": has(["Secreted"], loc)}
tmem = has(["Transmembrane"], kw)
enz = has(["Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase", "Ligase"], kw)
seqlen = {}
for fn in (f"{ROOT}/data/pd_data/sequences.fasta", f"{GA}/new_sequences.fasta"):
    k = None
    for l in open(fn):
        if l[0] == ">": k = l[1:].strip().split()[0].split("|")[0]; seqlen[k] = 0
        else: seqlen[k] += len(l.strip())
loglen = np.array([np.log2(seqlen[a]) if seqlen.get(a) else np.nan for a in acc])

# ---------------- 2. matched properties
print("\n2. slope on input at matched properties (35 C, average of the two baits)", flush=True)
y = ip[("avg", "35")]
zt = np.where(np.isfinite(tm), (tm - np.nanmean(tm)) / np.nanstd(tm), np.nan)
props = {"Tm (linear + quadratic)": [zt, zt ** 2], "disorder (SAE 654)": [dis], "membrane helix (SAE 10715) + transmembrane": [mhx, tmem],
         "location (6 classes)": list(LOC.values()), "enzyme (EC keyword)": [enz], "length": [loglen]}
allp = [v for vs in props.values() for v in vs]
base = np.isfinite(y) & np.isfinite(inp) & ~con & np.all([np.isfinite(v) for v in allp], 0)
nonTm = [v for k, vs in props.items() if not k.startswith("Tm") for v in vs]
base2 = np.isfinite(y) & np.isfinite(inp) & ~con & np.all([np.isfinite(v) for v in nonTm], 0)
tab2 = {}
for subset, m, plist in (("with a Meltome Tm", base, props), ("all (no Tm)", base2, {k: v for k, v in props.items() if not k.startswith("Tm")})):
    b0 = fit_ci(np.column_stack([one(len(y)), inp]), y, m, [1])[0]
    rows = [("input only", b0)]
    for k, vs in plist.items():
        rows.append((f"+ {k}", fit_ci(np.column_stack([one(len(y)), inp] + vs), y, m, [1], B=500)[0]))
    vs = [v for vv in plist.values() for v in vv]
    bx = fit_ci(np.column_stack([one(len(y)), inp] + vs), y, m, [1])[0]
    rows.append(("+ all of these", bx))
    # share of the property set's own explanatory power: R2 of the property set alone for enrichment
    e = y - inp; Xp = np.column_stack([one(len(y))] + vs)
    w = np.linalg.lstsq(Xp[m], e[m], rcond=None)[0]; r2p = 1 - np.var(e[m] - Xp[m] @ w) / np.var(e[m])
    print(f"  [{subset}, n {m.sum()}]  properties explain {100*r2p:.1f}% of enrichment", flush=True)
    for nm, t in rows:
        moved = (t[0] - b0[0]) / (1 - b0[0])
        print(f"    {nm:<46} slope {f3(t)}   share of the gap to 1 closed {100*moved:+.0f}%", flush=True)
    tab2[subset] = dict(n=int(m.sum()), r2_props=float(r2p), rows=[(nm, t) for nm, t in rows])
res["matched"] = tab2

# ---------------- 3. typical vs lysate-specific abundance
print("\n3. input split into typical abundance (predicted from elsewhere) and the lysate-specific residual", flush=True)
PX = f"{ROOT}/data/external/paxdb"
e2u = collections.defaultdict(set)
for l in gzip.open(f"{PX}/map.tsv.gz", "rt"):
    p = l.rstrip("\n").split("\t")
    if len(p) >= 3: e2u[p[2]].add(p[1].split("|")[0])
def paxdb(fn):
    d = collections.defaultdict(list)
    for l in open(fn):
        if l[0] == "#": continue
        p = l.split()
        if len(p) < 2: continue
        v = float(p[1])
        if v > 0:
            for u in e2u.get(p[0], ()): d[u].append(np.log2(v))
    return {u: max(v) for u, v in d.items()}
D = f"{PX}/paxdb-abundance-files-v5.0/9606"
CELL = ["iBAQ_HEK293_Geiger_2012_uniprot", "iBAQ_U2OS_Geiger_2012_uniprot", "iBAQ_Jurkat_Geiger_2012_uniprot",
        "iBAQ_LnCap_Geiger_2012_uniprot", "hela_Nagaraj_2011_iBAQ", "K562_Geiger_2012", "A549_Geiger_2012", "RKO_Geiger_2012"]
cell = {k: paxdb(f"{D}/9606-{k}.txt") for k in CELL}
whole = paxdb(f"{D}/9606-WHOLE_ORGANISM-integrated.txt")
def consensus(dsets, min_n=2):
    vals = collections.defaultdict(list)
    for d in dsets:
        med = np.median(list(d.values()))
        for u, v in d.items(): vals[u].append(v - med)
    return {u: float(np.mean(v)) for u, v in vals.items() if len(v) >= min_n}
cellc = consensus(cell.values())
proxies = {"PaxDb, 8 cell lines": cellc, "PaxDb, whole organism": whole, "PaxDb, HEK293": cell["iBAQ_HEK293_Geiger_2012_uniprot"]}
oof = f"{GA}/ad13_oof_input_abundance.npz"
if os.path.exists(oof):
    O = np.load(oof, allow_pickle=True); proxies["ESMC sequence (out-of-fold)"] = {str(a): float(p) for a, p in zip(O["accession"], O["pred"])}
P = {k: np.array([d.get(a, np.nan) for a in acc]) for k, d in proxies.items()}
Xall = np.column_stack(list(P.values()))
tab3 = {}
for c in ("avg", "21A", "24"):
    for T in ("35", "43"):
        y = ip[(c, T)]
        sets = list(P.items()) + ([("all proxies together", None)] if c == "avg" else [])
        for k, p in sets:
            Z = Xall if p is None else p[:, None]
            m = np.isfinite(y) & np.isfinite(inp) & ~con & np.all(np.isfinite(Z), 1)
            Zm = np.column_stack([one(len(y)), np.nan_to_num(Z)])
            w = np.linalg.lstsq(Zm[m], inp[m], rcond=None)[0]
            typ = Zm @ w; spec = inp - typ
            r2 = 1 - np.var(spec[m]) / np.var(inp[m])
            t = fit_ci(np.column_stack([one(len(y)), typ, spec]), y, m, [1, 2])
            b0 = fit_ci(np.column_stack([one(len(y)), inp]), y, m, [1], B=300)[0]
            tab3[f"{c}_{T}_{k}"] = dict(n=int(m.sum()), r2_input=float(r2), slope_input=b0, typical=t[0], specific=t[1])
            if c == "avg" or k == "PaxDb, 8 cell lines":
                print(f"  {NAME.get(c, 'average'):<8} {T} C, {k:<30} n {m.sum():>5}, explains {100*r2:>4.0f}% of input: slope overall {b0[0]:+.2f}; "
                      f"typical part {f3(t[0])}, lysate-specific part {f3(t[1])}", flush=True)
res["typical_vs_specific"] = tab3

# ---------------- 4. load
print("\n4. load: raw intensity per pull-down run and that run's slope on input", flush=True)
df = pd.read_csv(f"{ROOT}/pd_data/GA_33_report_out.tsv", sep="\t", low_memory=False)
assert len(df) == len(acc) and (df["Protein.Group"].astype(str).values == blk["accession_raw"].astype(str)).all()
runs = list(df.columns[8:])
lab = [re.search(r"GA_33_(21A|24)_STAU_(\d+)_T_(\d+)_BR(\d+)", r).groups() for r in runs]
V = df[runs].apply(pd.to_numeric, errors="coerce").values.astype(float)
gn = df["Genes"].astype(str).values
bait = np.array([g in ("DNAJA1", "DNAJB11") for g in gn])
prey = ~con & ~bait
R = []
for j, (c, s, T, br) in enumerate(lab):
    v = V[:, j]; m = np.isfinite(v) & np.isfinite(inp) & prey
    tot = np.log2(np.nansum(2.0 ** v[prey & np.isfinite(v)]))
    R.append(dict(chap=c, stau=s, temp=T, br=br, total=tot, median=float(np.nanmedian(v[prey])), ndet=int((np.isfinite(v) & prey).sum()),
                  bait=float(np.nanmean(v[bait & (gn == ("DNAJA1" if c == "21A" else "DNAJB11"))])), slope=float(np.polyfit(inp[m], v[m], 1)[0])))
R = pd.DataFrame(R)
# input runs, raw, for scale
dI = pd.read_csv(f"{ROOT}/pd_data/GA_33_Proteome_report_out.tsv", sep="\t", low_memory=False)
VI = dI.iloc[:, 8:].apply(pd.to_numeric, errors="coerce").values.astype(float)
conI = np.array([str(p).startswith("contam") for p in dI["Protein.Group"]])
tot_in = np.array([np.log2(np.nansum(2.0 ** VI[~conI & np.isfinite(VI[:, j]), j])) for j in range(VI.shape[1])])
print(f"  input runs: total log2 intensity {np.median(tot_in):.2f} (range {tot_in.min():.2f}-{tot_in.max():.2f}); "
      f"pull-down runs {R.total.median():.2f} ({R.total.min():.2f}-{R.total.max():.2f})", flush=True)
g = R.groupby(["chap", "stau", "temp"]).agg(total=("total", "mean"), slope=("slope", "mean"), slope_sd=("slope", "std"), ndet=("ndet", "mean"), bait=("bait", "mean"))
print(g.round(3).to_string(), flush=True)
R["cond"] = R.chap + "_" + R.stau + "_" + R.temp
rt = R.total - R.groupby("cond").total.transform("mean"); rs = R.slope - R.groupby("cond").slope.transform("mean")
bw = float(np.polyfit(rt, rs, 1)[0]); rw = float(np.corrcoef(rt, rs)[0, 1])
bs = []
for _ in range(2000):
    s = rng.integers(0, len(R), len(R)); bs.append(np.polyfit(rt.values[s], rs.values[s], 1)[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  within condition ({len(R)} runs, condition means removed): load sd {rt.std():.2f} log2; slope per log2 of load {bw:+.3f} [{lo:+.3f}, {hi:+.3f}], r {rw:+.2f}", flush=True)
# between conditions: regress condition slope on load with bait and temperature terms
gg = g.reset_index()
Xb = np.column_stack([one(len(gg)), gg.total, gg.chap == "24", gg.temp == "37", gg.temp == "43"]).astype(float)
wb = np.linalg.lstsq(Xb, gg.slope, rcond=None)[0]
w1 = np.polyfit(gg.total, gg.slope, 1)[0]
print(f"  between conditions: slope per log2 of load {w1:+.3f} alone; {wb[1]:+.3f} with bait and temperature terms "
      f"(43 C term {wb[4]:+.3f}, DNAJB11 term {wb[2]:+.3f})", flush=True)
# the load each run would need, if slope tracked load at the within-condition rate, to explain the 35->43 change
d_slope = g.xs("43", level="temp").slope.mean() - g.xs("35", level="temp").slope.mean()
d_load = g.xs("43", level="temp").total.mean() - g.xs("35", level="temp").total.mean()
print(f"  35 -> 43 C: slope {d_slope:+.3f}, load {d_load:+.2f} log2; within-condition rate predicts {bw*d_load:+.3f}", flush=True)
res["load"] = dict(input_total=float(np.median(tot_in)), ip_total=float(R.total.median()), within=dict(slope=bw, ci=[float(lo), float(hi)], r=rw, load_sd=float(rt.std())),
                   between=dict(alone=float(w1), adjusted=float(wb[1]), t43=float(wb[4]), b24=float(wb[2])), d_slope_43=float(d_slope), d_load_43=float(d_load),
                   table=g.reset_index().to_dict("records"))
R.to_csv(f"{GA}/ad15_runs.tsv", sep="\t", index=False)
json.dump(res, open(f"{GA}/ad15_results.json", "w"), indent=1)
np.savez(f"{GA}/ad15_fig.npz", **{f"curve_{k}": v for k, v in curve.items()})
print("\nAD15_DONE", flush=True)
