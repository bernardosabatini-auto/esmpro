"""ad01b: is the inverted-U relation between melting temperature and the heat response real?

Checks: (a) the same curve from HEK293T and from K562 Tm alone; (b) within protein classes defined
without Tm (UniProt location: nuclear-only, cytoplasmic, membrane) and after adjusting for the SAE
disorder/membrane features; (c) detection: do the least stable proteins drop out of the 43 C
pull-downs (more missing replicates), as aggregation and loss from the soluble lysate would predict;
(d) the 37 vs 35 C response; (e) what the proteins without a Meltome value are.
"""
import os, csv, json, collections, re
import numpy as np
from scipy import stats
ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
d = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in d:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
rm = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
cen = {r: float(np.median(list(rm[r].values()))) for r in HUM}; grand = float(np.median(list(cen.values())))
acc_l = collections.defaultdict(list)
for r in HUM:
    for a, t in rm[r].items(): acc_l[a].append(t - cen[r])
TM = {a: float(np.median(v)) + grand for a, v in acc_l.items()}
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True); spos = {str(a): i for i, a in enumerate(S["accession"])}


def curve(tm, y, nb=10):
    ok = np.isfinite(tm); q = np.percentile(tm[ok], np.linspace(0, 100, nb + 1))
    b = np.clip(np.digitize(tm, q[1:-1]), 0, nb - 1)
    return [float(y[ok & (b == i)].mean()) for i in range(nb)], q


def fmt(c):
    return " ".join(f"{v:+.2f}" for v in c)


out = {}
for t in ("temp_avg_43v35", "temp_avg_37v35"):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]
    tm = np.array([TM.get(a, np.nan) for a in a_])
    c, q = curve(tm, y)
    print(f"\n{t}: response by Tm decile (all ten cell lines): {fmt(c)}", flush=True)
    for r in ("HEK293T", "K562", "Jurkat"):
        tr = np.array([rm[r].get(a, np.nan) for a in a_])
        cr, _ = curve(tr, y)
        print(f"  Tm from {r:<8} only ({np.isfinite(tr).sum()} proteins):     {fmt(cr)}", flush=True)
    out[t] = dict(all=c, edges=q.tolist())
    if t != "temp_avg_43v35":
        continue
    # (b) within classes
    loc = [ann.get(a, {}).get("Subcellular location [CC]", "") for a in a_]
    kw = [ann.get(a, {}).get("Keywords", "") for a in a_]
    mem = np.array(["Transmembrane" in k for k in kw])
    nuc = np.array([("Nucleus" in l) and ("Cytoplasm" not in l) for l in loc])
    cyt = np.array([("Cytoplasm" in l) and ("Nucleus" not in l) and not m for l, m in zip(loc, mem)])
    for nm, m in (("membrane (transmembrane keyword)", mem), ("nuclear only", nuc), ("cytoplasmic, not nuclear, not membrane", cyt)):
        cm, _ = curve(np.where(m, tm, np.nan), y, nb=5)
        print(f"  within {nm:<40} ({(m & np.isfinite(tm)).sum()}), Tm quintiles: {fmt(cm)}", flush=True)
    ix = np.array([spos[a] for a in a_])
    F = np.column_stack([np.log1p(S["max"][ix, f].astype(np.float64)) for f in (654, 10715, 5659)])
    ok = np.isfinite(tm)
    A = np.column_stack([np.ones(ok.sum()), F[ok]]); res = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
    cr, _ = curve(tm[ok], res)
    print(f"  after regressing out the disorder and membrane SAE features:      {fmt(cr)}", flush=True)
    # (c) detection by Tm decile: missing replicates at 43 vs 35 C in the control pull-downs
    miss = {}
    for T in ("35", "43"):
        miss[T] = np.mean([np.isnan(blk[f"br_{c}_0_{T}"][rows]).sum(1) for c in ("21A", "24")], 0)
    allrows = np.arange(len(acc))
    tm_all = np.array([TM.get(a, np.nan) for a in acc])
    det = {T: np.mean([np.isfinite(blk[f"br_{c}_0_{T}"]).sum(1) >= 4 for c in ("21A", "24")], 0) for T in ("35", "43")}
    q_all = np.percentile(tm_all[np.isfinite(tm_all)], np.linspace(0, 100, 11))
    b_all = np.clip(np.digitize(tm_all, q_all[1:-1]), 0, 9)
    lines = []
    for i in range(10):
        m = np.isfinite(tm_all) & (b_all == i) & ((det["35"] > 0) | (det["43"] > 0))
        lines.append((float(det["35"][m].mean()), float(det["43"][m].mean())))
    print("  detected (>= 4 of 6) at 35 / 43 C, by Tm decile, all proteins: " + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in lines), flush=True)
    out["detection"] = lines
    # (e) proteins without Tm
    noT = ~np.isfinite(tm)
    kwc = collections.Counter(k.strip() for k, h in zip(kw, noT) if h for k in k.split(";"))
    allc = collections.Counter(k.strip() for k in kw for k in k.split(";"))
    enr = sorted(((np.log((kwc[k] + .5) / (noT.sum() + 1)) - np.log((allc[k] + .5) / (len(kw) + 1)), k, kwc[k]) for k in kwc if kwc[k] >= 20), reverse=True)[:8]
    L = np.array([np.log10(max(len(ann.get(a, {}).get("Entry", "x")), 1)) for a in a_])
    print(f"  no Tm ({noT.sum()}): over-represented keywords: " + ", ".join(f"{k} ({n})" for _, k, n in enr), flush=True)
    dis = np.log1p(S["max"][ix, 654].astype(np.float64)) > 0.5
    print(f"  no Tm: membrane {mem[noT].mean():.2f} vs {mem[~noT].mean():.2f}; disorder feature active {dis[noT].mean():.2f} vs {dis[~noT].mean():.2f}", flush=True)
json.dump(out, open(f"{ROOT}/data/external/meltome/ad01b_results.json", "w"), indent=1)
print("\nAD01B_DONE")

# ---- shape statistics with cluster-bootstrap intervals (appended)
rng = np.random.default_rng(0)
print("\nSHAPE STATISTICS (standardised Tm; 95 % intervals resampling sequence clusters)", flush=True)
shape = {}
for t in ("temp_avg_43v35", "temp_avg_37v35"):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]; G = FG["groups"][rows]
    tm = np.array([TM.get(a, np.nan) for a in a_]); ok = np.isfinite(tm)
    ix = np.array([spos[a] for a in a_])
    Fc = np.column_stack([np.log1p(S["max"][ix, f].astype(np.float64)) for f in (654, 10715, 5659)])
    z = (tm - np.nanmean(tm)) / np.nanstd(tm)
    ug, inv = np.unique(G[ok], return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    for adj in (False, True):
        X = np.column_stack([np.ones(ok.sum()), z[ok], z[ok] ** 2] + ([Fc[ok]] if adj else []))
        yy = y[ok]
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        bs = []
        for _ in range(1000):
            s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))])
            bs.append(np.linalg.lstsq(X[s], yy[s], rcond=None)[0][:3])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        peak = 51.6 + (-b[1] / (2 * b[2])) * np.nanstd(tm) if b[2] < 0 else float("nan")
        key = f"{t}{' adjusted' if adj else ''}"
        shape[key] = dict(linear=[float(b[1]), float(lo[1]), float(hi[1])], quadratic=[float(b[2]), float(lo[2]), float(hi[2])], peak_tm=float(peak))
        print(f"  {key:<28} linear {b[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]  quadratic {b[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]"
              + (f"  peak near {peak:.1f} C" if b[2] < 0 else ""), flush=True)
json.dump(shape, open(f"{ROOT}/data/external/meltome/ad01_shape.json", "w"), indent=1)
