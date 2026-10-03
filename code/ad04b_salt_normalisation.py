"""ad04b: the same normalisation check for the salt titration (pd_data).

The workbook arrived as log2 intensities, processed upstream, and has been used as delivered. Here:
dynamic range per run and per salt block, and the 150 and 75 mM fold changes rebuilt under median,
scale and quantile normalisation of the 30 runs, with the reliability, the dependence on baseline
abundance, the ESMC (layer 50) ridge R2 on cluster folds, and the transmembrane effect.
"""
import os, csv, json
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
PD = f"{ROOT}/data/pd_data"
blk = np.load(f"{PD}/blocks.npz", allow_pickle=True)
RAW = np.hstack([blk["raw_base"], blk["raw_s75"], blk["raw_s150"]]).astype(float)       # 30 runs
BLOCK = ["base"] * 10 + ["s75"] * 10 + ["s150"] * 10
det = np.isfinite(RAW).mean(1) >= 0.9
print("1. DYNAMIC RANGE (proteins detected in >= 90 % of runs)")
for b in ("base", "s75", "s150"):
    js = [j for j, x in enumerate(BLOCK) if x == b]
    print(f"  {b:<5} median {np.nanmedian(RAW[det][:, js]):.2f}  mean IQR {np.mean([np.subtract(*np.nanpercentile(RAW[det, j], [75, 25])) for j in js]):.3f}"
          f"  run medians {np.nanmin(np.nanmedian(RAW[det][:, js], 0)):.2f}-{np.nanmax(np.nanmedian(RAW[det][:, js], 0)):.2f}")


def norm(kind):
    X = RAW.copy()
    if kind == "as delivered":
        return X
    if kind == "median":
        return X - np.nanmedian(X[det] - np.nanmean(X[det], 1, keepdims=True), 0)
    if kind == "scale":
        med = np.nanmedian(X[det], 0); iqr = np.subtract(*np.nanpercentile(X[det], [75, 25], 0))
        return (X - med) / iqr * np.mean(iqr)
    ps = np.linspace(0, 1, 2001); ref = np.mean([np.nanquantile(X[:, j], ps) for j in range(30)], 0)
    out = np.full_like(X, np.nan)
    for j in range(30):
        ok = np.isfinite(X[:, j]); r = stats.rankdata(X[ok, j]); out[ok, j] = np.interp((r - .5) / ok.sum(), ps, ref)
    return out


def brmeans(Xb, cols=range(5)):
    pair = np.nanmean(np.stack([Xb[:, 0:5], Xb[:, 5:10]]), 0)
    return pair[:, list(cols)]


st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{PD}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
FP = np.load(f"{PD}/folds.npz", allow_pickle=True)
L = np.load(f"{PD}/embeddings_layers.npz", allow_pickle=True)
fa = [l[1:].strip().split()[0].split("|")[0] for l in open(f"{PD}/sequences.fasta") if l[0] == ">"]
pos = {a: i for i, a in enumerate(fa)}; li = [int(x) for x in L["layers"]].index(50)
E = L["pool"][np.array([pos[a] for a in acc]), li]
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
TM = np.array(["Transmembrane" in ann.get(a, {}).get("Keywords", "") for a in acc])
print("\n2. FOLD CHANGES UNDER EACH NORMALISATION")
ref = {}; out = {}
for kind in ("as delivered", "median", "scale", "quantile"):
    X = norm(kind); out[kind] = {}
    for c in ("s150", "s75"):
        b = brmeans(X[:, :10]); s = brmeans(X[:, 10:20] if c == "s75" else X[:, 20:30])
        with np.errstate(invalid="ignore"):
            y = np.nanmean(s, 1) - np.nanmean(b, 1); base = np.nanmean(b, 1)
        rows, fid = FP[f"rows_{c}"], FP[f"fold_{c}"]
        yk, bk = y[rows], base[rows]
        rs = []
        for A_, B_ in (([0, 1], [3, 4]), ([0, 2], [1, 3]), ([1, 4], [0, 3])):
            with np.errstate(invalid="ignore"):
                ya = (np.nanmean(s[:, A_], 1) - np.nanmean(b[:, A_], 1))[rows]; yb = (np.nanmean(s[:, B_], 1) - np.nanmean(b[:, B_], 1))[rows]
            ok = np.isfinite(ya) & np.isfinite(yb); rs.append(np.corrcoef(ya[ok], yb[ok])[0, 1])
        rel = 2.5 * np.mean(rs) / (1 + 1.5 * np.mean(rs))
        if kind == "as delivered":
            ref[c] = yk
        p = np.zeros(len(yk))
        for f in range(5):
            tr, te = fid != f, fid == f
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=1e4)).fit(E[rows][tr], yk[tr]).predict(E[rows][te])
        r2 = 1 - np.sum((yk - p) ** 2) / np.sum((yk - yk.mean()) ** 2)
        ok = np.isfinite(bk)
        tm = TM[rows]
        out[kind][c] = dict(r_vs_delivered=float(np.corrcoef(yk, ref[c])[0, 1]), reliability=float(rel),
                            rho_abundance=float(stats.spearmanr(yk[ok], bk[ok])[0]), esmc_r2=float(r2), tm=float(yk[tm].mean() - yk[~tm].mean()))
        print(f"  {kind:<13} {c:<5} r vs as delivered {out[kind][c]['r_vs_delivered']:+.3f}  rel {rel:.3f}  rho(abundance) {out[kind][c]['rho_abundance']:+.3f}  "
              f"ESMC L50 {100*r2:5.1f}%  TM {out[kind][c]['tm']:+.3f}")
json.dump(out, open(f"{PD}/ad04b_results.json", "w"), indent=1)
print("\nAD04B_DONE")
