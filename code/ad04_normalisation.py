"""ad04: which GA_33 conclusions depend on how the runs are normalised?

Per-run constant normalisations (none, median, bait) differ only by one offset per run, so they leave
across-protein comparisons unchanged except where values are missing. What can matter is a
difference in dynamic range between runs: if the 43 C runs compress the intensity scale, abundant
proteins appear to fall and scarce ones to rise, which would mimic the heat response's dependence on
abundance. Two normalisations remove such differences: scale (each run centred and scaled to a
common interquartile range) and quantile (each run mapped onto a common intensity distribution).

1. Dynamic range per sample group, and the slope of 43 C on 35 C intensities.
2. Key targets rebuilt under five normalisations (none, median, bait, scale, quantile), and for each:
   reliability, correlation with the median-normalised target, dependence on abundance, ESMC ridge R2
   (final layer for heat and preference, layer 50 for staurosporine; cluster-grouped folds),
   transmembrane and disorder effects on heat, the kinase shift under staurosporine, the signal-
   peptide and nuclear effects on DNAJB11 preference.
"""
import os, re, csv, itertools, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
df = pd.read_csv(f"{ROOT}/pd_data/GA_33_report_out.tsv", sep="\t", low_memory=False)
runs = list(df.columns[8:]); pat = re.compile(r"GA_33_(21A|24)_STAU_(\d+)_T_(\d+)_BR(\d+)_TR\d+_.*_(\d+)$")
lab = [pat.search(c).groups() for c in runs]
V = df[runs].apply(pd.to_numeric, errors="coerce").values.astype(float)
gene = df["Genes"].astype(str).values
det = np.isfinite(V).mean(1) >= 0.9


def norm(kind):
    X = V.copy()
    if kind == "none":
        return X
    if kind == "median":
        return X - np.nanmedian(X[det] - np.nanmean(X[det], 1, keepdims=True), 0)
    if kind == "bait":
        out = X.copy()
        for j, l in enumerate(lab):
            b = np.nonzero(gene == ("DNAJA1" if l[0] == "21A" else "DNAJB11"))[0][0]
            out[:, j] = X[:, j] - X[b, j]
        return out
    if kind == "scale":
        med = np.nanmedian(X[det], 0); iqr = np.subtract(*np.nanpercentile(X[det], [75, 25], 0))
        return (X - med) / iqr * np.mean(iqr)
    if kind == "quantile":
        ps = np.linspace(0, 1, 2001)
        ref = np.mean([np.nanquantile(X[:, j], ps) for j in range(X.shape[1])], 0)
        out = np.full_like(X, np.nan)
        for j in range(X.shape[1]):
            v = X[:, j]; ok = np.isfinite(v)
            r = stats.rankdata(v[ok]); out[ok, j] = np.interp((r - 0.5) / ok.sum(), ps, ref)
        return out


def groups(X):
    G = {}
    for c, s, t in itertools.product(("21A", "24"), ("0", "10"), ("35", "37", "43")):
        A = np.full((len(X), 6), np.nan)
        for b in range(1, 7):
            cols = [j for j, l in enumerate(lab) if l[:4] == (c, s, t, str(b))]
            with np.errstate(invalid="ignore"):
                A[:, b - 1] = np.nanmean(X[:, cols], 1)
        G[(c, s, t)] = A
    return G


def terms(name, G):
    if name == "heat 43":
        return [(+.5, G[(c, "0", "43")]) for c in ("21A", "24")] + [(-.5, G[(c, "0", "35")]) for c in ("21A", "24")]
    if name == "heat 37":
        return [(+.5, G[(c, "0", "37")]) for c in ("21A", "24")] + [(-.5, G[(c, "0", "35")]) for c in ("21A", "24")]
    if name == "stau 37":
        return [(+.5, G[(c, "10", "37")]) for c in ("21A", "24")] + [(-.5, G[(c, "0", "37")]) for c in ("21A", "24")]
    return [(sg / 3, G[(c, "0", t)]) for t in ("35", "37", "43") for c, sg in (("24", 1), ("21A", -1))]


def build(name, G, cols=None):
    T = terms(name, G)
    keep = np.all([np.isfinite(m).sum(1) >= 4 for _, m in T], 0)
    with np.errstate(invalid="ignore"):
        y = sum(s * np.nanmean(m if cols is None else m[:, cols], 1) for s, m in T)
    return y, keep & np.isfinite(y)


# 1. dynamic range
print("1. DYNAMIC RANGE (raw log2 intensities, proteins detected in >= 90 % of runs)", flush=True)
for c in ("21A", "24"):
    for t in ("35", "37", "43"):
        js = [j for j, l in enumerate(lab) if l[0] == c and l[1] == "0" and l[2] == t]
        iqr = np.mean([np.subtract(*np.nanpercentile(V[det, j], [75, 25])) for j in js])
        sd = np.mean([np.nanstd(V[det, j]) for j in js])
        print(f"  {c:>3} control {t} C: median {np.nanmedian(V[det][:, js]):.2f}  IQR {iqr:.3f}  sd {sd:.3f}", flush=True)
m35 = np.nanmean(V[:, [j for j, l in enumerate(lab) if l[1] == "0" and l[2] == "35"]], 1)
m43 = np.nanmean(V[:, [j for j, l in enumerate(lab) if l[1] == "0" and l[2] == "43"]], 1)
ok = det & np.isfinite(m35) & np.isfinite(m43)
sl = np.polyfit(m35[ok], m43[ok], 1)[0]; sl_tls = np.std(m43[ok]) / np.std(m35[ok]) * np.sign(np.corrcoef(m35[ok], m43[ok])[0, 1])
print(f"  43 C on 35 C intensities (controls): least-squares slope {sl:.3f}; sd ratio {sl_tls:.3f} (1 = no compression)", flush=True)

# context for the protein-level tests
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(a.split(";")[0], a.split(";")[0]) for a in df["Protein.Group"].astype(str)])
has = np.array([st.get(a.split(";")[0], {"x": 1}) is not None and a.split(";")[0] in st and True for a in df["Protein.Group"].astype(str)])
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{GA}/clusters.tsv"), delimiter="\t")}
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
E = {50: np.load(f"{GA}/emb_L50.npy"), 80: np.load(f"{GA}/emb_L80.npy")}
emb_ok = np.isfinite(E[50]).all(1)
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True); spos = {str(a): i for i, a in enumerate(S["accession"])}
TM = np.array(["Transmembrane" in ann.get(a, {}).get("Keywords", "") for a in acc])
SIG = np.array(["Signal" in ann.get(a, {}).get("Keywords", "").split(";") or "Signal" in [k.strip() for k in ann.get(a, {}).get("Keywords", "").split(";")] for a in acc])
NUC = np.array([("Nucleus" in ann.get(a, {}).get("Subcellular location [CC]", "")) and ("Cytoplasm" not in ann.get(a, {}).get("Subcellular location [CC]", "")) for a in acc])
KN = np.array([a in KIN for a in acc])
F654 = S["max"][:, 654].astype(np.float64)           # read the array once: indexing an npz re-reads it
DIS = np.array([np.log1p(F654[spos[a]]) if a in spos else np.nan for a in acc])
bait = np.isin(gene, ["DNAJA1", "DNAJB11"])


def ridge_r2(X, y, G):
    fid = np.full(len(y), -1)
    for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), None, G)):
        fid[te] = f
        # one penalty, the value the earlier fits selected for these targets: the question here is whether
    # results change with normalisation, not how well each fit is tuned
    best = -9
    for al in (1e4,):
        p = np.zeros(len(y))
        for f in range(5):
            tr, te = fid != f, fid == f
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
        best = max(best, 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return best


print("\n2. KEY RESULTS UNDER EACH NORMALISATION", flush=True)
out = {"slope_43_on_35": float(sl), "sd_ratio": float(sl_tls)}
ref = {}
for kind in ("median", "none", "bait", "scale", "quantile"):
    Xn = norm(kind); G = groups(Xn)
    out[kind] = {}
    for name in ("heat 43", "heat 37", "stau 37", "DNAJB11 pref"):
        y, k = build(name, G)
        k &= emb_ok & ~bait
        rs = []
        for tri in itertools.combinations(range(6), 3):
            if 0 not in tri: continue
            o_ = [i for i in range(6) if i not in tri]
            a_, _ = build(name, G, list(tri)); b_, _ = build(name, G, o_)
            m = k & np.isfinite(a_) & np.isfinite(b_); rs.append(np.corrcoef(a_[m], b_[m])[0, 1])
        rel = 2 * np.mean(rs) / (1 + np.mean(rs))
        if kind == "median":
            ref[name] = (y, k)
        ry, kr = ref[name]; m = k & kr
        r_ref = np.corrcoef(y[m], ry[m])[0, 1]
        T = terms(name, G); base = [mm for s, mm in T if s < 0][0] if name != "DNAJB11 pref" else None
        with np.errstate(invalid="ignore"):
            ab = np.nanmean(np.hstack([mm for s, mm in T if s < 0]), 1) if name != "DNAJB11 pref" else np.nanmean(np.hstack([mm for _, mm in T]), 1)
        ok_ = k & np.isfinite(ab)
        r_ab = stats.spearmanr(y[ok_], ab[ok_])[0]
        L = 50 if name.startswith("stau") else 80
        rows = np.nonzero(k)[0]
        r2 = ridge_r2(E[L][rows], y[rows], np.array([cl.get(a, a) for a in acc[rows]]))
        yk = y[k]
        eff = {}
        if name.startswith("heat"):
            eff["TM"] = float(yk[TM[k]].mean() - yk[~TM[k]].mean())
            okd = np.isfinite(DIS[k]); eff["disorder r"] = float(np.corrcoef(DIS[k][okd], yk[okd])[0, 1])
        if name.startswith("stau"):
            eff["kinase shift (sd)"] = float((yk[KN[k]].mean() - yk[~KN[k]].mean()) / yk.std())
        if name.startswith("DNAJB11"):
            eff["signal"] = float(yk[SIG[k]].mean() - yk[~SIG[k]].mean()); eff["nucleus"] = float(yk[NUC[k]].mean() - yk[~NUC[k]].mean())
        out[kind][name] = dict(n=int(k.sum()), reliability=float(rel), r_with_median=float(r_ref), rho_abundance=float(r_ab), esmc_r2=float(r2), **eff)
        print(f"  {kind:<9} {name:<13} n {k.sum():>5}  rel {rel:.3f}  r vs median {r_ref:+.3f}  rho(abund) {r_ab:+.3f}  ESMC L{L} {100*r2:5.1f}%  "
              + "  ".join(f"{a} {b:+.3f}" for a, b in eff.items()), flush=True)
json.dump(out, open(f"{GA}/ad04_results.json", "w"), indent=1)
print("\nAD04_DONE", flush=True)
