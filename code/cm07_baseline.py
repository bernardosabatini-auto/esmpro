"""cm07: what protein properties predict how much of a protein an HSP pulls down with no perturbation?

Targets (cm02): the baseline pull-down level of every run (HSPB1: GA20, GA22, GA24, FS73, FS76; DNAJA1 and
DNAJB11: GA33), and the abundance-free versions where an abundance measurement exists: the bound:free
partition (GA22, GA24, same tube) and the enrichment over the input lysate (GA33).

Named properties, each a column a reader can interpret:
  abundance      lysate level measured here (GA33 input, contaminants excluded); PaxDb consensus over 8 cell lines
  stability      Meltome consensus Tm (10 human cell types, as ad01)
  sequence       log10 length, net charge / residue (pH 7.4), pI, GRAVY hydropathy, disorder-promoting fraction
  location       UniProt keywords: Nucleus, Cytoplasm, Mitochondrion, Membrane (transmembrane), Secreted, ER
  function       RNA-binding, Ribonucleoprotein, Ribosomal protein, DNA-binding, Kinase, Chaperone,
                 Ubl conjugation pathway, Coiled coil, Metal-binding; member of a Complex Portal complex
  sequence model the campaign MLP's out-of-fold prediction of the same target (ESMC-6B layers 50 + 80)

1. Univariate: Spearman rho of each property with each target, over proteins measured for both.
2. Blocks: ridge on growing sets of property blocks, R2 out-of-fold on the campaign's grouped folds (penalty
   on the inner split), against the target's ceiling; then the sequence model alone and the sequence model plus
   all named properties. Missing property values are imputed with the training-fold median plus a missingness
   indicator, so no protein is dropped.
"""
import os, re, csv, json, gzip, collections, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
ap = argparse.ArgumentParser(); ap.add_argument("--mlp", default=f"{C}/runs/s2/full"); ap.add_argument("--out", default=f"{C}/cm07_baseline.json")
a = ap.parse_args()
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist(); n = len(U)
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
meta = json.load(open(f"{C}/targets_meta.json")); F = np.load(f"{C}/folds.npz"); outer, inner = F["outer"], F["inner"]
TGT = [t for t in names if meta[t]["kind"] in ("baseline", "enrichment")]

# ---------------------------------------------------------------- named properties
P = pd.DataFrame(index=U)
P["abundance_here"] = Y[:, names.index("GA33_input")]
PX = f"{ROOT}/data/external/paxdb"; e2u = collections.defaultdict(set)
for l in gzip.open(f"{PX}/map.tsv.gz", "rt"):
    p = l.rstrip("\n").split("\t")
    if len(p) >= 3: e2u[p[2]].add(p[1].split("|")[0])
def paxdb(fn):
    d = collections.defaultdict(list)
    for l in open(fn):
        if l[0] == "#": continue
        p = l.split()
        if len(p) >= 2 and float(p[1]) > 0:
            for u in e2u.get(p[0], ()): d[u].append(np.log2(float(p[1])))
    return {u: max(v) for u, v in d.items()}
D = f"{PX}/paxdb-abundance-files-v5.0/9606"
CELL = ["iBAQ_HEK293_Geiger_2012_uniprot", "iBAQ_U2OS_Geiger_2012_uniprot", "iBAQ_Jurkat_Geiger_2012_uniprot", "iBAQ_LnCap_Geiger_2012_uniprot",
        "hela_Nagaraj_2011_iBAQ", "K562_Geiger_2012", "A549_Geiger_2012", "RKO_Geiger_2012"]
vals = collections.defaultdict(list)
for k in CELL:
    d = paxdb(f"{D}/9606-{k}.txt"); med = np.median(list(d.values()))
    for u, v in d.items(): vals[u].append(v - med)
P["abundance_paxdb"] = [np.mean(vals[u]) if len(vals.get(u, [])) >= 2 else np.nan for u in U]
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json")):
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None: per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
acc = collections.defaultdict(list)
for r in HUM:
    rm = {k: np.median(v) for k, v in per[r].items()}; c = np.median(list(rm.values()))
    for k, v in rm.items(): acc[k].append(v - c)
P["Tm"] = [np.median(acc[u]) if u in acc else np.nan for u in U]
S = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).reindex(U)
for c in ("log10_length", "charge_per_res", "pI", "gravy", "disorder_promoting"): P[c] = S[c].values
UA = pd.read_csv(f"{ROOT}/data/annot/uniprot_human_full.tsv.gz", sep="\t", index_col=0, usecols=["Entry", "Keywords"]).reindex(U)
KW = UA["Keywords"].fillna("").str.split(";")
for kw in ["Nucleus", "Cytoplasm", "Mitochondrion", "Transmembrane", "Secreted", "Endoplasmic reticulum", "RNA-binding", "Ribonucleoprotein",
           "Ribosomal protein", "DNA-binding", "Kinase", "Chaperone", "Ubl conjugation pathway", "Coiled coil", "Metal-binding"]:
    P[f"kw_{kw}"] = KW.apply(lambda l: float(kw in l)).values
P.loc[UA["Keywords"].isna().values, [c for c in P if c.startswith("kw_")]] = np.nan
member = set()
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    if len(mem) >= 2: member |= mem
P["in_complex"] = [float(u in member) for u in U]
BLOCKS = {"abundance": ["abundance_here", "abundance_paxdb"], "stability": ["Tm"],
          "sequence": ["log10_length", "charge_per_res", "pI", "gravy", "disorder_promoting"],
          "location": [c for c in P if c in ("kw_Nucleus", "kw_Cytoplasm", "kw_Mitochondrion", "kw_Transmembrane", "kw_Secreted", "kw_Endoplasmic reticulum")],
          "function": [c for c in P if c.startswith("kw_") and c not in ("kw_Nucleus", "kw_Cytoplasm", "kw_Mitochondrion", "kw_Transmembrane", "kw_Secreted", "kw_Endoplasmic reticulum")] + ["in_complex"]}
print({k: int(P[k].notna().sum()) for k in P}, flush=True)
P.to_csv(f"{C}/named_properties.tsv", sep="\t", index_label="accession")

# sequence-model out-of-fold predictions (campaign MLP)
mlp = np.full((n, len(names)), np.nan)
if os.path.exists(f"{a.mlp}_oof.npy"):
    rf = f"{a.mlp}_rows.npy"
    rows = np.load(rf) if os.path.exists(rf) else np.nonzero(F["has_features"] & np.isfinite(Y).any(1))[0]
    mlp[rows] = np.load(f"{a.mlp}_oof.npy")

# ---------------------------------------------------------------- 1. univariate
J = dict(univariate={}, blocks={}, n_with={k: int(P[k].notna().sum()) for k in P})
for t in TGT:
    y = Y[:, names.index(t)]
    J["univariate"][t] = {c: float(spearmanr(P[c], y, nan_policy="omit").correlation) for c in P}

# ---------------------------------------------------------------- 2. blocks, out-of-fold ridge
LAM = np.logspace(-2, 4, 13)


def design(cols, tr):
    X = P[cols].values.astype(float); med = np.nanmedian(X[tr], 0); miss = np.isnan(X)
    X = np.where(miss, med, X); keepm = miss[tr].any(0)
    X = np.hstack([X, miss[:, keepm].astype(float)]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    return (X - mu) / sd


def oof(cols, y, extra=None):
    ok = np.isfinite(y) & (np.isfinite(extra).all(1) if extra is not None else True); pred = np.full(n, np.nan)
    for f in range(5):
        tr = np.nonzero(ok & (outer != f))[0]; te = np.nonzero(ok & (outer == f))[0]; itr, iva = tr[inner[f, tr] > 0], tr[inner[f, tr] == 0]
        X = design(cols, tr) if cols else np.zeros((n, 0))
        if extra is not None: X = np.hstack([X, (extra - extra[tr].mean(0)) / (extra[tr].std(0) + 1e-9)])
        def fit(idx, lam):
            A = X[idx]; ym = y[idx].mean(); return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[idx] - ym)), ym
        err = [((X[iva] @ w + m - y[iva]) ** 2).mean() for w, m in (fit(itr, l) for l in LAM)]
        w, m = fit(tr, LAM[int(np.argmin(err))]); pred[te] = X[te] @ w + m
    k = np.isfinite(pred); return float(1 - ((y[k] - pred[k]) ** 2).sum() / ((y[k] - y[k].mean()) ** 2).sum())


steps = [("abundance", ["abundance"]), ("+ stability", ["abundance", "stability"]), ("+ sequence", ["abundance", "stability", "sequence"]),
         ("+ location", ["abundance", "stability", "sequence", "location"]), ("all named properties", list(BLOCKS)),
         ("named, without abundance", ["stability", "sequence", "location", "function"])]
for t in TGT:
    y = Y[:, names.index(t)]; ti = names.index(t); R = {}
    for lab, bl in steps: R[lab] = oof(sum((BLOCKS[b] for b in bl), []), y)
    if np.isfinite(mlp[:, ti]).any():
        R["sequence model (MLP)"] = oof([], y, mlp[:, [ti]])
        R["sequence model + named properties"] = oof(sum(BLOCKS.values(), []), y, mlp[:, [ti]])
    J["blocks"][t] = {k: dict(r2=v, r2_over_ceiling=v / meta[t]["ceiling"]) for k, v in R.items()}
    print(f"{t:18} " + "  ".join(f"{k}: {v:.3f}" for k, v in R.items()), flush=True)
json.dump(J, open(a.out, "w"), indent=1)
print("CM07_DONE")
