"""pd08: how much of the fold change is amino-acid composition, and what does ESMC add?

A nested decomposition, each level a superset of the last, scored the same way throughout:
ridge, penalty over a fixed grid, 5-fold cross-validation with folds grouped by MMseqs2 cluster
at 30 % identity. The reported quantity is predictive R-squared on held-out predictions
(1 - SSE/SS_tot), i.e. fraction of variance explained, since that is what "how much does this
feature set explain" means.

  length                      1-d   how big the protein is
  composition                20-d   amino-acid fractions, nothing else
  composition + length       21-d
  physicochemical            31-d   + net charge, charge per residue, K/R and D/E fractions,
                                    GRAVY, aromaticity, disorder-prone fraction, charge
                                    segregation, molecular weight
  ESMC-6B mean-pool        2560-d   the most complex representation used
  ESMC + composition       2580-d   does the embedding already contain composition?

Increments are reported with a paired bootstrap over proteins on the out-of-fold predictions, so
"ESMC adds X" carries an interval rather than being a difference of two point estimates.
"""
import os, csv, json
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/pd_data"
rng = np.random.default_rng(0)
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip("AVLIPFMWGSTCYNQDEKRH",
              [1.8, 4.2, 3.8, 4.5, -1.6, 2.8, 1.9, -0.9, -0.4, -0.8,
               -0.7, 2.5, -1.3, -3.5, -3.5, -3.5, -3.5, -3.9, -4.5, -3.2]))

blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
emb = np.load(f"{D}/embeddings.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
pos = {str(s): i for i, s in enumerate(emb["accession"])}
MEAN = emb["mean_pool"][np.array([pos[resolved.get(s, s)] for s in acc])]
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])

seq = {}
cur = None
for ln in open(f"{D}/sequences.fasta"):
    if ln[0] == ">":
        cur = ln[1:].strip().split()[0].split("|")[0]
        seq[cur] = ""
    else:
        seq[cur] += ln.strip()
S = [seq.get(resolved.get(a, a), "") for a in acc]

comp = np.array([[s.count(c) / max(len(s), 1) for c in AA] for s in S])
ln_len = np.log10(np.array([max(len(s), 1) for s in S]))[:, None]
chg = np.array([[(s.count("K") + s.count("R") - s.count("D") - s.count("E")),
                 (s.count("K") + s.count("R") - s.count("D") - s.count("E")) / max(len(s), 1),
                 (s.count("K") + s.count("R")) / max(len(s), 1),
                 (s.count("D") + s.count("E")) / max(len(s), 1),
                 (s.count("K") + s.count("R") + s.count("D") + s.count("E")) / max(len(s), 1),
                 np.mean([KD.get(c, 0) for c in s]) if s else 0.0,
                 (s.count("F") + s.count("W") + s.count("Y")) / max(len(s), 1),
                 sum(s.count(c) for c in "PESTQKRG") / max(len(s), 1),
                 np.std([1 if c in "KR" else (-1 if c in "DE" else 0)
                         for c in s[:: max(1, len(s) // 200)]]) if s else 0.0,
                 len(s) * 110.0 / 1000.0] for s in S])

FEATS = {"length": ln_len,
         "composition": comp,
         "composition + length": np.hstack([comp, ln_len]),
         "physicochemical": np.hstack([comp, ln_len, chg]),
         "ESMC-6B mean-pool": MEAN,
         "ESMC + composition": np.hstack([MEAN, comp])}


def oof(X, y, gr, alpha):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, gr):
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr]).predict(X[te])
    return p


def best(X, y, gr, alphas=(1, 10, 100, 1e3, 1e4, 1e5)):
    o = (-9e9, None, None)
    for al in alphas:
        p = oof(X, y, gr, al)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        if r2 > o[0]:
            o = (float(r2), al, p)
    return o


def r2_of(y, p, idx):
    yy, pp = y[idx], p[idx]
    return 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)


out = {}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    y = blk[f"y_{cond}"].astype(float)
    k = blk[f"keep_{cond}"].astype(bool) & np.isfinite(y)
    yk, gk = y[k], groups[k]
    print(f"\n{'='*92}\n{label} vs 0 mM   n={k.sum()}   sd(y)={yk.std():.3f} log2 units", flush=True)
    print(f"  {'feature set':<24}{'dim':>6}{'R2':>9}{'var expl':>10}{'increment over previous':>26}", flush=True)
    preds, rows, prev = {}, {}, None
    for nm, X in FEATS.items():
        r2, al, p = best(X[k], yk, gk)
        preds[nm] = p
        inc = ""
        if prev is not None:
            idx = np.arange(len(yk))
            d = [r2_of(yk, p, s) - r2_of(yk, preds[prev], s)
                 for s in (rng.choice(idx, len(idx), replace=True) for _ in range(2000))]
            inc = f"{r2 - rows[prev]['r2']:+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        rows[nm] = dict(dim=int(X.shape[1]), r2=r2, alpha=float(al),
                        r=float(stats.pearsonr(yk, p)[0]), increment=inc)
        print(f"  {nm:<24}{X.shape[1]:>6}{r2:>+9.3f}{100*r2:>9.1f}%{inc:>26}", flush=True)
        prev = nm

    # the two orderings that answer the question directly
    base = rows["composition"]["r2"]
    full = rows["ESMC-6B mean-pool"]["r2"]
    idx = np.arange(len(yk))
    d = [r2_of(yk, preds["ESMC-6B mean-pool"], s) - r2_of(yk, preds["composition"], s)
         for s in (rng.choice(idx, len(idx), replace=True) for _ in range(2000))]
    print(f"\n  composition alone explains            {100*base:.1f}% of the variance", flush=True)
    print(f"  the embedding explains                {100*full:.1f}%", flush=True)
    print(f"  so the embedding adds                 {100*(full-base):+.1f} points "
          f"[{100*np.percentile(d,2.5):+.1f},{100*np.percentile(d,97.5):+.1f}]  "
          f"= {full/max(base,1e-9):.1f}x composition", flush=True)
    d2 = [r2_of(yk, preds["ESMC + composition"], s) - r2_of(yk, preds["ESMC-6B mean-pool"], s)
          for s in (rng.choice(idx, len(idx), replace=True) for _ in range(2000))]
    print(f"  composition added ON TOP of the embedding adds "
          f"{100*(rows['ESMC + composition']['r2']-full):+.1f} points "
          f"[{100*np.percentile(d2,2.5):+.1f},{100*np.percentile(d2,97.5):+.1f}]  "
          f"<- near zero means the embedding already contains composition", flush=True)
    out[cond] = dict(n=int(k.sum()), sd=float(yk.std()), rows=rows,
                     composition_r2=base, esmc_r2=full,
                     esmc_gain=float(full - base),
                     esmc_gain_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     comp_on_top=float(rows["ESMC + composition"]["r2"] - full),
                     comp_on_top_ci=[float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))])

json.dump(out, open(f"{D}/pd08_results.json", "w"), indent=1)
print(f"\nwrote {D}/pd08_results.json\nPD08_DONE", flush=True)
