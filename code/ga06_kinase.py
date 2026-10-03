"""ga06: how much of the staurosporine (STAU_10 vs STAU_0) response is "is it a protein kinase"?

Protein kinases are depleted from both co-chaperone pull-downs by 10 uM staurosporine (shift -0.7
to -1.1 sd, kinases 5-6.5x over-represented among the 5 % most depleted), and kinase identity
alone explains 2-4 % of the variance. ESMC explains 5.7-6.5 %. This separates the two: ridge on
the kinase / ATP-binding indicators alone, ESMC alone, both, ESMC on non-kinases only, and ESMC
predicting the response WITHIN the kinases (which kinases go down, which go up). Same folds as
ga05 (scikit-learn GroupKFold over sequence clusters); annotations are UniProt human reviewed
keyword KW-0067 (ATP-binding) and the "Protein kinase" domain feature.
"""
import os, csv, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/ga_data"
rng = np.random.default_rng(0)
b = np.load(f"{D}/blocks.npz", allow_pickle=True)
F = np.load(f"{D}/folds_ga05.npz", allow_pickle=True)
E = np.load(f"{D}/emb_L50.npy")
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in b["accession"]])
gene = np.array([str(g) for g in b["gene"]])
KIN = set(open(f"{D}/uniprot_protein_kinase_dom.txt").read().split())
ATP = set(open(f"{D}/uniprot_atp_binding.txt").read().split())
kin = np.array([a in KIN for a in acc]).astype(float)
atp = (np.array([a in ATP for a in acc]) & (kin == 0)).astype(float)
G = F["groups"]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def cv(X, y, g, alphas, folds=None):
    folds = folds or list(GroupKFold(n_splits=5).split(np.zeros(len(y)), None, g))
    best = (-9e9, None)
    for al in alphas:
        p = np.full(len(y), np.nan)
        for tr, te in folds:
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
        v = r2(y, p)
        if v > best[0]:
            best = (v, p)
    return best


EA = (1e3, 3e3, 1e4, 3e4, 1e5)
out = {}
print(f"{'target':<13}{'indicators':>12}{'ESMC':>8}{'ESMC+ind':>10}{'ESMC, non-kinases':>19}{'ESMC within kinases':>21}{'n kin':>7}")
for t in ("stau_avg_35", "stau_avg_37", "stau_avg_43", "stau_21A_37", "stau_24_37"):
    rows, y, fid = F[f"rows_{t}"], F[f"y_{t}"].astype(float), F[f"fold_{t}"]
    folds = [(np.nonzero(fid != f)[0], np.nonzero(fid == f)[0]) for f in range(5)]
    I = np.column_stack([kin[rows], atp[rows]])
    X = E[rows]
    ri, _ = cv(I, y, None, (1e-3, 1, 10), folds)
    re_, _ = cv(X, y, None, EA, folds)
    rb, _ = cv(np.hstack([X, I * 50]), y, None, EA, folds)          # indicators scaled up so ridge keeps them
    nk = kin[rows] == 0
    rn, _ = cv(X[nk], y[nk], G[rows][nk], EA)
    kk = kin[rows] == 1
    rk, pk = cv(X[kk], y[kk], G[rows][kk], (1e2, 3e2, 1e3, 3e3, 1e4, 3e4))
    out[t] = dict(indicators=ri, esmc=re_, esmc_plus_ind=rb, esmc_nonkinase=rn, esmc_within_kinases=rk,
                  n_kinases=int(kk.sum()), n=int(len(y)))
    print(f"{t:<13}{100*ri:>11.2f}%{100*re_:>7.2f}%{100*rb:>9.2f}%{100*rn:>18.2f}%{100*rk:>20.2f}%{kk.sum():>7}")
json.dump(out, open(f"{D}/ga06_results.json", "w"), indent=1)
print("GA06_DONE")
