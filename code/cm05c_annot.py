"""cm05c: annotated table and figure of the SAE features most associated with each target family (cm05b).

For each listed feature: Biohub label and category (LLM-generated hypotheses from activation patterns across
millions of proteins, pd14; reported as such), its Spearman rho with the family score, and an independent label
check: the UniProt keywords most over-represented among the campaign proteins on which the feature is active
(activation above Biohub's threshold, else the top 5 %), Fisher exact test against all campaign proteins.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"; OUT = f"{C}/attrib"
FIG = f"{ROOT}/reports/figures/campaign"; os.makedirs(FIG, exist_ok=True)
J = json.load(open(f"{OUT}/assoc.json"))
ANN = {}
for l in open(f"{ROOT}/data/sae/feature_annot.jsonl"):
    d = json.loads(l); ANN[d["feature_index"]] = d
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist()
has = np.load(f"{C}/folds.npz")["has_features"]; X = np.load(f"{C}/sae_max.npy", mmap_mode="r")
KW = pd.read_csv(f"{ROOT}/data/annot/uniprot_human_full.tsv.gz", sep="\t", index_col=0, usecols=["Entry", "Keywords"]).reindex(U)["Keywords"].fillna("").str.split(";")
allkw = pd.Series([k for l in KW[has] for k in l if k]).value_counts(); allkw = allkw[allkw >= 20]
kwsets = {k: np.array([k in l for l in KW]) for k in allkw.index}


def check(i):
    x = X[:, i].astype(np.float32); thr = ANN.get(i, {}).get("threshold")
    act = (x > thr) if thr else (x >= np.quantile(x[has], .95)); act &= has; n1 = act.sum()
    out = []
    for k, m in kwsets.items():
        a = int((act & m).sum())
        if a < 5: continue
        b = int((~act & has & m).sum()); odds, p = fisher_exact([[a, n1 - a], [b, has.sum() - n1 - b]], "greater")
        out.append((p, k, a / max(n1, 1)))
    out.sort(); return int(n1), [f"{k} ({f:.0%})" for p, k, f in out[:2] if p < 1e-4]


rows = []
for fam, d in J.items():
    for sign in ("positive", "negative"):
        for r in d[sign][:8]:
            i = r["feature"]; a = ANN.get(i, {}); n1, kws = check(i)
            rows.append(dict(family=fam, direction="higher" if sign == "positive" else "lower", feature=i, rho=round(r["rho"], 3),
                             label=a.get("label") or "(no annotation)", category=a.get("category") or "", n_active=n1, keyword_check="; ".join(kws)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/top_features_annotated.tsv", sep="\t", index=False)
print(T.to_string(max_colwidth=60))

fams = list(J); f, ax = plt.subplots(len(fams), 1, figsize=(11, 3.1 * len(fams)))
for a_, fam in zip(ax, fams):
    t = T[T.family == fam].sort_values("rho")
    a_.barh(range(len(t)), t["rho"], color=[("#c0392b" if v > 0 else "#1f3b73") for v in t["rho"]])
    a_.set_yticks(range(len(t))); a_.set_yticklabels([f"{r.feature}: {r.label[:55]}" for r in t.itertuples()], fontsize=6.5)
    a_.axvline(0, color="k", lw=.5); a_.set_title(f"{fam} (stability between protein halves r = {J[fam]['halves_r']:.2f})", fontsize=9); a_.set_xlabel("Spearman rho with the family score", fontsize=8)
f.tight_layout(); f.savefig(f"{FIG}/sae_features.png", dpi=140); plt.close(f)
print("CM05C_DONE")
