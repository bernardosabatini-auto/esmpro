"""ad03: is there one axis shared by every response, and what is left of each response without it?

Membrane proteins rise with salt in the HSPB1 pull-down, prefer DNAJB11, and fall with heat;
disordered, charged proteins do the opposite. On the proteins measured in both experiments:
  1. correlations between the eight reproducible responses, raw and corrected for each response's
     measurement noise (r / sqrt(reliability_i * reliability_j));
  2. principal components of the standardised responses;
  3. a sequence-only axis: mean SAE activation of the strongest membrane features minus that of the
     strongest disorder features (features chosen from their Biohub category and label, not from the
     responses), and its correlation with each response;
  4. for each response, the part not shared with the others (residual after regressing it on the
     first principal component of the OTHER responses) and the part not explained by the sequence
     axis: how much variance remains, and how much of it the ESMC embedding still predicts
     (ridge, folds grouped by sequence cluster).
"""
import os, csv, json
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
PD, GA, SD = f"{ROOT}/data/pd_data", f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


pblk, pacc = resolved(PD); gblk, gacc = resolved(GA)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True); SP = np.load(f"{GA}/spec.npz", allow_pickle=True)
R = {}
for c, nm in (("s75", "salt 75"), ("s150", "salt 150")):
    k = pblk[f"keep_{c}"].astype(bool) & np.isfinite(pblk[f"y_{c}"])
    R[nm] = dict(zip(pacc[k], pblk[f"y_{c}"][k].astype(float)))
for t, nm in (("temp_avg_37v35", "heat 37"), ("temp_avg_43v35", "heat 43"), ("stau_avg_35", "stau 35"),
              ("stau_avg_37", "stau 37"), ("stau_avg_43", "stau 43")):
    R[nm] = dict(zip(gacc[FG[f"rows_{t}"]], FG[f"y_{t}"].astype(float)))
R["DNAJB11 pref"] = dict(zip(gacc[SP["rows_spec_avg"]], SP["y_spec_avg"].astype(float)))
ga5 = json.load(open(f"{GA}/ga05_results.json")); g12 = json.load(open(f"{GA}/ga12_summary.json"))
REL = {"salt 75": 0.986, "salt 150": 0.993, "heat 37": ga5["temp_avg_37v35"]["ceiling"], "heat 43": ga5["temp_avg_43v35"]["ceiling"],
       "stau 35": ga5["stau_avg_35"]["ceiling"], "stau 37": ga5["stau_avg_37"]["ceiling"], "stau 43": ga5["stau_avg_43"]["ceiling"],
       "DNAJB11 pref": g12["spec_avg"]["reliability"]}
names = list(R)
common = sorted(set.intersection(*[set(v) for v in R.values()]))
Y = np.array([[R[n][a] for n in names] for a in common])
Z = (Y - Y.mean(0)) / Y.std(0)
print(f"{len(common)} proteins measured in all eight responses", flush=True)

C = np.corrcoef(Z.T)
Cc = C / np.sqrt(np.outer([REL[n] for n in names], [REL[n] for n in names])); np.fill_diagonal(Cc, 1)
print("\n1. CORRELATIONS (upper: raw; lower: corrected for measurement noise)", flush=True)
print(" " * 13 + "".join(f"{n:>13}" for n in names), flush=True)
for i, n in enumerate(names):
    print(f"{n:<13}" + "".join(f"{(C[i,j] if j > i else (Cc[i,j] if j < i else 1)):>+13.2f}" for j in range(len(names))), flush=True)

w, V = np.linalg.eigh(C); o = np.argsort(-w); w, V = w[o], V[:, o]
print("\n2. PRINCIPAL COMPONENTS of the standardised responses", flush=True)
for k in range(3):
    s = np.sign(V[names.index("salt 150"), k]) or 1
    print(f"  PC{k+1}: {100*w[k]/w.sum():4.1f}% of variance; loadings " + ", ".join(f"{n} {s*V[i,k]:+.2f}" for i, n in enumerate(names)), flush=True)

# 3. sequence-only axis from SAE features chosen by category/label
A = {}
for l in open(f"{SD}/feature_annot.jsonl"):
    d = json.loads(l); A[d["feature_index"]] = d
memf = [f for f, d in A.items() if d.get("category") == "Membrane-associated" and d.get("label") and "transmembrane" in d["label"].lower()]
disf = [f for f, d in A.items() if d.get("category") in ("Disorder", "Compositional bias") and d.get("label") and
        any(w_ in d["label"].lower() for w_ in ("idr", "disorder", "low-complexity", "low complexity"))]
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True); spos = {str(a): i for i, a in enumerate(S["accession"])}
ix = np.array([spos[a] for a in common])
mem_s = np.log1p(S["max"][ix][:, memf].astype(np.float64)).mean(1)
dis_s = np.log1p(S["max"][ix][:, disf].astype(np.float64)).mean(1)
axis = (mem_s - mem_s.mean()) / mem_s.std() - (dis_s - dis_s.mean()) / dis_s.std()
pc1 = Z @ V[:, 0] * (np.sign(V[names.index("salt 150"), 0]) or 1)
print(f"\n3. SEQUENCE-ONLY AXIS: {len(memf)} transmembrane features minus {len(disf)} disorder/low-complexity features "
      f"(chosen by Biohub category and label among the {len(A)} annotated)", flush=True)
print(f"  correlation with PC1 of the responses: {np.corrcoef(axis, pc1)[0,1]:+.3f}", flush=True)
print("  correlation with each response: " + ", ".join(f"{n} {np.corrcoef(axis, Z[:, i])[0,1]:+.2f}" for i, n in enumerate(names)), flush=True)

# 4. residuals and their predictability
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{GA}/clusters.tsv"), delimiter="\t")}
G = np.array([cl.get(a, a) for a in common])
E = np.load(f"{GA}/emb_L50.npy"); gpos = {a: i for i, a in enumerate(gacc)}
X = E[np.array([gpos[a] for a in common])]
folds = list(GroupKFold(n_splits=5).split(X, None, G))


def ridge_r2(y):
    best = -9
    for al in (1e3, 1e4, 1e5):
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
        best = max(best, 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return best


print(f"\n4. WHAT IS LEFT OF EACH RESPONSE (ESMC layer 50 ridge, cluster-grouped folds, same {len(common)} proteins)", flush=True)
print(f"  {'response':<13}{'ESMC on response':>17}{'shared with others':>20}{'left':>7}{'ESMC on what is left':>22}{'seq. axis share':>17}{'ESMC, axis removed':>20}", flush=True)
res = {}
for i, n in enumerate(names):
    others = [j for j in range(len(names)) if j != i]
    Co = np.corrcoef(Z[:, others].T); wo, Vo = np.linalg.eigh(Co); v1 = Vo[:, -1]
    pco = Z[:, others] @ v1
    b = np.polyfit(pco, Z[:, i], 1); resid = Z[:, i] - np.polyval(b, pco)
    shared = 1 - resid.var() / Z[:, i].var()
    b2 = np.polyfit(axis, Z[:, i], 1); resid2 = Z[:, i] - np.polyval(b2, axis)
    r_full, r_res, r_res2 = ridge_r2(Z[:, i]), ridge_r2(resid), ridge_r2(resid2)
    res[n] = dict(esmc=r_full, shared_with_others=float(shared), esmc_on_residual=r_res, axis_share=float(1 - resid2.var() / Z[:, i].var()),
                  esmc_axis_removed=r_res2)
    print(f"  {n:<13}{100*r_full:>16.1f}%{100*shared:>19.1f}%{100*(1-shared):>6.0f}%{100*r_res:>21.1f}%{100*res[n]['axis_share']:>16.1f}%{100*r_res2:>19.1f}%", flush=True)
json.dump(dict(n=len(common), names=names, corr=C.tolist(), corr_corrected=Cc.tolist(), pc_var=(w / w.sum()).tolist(),
               pc_loadings=V[:, :3].tolist(), n_mem_features=len(memf), n_dis_features=len(disf),
               axis_pc1=float(np.corrcoef(axis, pc1)[0, 1]),
               axis_corr={n: float(np.corrcoef(axis, Z[:, i])[0, 1]) for i, n in enumerate(names)}, residuals=res),
          open(f"{ROOT}/data/ga_data/ad03_results.json", "w"), indent=1)
print("\nAD03_DONE", flush=True)
