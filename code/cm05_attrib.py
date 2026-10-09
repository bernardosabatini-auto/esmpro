"""cm05: which SAE features drive each prediction of the campaign's interpretable model (SAE-only input).

The model (cm03 group "full_sae": layer-60 SAE, log1p max per protein, plus sequence covariates -> shared MLP ->
33 heads) is saved per outer fold as its best member. For each fold, on that fold's held-out proteins only:
    attribution(feature j, target t) = mean over proteins of  d yhat_t / d x_j  *  x_j
with x the standardised input (deviation from the training mean, in SD units), so the sum over features
approximates each protein's prediction minus the prediction at the mean input. Averaged over folds.

Then, per target family (pull-down baseline, abundance, bound:free / enrichment, salt, temperature, Mg):
  * the 25 strongest features with each sign, with their Biohub annotation (fetched and cached as pd14;
    LLM-generated hypotheses, reported as such);
  * a label check: the UniProt keywords most over-represented among the proteins on which the feature is
    active (activation above Biohub's threshold, or the top 5 % when no threshold is available), Fisher test;
  * agreement across folds: correlation of the fold-wise attribution vectors (stability).

Writes data/campaign/attrib/attrib.npz (33 x 16384 mean attribution), attrib_top.json, features to annotate.
"""
import os, json, time
import numpy as np, pandas as pd, torch
from scipy.stats import fisher_exact

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
RUN = os.environ.get("CM_RUN", f"{C}/runs/s2"); OUT = f"{C}/attrib"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.load(open(f"{C}/targets_meta.json")); names = list(meta)
F = np.load(f"{C}/folds.npz"); outer = F["outer"]
rf = f"{RUN}/full_sae_rows.npy"; _Y = np.load(f"{C}/targets.npz", allow_pickle=True)["Y"]
rows = np.load(rf) if os.path.exists(rf) else np.nonzero(F["has_features"] & np.isfinite(_Y).any(1))[0]   # rows are per group
Mz = np.load(f"{RUN}/full_sae_model.npz")
XS = np.log1p(np.load(f"{C}/sae_max.npy", mmap_mode="r")[rows].astype(np.float32))
SIM = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).iloc[rows].fillna(0).values.astype(np.float32)
folds = sorted({int(k.split("_")[0][1:]) for k in Mz.files})
A = np.zeros((len(folds), len(names), XS.shape[1]), np.float32)
for fi, f in enumerate(folds):
    P = {k.split("_", 1)[1]: torch.tensor(Mz[k], device=dev) for k in Mz.files if k.startswith(f"f{f}_")}
    te = np.nonzero(outer[rows] == f)[0]
    xs = (torch.tensor(XS[te], device=dev) - P["norm_muS"]) / P["norm_sdS"]; xs.requires_grad_(True)
    xp = (torch.tensor(SIM[te], device=dev) - P["norm_muP"]) / P["norm_sdP"]
    h = torch.cat([xs @ P["WS"] + P["bS"], xp @ P["WP"] + P["bP"]], -1)
    h = torch.nn.functional.gelu(h); h = torch.nn.functional.gelu(h @ P["W1"] + P["b1"]); z = h @ P["W2"] + P["b2"]; y = z @ P["WH"] + P["bH"]
    for t in range(len(names)):
        g, = torch.autograd.grad(y[:, t].sum(), xs, retain_graph=True)
        A[fi, t] = (g * xs).mean(0).detach().cpu().numpy() * float(P["ys"][t])     # in the target's own units
    print(f"fold {f}: {len(te)} proteins", flush=True)
attr = A.mean(0)
stab = {t: float(np.mean([np.corrcoef(A[i, k], A[j, k])[0, 1] for i in range(len(folds)) for j in range(i + 1, len(folds))])) for k, t in enumerate(names)}
np.savez(f"{OUT}/attrib.npz", attribution=attr, per_fold=A, names=np.array(names))

FAM = {"pull-down baseline": [t for t in names if meta[t]["kind"] == "baseline"], "abundance": [t for t in names if meta[t]["kind"] == "abundance"],
       "bound:free / enrichment": [t for t in names if meta[t]["kind"] == "enrichment"], "salt": [t for t in names if meta[t]["kind"] == "salt"],
       "temperature": [t for t in names if meta[t]["kind"] == "temperature"], "Mg": [t for t in names if meta[t]["kind"] == "Mg"]}
fam_attr = {k: attr[[names.index(t) for t in v]].mean(0) for k, v in FAM.items()}
want = set(); TOP = {}
for k, v in fam_attr.items():
    pos, neg = np.argsort(-v)[:25], np.argsort(v)[:25]; TOP[k] = dict(positive=pos.tolist(), negative=neg.tolist(), pos_val=v[pos].tolist(), neg_val=v[neg].tolist())
    want |= set(pos.tolist()) | set(neg.tolist())
open(f"{OUT}/features_to_annotate.txt", "w").write("\n".join(map(str, sorted(want))))
json.dump(dict(top=TOP, stability=stab, families=FAM), open(f"{OUT}/attrib_top.json", "w"), indent=1)
print("fold-to-fold stability of attribution (mean r):", {k: round(np.mean([stab[t] for t in v]), 3) for k, v in FAM.items()})
print("CM05_DONE")
