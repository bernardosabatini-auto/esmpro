"""pd16: within the protein kinases, which SAE features separate the kinases staurosporine depletes
from those it enriches?

Kinases as a class fall with staurosporine; the open question is direction within the class, which
the dense embedding barely predicts (1-3.5 % of the variance within kinases). Among the ~300
kinases kept for each staurosporine target: correlation of every SAE feature (log1p max-pooled)
with the response, BH q over the features active in at least 10 kinases, and the distinct strongest
features each way. Writes the features that need annotations.
"""
import os, csv, json
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True)
spos = {str(a): i for i, a in enumerate(S["accession"])}
MAX = S["max"]
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
KIN = set(open(f"{GA}/uniprot_protein_kinase_dom.txt").read().split())
ann = {r["Entry"]: r.get("Gene Names (primary)", "") for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
out, want = {}, set()
for t in ("stau_avg_37", "stau_avg_43", "stau_avg_35"):
    a_ = acc[FG[f"rows_{t}"]]; y = FG[f"y_{t}"].astype(float)
    k = np.array([x in KIN for x in a_])
    ak, yk = a_[k], y[k]
    X = np.log1p(MAX[np.array([spos[x] for x in ak])].astype(np.float64))
    act = (X > 0).sum(0)
    ok = act >= 10
    Xo = X[:, ok] - X[:, ok].mean(0)
    yc = yk - yk.mean()
    r = (Xo.T @ yc) / (np.sqrt((Xo ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    tt = r * np.sqrt((len(yk) - 2) / np.maximum(1 - r ** 2, 1e-12))
    p = 2 * stats.t.sf(np.abs(tt), len(yk) - 2)
    o = np.argsort(p); q = np.empty_like(p); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    fidx = np.nonzero(ok)[0]
    res = {}
    for sign in (1, -1):
        order = np.argsort(-sign * r)
        picked = []
        for j in order[:200]:
            if any(abs(np.corrcoef(Xo[:, j], Xo[:, i])[0, 1]) > 0.7 for i in picked):
                continue
            picked.append(j)
            if len(picked) == 6:
                break
        res["up" if sign > 0 else "down"] = [dict(feature=int(fidx[j]), r=float(r[j]), q=float(q[j]), n_active=int(act[fidx[j]]),
                                                  top=[ann.get(ak[i], ak[i]) for i in np.argsort(-X[:, fidx[j]])[:6]]) for j in picked]
        want |= {int(fidx[j]) for j in picked}
    out[t] = dict(n_kinases=int(k.sum()), n_tested=int(ok.sum()), n_fdr05=int((q < 0.05).sum()), n_fdr20=int((q < 0.2).sum()), **res)
    print(f"{t}: {k.sum()} kinases, {ok.sum()} features tested, FDR<0.05: {(q<0.05).sum()}, FDR<0.2: {(q<0.2).sum()}", flush=True)
json.dump(out, open(f"{SD}/pd16_results.json", "w"), indent=1)
open(f"{SD}/features_kinase.txt", "w").write("\n".join(map(str, sorted(want))) + "\n")
P = json.load(open(f"{SD}/pd15_results.json"))
miss = {d["feature"] for v in P.values() for d in v["top"] if d["label"] is None}
open(f"{SD}/features_missing.txt", "w").write("\n".join(map(str, sorted(miss))) + "\n")
print(f"{len(want)} within-kinase features, {len(miss)} unlabelled top features to fetch\nPD16_DONE")
