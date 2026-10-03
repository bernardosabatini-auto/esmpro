"""ad05b: are the proteins that appear (heat 43 C) or vanish (salt 150 mM) the extremes of the same
sequence-determined response, and what are they?

Events (ad05): heat 43 C, detected in >= 4/6 replicates at 43 C and <= 1/6 at 35 C in BOTH co-chaperone
pull-downs (131 proteins); salt 150 mM, >= 7/10 at 0 mM and <= 2/10 at 150 mM (114).
  1. Generalisation. Ridge on ESMC (final layer for heat, layer 50 for salt) trained on the kept proteins'
     fold changes only; predicted response of the event proteins against kept proteins matched on the
     intensity of the side where they are detected (five nearest kept proteins each). AUC.
  2. Classification. Event vs intensity-matched kept proteins, logistic regression on ESMC, folds grouped
     by sequence cluster; against intensity alone.
  3. What they are: UniProt keywords, compartments, Meltome Tm, SAE disorder/membrane features.
"""
import os, csv, json, collections, re
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, PD, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/pd_data", f"{ROOT}/data/sae"
rng = np.random.default_rng(0)
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


gb, gacc = resolved(GA); pb, pacc = resolved(PD)
cl_g = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{GA}/clusters.tsv"), delimiter="\t")}
cl_p = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{PD}/clusters.tsv"), delimiter="\t")}
E80, E50g = np.load(f"{GA}/emb_L80.npy"), np.load(f"{GA}/emb_L50.npy")
L = np.load(f"{PD}/embeddings_layers.npz", allow_pickle=True)
fa = [l[1:].strip().split()[0].split("|")[0] for l in open(f"{PD}/sequences.fasta") if l[0] == ">"]
ppos = {a: i for i, a in enumerate(fa)}
E50p = L["pool"][np.array([ppos[a] for a in pacc])][:, [int(x) for x in L["layers"]].index(50)]
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)

# ---- define events and the training sets
n = lambda c, s, t: np.isfinite(gb[f"br_{c}_{s}_{t}"]).sum(1)
heat_ev = np.all([(n(c, "0", "43") >= 4) & (n(c, "0", "35") <= 1) for c in ("21A", "24")], 0) & np.isfinite(E80).all(1)
heat_I = np.nanmean(np.hstack([gb[f"br_{c}_0_43"] for c in ("21A", "24")]), 1)
heat_rows = FG["rows_temp_avg_43v35"]; heat_y = FG["y_temp_avg_43v35"].astype(float)
n1 = np.isfinite(pb["raw_s150"]).sum(1); n0 = np.isfinite(pb["raw_base"]).sum(1)
salt_ev = (n0 >= 7) & (n1 <= 2)
salt_I = np.nanmean(pb["raw_base"], 1)
salt_keep = pb["keep_s150"].astype(bool) & np.isfinite(pb["y_s150"])
SETS = {"heat 43 appears": dict(E=E80, ev=heat_ev, I=heat_I, rows=heat_rows, y=heat_y, acc=gacc, cl=cl_g, sign=+1),
        "salt 150 vanishes": dict(E=E50p, ev=salt_ev, I=salt_I, rows=np.nonzero(salt_keep)[0], y=pb["y_s150"].astype(float)[salt_keep],
                                  acc=pacc, cl=cl_p, sign=-1)}
out = {}
for nm, d in SETS.items():
    E, ev, I, rows, y, acc, cl = d["E"], d["ev"], d["I"], d["rows"], d["y"], d["acc"], d["cl"]
    evi = np.nonzero(ev)[0]
    # matched controls: kept proteins with nearest intensity on the detected side
    kI = I[rows]
    ctrl = []
    for i in evi:
        o = np.argsort(np.abs(kI - I[i]))
        ctrl += [rows[j] for j in o[:5] if rows[j] not in ctrl][:5]
    ctrl = np.array(sorted(set(ctrl)))
    # 1. generalisation: ridge trained on kept proteins, excluding the matched controls' clusters? no -
    #    controls are scored out of fold: fit on folds that exclude each control's cluster
    G = np.array([cl.get(a, a) for a in acc[rows]])
    pk = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), None, G):
        pk[te] = make_pipeline(StandardScaler(), Ridge(alpha=1e4)).fit(E[rows[tr]], y[tr]).predict(E[rows[te]])
    model = make_pipeline(StandardScaler(), Ridge(alpha=1e4)).fit(E[rows], y)
    ev_clusters = {cl.get(a, a) for a in acc[evi]}
    shared = np.isin(G, list(ev_clusters))
    model_ex = make_pipeline(StandardScaler(), Ridge(alpha=1e4)).fit(E[rows[~shared]], y[~shared])   # no relative of any event protein
    pe = model_ex.predict(E[evi])
    pos = {r: k for k, r in enumerate(rows)}
    pc = np.array([pk[pos[r]] for r in ctrl])
    lab = np.r_[np.ones(len(pe)), np.zeros(len(pc))]
    auc = roc_auc_score(lab, d["sign"] * np.r_[pe, pc])
    # bootstrap AUC
    bs = []
    for _ in range(1000):
        a_ = rng.choice(len(pe), len(pe)); b_ = rng.choice(len(pc), len(pc))
        bs.append(roc_auc_score(lab, d["sign"] * np.r_[pe[a_], pc[b_]]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rec = dict(n_events=int(len(evi)), n_controls=int(len(ctrl)), mean_pred_events=float(pe.mean()), mean_pred_controls=float(pc.mean()),
               mean_obs_controls=float(np.mean([y[pos[r]] for r in ctrl])), generalisation_auc=[float(auc), float(lo), float(hi)],
               intensity_events=float(np.nanmedian(I[evi])), intensity_controls=float(np.nanmedian(I[ctrl])))
    print(f"\n{nm}: {len(evi)} events, {len(ctrl)} intensity-matched kept proteins (median intensity {rec['intensity_events']:.2f} vs {rec['intensity_controls']:.2f})", flush=True)
    print(f"  1. sequence model trained on kept proteins only (no relative of any event protein in training):", flush=True)
    print(f"     predicted response, events {pe.mean():+.3f} vs matched kept {pc.mean():+.3f} (their observed {rec['mean_obs_controls']:+.3f}); "
          f"AUC {auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    # 2. direct classification, grouped CV, vs intensity alone
    allx = np.r_[evi, ctrl]; lab2 = np.r_[np.ones(len(evi)), np.zeros(len(ctrl))]
    G2 = np.array([cl.get(a, a) for a in acc[allx]])
    p_seq = np.zeros(len(allx)); p_int = np.zeros(len(allx))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(allx)), None, G2):
        p_seq[te] = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000)).fit(E[allx[tr]], lab2[tr]).predict_proba(E[allx[te]])[:, 1]
        Xi = I[allx][:, None]
        p_int[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xi[tr], lab2[tr]).predict_proba(Xi[te])[:, 1]
    rec["classification_auc_seq"] = float(roc_auc_score(lab2, p_seq)); rec["classification_auc_intensity"] = float(roc_auc_score(lab2, p_int))
    print(f"  2. classifying events vs matched kept proteins, cluster-grouped folds: sequence AUC {rec['classification_auc_seq']:.3f}, "
          f"intensity alone {rec['classification_auc_intensity']:.3f}", flush=True)
    # 3. what they are
    kw = lambda a: set(k.strip() for k in ann.get(a, {}).get("Keywords", "").split(";") if k.strip())
    ek = collections.Counter(k for a in acc[evi] for k in kw(a)); ck = collections.Counter(k for a in acc[ctrl] for k in kw(a))
    enr = []
    for k in set(ek) | set(ck):
        a_, b_ = ek[k], ck[k]
        if a_ + b_ >= 8:
            o, p = stats.fisher_exact([[a_, len(evi) - a_], [b_, len(ctrl) - b_]])
            enr.append((p, k, a_, b_, o))
    enr.sort()
    rec["keywords"] = [dict(keyword=k, events=a_, controls=b_, odds=float(o), p=float(p)) for p, k, a_, b_, o in enr[:15]]
    print("  3. keywords vs matched kept proteins: " + "; ".join(f"{k} {a_}/{len(evi)} vs {b_}/{len(ctrl)} (OR {o:.1f}, p {p:.0e})" for p, k, a_, b_, o in enr[:8]), flush=True)
    genes = [ann.get(a, {}).get("Gene Names (primary)", a) for a in acc[evi]]
    rec["genes"] = genes
    print(f"     examples: {', '.join(genes[:30])}", flush=True)
    out[nm] = rec

# Meltome Tm for the heat events
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
dm = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in dm:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
rm = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
cen = {r: float(np.median(list(rm[r].values()))) for r in HUM}; grand = float(np.median(list(cen.values())))
LL = collections.defaultdict(list)
for r in HUM:
    for a, t in rm[r].items(): LL[a].append(t - cen[r])
TM = {a: float(np.median(v)) + grand for a, v in LL.items()}
evi = np.nonzero(heat_ev)[0]; kept_all = heat_rows
te = np.array([TM.get(a, np.nan) for a in gacc[evi]]); tk = np.array([TM.get(a, np.nan) for a in gacc[kept_all]])
print(f"\nheat 43 appearing proteins: Tm available {np.isfinite(te).sum()}/{len(te)}; median Tm {np.nanmedian(te):.1f} C vs kept {np.nanmedian(tk):.1f} C "
      f"(Mann-Whitney p {stats.mannwhitneyu(te[np.isfinite(te)], tk[np.isfinite(tk)]).pvalue:.0e})", flush=True)
out["heat 43 appears"]["tm"] = dict(n=int(np.isfinite(te).sum()), median=float(np.nanmedian(te)), median_kept=float(np.nanmedian(tk)))
json.dump(out, open(f"{GA}/ad05b_results.json", "w"), indent=1)
print("\nAD05B_DONE")
