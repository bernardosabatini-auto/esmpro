"""pd15: what do the SAE features that predict each response mean, and do the labels hold up?

For each response:
  1. Distinct top features. Features ranked by |correlation| with the response (log1p max-pooled
     activation over all proteins, pd13); a feature is skipped when its activation pattern correlates
     above 0.7 with one already listed, so near-duplicates appear once, with the number they stand for.
  2. Label check. For each listed feature, the UniProt keywords most over-represented among the
     proteins in THIS dataset on which it is active (above Biohub's own threshold), against all
     proteins. The Biohub label is an LLM-generated hypothesis; the keywords are an independent read.
  3. Category enrichment. The Biohub categories of the 100 strongest features, positive and negative
     separately, against 500 random features (Fisher's exact test).
"""
import os, csv, json, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
PD, GA, SD = f"{ROOT}/data/pd_data", f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
R = json.load(open(f"{SD}/pd13_results.json"))
A = {}
for line in open(f"{SD}/feature_annot.jsonl"):
    d = json.loads(line); A[d["feature_index"]] = d
base = [int(x) for x in open(f"{SD}/features_baseline_sample.txt").read().split()]
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True)
spos = {str(a): i for i, a in enumerate(S["accession"])}
MAX = S["max"]
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
kw_of = {a: set(k.strip() for k in r.get("Keywords", "").split(";") if k.strip()) for a, r in ann.items()}
gene_of = {a: r.get("Gene Names (primary)", "") for a, r in ann.items()}


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


pblk, pacc = resolved(PD); gblk, gacc = resolved(GA)
FP = np.load(f"{PD}/folds.npz", allow_pickle=True); FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
T = {"salt_75": (pacc[FP["rows_s75"]], pblk["y_s75"].astype(float)[FP["rows_s75"]]),
     "salt_150": (pacc[FP["rows_s150"]], pblk["y_s150"].astype(float)[FP["rows_s150"]])}
for t in ("temp_avg_43v35", "temp_avg_37v35", "stau_avg_35", "stau_avg_37", "stau_avg_43"):
    T[t] = (gacc[FG[f"rows_{t}"]], FG[f"y_{t}"].astype(float))

out = {}
for t in ("salt_150", "temp_avg_43v35", "stau_avg_37", "salt_75", "temp_avg_37v35", "stau_avg_43"):
    accs, y = T[t]
    ix = np.array([spos[a] for a in accs])
    top = R[t]["top_features"]
    Xf = {d["feature"]: np.log1p(MAX[ix, d["feature"]].astype(np.float64)) for d in top[:150]}
    picked, absorbed = [], collections.Counter()
    for d in top[:150]:
        f = d["feature"]; v = Xf[f]
        dup = None
        for p in picked:
            if np.std(v) > 0 and np.std(Xf[p["feature"]]) > 0 and abs(np.corrcoef(v, Xf[p["feature"]])[0, 1]) > 0.7:
                dup = p["feature"]; break
        if dup is not None:
            absorbed[dup] += 1; continue
        if len(picked) < 12:
            picked.append(d)
    print(f"\n{'='*118}\n{t}: {R[t]['n_fdr05']} features at FDR < 0.05; distinct top features "
          f"(r > 0: proteins carrying the feature rise relative to control)", flush=True)
    rows = []
    for d in picked:
        f = d["feature"]; a = A.get(f, {})
        thr = a.get("threshold") or 0.0
        act = MAX[ix, f].astype(np.float32) > thr
        # independent label check: UniProt keywords enriched among active proteins in this dataset
        kws = collections.Counter(k for x in accs[act] for k in kw_of.get(x, ()))
        allk = collections.Counter(k for x in accs for k in kw_of.get(x, ()))
        enr = []
        for k, c in kws.items():
            if c >= 5:
                lor = np.log((c + .5) / (act.sum() - c + .5)) - np.log((allk[k] + .5) / (len(accs) - allk[k] + .5))
                enr.append((lor, k, c))
        enr.sort(reverse=True)
        top_prot = [gene_of.get(accs[i], accs[i]) for i in np.argsort(-MAX[ix, f].astype(np.float32))[:4]]
        row = dict(feature=f, r=d["r"], q=d["q"], n_active=int(act.sum()), mean_y_active=float(y[act].mean()) if act.any() else None,
                   mean_y_other=float(y[~act].mean()), stands_for=1 + absorbed[f], label=a.get("label"), category=a.get("category"),
                   summary=a.get("summary"), uniprot_keywords=[k for _, k, _ in enr[:4]], top_proteins=top_prot)
        rows.append(row)
        print(f"  {f:>5} r {d['r']:+.3f} (x{1+absorbed[f]:<2}) n {act.sum():>4}  y {row['mean_y_active']:+.2f} vs {row['mean_y_other']:+.2f}"
              f" | {str(a.get('category'))[:22]:<22} | {str(a.get('label'))[:52]:<52}", flush=True)
        print(f"        UniProt keywords among active: {', '.join(row['uniprot_keywords']) or '-'};  top: {', '.join(top_prot)}", flush=True)
    # category enrichment
    cb = collections.Counter(A[f]["category"] for f in base if f in A and A[f].get("category"))
    nb = sum(cb.values())
    enr_out = {}
    for sign, sel in (("positive", [d["feature"] for d in top if d["r"] > 0][:100]),
                      ("negative", [d["feature"] for d in top if d["r"] < 0][:100])):
        ct = collections.Counter(A[f]["category"] for f in sel if f in A and A[f].get("category"))
        ns = sum(ct.values())
        res = []
        for c in set(ct) | set(cb):
            o, p = stats.fisher_exact([[ct[c], ns - ct[c]], [cb[c], nb - cb[c]]])
            res.append((p, c, ct[c], ns, cb[c], nb, o))
        res.sort()
        enr_out[sign] = [dict(category=c, n=k, of=ns, base=b, base_of=bn, odds=float(o), p=float(p)) for p, c, k, ns, b, bn, o in res]
        sig = [r for r in res if r[0] < 0.05]
        print(f"  categories, {sign} features ({ns}): " + ("; ".join(f"{c} {k}/{ns} vs {b}/{bn} (OR {o:.1f}, p {p:.0e})" for p, c, k, ns, b, bn, o in sig) or "no category differs from random (p >= 0.05)"), flush=True)
    out[t] = dict(top=rows, categories=enr_out, n_fdr05=R[t]["n_fdr05"])
json.dump(out, open(f"{SD}/pd15_results.json", "w"), indent=1)
print("\nPD15_DONE")
