"""pd05c: is the baseline-abundance predictor real, or an artefact of sharing a term with y?

The target is y = mean(log2 cond) - mean(log2 base), a log2 fold change. Baseline abundance
scored r ~ 0.40-0.42 as a predictor of it -- but the baseline mean is literally the second term
of y, so any noise in it appears in the feature and (with a minus sign) in the target, which
manufactures correlation on its own. The same worry applies to the detection-floor and
decomposition statistics in pd05b.

The clean test: estimate the FEATURE from one set of biological replicates and the TARGET from
the disjoint remainder. Shared noise then cannot contribute, and whatever correlation survives
is a real dependence of salt sensitivity on expression level. Run both ways round (1,2 | 3,4,5
and 4,5 | 1,2,3) so the answer is not a property of one split.

The ESMC-only model needs no such correction -- the embedding is a function of sequence and
knows nothing about this experiment -- so it also serves as the reference that both the shared
and disjoint targets are scored against.
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
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
emb = np.load(f"{D}/embeddings.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
pos = {str(s): i for i, s in enumerate(emb["accession"])}
MEAN = emb["mean_pool"][np.array([pos[resolved.get(s, s)] for s in acc])]
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])
npep = np.log10(np.maximum(blk["npeptides"], 1))
npep[~np.isfinite(npep)] = np.nanmedian(npep[np.isfinite(npep)])


def oof(X, y, gr, alpha, folds=5):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, gr):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def best(X, y, gr, alphas=(1, 10, 100, 1e3, 1e4, 1e5)):
    o = (-9, None)
    for al in alphas:
        r = stats.pearsonr(y, oof(X, y, gr, al))[0]
        if r > o[0]:
            o = (r, al)
    return o


SPLITS = (("BR1,2 -> BR3,4,5", [0, 1], [2, 3, 4]), ("BR4,5 -> BR1,2,3", [3, 4], [0, 1, 2]))
out = {}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    bb, bc = blk["br_base"], blk[f"br_{cond}"]
    print(f"\n{'='*84}\n{label} vs 0 mM", flush=True)
    rec = {}
    for nm, F, T in SPLITS:
        with np.errstate(invalid="ignore"):
            feat = np.nanmean(bb[:, F], 1)                                  # baseline, held out
            ytgt = np.nanmean(bc[:, T], 1) - np.nanmean(bb[:, T], 1)        # target, disjoint
            shar = np.nanmean(bb[:, T], 1)                                  # baseline, SHARED
        k = (np.isfinite(feat) & np.isfinite(ytgt) & np.isfinite(shar)
             & (np.isfinite(bb[:, F]).sum(1) == len(F))
             & (np.isfinite(bc[:, T]).sum(1) == len(T))
             & (np.isfinite(bb[:, T]).sum(1) == len(T)))
        f1, f0, yk, gk, Mk, nk = feat[k], shar[k], ytgt[k], groups[k], MEAN[k], npep[k]
        r_sh = stats.pearsonr(yk, f0)[0]
        r_dj = stats.pearsonr(yk, f1)[0]
        print(f"\n  {nm}   ({k.sum()} proteins, target sd {yk.std():.3f})", flush=True)
        print(f"    corr(y, baseline) using the SAME replicates as y  {r_sh:+.3f}   <- shares a term", flush=True)
        print(f"    corr(y, baseline) from DISJOINT replicates        {r_dj:+.3f}   "
              f"<- real dependence ({100*abs(r_dj/r_sh):.0f}% of it survives)", flush=True)
        rows = {}
        for mn, X in (("abundance (shared replicates)", np.column_stack([f0, nk])),
                      ("abundance (disjoint replicates)", np.column_stack([f1, nk])),
                      ("ESMC alone", Mk),
                      ("ESMC + abundance (shared)", np.hstack([Mk, np.column_stack([f0, nk])])),
                      ("ESMC + abundance (disjoint)", np.hstack([Mk, np.column_stack([f1, nk])]))):
            r, al = best(X, yk, gk)
            rows[mn] = dict(r=float(r), alpha=float(al))
            print(f"      {mn:34s} r = {r:+.3f}  (alpha {al:g})", flush=True)

        # Baseline abundance is a nuisance variable: it says how much protein was in the
        # pull-down at 0 mM, not how salt-sensitive the association is, so it has no place in
        # the headline model. It cannot just be ignored either, because sequence partly
        # PREDICTS abundance (length, composition, expression propensity), so the ESMC score
        # could be abundance in disguise. These two rows settle that.
        r, al = best(Mk, f1, gk)
        rows["ESMC -> baseline abundance"] = dict(r=float(r), alpha=float(al))
        print(f"      {'ESMC -> baseline abundance':34s} r = {r:+.3f}  (alpha {al:g})"
              f"   <- how much of abundance is sequence-predictable", flush=True)
        A = np.column_stack([f1, nk])
        ra, ala = best(A, yk, gk)
        resid = yk - oof(A, yk, gk, ala)
        r, al = best(Mk, resid, gk)
        rows["ESMC -> abundance-free residual"] = dict(r=float(r), alpha=float(al),
                                                      resid_sd=float(resid.std()))
        print(f"      {'ESMC -> abundance-free residual':34s} r = {r:+.3f}  (alpha {al:g})"
              f"   <- residual sd {resid.std():.3f} = {100*resid.std()/yk.std():.0f}% of y", flush=True)
        rec[nm] = dict(n=int(k.sum()), sd=float(yk.std()), r_shared=float(r_sh),
                       r_disjoint=float(r_dj), models=rows)
    out[cond] = rec

json.dump(out, open(f"{D}/pd05c_results.json", "w"), indent=1)
print(f"\nwrote {D}/pd05c_results.json\nPD05C_DONE", flush=True)
