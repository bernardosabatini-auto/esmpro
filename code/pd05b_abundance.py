"""pd05b: why does baseline abundance predict the fold change at r = 0.81, and does the
sequence embedding add anything on top of it?

pd05 found abundance + peptide count (r ~ 0.81) far ahead of the ESMC embedding (r ~ 0.40).
That ordering needs explaining before either number can be reported, because the target
is y = mean(condition) - mean(baseline) and the feature IS the second term. Three checks:

  1. Decompose. How much of y is explained by mean_baseline alone, and is the relationship
     the near-identity that sharing a term would produce?
  2. Floor effect. A pull-down has a detection limit. A protein starting high can fall a long
     way; one starting near the floor cannot. If so, y is partly an artefact of where each
     protein starts, not of how salt-sensitive it is.
  3. Incremental value. Does the embedding improve on abundance when both are given to the
     model, and what does it predict about the part of y that abundance cannot explain?

Only the third question is about sequence, and it is the one the project actually asked.
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

blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
emb = np.load(f"{D}/embeddings.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
pos = {str(s): i for i, s in enumerate(emb["accession"])}
MEAN = emb["mean_pool"][np.array([pos[resolved.get(s, s)] for s in acc])]
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])

with np.errstate(invalid="ignore"):
    base = np.nanmean(blk["br_base"], 1)


def oof(X, y, gr, alpha, folds=5):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, gr):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def best_r(X, y, gr, alphas=(1, 10, 100, 1e3, 1e4, 1e5)):
    out = (-9, None, None)
    for al in alphas:
        p = oof(X, y, gr, al)
        r = stats.pearsonr(y, p)[0]
        if r > out[0]:
            out = (r, al, p)
    return out


def ci(y, p, n=2000):
    i = np.arange(len(y))
    bs = [stats.pearsonr(y[s], p[s])[0] for s in (rng.choice(i, len(i), replace=True) for _ in range(n))]
    return np.percentile(bs, [2.5, 97.5])


out = {}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    y = blk[f"y_{cond}"].astype(float)
    with np.errstate(invalid="ignore"):
        cmean = np.nanmean(blk[f"br_{cond}"], 1)
    k = blk[f"keep_{cond}"].astype(bool) & np.isfinite(y) & np.isfinite(base) & np.isfinite(cmean)
    yk, bk, ck, gk, Mk = y[k], base[k], cmean[k], groups[k], MEAN[k]
    print(f"\n{'='*78}\n{label} vs 0 mM — {k.sum()} proteins", flush=True)

    # 1. decomposition
    r_yb = stats.pearsonr(yk, bk)[0]
    r_yc = stats.pearsonr(yk, ck)[0]
    r_bc = stats.pearsonr(bk, ck)[0]
    sl, ic = np.polyfit(bk, yk, 1)
    print(f"\n  1. WHERE THE TARGET COMES FROM", flush=True)
    print(f"     corr(y, baseline)          {r_yb:+.3f}   slope {sl:+.3f}  (a shared term predicts slope -1)", flush=True)
    print(f"     corr(y, condition mean)    {r_yc:+.3f}", flush=True)
    print(f"     corr(baseline, condition)  {r_bc:+.3f}", flush=True)
    print(f"     sd(baseline) {bk.std():.3f}   sd(condition) {ck.std():.3f}   sd(y) {yk.std():.3f}", flush=True)
    print(f"     -> the condition block has {'LESS' if ck.std() < bk.std() else 'MORE'} dynamic range than the "
          f"baseline, so y is largely {'minus the baseline' if ck.std() < bk.std() else 'the condition level'}", flush=True)

    # 2. floor effect: does the drop depend on how much room there was to fall?
    lo, hi = np.percentile(bk, [25, 75])
    print(f"\n  2. DETECTION-FLOOR CHECK", flush=True)
    for nm, m in (("lowest quartile of baseline", bk <= lo), ("highest quartile", bk >= hi)):
        print(f"     {nm:28s} mean y {yk[m].mean():+.3f}  sd {yk[m].std():.3f}  "
              f"min {yk[m].min():+.2f}", flush=True)
    nz = 10 - np.isfinite(blk[f"raw_{cond}"]).sum(1)
    print(f"     proteins with >=1 undetected replicate in this condition: {(nz[k] > 0).sum()}; "
          f"their mean y {yk[nz[k] > 0].mean():+.3f} vs {yk[nz[k] == 0].mean():+.3f} for fully detected", flush=True)

    # 3. does sequence add anything to abundance?
    A = bk[:, None]
    print(f"\n  3. DOES SEQUENCE ADD TO ABUNDANCE?", flush=True)
    rows = {}
    for nm, X in (("baseline abundance only", A),
                  ("ESMC only", Mk),
                  ("abundance + ESMC", np.hstack([A, Mk]))):
        r, al, p = best_r(X, yk, gk)
        c = ci(yk, p)
        rows[nm] = dict(r=float(r), lo=float(c[0]), hi=float(c[1]), alpha=float(al))
        print(f"     {nm:26s} r = {r:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]  (alpha {al:g})", flush=True)
    d = rows["abundance + ESMC"]["r"] - rows["baseline abundance only"]["r"]
    print(f"     -> adding 2,560 sequence features to abundance changes r by {d:+.4f}", flush=True)

    # the part of y abundance cannot explain, predicted from sequence alone
    resid = yk - oof(A, yk, gk, rows["baseline abundance only"]["alpha"])
    r, al, p = best_r(Mk, resid, gk)
    c = ci(resid, p)
    rows["ESMC on abundance-residual"] = dict(r=float(r), lo=float(c[0]), hi=float(c[1]), alpha=float(al))
    print(f"\n     residual of y after abundance: sd {resid.std():.3f} "
          f"({100*resid.std()/yk.std():.0f}% of the original spread)", flush=True)
    print(f"     ESMC predicting that residual  r = {r:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]  "
          f"<- the honest sequence signal", flush=True)
    out[cond] = {"decomposition": dict(r_y_base=float(r_yb), r_y_cond=float(r_yc), r_base_cond=float(r_bc),
                                       slope=float(sl), sd_base=float(bk.std()), sd_cond=float(ck.std()),
                                       sd_y=float(yk.std())),
                 "models": rows, "n": int(k.sum())}

json.dump(out, open(f"{D}/pd05b_results.json", "w"), indent=1)
print(f"\nwrote {D}/pd05b_results.json\nPD05B_DONE")
