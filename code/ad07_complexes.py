"""ad07: does complex membership explain the heat response better than thermal stability?

Hypothesis: heat disassembles complexes, and released subunits become DNAJ clients regardless of
their intrinsic stability. Complexes: EBI Complex Portal, human (curated; participants from the
expanded participant list, so sub-complexes are resolved to proteins). Predictions:
  1. at matched Tm, abundance and compartment, complex subunits gain more co-chaperone association
     at 43 C than non-members;
  2. subunits of the same complex move together (their responses are more alike than random sets of
     the same size);
  3. heat 37 C, where little is perturbed, shows less of both.
"""
import os, re, csv, json, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
rng = np.random.default_rng(0)
# ---- complexes
cplx = {}
for r in csv.reader(open(f"{ROOT}/data/external/complexes/complexportal_9606.tsv"), delimiter="\t"):
    if r[0].startswith("#"): continue
    mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
    if len(mem) >= 2:
        cplx[r[0]] = dict(name=r[1], members=mem)
member_of = collections.defaultdict(set)
for c, d in cplx.items():
    for a in d["members"]: member_of[a].add(c)
print(f"Complex Portal: {len(cplx)} human complexes with >= 2 protein subunits; {len(member_of)} proteins in at least one", flush=True)
# ---- Meltome Tm (as in ad01)
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
# ---- data
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]]); gene = np.array([str(g) for g in blk["gene"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
ann = {r["Entry"]: r for r in csv.DictReader(open(f"{ROOT}/data/annot/uniprot_annot.tsv"), delimiter="\t")}
loc = lambda a: ann.get(a, {}).get("Subcellular location [CC]", "")
out = {"n_complexes": len(cplx)}
for t, T in (("temp_avg_43v35", "43"), ("temp_avg_37v35", "37")):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]; G = FG["groups"][rows]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_35"][rows] for c in ("21A", "24")]), 1)
    tm = np.array([TM.get(a, np.nan) for a in a_])
    mem = np.array([a in member_of for a in a_])
    nuc = np.array(["Nucleus" in loc(a) for a in a_]); cyt = np.array(["Cytoplasm" in loc(a) for a in a_])
    tmem = np.array(["Transmembrane" in ann.get(a, {}).get("Keywords", "") for a in a_])
    ok = np.isfinite(tm) & np.isfinite(lev)
    print(f"\n{'='*100}\n{t}: {len(y)} proteins, {mem.sum()} complex subunits ({ok.sum()} with Tm and level)", flush=True)
    # 1. raw and adjusted membership effect
    raw = y[mem].mean() - y[~mem].mean(); p_raw = stats.mannwhitneyu(y[mem], y[~mem]).pvalue
    z = (tm - np.nanmean(tm)) / np.nanstd(tm); lz = (lev - np.nanmean(lev)) / np.nanstd(lev)
    def design(m):
        return np.column_stack([np.ones(m.sum()), mem[m].astype(float), z[m], z[m] ** 2, lz[m], lz[m] ** 2,
                                nuc[m].astype(float), cyt[m].astype(float), tmem[m].astype(float)])
    X = design(ok); yy = y[ok]; b = np.linalg.lstsq(X, yy, rcond=None)[0]
    ug, inv = np.unique(G[ok], return_inverse=True); mm = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    bs = [np.linalg.lstsq(X[s], yy[s], rcond=None)[0][1] for s in (np.concatenate([mm[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(1000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # Tm terms with and without membership
    Xn = np.delete(X, 1, axis=1); bq = np.linalg.lstsq(Xn, yy, rcond=None)[0]
    rec = dict(n=int(len(y)), n_members=int(mem.sum()), raw_diff=float(raw), p_raw=float(p_raw), adjusted=[float(b[1]), float(lo), float(hi)],
               tm_quadratic_with_membership=float(b[3]), tm_quadratic_without=float(bq[2]))
    print(f"  1. complex subunits vs non-members: raw {raw:+.3f} (p {p_raw:.0e}); adjusted for Tm (quadratic), level (quadratic), "
          f"nucleus/cytoplasm/membrane: {b[1]:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    print(f"     Tm curvature with membership in the model {b[3]:+.3f} (without {bq[2]:+.3f})", flush=True)
    # by Tm tertile
    q = np.nanpercentile(tm[ok], [33.3, 66.7]); bins = np.digitize(tm, q)
    tt = []
    for k, nm in enumerate(("least stable third", "middle third", "most stable third")):
        s = ok & (bins == k)
        tt.append(dict(bin=nm, members=float(y[s & mem].mean()), others=float(y[s & ~mem].mean()), n_members=int((s & mem).sum())))
        print(f"     {nm:<20} subunits {y[s & mem].mean():+.3f} (n {(s & mem).sum():>4})   non-members {y[s & ~mem].mean():+.3f} (n {(s & ~mem).sum():>4})", flush=True)
    rec["by_tm_third"] = tt
    # 2. coherence within complexes: variance of responses among a complex's detected subunits vs random sets of the same size
    pos = {a: i for i, a in enumerate(a_)}
    sets = []
    for c, d in cplx.items():
        idx = sorted({pos[a] for a in d["members"] if a in pos})
        if len(idx) >= 3: sets.append((c, idx))
    # de-duplicate near-identical member sets
    seen, uniq = set(), []
    for c, idx in sorted(sets, key=lambda z: -len(z[1])):
        key = tuple(idx)
        if key in seen: continue
        seen.add(key); uniq.append((c, idx))
    obs = np.mean([y[idx].var(ddof=1) for _, idx in uniq])
    null = []
    allidx = np.arange(len(y))
    for _ in range(1000):
        null.append(np.mean([y[rng.choice(allidx, len(idx), replace=False)].var(ddof=1) for _, idx in uniq]))
    null = np.array(null)
    # null that keeps protein class: random sets drawn from complex subunits of other complexes
    pool = np.nonzero(mem)[0]
    null2 = np.array([np.mean([y[rng.choice(pool, len(idx), replace=False)].var(ddof=1) for _, idx in uniq]) for _ in range(1000)])
    icc = 1 - obs / null.mean(); icc2 = 1 - obs / null2.mean()
    rec["coherence"] = dict(n_complexes=len(uniq), within_var=float(obs), random_var=float(null.mean()), random_subunit_var=float(null2.mean()),
                            share_between=float(icc), share_between_vs_subunits=float(icc2), p=float((null <= obs).mean()), p2=float((null2 <= obs).mean()))
    print(f"  2. coherence: {len(uniq)} complexes with >= 3 detected subunits; within-complex variance {obs:.3f} vs random sets {null.mean():.3f} "
          f"(p {max((null <= obs).mean(), 1e-3):.3f}) and vs random subunits of other complexes {null2.mean():.3f} (p {max((null2 <= obs).mean(), 1e-3):.3f})", flush=True)
    print(f"     share of response variance shared within a complex: {100*icc:.0f}% (against random proteins), {100*icc2:.0f}% (against other complexes' subunits)", flush=True)
    # which complexes move
    comp = sorted([(float(np.mean(y[idx])), c, len(idx)) for c, idx in uniq], reverse=True)
    rec["top_complexes"] = [dict(complex=cplx[c]["name"], mean=m, n=n) for m, c, n in comp[:10]]
    rec["bottom_complexes"] = [dict(complex=cplx[c]["name"], mean=m, n=n) for m, c, n in comp[-10:]]
    print("     most gained: " + "; ".join(f"{cplx[c]['name'][:40]} {m:+.2f} (n {n})" for m, c, n in comp[:6]), flush=True)
    print("     most lost:   " + "; ".join(f"{cplx[c]['name'][:40]} {m:+.2f} (n {n})" for m, c, n in comp[-6:]), flush=True)
    out[t] = rec
json.dump(out, open(f"{GA}/ad07_results.json", "w"), indent=1)
print("\nAD07_DONE")

# ---- coherence after removing what subunits share anyway (appended)
print("\n3. COHERENCE AFTER REMOVING BASELINE LEVEL, Tm AND COMPARTMENT", flush=True)
for t, T in (("temp_avg_43v35", "43"), ("temp_avg_37v35", "37")):
    rows = FG[f"rows_{t}"]; y = FG[f"y_{t}"].astype(float); a_ = acc[rows]
    lev = np.nanmean(np.hstack([blk[f"br_{c}_0_35"][rows] for c in ("21A", "24")]), 1)
    lev35 = lev
    tm = np.array([TM.get(a, np.nan) for a in a_]); z = np.where(np.isfinite(tm), (tm - np.nanmean(tm)) / np.nanstd(tm), 0.0); hz = np.isfinite(tm).astype(float)
    nuc = np.array(["Nucleus" in loc(a) for a in a_]); cyt = np.array(["Cytoplasm" in loc(a) for a in a_])
    tmem = np.array(["Transmembrane" in ann.get(a, {}).get("Keywords", "") for a in a_])
    ok = np.isfinite(lev)
    lz = (lev - np.nanmean(lev)) / np.nanstd(lev)
    X = np.column_stack([np.ones(len(y)), lz, lz ** 2, lz ** 3, z, z ** 2, hz, nuc, cyt, tmem])[ok]
    res = np.full(len(y), np.nan); res[ok] = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    pos = {a: i for i, a in enumerate(a_)}
    uniq, seen = [], set()
    for c, d in sorted(cplx.items(), key=lambda kv: -len(kv[1]["members"])):
        idx = tuple(sorted({pos[a] for a in d["members"] if a in pos and ok[pos[a]]}))
        if len(idx) >= 3 and idx not in seen:
            seen.add(idx); uniq.append(idx)
    mem = np.array([a in member_of for a in a_]) & ok
    pool = np.nonzero(mem)[0]; allidx = np.nonzero(ok)[0]
    obs = np.mean([res[list(i)].var(ddof=1) for i in uniq])
    n1 = np.array([np.mean([res[rng.choice(allidx, len(i), replace=False)].var(ddof=1) for i in uniq]) for _ in range(500)])
    n2 = np.array([np.mean([res[rng.choice(pool, len(i), replace=False)].var(ddof=1) for i in uniq]) for _ in range(500)])
    shared = 1 - res[ok].var() / y[ok].var()
    print(f"  {t}: covariates explain {100*shared:.0f}% of the response; on what is left, within-complex variance {obs:.3f} vs random {n1.mean():.3f}, "
          f"vs other complexes' subunits {n2.mean():.3f} -> shared within complex {100*(1-obs/n1.mean()):.0f}% / {100*(1-obs/n2.mean()):.0f}% (p {max((n2 <= obs).mean(), 1/500):.3f})", flush=True)
    out[t]["coherence_residual"] = dict(covariates_r2=float(shared), within=float(obs), random=float(n1.mean()), random_subunits=float(n2.mean()),
                                        share=float(1 - obs / n1.mean()), share_vs_subunits=float(1 - obs / n2.mean()))
json.dump(out, open(f"{GA}/ad07_results.json", "w"), indent=1)
