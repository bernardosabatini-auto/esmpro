"""ad10: the GA_33 heat response read against the input lysate.

The heat response itself (43 or 37 vs 35 C, within one co-chaperone) is unchanged by the input: the same
input sits under both temperatures (ad09: matched-input version r 0.986). What the input changes is the
reading of the 35 C pull-down level, which is input abundance + enrichment at 35 C.

  1. Starting level. Which part of the 35 C level carries the starting-level effect on the heat
     response (r -0.43): lysate abundance, or 35 C enrichment? The 35 C level and its enrichment come
     from biological replicates BR1-3, the response from BR4-6 (and the reverse), so no measurement noise
     is shared. The input is a separate set of runs. 95 % intervals resample sequence clusters.
  2. Stability. By Meltome melting-temperature decile: input abundance, 35 C enrichment and the heat
     response. "The least stable proteins did not start out more bound" was stated on the pull-down
     scale; here it is re-tested on the enrichment scale.
  3. The 131 proteins that appear in the pull-downs only at 43 C: are they in the input, and how
     abundant are they?
"""
import os, csv, json, collections
import numpy as np

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
CH = ("21A", "24")
rng = np.random.default_rng(0)
res = {}


def bmean(T, cols, chs=CH, s="0"):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.hstack([blk[f"br_{c}_{s}_{T}"][:, cols] for c in chs]), 1)


def z(x):
    return (x - x.mean()) / x.std()


def boot(X, y, G, n=1000):
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    A = np.column_stack([np.ones(len(y)), X]); out = []
    for _ in range(n):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))])
        out.append(np.linalg.lstsq(A[s], y[s], rcond=None)[0][1:])
    return np.percentile(out, [2.5, 97.5], axis=0)


def r2(X, y):
    A = np.column_stack([np.ones(len(y)), X]); p = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ---------------- 1. starting level
print("1. what carries the starting-level effect (standardised slopes; R2 of the response)", flush=True)
for t, T in (("temp_avg_43v35", "43"), ("temp_avg_37v35", "37")):
    rows = FG[f"rows_{t}"]; G = FG["groups"][rows]
    for nm, F, R in (("BR1-3 level, BR4-6 response", [0, 1, 2], [3, 4, 5]), ("BR4-6 level, BR1-3 response", [3, 4, 5], [0, 1, 2])):
        lev = bmean("35", F)[rows]; ab = inp[rows]; en = lev - ab
        y = (bmean(T, R) - bmean("35", R))[rows]
        ok = np.isfinite(lev) & np.isfinite(ab) & np.isfinite(y) & ~con[rows]
        y_, l_, a_, e_, g_ = y[ok], lev[ok], ab[ok], en[ok], G[ok]
        rl, ra, re_ = (np.corrcoef(v, y_)[0, 1] for v in (l_, a_, e_))
        X = np.column_stack([z(a_), z(e_)])
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y_)), X]), y_ / y_.std(), rcond=None)[0][1:]
        ci = boot(X, y_ / y_.std(), g_)
        key = f"{t} {nm}"
        res[key] = dict(n=int(ok.sum()), r_level=float(rl), r_input=float(ra), r_enrich=float(re_),
                        r2_level=float(r2(l_[:, None], y_)), r2_input=float(r2(a_[:, None], y_)), r2_enrich=float(r2(e_[:, None], y_)),
                        r2_both=float(r2(X, y_)), beta_input=float(b[0]), beta_enrich=float(b[1]),
                        ci_input=ci[:, 0].tolist(), ci_enrich=ci[:, 1].tolist(), r_input_enrich=float(np.corrcoef(a_, e_)[0, 1]))
        print(f"  {t} [{nm}] n {ok.sum()}: corr with response: 35 C level {rl:+.3f}, input {ra:+.3f}, enrichment {re_:+.3f} "
              f"(input vs enrichment {np.corrcoef(a_, e_)[0,1]:+.2f})", flush=True)
        print(f"     R2: level {100*r2(l_[:,None],y_):.1f}%  input {100*r2(a_[:,None],y_):.1f}%  enrichment {100*r2(e_[:,None],y_):.1f}%  both {100*r2(X,y_):.1f}%;"
              f"  joint slopes input {b[0]:+.3f} [{ci[0,0]:+.3f}, {ci[1,0]:+.3f}], enrichment {b[1]:+.3f} [{ci[0,1]:+.3f}, {ci[1,1]:+.3f}]", flush=True)
    # per co-chaperone, full replicates for the response, input only (no shared noise: separate runs)
    for c in CH:
        tt = f"temp_{c}_{T}v35"; rr = FG[f"rows_{tt}"]; yy = FG[f"y_{tt}"].astype(float)
        ok = np.isfinite(inp[rr]) & ~con[rr]
        print(f"     {c}: full-replicate response vs input r {np.corrcoef(yy[ok], inp[rr][ok])[0,1]:+.3f}", flush=True)
        res[f"{tt}_r_input_full"] = float(np.corrcoef(yy[ok], inp[rr][ok])[0, 1])

# ---------------- 1b. does the pull-down become more like the input as temperature rises?
# Slope of pull-down level on input (input reliability 0.997, so the slope is not attenuated), the
# spread of enrichment, and a mixing fit: level_T = a + beta * level_35 + gamma * input, with level_35
# from replicates disjoint from level_T and beta corrected for the noise in level_35 (its split-half
# reliability). A pull-down that becomes a mixture of its 35 C self and the lysate gives beta + gamma = 1.
print("\n1b. selectivity by temperature (control pull-downs)", flush=True)
for c in CH:
    for T in ("35", "37", "43"):
        p = bmean(T, list(range(6)), chs=(c,)); ok = np.isfinite(p) & np.isfinite(inp) & ~con
        sl = np.polyfit(inp[ok], p[ok], 1)[0]; e = p[ok] - inp[ok]
        res[f"slope_{c}_{T}"] = float(sl)
        print(f"  {c} {T} C: slope on input {sl:.3f}, r {np.corrcoef(inp[ok], p[ok])[0,1]:.3f}, sd of enrichment {e.std():.3f}", flush=True)
    for T in ("37", "43"):
        a35, b35 = bmean("35", [0, 1, 2], chs=(c,)), bmean("35", [3, 4, 5], chs=(c,))
        pT = bmean(T, [3, 4, 5], chs=(c,))
        ok = np.isfinite(a35) & np.isfinite(b35) & np.isfinite(pT) & np.isfinite(inp) & ~con
        rel = np.corrcoef(a35[ok], b35[ok])[0, 1]                      # reliability of a three-replicate mean
        X = np.column_stack([np.ones(ok.sum()), a35[ok], inp[ok]])
        w = np.linalg.lstsq(X, pT[ok], rcond=None)[0]
        # errors-in-variables correction for the noisy level_35 (input is near noise-free)
        S = np.cov(np.vstack([a35[ok], inp[ok]])); S[0, 0] *= rel
        bc = np.linalg.solve(S, np.cov(np.vstack([a35[ok], inp[ok], pT[ok]]))[:2, 2])
        r_e = np.corrcoef((bmean("35", list(range(6)), chs=(c,)) - inp)[ok], (bmean(T, list(range(6)), chs=(c,)) - inp)[ok])[0, 1]
        res[f"mix_{c}_{T}"] = dict(beta=float(bc[0]), gamma=float(bc[1]), beta_raw=float(w[1]), gamma_raw=float(w[2]), rel=float(rel), r_enrich_35_T=float(r_e))
        print(f"  {c} {T} C from 35 C: beta {bc[0]:.3f}, gamma {bc[1]:.3f} (sum {bc.sum():.3f}; uncorrected {w[1]:.3f}/{w[2]:.3f}; "
              f"reliability of level_35 {rel:.3f}); enrichment 35 vs {T} C r {r_e:.3f}", flush=True)

# ---------------- 2. stability
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
    for a, tm_ in rm[r].items(): LL[a].append(tm_ - cen[r])
TM = {a: float(np.median(v)) + grand for a, v in LL.items()}

t = "temp_avg_43v35"; rows = FG[f"rows_{t}"]; y43 = FG[f"y_{t}"].astype(float)
tm = np.array([TM.get(a, np.nan) for a in acc[rows]])
lev = bmean("35", list(range(6)))[rows]; ab = inp[rows]; en = lev - ab
ok = np.isfinite(tm) & np.isfinite(ab) & ~con[rows]
mid_l, mid_a, mid_e = np.median(lev[ok]), np.median(ab[ok]), np.median(en[ok])
sl, ic = np.polyfit(ab[ok], en[ok], 1)
det = en - (ic + sl * ab)
edges = np.percentile(tm[np.isfinite(tm)], np.linspace(0, 100, 11))
bins = np.clip(np.digitize(tm, edges[1:-1]), 0, 9)
print(f"\n2. by melting-temperature decile ({ok.sum()} proteins with Tm and input; medians relative to all of them; 43 C response mean)", flush=True)
rowsT = collections.defaultdict(list)
for i in range(10):
    s = ok & (bins == i)
    rowsT["Tm upper edge"].append(edges[i + 1]); rowsT["n"].append(int(s.sum()))
    rowsT["35 C pull-down level"].append(np.median(lev[s]) - mid_l)
    rowsT["input abundance"].append(np.median(ab[s]) - mid_a)
    rowsT["35 C enrichment"].append(np.median(en[s]) - mid_e)
    rowsT["35 C enrichment, abundance trend removed"].append(np.median(det[s]))
    rowsT["43 vs 35 C response"].append(float(y43[s].mean()))
for k, v in rowsT.items():
    print(f"  {k:<42}" + " ".join(f"{x:>+6.2f}" if k != "n" else f"{x:>6d}" for x in v), flush=True)
res["tm_deciles"] = {k: [float(x) for x in v] for k, v in rowsT.items()}
# least stable decile vs window (deciles 3-6) on enrichment, cluster bootstrap of the difference
G = FG["groups"][rows]
d_low = ok & (bins == 0); d_win = ok & np.isin(bins, [2, 3, 4, 5])
for nm, v in (("35 C enrichment", en), ("abundance-detrended enrichment", det), ("input abundance", ab)):
    obs = np.median(v[d_low]) - np.median(v[d_win])
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    bs = []
    for _ in range(500):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))])
        bs.append(np.median(v[s][d_low[s]]) - np.median(v[s][d_win[s]]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  least stable decile minus window (Tm 48-52): {nm} {obs:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    res[f"low_minus_window_{nm}"] = [float(obs), float(lo), float(hi)]
# does input abundance or 35 C enrichment change the U?  adjust the response for both
X = np.column_stack([np.ones(ok.sum()), ab[ok], en[ok]])
yr = y43[ok] - X @ np.linalg.lstsq(X, y43[ok], rcond=None)[0] + y43[ok].mean()
adj = [float(yr[bins[ok] == i].mean()) for i in range(10)]
print("  43 vs 35 C response adjusted for input abundance and 35 C enrichment: " + " ".join(f"{x:+.2f}" for x in adj), flush=True)
res["tm_deciles"]["43 vs 35 C response, adjusted"] = adj

# ---------------- 3. appearing proteins
app = [r for r in csv.DictReader(open(f"{ROOT}/reports/additional/heat43_appearing_proteins.tsv"), delimiter="\t")]
pos = {a: i for i, a in enumerate(acc)}
ix = np.array([pos.get(r["uniprot"], -1) for r in app])
ix = ix[ix >= 0]
allpd = np.isfinite(bmean("35", list(range(6)))) | np.isfinite(bmean("43", list(range(6))))
pct = lambda v: 100 * (np.sum(inp[allpd & np.isfinite(inp) & ~con] < v) / np.sum(allpd & np.isfinite(inp) & ~con))
has = np.isfinite(inp[ix])
print(f"\n3. heat-appearing proteins: {len(ix)} of {len(app)} mapped; in the input {has.sum()} ({100*has.mean():.0f}%) "
      f"against {100*np.mean(np.isfinite(inp[allpd])):.0f}% of all pull-down proteins", flush=True)
pc = np.array([pct(v) for v in inp[ix][has]])
print(f"  input abundance percentile among pull-down proteins: median {np.median(pc):.0f} (IQR {np.percentile(pc,25):.0f}-{np.percentile(pc,75):.0f})", flush=True)
ns = np.load(f"{GA}/input_reference.npz", allow_pickle=True)["n_samples"]
print(f"  input samples (of 10) detecting them: median {np.median(ns[ix][has]):.0f}; all pull-down proteins with input: {np.median(ns[allpd & np.isfinite(inp)]):.0f}", flush=True)
res["appearing"] = dict(n=int(len(ix)), in_input=int(has.sum()), frac_all=float(np.mean(np.isfinite(inp[allpd]))),
                        pct_median=float(np.median(pc)), pct_iqr=[float(np.percentile(pc, 25)), float(np.percentile(pc, 75))])
json.dump(res, open(f"{GA}/ad10_results.json", "w"), indent=1)
print("\nAD10_DONE", flush=True)
