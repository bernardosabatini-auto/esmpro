"""ad01: does the heat response follow measured protein thermal stability?

The sequence and SAE analysis read the 43 vs 35 C response as "folded proteins that unfold with heat
are drawn into the co-chaperone pull-downs; disordered and membrane proteins are not". The Meltome
atlas (Jarzab et al. 2020, Nat Methods; full dataset as redistributed by FLIP, J-SNACKKB/FLIP,
downloaded from the atlas server) gives measured melting temperatures (Tm) for human proteins in ten
human cell lines and cell types, so the reading becomes a test: lower Tm should mean a larger rise.

Tm per protein: median of replicate entries within each cell line; each cell line centred on its
own median (cell lines differ by ~1 C overall); consensus = median across cell lines plus the grand
median. Tm reliability: two consensus values from disjoint halves of the cell lines.

Tests, on GA_33 proteins with a consensus Tm:
  1. correlation of Tm with the heat responses, and with staurosporine and bait preference as
     controls that should not depend on Tm;
  2. heat response by Tm decile;
  3. coverage bias: heat response of proteins with and without a Tm;
  4. Tm against sequence: cross-validated R2 of Tm alone, of the out-of-fold sequence prediction
     (ga10, MLP + ridge), and of both, on the same proteins and folds; cluster-bootstrap intervals;
  5. the SAE interpretation: do the disorder, membrane and enzyme-core features still predict the
     heat response once Tm is in the model?
"""
import os, csv, json, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
rng = np.random.default_rng(0)
HUM = ["HepG2", "Jurkat", "K562", "U937", "HAOEC", "HL60", "HEK293T", "colon_cancer_spheroids", "HaCaT", "pTcells"]
d = json.load(open(f"{ROOT}/data/external/meltome/full_dataset.json"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
for x in d:
    if x["runName"] in HUM and x["uniprotAccession"] and x["meltingPoint"] is not None:
        per[x["runName"]][x["uniprotAccession"].split("-")[0]].append(x["meltingPoint"])
run_med = {r: {a: float(np.median(v)) for a, v in per[r].items()} for r in HUM}
center = {r: float(np.median(list(run_med[r].values()))) for r in HUM}
grand = float(np.median(list(center.values())))


def consensus(runs):
    acc = collections.defaultdict(list)
    for r in runs:
        for a, t in run_med[r].items():
            acc[a].append(t - center[r])
    return {a: float(np.median(v)) + grand for a, v in acc.items()}, {a: len(v) for a, v in acc.items()}


TM, NR = consensus(HUM)
h1, _ = consensus(HUM[0::2]); h2, _ = consensus(HUM[1::2])
both = [a for a in h1 if a in h2]
r_half = stats.pearsonr([h1[a] for a in both], [h2[a] for a in both])[0]
rel_tm = 2 * r_half / (1 + r_half)
hek = run_med["HEK293T"]
print(f"Meltome: {len(TM)} human proteins with a consensus Tm (median {np.median(list(TM.values())):.1f} C, "
      f"IQR {np.percentile(list(TM.values()),25):.1f}-{np.percentile(list(TM.values()),75):.1f}); "
      f"cell lines per protein: median {np.median(list(NR.values())):.0f}", flush=True)
print(f"  Tm reliability: half-vs-half cell lines r = {r_half:.3f} on {len(both)} proteins -> {rel_tm:.3f} for all ten", flush=True)

blk = np.load(f"{GA}/blocks.npz", allow_pickle=True)
st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{GA}/sequences_status.tsv"), delimiter="\t")}
acc = np.array([st.get(str(a), str(a)) for a in blk["accession"]])
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
SP = np.load(f"{GA}/spec.npz", allow_pickle=True)


def target(t):
    if t.startswith("spec"):
        return SP[f"rows_{t}"], SP[f"y_{t}"].astype(float), SP[f"fold_{t}"]
    return FG[f"rows_{t}"], FG[f"y_{t}"].astype(float), FG[f"fold_{t}"]


out = {"n_meltome": len(TM), "tm_reliability": rel_tm, "tm_half_r": r_half}
print(f"\n1. CORRELATION WITH Tm (Spearman; negative = less stable proteins rise more)", flush=True)
print(f"  {'target':<17}{'n with Tm':>10}{'coverage':>10}{'rho':>8}{'p':>10}{'Pearson r':>11}{'mean y, Tm / no Tm':>22}", flush=True)
corr = {}
for t in ("temp_avg_43v35", "temp_21A_43v35", "temp_24_43v35", "temp_avg_37v35", "stau_avg_37", "stau_avg_43", "spec_avg"):
    rows, y, _ = target(t)
    a_ = acc[rows]
    has = np.array([a in TM for a in a_])
    tm = np.array([TM.get(a, np.nan) for a in a_])
    rho, p = stats.spearmanr(tm[has], y[has])
    r = stats.pearsonr(tm[has], y[has])[0]
    mw = stats.mannwhitneyu(y[has], y[~has]).pvalue
    corr[t] = dict(n=int(has.sum()), coverage=float(has.mean()), rho=float(rho), p=float(p), r=float(r),
                   mean_with=float(y[has].mean()), mean_without=float(y[~has].mean()), p_cov=float(mw))
    print(f"  {t:<17}{has.sum():>10}{100*has.mean():>9.0f}%{rho:>+8.3f}{p:>10.0e}{r:>+11.3f}{y[has].mean():>+11.2f} / {y[~has].mean():+.2f} (p {mw:.0e})", flush=True)
out["correlations"] = corr

t = "temp_avg_43v35"
rows, y, fid = target(t)
a_ = acc[rows]; has = np.array([a in TM for a in a_]); tm = np.array([TM.get(a, np.nan) for a in a_])
print(f"\n2. HEAT RESPONSE (43 vs 35 C) BY Tm DECILE", flush=True)
qs = np.percentile(tm[has], np.linspace(0, 100, 11))
dec = []
for i in range(10):
    m = has & (tm >= qs[i]) & (tm <= qs[i + 1] if i == 9 else tm < qs[i + 1])
    dec.append(dict(tm_lo=float(qs[i]), tm_hi=float(qs[i + 1]), n=int(m.sum()), mean=float(y[m].mean()), sem=float(y[m].std() / np.sqrt(m.sum()))))
    print(f"  Tm {qs[i]:5.1f}-{qs[i+1]:5.1f} C  n {m.sum():>4}  mean heat response {y[m].mean():+.3f} +/- {y[m].std()/np.sqrt(m.sum()):.3f}", flush=True)
out["deciles"] = dec

# 4. Tm vs sequence, cross-validated on the proteins with Tm
Z = np.load(f"{GA}/ga10_heat.npz", allow_pickle=True)
zrows = Z["rows"]; k43 = [str(x) for x in Z["targets"]].index("temp_avg_43v35")
seqp = dict(zip(zrows.tolist(), (0.5 * Z["separate"][:, k43] + 0.5 * Z["ridge"][:, k43]).tolist()))
ok = has & np.array([r in seqp for r in rows])
yy, tt, sp, ff = y[ok], tm[ok], np.array([seqp[r] for r in rows[ok]]), fid[ok]
G = FG["groups"][rows[ok]]


def cv(X, y, f):
    p = np.zeros(len(y))
    for k in np.unique(f):
        tr, te = f != k, f == k
        A = np.column_stack([np.ones(tr.sum()), X[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        p[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ w
    return p


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def cb(y, p1, p0, G, nb=1000):
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    dd = []
    for _ in range(nb):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); q = y[s]
        dd.append((((q - p0[s]) ** 2).sum() - ((q - p1[s]) ** 2).sum()) / ((q - q.mean()) ** 2).sum())
    return [float(x) for x in np.percentile(dd, [2.5, 97.5])]


tz = (tt - tt.mean()) / tt.std()
Xtm = np.column_stack([tz, tz ** 2, tz ** 3])
p_tm = cv(Xtm, yy, ff); p_seq = cv(sp[:, None], yy, ff); p_both = cv(np.column_stack([sp, Xtm]), yy, ff)
res = dict(n=int(ok.sum()), tm=r2(yy, p_tm), seq=r2(yy, p_seq), both=r2(yy, p_both),
           tm_adds=cb(yy, p_both, p_seq, G), seq_adds=cb(yy, p_both, p_tm, G),
           r_seq_tm=float(stats.pearsonr(sp, tt)[0]))
print(f"\n4. Tm AND SEQUENCE ({ok.sum()} proteins with both; cross-validated, cubic in Tm)", flush=True)
print(f"  Tm alone {100*res['tm']:.1f}%   sequence model alone {100*res['seq']:.1f}%   both {100*res['both']:.1f}%", flush=True)
print(f"  Tm adds to sequence {100*(res['both']-res['seq']):+.1f} [{100*res['tm_adds'][0]:+.1f},{100*res['tm_adds'][1]:+.1f}];  "
      f"sequence adds to Tm {100*(res['both']-res['tm']):+.1f} [{100*res['seq_adds'][0]:+.1f},{100*res['seq_adds'][1]:+.1f}]", flush=True)
print(f"  correlation of the sequence prediction with Tm: {res['r_seq_tm']:+.3f}", flush=True)
out["joint"] = res

# 5. the SAE features behind the interpretation, with and without Tm
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True); spos = {str(a): i for i, a in enumerate(S["accession"])}
ix = np.array([spos[a] for a in a_[ok]])
FEATS = {"654 long low-complexity IDRs (disorder)": 654, "10715 mixed-charge IDRs (disorder)": 10715,
         "5659 TM helical segments (membrane)": 5659, "4518 IPMDH beta-strand (enzyme core)": 4518,
         "5338 basic/polar N-terminal segments (enzyme)": 5338}
print(f"\n5. DO THE SAE FEATURES STILL PREDICT THE HEAT RESPONSE WITH Tm IN THE MODEL? (standardised slopes)", flush=True)
sae_rows = []
for nm, f in FEATS.items():
    v = np.log1p(S["max"][ix, f].astype(np.float64)); v = (v - v.mean()) / (v.std() + 1e-12)
    A1 = np.column_stack([np.ones(len(yy)), v]); b1 = np.linalg.lstsq(A1, yy, rcond=None)[0][1]
    A2 = np.column_stack([np.ones(len(yy)), v, Xtm]); b2 = np.linalg.lstsq(A2, yy, rcond=None)[0][1]
    rtm = stats.spearmanr(v, tt)[0]
    sae_rows.append(dict(feature=nm, slope=float(b1), slope_with_tm=float(b2), rho_with_tm=float(rtm)))
    print(f"  {nm:<46} slope {b1:+.3f} -> {b2:+.3f} with Tm ({100*(1-b2/b1) if b1 else 0:+.0f}% explained); feature-Tm rho {rtm:+.2f}", flush=True)
out["sae_with_tm"] = sae_rows
json.dump(out, open(f"{ROOT}/data/external/meltome/ad01_results.json", "w"), indent=1)
print("\nAD01_DONE", flush=True)
