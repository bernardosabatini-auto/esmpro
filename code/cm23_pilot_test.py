"""cm23: does a predicted HSPB1-client interface predict measured HSPB1 binding?

Tier 1 of the complex-structure plan, and the go/no-go for everything beyond it.

WHAT IS BEING ASKED. For each of 200 proteins spanning the measured bound:free partition (cm20), the
whole sequence was tiled into non-overlapping 16-mers and each window was co-folded with the HSPB1
ACD dimer (cm21). The per-window interface confidence is aggregated to a per-protein score. Does
that score track how strongly the protein actually partitions onto HSPB1?

PREDICTIONS, FIXED BEFORE ANY RESULT WAS SEEN (this file was written while the co-folds were still
running; the validation gate cm22 had been scored, nothing else):
  P1  per-protein interface score correlates POSITIVELY with the measured partition, bootstrap CI
      excluding zero.
  P2  real windows score above their own composition-matched shuffled twins (paired, one-sided).
      If this fails, the co-fold is reading amino-acid composition and nothing else, and every other
      result here is a hydrophobicity scale in an expensive costume.
  P3  the structure score adds predictive power over a window-composition model.
  P4  the structure score adds over the ESMC-6B + SAE sequence model's out-of-fold prediction of the
      same target. This is the bar tiers 0 and the AlphaFold test were held to, and the only one
      that matters for whether this is worth scaling.

A failure of P1 kills the tier. A pass of P1 with a failure of P2 or P3 means the signal is
composition, which the sequence model already has, so it also kills the tier -- P4 is then the
arbiter.

HONEST LIMIT ON EFFECT SIZE. cm20 sampled 20 proteins per decile of the outcome, which spreads the
x-axis and inflates any correlation relative to the population. Every correlation is therefore
reported twice: as measured on the stratified sample, and reweighted by the inverse sampling
probability to the population the proteins were drawn from. The reweighted number is the one to
quote.
"""
import os, json, glob, collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pandas as pd, gemmi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
OUT = f"{C}/boltz/out"
EXP = f"{ROOT}/data/external/pdb_assemblies/cif/9ef3.cif.gz"
CUT, NBOOT, SEED = 4.5, 4000, 0
HYD, ARO, CHG, POS, NEG = set("AVLIMFWYC"), set("FWY"), set("DEKR"), set("KR"), set("DE")
KD = dict(zip("AVLIMFWYCGPSTNQHDEKR",
              [1.8, 4.2, 3.8, 4.5, 1.9, 2.8, -0.9, -1.3, 2.5, -0.4, -1.6, -0.8, -0.7,
               -3.5, -3.5, -3.2, -3.5, -3.5, -3.9, -4.5]))


def one(n):
    r = gemmi.find_tabulated_residue(n)
    return r.one_letter_code.upper() if r and r.is_amino_acid() else None


def groove_set():
    st = gemmi.read_structure(EXP); st.setup_entities()
    ch = {c.name: [r for r in c if one(r.name)] for c in st[0]}
    pep = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch["D"] for a in r])
    t = cKDTree(pep)
    g = set()
    for i, r in enumerate(ch["A"], 1):
        x = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])
        if t.query(x, k=1, distance_upper_bound=CUT)[0].min() < CUT:
            g.add(i)
    return g


GROOVE = groove_set()


def parse_one(d):
    """one prediction directory -> interface confidence + where on the ACD the peptide sat"""
    name = os.path.basename(d)
    cif, cjs = f"{d}/{name}_model_0.cif", f"{d}/confidence_{name}_model_0.json"
    if not os.path.exists(cif) or not os.path.exists(cjs):
        return None
    cf = json.load(open(cjs))
    r = dict(name=name, iptm=cf.get("iptm"), ptm=cf.get("ptm"),
             plddt=cf.get("complex_plddt"), iplddt=cf.get("complex_iplddt"),
             ipde=cf.get("complex_ipde"), conf=cf.get("confidence_score"))
    pci = cf.get("pair_chains_iptm", {})
    try:
        r["iptm_pep"] = max(float(pci["0"]["2"]), float(pci["2"]["0"]),
                            float(pci["1"]["2"]), float(pci["2"]["1"]))
    except Exception:
        r["iptm_pep"] = np.nan
    st = gemmi.read_structure(cif); st.setup_entities()
    ch = {c.name: [rr for rr in c if one(rr.name)] for c in st[0]}
    if "C" not in ch or not ch["C"]:
        return r
    pep = np.array([[a.pos.x, a.pos.y, a.pos.z] for rr in ch["C"] for a in rr])
    t = cKDTree(pep)
    best = None
    for cn in ("A", "B"):
        hit = set()
        for i, rr in enumerate(ch.get(cn, []), 1):
            x = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in rr])
            if t.query(x, k=1, distance_upper_bound=CUT)[0].min() < CUT:
                hit.add(i)
        if best is None or len(hit) > len(best[1]):
            best = (cn, hit)
    cn, hit = best
    r["bound_chain"], r["n_contact"] = cn, len(hit)
    r["groove_frac"] = len(hit & GROOVE) / max(len(hit), 1)
    r["groove_recall"] = len(hit & GROOVE) / len(GROOVE)
    # buried peptide area proxy: contacts per peptide residue
    r["contact_per_res"] = len(hit) / len(ch["C"])
    return r


def comp_feats(p):
    n = len(p)
    c = collections.Counter(p)
    f = dict(w_hydroph=sum(c[a] for a in HYD) / n, w_aromatic=sum(c[a] for a in ARO) / n,
             w_charged=sum(c[a] for a in CHG) / n,
             w_netcharge=(sum(c[a] for a in POS) - sum(c[a] for a in NEG)) / n,
             w_gp=(c["G"] + c["P"]) / n, w_kd=float(np.mean([KD.get(a, 0) for a in p])))
    # the cm18 APR rule: a 7-mer with >=5 hydrophobics, no charge, <=1 of P/G
    apr = 0
    for i in range(max(n - 6, 0)):
        s = p[i:i + 7]
        if (sum(a in HYD for a in s) >= 5 and not any(a in CHG for a in s)
                and sum(a in "PG" for a in s) <= 1):
            apr += 1
    f["w_apr"] = apr / max(n - 6, 1)
    return f


# ---------------------------------------------------------------- window table
if os.environ.get("CM23_RELOAD", "1") == "1" or not os.path.exists(f"{C}/pilot_windows.tsv"):
    dirs = sorted(glob.glob(f"{OUT}/shard_*/boltz_results_shard_*/predictions/*"))
    print(f"parsing {len(dirs)} predictions ...", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("CM23_PROC", "12"))) as ex:
        recs = [r for r in ex.map(parse_one, dirs, chunksize=32) if r]
    W = pd.DataFrame(recs)
    J = pd.read_csv(f"{C}/boltz_jobs.tsv", sep="\t")
    W = J.merge(W, on="name", how="left")
    W = pd.concat([W, pd.DataFrame([comp_feats(p) if isinstance(p, str) and p else {}
                                    for p in W.pep]).set_index(W.index)], axis=1)
    W.to_csv(f"{C}/pilot_windows.tsv", sep="\t", index=False)
else:
    W = pd.read_csv(f"{C}/pilot_windows.tsv", sep="\t")

done = W.iptm_pep.notna()
print(f"{len(W)} jobs, {done.sum()} scored, {(~done).sum()} missing")
P = pd.read_csv(f"{C}/pilot_proteins.tsv", sep="\t")
win = W[(W.kind == "window") & done].copy()
shuf = W[(W.kind == "shuffled") & done].copy()
ctrl = W[W.kind.isin(["pos_heldout", "pos_intrain", "neg"]) & done].copy()
print(f"real windows {len(win)}, shuffled {len(shuf)}, controls {len(ctrl)}")

print("\n---- controls on the window scale ----")
q = win.iptm_pep.values
for _, r in ctrl.sort_values("iptm_pep", ascending=False).iterrows():
    print(f"  {r['name']:22s} {r.kind:12s} iptm {r.iptm_pep:.3f}  "
          f"percentile among real windows {100*(q < r.iptm_pep).mean():5.1f}  "
          f"groove_recall {r.groove_recall:.2f}")

# ---------------------------------------------------------------- aggregation
SCORES = ["iptm_pep", "groove_recall", "n_contact", "iplddt"]


def agg(df, col):
    g = df.groupby("accession")[col]
    a = pd.DataFrame({f"{col}_max": g.max(), f"{col}_mean": g.mean(),
                      f"{col}_top3": g.apply(lambda v: v.nlargest(3).mean())})
    hi = df[col].quantile(0.90)
    a[f"{col}_fhi"] = g.apply(lambda v: float((v >= hi).mean()))
    return a


A = pd.concat([agg(win, c) for c in SCORES], axis=1)
A["n_win"] = win.groupby("accession").size()
A["gated_max"] = win.assign(x=win.iptm_pep * win.groove_recall).groupby("accession").x.max()
COMP = [c for c in win.columns if c.startswith("w_")]
A = A.join(win.groupby("accession")[COMP].max().add_suffix("_max"))
A = A.join(win.groupby("accession")[COMP].mean().add_suffix("_mean"))

D = P.set_index("accession").join(A, how="inner")
D = D[D.n_win.notna()]
print(f"\n{len(D)} proteins aggregated, median {D.n_win.median():.0f} windows each")

R = np.load(f"{C}/ridge_oof.npz", allow_pickle=True)
T = np.load(f"{C}/targets.npz", allow_pickle=True)
acc2i = {a: i for i, a in enumerate(T["accession"])}
ii = np.array([acc2i[a] for a in D.index])
sae = (R["sae|GA22_part_base"][ii] + R["sae|GA24_part_base"][ii]) / 2
D["seqmodel"] = sae

y = D.part_mean.values
w = D.weight.values
rng = np.random.default_rng(SEED)
BOOT = rng.integers(0, len(D), size=(NBOOT, len(D)))


def rho_ci(x, yy, weights=None):
    k = np.isfinite(x) & np.isfinite(yy)
    if k.sum() < 20:
        return np.nan, (np.nan, np.nan)
    r = spearmanr(x[k], yy[k]).statistic
    bs = []
    for b in BOOT:
        bb = b[k[b]]
        if len(bb) < 20 or np.std(x[bb]) == 0:
            continue
        if weights is None:
            bs.append(spearmanr(x[bb], yy[bb]).statistic)
        else:                                     # weighted: resample proportional to IPW
            p = weights[bb] / weights[bb].sum()
            s = rng.choice(bb, size=len(bb), p=p)
            if np.std(x[s]) > 0:
                bs.append(spearmanr(x[s], yy[s]).statistic)
    return r, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


print("\n---- P1: does the interface score track the measured partition? ----")
print(f"{'score':24s} {'rho (stratified)':>18s} {'95% CI':>18s} {'rho (reweighted)':>18s}")
p1 = {}
cand = [c for c in A.columns if not c.startswith("w_")]
for c in cand:
    r, ci = rho_ci(D[c].values, y)
    rw, _ = rho_ci(D[c].values, y, weights=w)
    p1[c] = dict(rho=r, lo=ci[0], hi=ci[1], rho_w=rw)
    print(f"{c:24s} {r:18.3f} {f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18s} {rw:18.3f}")
rs, _ = rho_ci(D.seqmodel.values, y)
print(f"{'(ESMC-6B+SAE seq model)':24s} {rs:18.3f}")

print("\n---- P2: real windows vs their own shuffled twins ----")
m = shuf.merge(win, on=["accession", "start"], suffixes=("_s", "_r"))
print(f"{len(m)} matched pairs")
if len(m) > 20:
    d = (m.iptm_pep_r - m.iptm_pep_s).values
    st = wilcoxon(d, alternative="greater")
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)]
    print(f"  mean iptm  real {m.iptm_pep_r.mean():.3f}  shuffled {m.iptm_pep_s.mean():.3f}  "
          f"difference {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
    print(f"  Wilcoxon one-sided (real > shuffled) p = {st.pvalue:.3g}")
    dg = (m.groove_recall_r - m.groove_recall_s).values
    print(f"  groove recall  real {m.groove_recall_r.mean():.3f}  "
          f"shuffled {m.groove_recall_s.mean():.3f}  difference {dg.mean():+.4f}")
    p2 = dict(d_iptm=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
              hi=float(np.percentile(bs, 97.5)), p=float(st.pvalue),
              d_groove=float(dg.mean()))
else:
    p2 = {}

# ---------------------------------------------------------------- P3 / P4
def cvpred(X, yy, folds=5):
    X = np.asarray(X, float)
    X = np.where(np.isfinite(X), X, 0.0)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    out = np.full(len(yy), np.nan)
    for tr, te in KFold(folds, shuffle=True, random_state=SEED).split(X):
        mo = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X[tr], yy[tr])
        out[te] = mo.predict(X[te])
    return out


STRUCT = [c for c in A.columns if not c.startswith("w_")]
COMPA = [c for c in A.columns if c.startswith("w_")]
blocks = {"composition": COMPA, "structure": STRUCT,
          "composition+structure": COMPA + STRUCT,
          "seqmodel": ["seqmodel"], "seqmodel+composition": ["seqmodel"] + COMPA,
          "seqmodel+structure": ["seqmodel"] + STRUCT,
          "seqmodel+composition+structure": ["seqmodel"] + COMPA + STRUCT}
print("\n---- P3/P4: cross-validated prediction of the measured partition ----")
print(f"{'block':34s} {'rho':>8s} {'R2':>8s}")
p34 = {}
for nm, cols in blocks.items():
    pr = cvpred(D[cols].values, y)
    rr = spearmanr(pr, y).statistic
    r2 = 1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p34[nm] = dict(rho=float(rr), r2=float(r2))
    print(f"{nm:34s} {rr:8.3f} {r2:8.3f}")

json.dump(dict(p1=p1, p2=p2, p34=p34, n=len(D),
               seqmodel_rho=float(rs), run_agreement=float(P.part_ga22.corr(P.part_ga24))),
          open(f"{C}/cm23_pilot.json", "w"), indent=1)
D.to_csv(f"{C}/pilot_scores.tsv", sep="\t")
print(f"\nwrote {C}/cm23_pilot.json and pilot_scores.tsv")
print("CM23_DONE")
