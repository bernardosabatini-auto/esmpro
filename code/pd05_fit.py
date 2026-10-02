"""pd05: measurement ceiling, baselines, and linear models for salt-driven fold change.

Runs in `dendritic_env`, which has scikit-learn; the embedding half ran in `proteinae`, which
has torch. The two halves meet at data/pd_data/*.npz.

Order matters here. The ceiling is computed FIRST, because a model correlation of 0.4 means
something entirely different against a target whose own reproducibility is 0.5 than against one
at 0.95. The ceiling comes from the five biological replicates: split them into disjoint halves,
compute the fold change twice, and see how well the experiment agrees with itself. Splitting the
ten columns at random instead would separate technical pairs and measure instrument noise, which
would flatter the experiment.

Then baselines, in increasing order of information, before anything with 2,560 features is
allowed to claim credit. For a salt titration the physicochemical baseline is not a formality:
ionic strength acts on electrostatics, so net charge and charge fraction are the mechanistically
motivated predictors, and the sharp question is whether the embedding beats them.

  dendritic_env/bin/python pd05_fit.py
"""
import os, sys, csv, json, argparse, collections
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/pd_data"
ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("--out", default=f"{D}/pd05_results.json")
a = ap.parse_args()
rng = np.random.default_rng(0)

blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
emb = np.load(f"{D}/embeddings.npz", allow_pickle=True)
acc_b = np.array([str(x) for x in blk["accession"]])
acc_e = np.array([str(x) for x in emb["accession"]])
# the 8 accessions the find/replace damaged are keyed under their REPAIRED form in the
# embedding file, so go through the status table rather than matching the sheet string
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
pos = {s: i for i, s in enumerate(acc_e)}
ei = np.array([pos.get(resolved.get(s, s), -1) for s in acc_b])
assert (ei >= 0).all(), f"{(ei < 0).sum()} proteins have no embedding"

cl = {}
for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t"):
    cl[r["accession"]] = r["cluster"]
groups = np.array([cl.get(s, s) for s in acc_b])


# ------------------------------------------------------------------ the ceiling
def half_split_reliability(br_cond, br_base, keep, n_draw=40):
    """Correlation between two fold-change estimates from DISJOINT biological replicates.

    br_* are (n, 5): technical pairs already averaged, so each column is one biological
    replicate. Two disjoint pairs of replicates give two independent estimates of the same
    quantity; their correlation across proteins is the experiment agreeing with itself.
    Spearman-Brown then rescales that to the reliability of the 5-replicate estimate we
    actually model, and its square root is the highest correlation any predictor could reach
    against the observed target.
    """
    rs = []
    idx = np.arange(5)
    for _ in range(n_draw):
        p = rng.permutation(idx)
        A, B = p[:2], p[2:4]
        ya = np.nanmean(br_cond[:, A], 1) - np.nanmean(br_base[:, A], 1)
        yb = np.nanmean(br_cond[:, B], 1) - np.nanmean(br_base[:, B], 1)
        m = keep & np.isfinite(ya) & np.isfinite(yb)
        if m.sum() > 100:
            rs.append(stats.pearsonr(ya[m], yb[m])[0])
    r_half = float(np.mean(rs))
    k = 2.5                                      # 5 replicates vs the 2 used per half
    r_full = k * r_half / (1 + (k - 1) * r_half)
    return r_half, float(r_full), float(np.sqrt(max(r_full, 0))), len(rs)


# ------------------------------------------------- physicochemical features
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
MW = dict(zip(AA, [71.08, 103.14, 115.09, 129.12, 147.18, 57.05, 137.14, 113.16, 128.17, 113.16,
                   131.19, 114.10, 97.12, 128.13, 156.19, 87.08, 101.10, 99.13, 186.21, 163.18]))
PKA = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
POS, NEG = "KR", "DE"
DISORDER = set("PESKQRGAND")          # disorder-promoting; the rest are order-promoting


def read_fasta(path):
    out, acc, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(buf)
            acc = line[1:].split("|")[0].strip(); buf = []
        else:
            buf.append(line.strip())
    if acc:
        out[acc] = "".join(buf)
    return out


def charge_at_ph(seq, ph=7.0):
    c = 0.0
    for r, pk in (("K", PKA["K"]), ("R", PKA["R"]), ("H", PKA["H"])):
        c += seq.count(r) / (1 + 10 ** (ph - pk))
    for r, pk in (("D", PKA["D"]), ("E", PKA["E"]), ("C", PKA["C"]), ("Y", PKA["Y"])):
        c -= seq.count(r) / (1 + 10 ** (pk - ph))
    return c


def kappa(seq):
    """Charge segregation: 1 means + and - residues are separated into blocks, 0 mixed."""
    v = np.array([1 if c in POS else (-1 if c in NEG else 0) for c in seq], float)
    if np.abs(v).sum() < 2:
        return 0.0
    w = 10
    blocks = [v[i:i + w] for i in range(0, len(v) - w + 1, w)]
    if len(blocks) < 2:
        return 0.0
    sigma = [((b > 0).sum() - (b < 0).sum()) ** 2 / max((np.abs(b)).sum(), 1) for b in blocks]
    return float(np.mean(sigma))


def physchem(seq):
    L = max(len(seq), 1)
    comp = np.array([seq.count(c) / L for c in AA])
    q = charge_at_ph(seq)
    return np.concatenate([[
        np.log10(L),
        sum(MW.get(c, 110.0) for c in seq) / 1000.0,
        q, q / L,
        (seq.count("K") + seq.count("R")) / L,
        (seq.count("D") + seq.count("E")) / L,
        (seq.count("K") + seq.count("R") + seq.count("D") + seq.count("E")) / L,
        np.mean([KD.get(c, 0.0) for c in seq]) if seq else 0.0,
        (seq.count("F") + seq.count("W") + seq.count("Y")) / L,
        sum(1 for c in seq if c in DISORDER) / L,
        kappa(seq),
    ], comp])


PC_NAMES = ["log_len", "mw_kda", "net_charge", "charge_per_res", "frac_KR", "frac_DE",
            "frac_charged", "gravy", "aromatic", "frac_disorder_aa", "charge_segregation"] + [f"aa_{c}" for c in AA]
seqs = read_fasta(f"{D}/sequences.fasta")
PC = np.array([physchem(seqs.get(resolved.get(s, s), "")) for s in acc_b])

MEAN = emb["mean_pool"][ei]
MAXP = emb["max_pool"][ei]
def clean(M):
    """Replace any non-finite entry with that column's median; ridge cannot take NaN."""
    M = np.asarray(M, dtype=float).copy()
    for j in range(M.shape[1]):
        c = M[:, j]
        bad = ~np.isfinite(c)
        if bad.any():
            c[bad] = np.nanmedian(c[~bad]) if (~bad).any() else 0.0
    return M


# The XIC column is a total across ALL runs: it correlates better with the mean of all three
# conditions (0.807) than with the baseline alone (0.773), and together with the baseline it
# reconstructs the condition means at r = 0.89 / 0.79. Predicting a fold change computed from
# those same runs with it is leakage, so it is excluded. (It was what lifted a three-feature
# "abundance" model to r = 0.81 while its parts scored 0.40, 0.15 and 0.09 alone.)
with np.errstate(invalid="ignore"):
    ABUND = clean(np.column_stack([np.nanmean(blk["br_base"], 1),
                                   np.log10(np.maximum(blk["npeptides"], 1))]))
PC = clean(PC)

print(f"{len(acc_b)} proteins | embedding {MEAN.shape[1]}d | physchem {PC.shape[1]}d | "
      f"{len(set(groups))} sequence clusters", flush=True)


# ------------------------------------------------------------------ evaluation
def grouped_pred(X, y, w, gr, model, folds):
    """Out-of-fold predictions with folds grouped by sequence cluster."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, gr):
        m = model()
        try:
            m.fit(X[tr], y[tr], **({"ridge__sample_weight": w[tr]} if w is not None and hasattr(m, "named_steps") else {}))
        except TypeError:
            m.fit(X[tr], y[tr])
        oof[te] = np.asarray(m.predict(X[te])).ravel()
    return oof


def score(y, p, nboot):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    r = stats.pearsonr(y, p)[0]
    rho = stats.spearmanr(y, p)[0]
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    idx = np.arange(len(y))
    bs = [stats.pearsonr(y[s], p[s])[0] for s in (rng.choice(idx, len(idx), replace=True) for _ in range(nboot))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(r=float(r), r_lo=float(lo), r_hi=float(hi), spearman=float(rho), r2=float(r2), n=int(len(y)))


results = {}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    y = blk[f"y_{cond}"].astype(float)
    se = blk[f"se_{cond}"].astype(float)
    keep = blk[f"keep_{cond}"].astype(bool)
    print(f"\n{'='*78}\n{label} vs 0 mM  —  {int(keep.sum())} proteins pass the 7-of-10 rule", flush=True)

    r_half, r_full, ceil, nd = half_split_reliability(blk[f"br_{cond}"], blk["br_base"], keep)
    print(f"\n  MEASUREMENT CEILING (from {nd} disjoint biological-replicate splits)", flush=True)
    print(f"    half-vs-half agreement        r = {r_half:.3f}", flush=True)
    print(f"    implied reliability of target R = {r_full:.3f}", flush=True)
    print(f"    best correlation any model    r = {ceil:.3f}  <- read every score below against this", flush=True)

    k = keep & np.isfinite(y) & np.isfinite(se) & (se > 0)
    yk, sek, grk = y[k], se[k], groups[k]
    w = 1.0 / np.maximum(sek, 1e-3) ** 2
    w = w / w.mean()

    feats = {
        "null (global mean)":        np.zeros((k.sum(), 1)),
        "baseline abundance":        ABUND[k],
        "physicochemical":           PC[k],
        "ESMC mean-pool":            MEAN[k],
        "ESMC mean+max":             np.hstack([MEAN[k], MAXP[k]]),
        "ESMC + physicochemical":    np.hstack([MEAN[k], PC[k]]),
        "ESMC + abundance":          np.hstack([MEAN[k], ABUND[k]]),
        "ESMC + abundance + physchem": np.hstack([MEAN[k], ABUND[k], PC[k]]),
    }
    alphas = [10.0, 100.0, 1000.0, 1e4, 1e5]
    res = {}
    for name, X in feats.items():
        X = clean(X)
        if name.startswith("null"):
            p = grouped_pred(X, yk, None, grk, lambda: DummyRegressor(strategy="mean"), a.folds)
        else:
            best, bestr = None, -9
            for al in alphas:
                mk = lambda al=al: make_pipeline(StandardScaler(), Ridge(alpha=al))
                pp = grouped_pred(X, yk, None, grk, mk, a.folds)
                rr = stats.pearsonr(yk, pp)[0]
                if rr > bestr:
                    best, bestr, p = al, rr, pp
            res[name] = score(yk, p, a.boot); res[name]["alpha"] = best
            print(f"    {name:26s} r = {res[name]['r']:+.3f} [{res[name]['r_lo']:+.3f},{res[name]['r_hi']:+.3f}]  "
                  f"rho {res[name]['spearman']:+.3f}  R2 {res[name]['r2']:+.3f}  (alpha {best:g})", flush=True)
            continue
        res[name] = score(yk, p, a.boot)
        print(f"\n  MODELS (5-fold, folds grouped by sequence cluster)", flush=True)
        print(f"    {name:26s} r = {res[name]['r']:+.3f}  R2 {res[name]['r2']:+.3f}  (the null)", flush=True)

    # k-NN in embedding space: the family-lookup control
    p = grouped_pred(clean(MEAN[k]), yk, None, grk,
                     lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")), a.folds)
    res["kNN in embedding space"] = score(yk, p, a.boot)
    print(f"    {'kNN in embedding space':26s} r = {res['kNN in embedding space']['r']:+.3f} "
          f"[{res['kNN in embedding space']['r_lo']:+.3f},{res['kNN in embedding space']['r_hi']:+.3f}]", flush=True)

    # PLS
    p = grouped_pred(clean(MEAN[k]), yk, None, grk, lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=12)), a.folds)
    res["PLS (12 components)"] = score(yk, p, a.boot)
    print(f"    {'PLS (12 components)':26s} r = {res['PLS (12 components)']['r']:+.3f} "
          f"[{res['PLS (12 components)']['r_lo']:+.3f},{res['PLS (12 components)']['r_hi']:+.3f}]", flush=True)

    # weighted ridge
    al = res["ESMC mean-pool"]["alpha"]
    oof = np.full(k.sum(), np.nan)
    MK = clean(MEAN[k])
    for tr, te in GroupKFold(n_splits=a.folds).split(MK, yk, grk):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=al))
        pipe.fit(MK[tr], yk[tr], ridge__sample_weight=w[tr])
        oof[te] = pipe.predict(MK[te])
    res["ESMC ridge, SE-weighted"] = score(yk, oof, a.boot)
    print(f"    {'ESMC ridge, SE-weighted':26s} r = {res['ESMC ridge, SE-weighted']['r']:+.3f} "
          f"[{res['ESMC ridge, SE-weighted']['r_lo']:+.3f},{res['ESMC ridge, SE-weighted']['r_hi']:+.3f}]", flush=True)

    # leakage check: the same ridge on RANDOM folds, and a label-shuffle control
    rand_groups = rng.permutation(np.arange(k.sum()) % a.folds)
    p_rand = grouped_pred(MK, yk, None, rand_groups, lambda: make_pipeline(StandardScaler(), Ridge(alpha=al)), a.folds)
    res["[random folds, leaky]"] = score(yk, p_rand, a.boot)
    ysh = rng.permutation(yk)
    p_sh = grouped_pred(MK, ysh, None, grk, lambda: make_pipeline(StandardScaler(), Ridge(alpha=al)), a.folds)
    res["[shuffled labels]"] = score(ysh, p_sh, a.boot)
    print(f"\n  CONTROLS", flush=True)
    print(f"    {'random (leaky) folds':26s} r = {res['[random folds, leaky]']['r']:+.3f}  "
          f"<- inflation from ignoring homology: {res['[random folds, leaky]']['r'] - res['ESMC mean-pool']['r']:+.3f}", flush=True)
    print(f"    {'shuffled labels':26s} r = {res['[shuffled labels]']['r']:+.3f}  <- must be ~0", flush=True)

    best = max((v["r"], kk) for kk, v in res.items() if not kk.startswith("["))
    print(f"\n  best honest model: {best[1]} at r = {best[0]:.3f}, "
          f"which is {100*best[0]/ceil:.0f}% of the ceiling ({ceil:.3f})", flush=True)
    results[cond] = {"ceiling": {"r_half": r_half, "reliability": r_full, "max_r": ceil},
                     "n": int(k.sum()), "models": res}

json.dump(results, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
print("PD05_DONE")
