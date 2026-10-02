"""ga05: what does sequence explain about the GA_33 fold changes?

Same protocol as the salt titration (pd05/pd08): ridge, penalty over a fixed grid, 5-fold
cross-validation with folds grouped by MMseqs2 cluster at 30 % identity, predictive R2 on held-out
proteins with 2,000-resample bootstrap intervals and paired bootstrap intervals on increments.
Every R2 is read against the measurement ceiling, which for an R2 is the target's reliability.

Targets (ga01):
  stau_<chap>_<T>      STAU_10 vs control, per co-chaperone and temperature
  stau_avg_<T>         the same effect averaged over the two co-chaperones. Their true effects
                       correlate at 0.83-1.0 after correcting for noise, so the average is the
                       same quantity measured with twice the replicates
  temp_<chap>_<T>v35   temperature vs 35 C in the controls
  temp_avg_<T>v35      averaged over the two co-chaperones

Feature sets, nested: amino-acid composition (20-d), physicochemical (31-d: composition, length,
charge, GRAVY, aromaticity, disorder-prone fraction, charge segregation, MW), ESMC-6B layer 50
mean-pool (2560-d, the best layer on the salt data), and ESMC + composition.
"""
import os, csv, json, itertools
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D, P = f"{ROOT}/data/ga_data", f"{ROOT}/data/pd_data"
rng = np.random.default_rng(0)
NB = 2000
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip("AVLIPFMWGSTCYNQDEKRH", [1.8, 4.2, 3.8, 4.5, -1.6, 2.8, 1.9, -0.9, -0.4, -0.8,
                                       -0.7, 2.5, -1.3, -3.5, -3.5, -3.5, -3.5, -3.9, -4.5, -3.2]))

blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
status = {r["sheet_accession"]: r for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
res = np.array([status[s]["resolved_accession"] if s in status else s for s in acc])
has = np.array([status.get(s, {}).get("status", "notfound") != "notfound" for s in acc])


# --- embeddings: proteins shared with the salt data come from its pass, the rest from ga03
def fasta_keys(fn):
    return [ln[1:].strip().split()[0].split("|")[0] for ln in open(fn) if ln[0] == ">"]


old = np.load(f"{P}/embeddings_layers.npz", allow_pickle=True)
new = np.load(f"{D}/embeddings_layers_new.npz", allow_pickle=True)
LAY = [int(x) for x in old["layers"]]
assert LAY == [int(x) for x in new["layers"]]
ko, kn = fasta_keys(f"{P}/sequences.fasta"), fasta_keys(f"{D}/new_sequences.fasta")
assert len(ko) == old["pool"].shape[0] and len(kn) == new["pool"].shape[0]
li50, li80 = LAY.index(50), LAY.index(80)
src = {k: ("o", i) for i, k in enumerate(ko)}
src.update({k: ("n", i) for i, k in enumerate(kn)})
E50 = np.full((len(acc), 2560), np.nan, np.float32)
E80 = np.full((len(acc), 2560), np.nan, np.float32)
po, pn = old["pool"], new["pool"]
for i, r in enumerate(res):
    if has[i] and r in src:
        w, j = src[r]
        E50[i] = (po if w == "o" else pn)[j, li50]
        E80[i] = (po if w == "o" else pn)[j, li80]
emb_ok = np.isfinite(E50).all(1)
print(f"{len(acc)} protein groups; {emb_ok.sum()} with an embedding "
      f"({sum(1 for r in res[emb_ok] if src[r][0]=='o')} from the salt pass, "
      f"{sum(1 for r in res[emb_ok] if src[r][0]=='n')} new)", flush=True)

seq, cur = {}, None
for ln in open(f"{D}/sequences.fasta"):
    if ln[0] == ">":
        cur = ln[1:].strip().split()[0].split("|")[0]; seq[cur] = []
    else:
        seq[cur].append(ln.strip())
S = ["".join(seq.get(r, [])) for r in res]
L_ = np.array([max(len(s), 1) for s in S])
comp = np.array([[s.count(c) / max(len(s), 1) for c in AA] for s in S])
phys = np.hstack([comp, np.log10(L_)[:, None], np.array([[
    (s.count("K") + s.count("R") - s.count("D") - s.count("E")),
    (s.count("K") + s.count("R") - s.count("D") - s.count("E")) / max(len(s), 1),
    (s.count("K") + s.count("R")) / max(len(s), 1), (s.count("D") + s.count("E")) / max(len(s), 1),
    (s.count("K") + s.count("R") + s.count("D") + s.count("E")) / max(len(s), 1),
    np.mean([KD.get(c, 0) for c in s]) if s else 0.0,
    (s.count("F") + s.count("W") + s.count("Y")) / max(len(s), 1),
    sum(s.count(c) for c in "PESTQKRG") / max(len(s), 1),
    np.std([1 if c in "KR" else (-1 if c in "DE" else 0) for c in s[:: max(1, len(s) // 200)]]) if s else 0.0,
    len(s) * 110.0 / 1000.0] for s in S])])
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(r, r) for r in res])
FEATS = {"composition": comp, "physicochemical": phys, "ESMC L50": E50, "ESMC L50 + composition":
         np.hstack([E50, comp]), "ESMC L80 (last)": E80}


# --- targets, including the co-chaperone averages and their ceilings
def split_rel(pairs, rows):
    """pairs: list of (treated (n,6), control (n,6)) averaged into one target; split-half by BR."""
    rs = []
    for tri in itertools.combinations(range(6), 3):
        if 0 not in tri:
            continue
        o = [i for i in range(6) if i not in tri]
        with np.errstate(invalid="ignore"):
            y1 = np.mean([np.nanmean(B[rows][:, list(tri)], 1) - np.nanmean(A[rows][:, list(tri)], 1) for B, A in pairs], 0)
            y2 = np.mean([np.nanmean(B[rows][:, o], 1) - np.nanmean(A[rows][:, o], 1) for B, A in pairs], 0)
        k = np.isfinite(y1) & np.isfinite(y2)
        rs.append(np.corrcoef(y1[k], y2[k])[0, 1])
    rh = float(np.mean(rs))
    return 2 * rh / (1 + rh)


TG = {}
for c, t in itertools.product(("21A", "24"), ("35", "37", "43")):
    TG[f"stau_{c}_{t}"] = [(blk[f"br_{c}_10_{t}"], blk[f"br_{c}_0_{t}"])]
for t in ("35", "37", "43"):
    TG[f"stau_avg_{t}"] = [(blk[f"br_{c}_10_{t}"], blk[f"br_{c}_0_{t}"]) for c in ("21A", "24")]
for c, t in itertools.product(("21A", "24"), ("37", "43")):
    TG[f"temp_{c}_{t}v35"] = [(blk[f"br_{c}_0_{t}"], blk[f"br_{c}_0_35"])]
for t in ("37", "43"):
    TG[f"temp_avg_{t}v35"] = [(blk[f"br_{c}_0_{t}"], blk[f"br_{c}_0_35"]) for c in ("21A", "24")]


def keep_for(name):
    if "_avg_" in name:
        kind, _, t = name.split("_", 2)
        return np.all([blk[f"keep_{kind}_{c}_{t}"] for c in ("21A", "24")], 0)
    return blk[f"keep_{name}"].astype(bool)


def yof(name):
    if "_avg_" in name:
        kind, _, t = name.split("_", 2)
        return np.mean([blk[f"y_{kind}_{c}_{t}"] for c in ("21A", "24")], 0)
    return blk[f"y_{name}"].astype(float)


def oof(X, y, gr, al, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr]).predict(X[te])
    return p


def r2s(y, p, s=None):
    if s is not None:
        y, p = y[s], p[s]
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def fit(X, y, folds, alphas):
    best = (-9e9, None, None)
    for al in alphas:
        p = oof(X, y, None, al, folds)
        v = r2s(y, p)
        if v > best[0]:
            best = (v, al, p)
    return best


ALPHAS = {"composition": (1, 10, 100, 1e3), "physicochemical": (1, 10, 100, 1e3),
          "ESMC L50": (1e3, 3e3, 1e4, 3e4, 1e5), "ESMC L50 + composition": (1e3, 3e3, 1e4, 3e4, 1e5),
          "ESMC L80 (last)": (1e3, 3e3, 1e4, 3e4, 1e5)}
out = {}
hdr = (f"{'target':<17}{'n':>6}{'sd':>7}{'ceiling':>9}{'comp':>8}{'physchem':>10}{'ESMC':>8}"
       f"{'ESMC-comp [95% CI]':>24}{'+comp':>8}{'L80':>8}{'of ceiling':>12}")
print("\nvariance explained (predictive R2, held-out sequence clusters); ceiling = reliability\n" + hdr, flush=True)
for name, pairs in TG.items():
    y = yof(name)
    k = keep_for(name) & np.isfinite(y) & emb_ok
    rows = np.nonzero(k)[0]
    yk, gk = y[k], groups[k]
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(yk)), None, gk))
    ceil = split_rel(pairs, rows)
    R, P_ = {}, {}
    for fn, X in FEATS.items():
        v, al, p = fit(X[k], yk, folds, ALPHAS[fn])
        R[fn], P_[fn] = dict(r2=float(v), alpha=float(al), r=float(stats.pearsonr(yk, p)[0])), p
    i = np.arange(len(yk))
    bs = [rng.choice(i, len(i), replace=True) for _ in range(NB)]
    d = np.array([r2s(yk, P_["ESMC L50"], s) - r2s(yk, P_["composition"], s) for s in bs])
    ci_e = [float(np.percentile([r2s(yk, P_["ESMC L50"], s) for s in bs[:500]], q)) for q in (2.5, 97.5)]
    inc = R["ESMC L50"]["r2"] - R["composition"]["r2"]
    on_top = R["ESMC L50 + composition"]["r2"] - R["ESMC L50"]["r2"]
    out[name] = dict(n=int(k.sum()), sd=float(yk.std()), ceiling=float(ceil), models=R,
                     esmc_ci=ci_e, esmc_minus_comp=float(inc),
                     esmc_minus_comp_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     comp_on_top=float(on_top), frac_of_ceiling=float(R["ESMC L50"]["r2"] / ceil))
    print(f"{name:<17}{k.sum():>6}{yk.std():>7.3f}{100*ceil:>8.1f}%{100*R['composition']['r2']:>7.1f}%"
          f"{100*R['physicochemical']['r2']:>9.1f}%{100*R['ESMC L50']['r2']:>7.1f}%"
          f"{100*inc:>+8.1f} [{100*np.percentile(d,2.5):+.1f},{100*np.percentile(d,97.5):+.1f}]"
          f"{100*on_top:>+8.2f}{100*R['ESMC L80 (last)']['r2']:>7.1f}%{100*R['ESMC L50']['r2']/ceil:>11.0f}%",
          flush=True)

# shuffled-label control on the cleanest target
name = "temp_avg_43v35"
y = yof(name); k = keep_for(name) & np.isfinite(y) & emb_ok
yk = y[k]; sh = rng.permutation(len(yk))
folds = list(GroupKFold(n_splits=5).split(np.zeros(len(yk)), None, groups[k]))
v, al, _ = fit(E50[k], yk[sh], folds, (1e4,))
out["_control_shuffled"] = dict(target=name, r2=float(v))
print(f"\nshuffled-label control ({name}, ESMC L50): R2 = {v:+.4f}  (must be <= 0)", flush=True)
json.dump(out, open(f"{D}/ga05_results.json", "w"), indent=1)
print(f"wrote {D}/ga05_results.json\nGA05_DONE", flush=True)
