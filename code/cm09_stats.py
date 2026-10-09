"""cm09: cluster-bootstrap confidence intervals for the campaign's key comparisons.

All comparisons are paired: the same proteins, the same out-of-fold predictions, resampled together by 30 %
identity cluster (1,000 bootstrap replicates), so homologs move together and the interval reflects how many
independent protein families the result rests on. Statistic: mean over targets of R2 / ceiling (all targets, and
per target kind); for each pair, the difference and its 95 % interval.
"""
import os, json, glob
import numpy as np

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
F = np.load(f"{C}/folds.npz"); grp = F["group"]; has = F["has_features"]; n = len(Y)


def full_rows(targets):
    tix = [names.index(t) for t in targets]; return np.nonzero(has & np.isfinite(Y[:, tix]).any(1))[0]


def load(path, targets=None):
    """out-of-fold predictions placed in union rows (n x all targets, NaN elsewhere)"""
    P = np.full((n, len(names)), np.nan); oof = np.load(path); targets = targets or names
    rf = path.replace("_oof.npy", "_rows.npy"); rows = np.load(rf) if os.path.exists(rf) else full_rows(targets)
    assert len(rows) == len(oof), (path, len(rows), len(oof))
    for j, t in enumerate(targets): P[rows, names.index(t)] = oof[:, j]
    return P


M = {}
rd = np.load(f"{C}/ridge_oof.npz"); M["ridge layer 80"] = np.stack([rd[f"layer80|{t}"] for t in names], 1)
M["MLP (layers 50+80)"] = load(f"{C}/runs/s2/full_oof.npy")
for g in glob.glob(f"{C}/runs/attn_a1/*_oof.npy")[:1]: M["attention MLP"] = load(g)
T32 = [t for t in names if t != "GA33_input"]
for k, lab in (("seqonly", "sequence only (32 targets)"), ("covonly", "measured abundance + Tm only"), ("seqcov", "sequence + measured abundance + Tm")):
    if os.path.exists(f"{C}/runs/s3/{k}_oof.npy"): M[lab] = load(f"{C}/runs/s3/{k}_oof.npy", T32)
print({k: int(np.isfinite(v).any(1).sum()) for k, v in M.items()}, flush=True)

ug, ginv = np.unique(grp, return_inverse=True); rng = np.random.default_rng(0); B = 1000
W = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=B)        # cluster weights per replicate
KINDS = ["all", "abundance", "baseline", "enrichment", "salt", "temperature", "Mg"]


def stat(P, w, targets):
    """weighted R2/ceiling per target -> mean over targets of each kind"""
    out = {}
    vals = {}
    for t in targets:
        j = names.index(t); y = Y[:, j]; p = P[:, j]; k = np.isfinite(y) & np.isfinite(p)
        if k.sum() < 50: continue
        ww = w[ginv[k]].astype(float); yy, pp = y[k], p[k]; mu = (ww * yy).sum() / ww.sum()
        vals[t] = (1 - (ww * (yy - pp) ** 2).sum() / (ww * (yy - mu) ** 2).sum()) / meta[t]["ceiling"]
    for kd in KINDS:
        ts = [t for t in vals if kd == "all" or meta[t]["kind"] == kd]
        out[kd] = np.mean([vals[t] for t in ts]) if ts else np.nan
    return out


def common_targets(a, b): return [t for t in names if np.isfinite(M[a][:, names.index(t)]).any() and np.isfinite(M[b][:, names.index(t)]).any()]


PAIRS = [("MLP (layers 50+80)", "ridge layer 80"), ("attention MLP", "MLP (layers 50+80)"),
         ("sequence + measured abundance + Tm", "measured abundance + Tm only"), ("sequence + measured abundance + Tm", "sequence only (32 targets)"),
         ("measured abundance + Tm only", "sequence only (32 targets)")]
J = {}
ones = np.ones(len(ug))
for a, b in PAIRS:
    if a not in M or b not in M: continue
    ts = common_targets(a, b); obs_a, obs_b = stat(M[a], ones, ts), stat(M[b], ones, ts)
    diffs = {k: [] for k in KINDS}
    for r in range(B):
        sa, sb = stat(M[a], W[r], ts), stat(M[b], W[r], ts)
        for k in KINDS: diffs[k].append(sa[k] - sb[k])
    J[f"{a} minus {b}"] = {k: dict(a=float(obs_a[k]), b=float(obs_b[k]), diff=float(obs_a[k] - obs_b[k]),
                                   ci=[float(np.nanpercentile(diffs[k], 2.5)), float(np.nanpercentile(diffs[k], 97.5))]) for k in KINDS}
    print(f"{a} minus {b}: " + "  ".join(f"{k} {J[f'{a} minus {b}'][k]['diff']:+.3f} [{J[f'{a} minus {b}'][k]['ci'][0]:+.3f}, {J[f'{a} minus {b}'][k]['ci'][1]:+.3f}]" for k in KINDS), flush=True)
json.dump(J, open(f"{C}/cm09_stats.json", "w"), indent=1)
print("CM09_DONE")
