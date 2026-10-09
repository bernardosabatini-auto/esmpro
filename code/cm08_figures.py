"""cm08: figures and tables for the campaign report, from whatever results exist (missing pieces are skipped).

Figures (reports/figures/campaign/):
  overview.png       per-target R2 / ceiling: ridge (layer 80), best MLP, attention MLP, MLP + measured covariates
  scatter.png        predicted vs measured for one target of each kind (best MLP, out-of-fold), with OLS fit
  single_multi.png   per-target R2: one-target MLPs against the multi-task MLP
  transfer.png       leave-one-run-out: run's targets read from a trunk that never saw them, vs trained with them,
                     vs predicted from the other runs' MEASURED targets
  regress_out.png    effect targets: R2 from the full latent vs with the baseline / abundance subspace removed
  geometry.png       cosine between targets' read-out directions in the latent (clustered)
  baseline.png       named properties: univariate rho with each baseline target; block R2
"""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from pdpipe import plots as pl

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"; os.makedirs(FIG, exist_ok=True)
meta = json.load(open(f"{C}/targets_meta.json")); names = list(meta); ceil = np.array([meta[t]["ceiling"] for t in names])
KIND_ORDER = ["abundance", "baseline", "enrichment", "salt", "temperature", "Mg"]
order = sorted(names, key=lambda t: (KIND_ORDER.index(meta[t]["kind"]), meta[t]["run"], t))
ridge = json.load(open(f"{C}/ridge.json"))["per_target"]
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); Y = Tz["Y"]
OUT = {}


def best_group(pattern_dirs, exclude=("single_", "loro_", "smoke")):
    best, bj = None, -1
    for d in pattern_dirs:
        for f in glob.glob(f"{d}/*.json"):
            if any(x in os.path.basename(f) for x in exclude) or f.endswith("summary.json") or "/cm04" in f: continue
            r = json.load(open(f))
            if "r2" in r and r["mean_r2_over_ceiling"] > bj: best, bj = (r, f), r["mean_r2_over_ceiling"]
    return best


R = {"ridge (ESMC layer 80)": {t: ridge[t]["layer80"] for t in names}}
b = best_group([f"{C}/runs/s1", f"{C}/runs/s2"])
if b: R[f"MLP, best ({b[0]['group']})"] = b[0]["r2"]; OUT["best_mlp"] = b[0]["group"]; best_oof_file = b[1].replace(".json", "_oof.npy"); best_dir = os.path.dirname(b[1])
ba = best_group([f"{C}/runs/attn_a1"])
if ba: R[f"attention MLP ({ba[0]['group']})"] = ba[0]["r2"]
if os.path.exists(f"{C}/runs/s3/seqcov.json"): R["MLP + measured abundance, Tm"] = json.load(open(f"{C}/runs/s3/seqcov.json"))["r2"]
if os.path.exists(f"{C}/runs/s3/covonly.json"): R["measured abundance, Tm only (MLP)"] = json.load(open(f"{C}/runs/s3/covonly.json"))["r2"]
OUT["mean_r2_over_ceiling"] = {k: float(np.nanmean([v.get(t, np.nan) / meta[t]["ceiling"] for t in names])) for k, v in R.items()}

# overview
f, ax = plt.subplots(figsize=(16, 6)); x = np.arange(len(order)); w = .8 / len(R)
cols = [pl.BASE, pl.HIT, "#6a3d9a", pl.LINE, "#e08e0b"]
for i, (k, v) in enumerate(R.items()):
    ax.bar(x + (i - (len(R) - 1) / 2) * w, [v.get(t, np.nan) / meta[t]["ceiling"] for t in order], w, color=cols[i % len(cols)], label=k)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=70, ha="right", fontsize=7); ax.set_ylabel("R2 / split-half ceiling (out-of-fold)")
prev = None
for i, t in enumerate(order):
    if meta[t]["kind"] != prev: ax.axvline(i - .5, color="k", lw=.4); ax.text(i - .4, .77, meta[t]["kind"], fontsize=8, va="top"); prev = meta[t]["kind"]
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.42)); ax.set_title("How much of each measured quantity sequence predicts, against what the measurement allows")
f.tight_layout(); f.savefig(f"{FIG}/overview.png", dpi=150); plt.close(f)

# scatter: predicted vs measured, best MLP
if b:
    oof = np.load(best_oof_file); rf = best_oof_file.replace("_oof.npy", "_rows.npy")
    rows = np.load(rf) if os.path.exists(rf) else np.nonzero(np.load(f"{C}/folds.npz")["has_features"] & np.isfinite(Y).any(1))[0]
    pick = ["GA22_sup_base", "GA33_A1_enrich", "GA22_part_base", "GA20_ip_base", "GA24_ip_salt120", "GA33_B11_heat43"]
    f, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.8))
    for a_, t in zip(ax, pick):
        y = Y[rows, names.index(t)]; p = oof[:, names.index(t)]
        pl.scatter_reg(a_, p, y, "predicted from sequence (out-of-fold)", f"measured, {t}", f"{t}\n{meta[t]['what']}", s=3)
    f.tight_layout(); f.savefig(f"{FIG}/scatter.png", dpi=140); plt.close(f)

# single vs multi-task
sing = {t: json.load(open(f"{C}/runs/s2/single_{t}.json"))["r2"][t] for t in names if os.path.exists(f"{C}/runs/s2/single_{t}.json")}
if sing and os.path.exists(f"{C}/runs/s2/full.json"):
    full = json.load(open(f"{C}/runs/s2/full.json"))["r2"]; ts = list(sing)
    f, a_ = plt.subplots(figsize=(5.2, 4.8))
    pl.scatter_reg(a_, [sing[t] for t in ts], [full[t] for t in ts], "R2, one MLP per target", "R2, one multi-task MLP (same architecture)",
                   "Does sharing across conditions help?", identity=True, robust_lim=False, s=18, labels=np.array(ts), highlight=np.ones(len(ts), bool))
    f.tight_layout(); f.savefig(f"{FIG}/single_multi.png", dpi=150); plt.close(f)
    OUT["single_vs_multi"] = dict(mean_single=float(np.mean([sing[t] for t in ts])), mean_multi=float(np.mean([full[t] for t in ts])),
                                  n_multi_better=int(sum(full[t] > sing[t] for t in ts)), n=len(ts))

# latent analyses (cm04)
c4 = f"{C}/runs/s2/cm04_full.json"
if os.path.exists(c4):
    L = json.load(open(c4))
    if "transfer" in L:
        ts = [t for t in order if t in L["transfer"]]; x = np.arange(len(ts))
        f, a_ = plt.subplots(figsize=(15, 4.4))
        a_.bar(x - .27, [L["full_readout"][t]["r2"] for t in ts], .27, color=pl.HIT, label="latent of the model trained WITH the run")
        a_.bar(x, [L["transfer"][t]["left_out"] for t in ts], .27, color=pl.LINE, label="latent of a model that never saw the run")
        a_.bar(x + .27, [L["measured_cross_run"][t]["r2"] for t in ts], .27, color="#e08e0b", label="the other runs' MEASURED targets (no sequence)")
        a_.set_xticks(x); a_.set_xticklabels(ts, rotation=70, ha="right", fontsize=7); a_.set_ylabel("R2 (out-of-fold)"); a_.legend(fontsize=7)
        a_.set_title("Transfer across conditions: is a run predictable from what the model learned on the others?")
        f.tight_layout(); f.savefig(f"{FIG}/transfer.png", dpi=150); plt.close(f)
    ro = L["regress_out"]; ts = [t for t in order if t in ro]; x = np.arange(len(ts))
    f, a_ = plt.subplots(figsize=(12, 4.2))
    a_.bar(x - .27, [ro[t]["full"] for t in ts], .27, color=pl.HIT, label="full latent")
    a_.bar(x, [ro[t]["minus_first_baseline_axis"] for t in ts], .27, color="#e08e0b", label="minus the first baseline axis")
    a_.bar(x + .27, [ro[t]["minus_baseline_subspace"] for t in ts], .27, color=pl.LINE, label="minus the whole baseline / abundance subspace")
    a_.set_xticks(x); a_.set_xticklabels(ts, rotation=60, ha="right", fontsize=7); a_.set_ylabel("R2 (out-of-fold)"); a_.legend(fontsize=7)
    a_.set_title("Regressing the baseline out of the latent: what of each condition's effect is condition-specific?")
    f.tight_layout(); f.savefig(f"{FIG}/regress_out.png", dpi=150); plt.close(f)
    D = np.array(L["direction_cosine"]["matrix"]); lk = leaves_list(linkage(D, "average", metric="correlation")); nm = np.array(L["direction_cosine"]["names"])[lk]
    f, a_ = plt.subplots(figsize=(10, 9)); im = a_.imshow(D[np.ix_(lk, lk)], cmap="RdBu_r", vmin=-1, vmax=1)
    a_.set_xticks(range(len(nm))); a_.set_xticklabels(nm, rotation=90, fontsize=6); a_.set_yticks(range(len(nm))); a_.set_yticklabels(nm, fontsize=6)
    plt.colorbar(im, ax=a_, shrink=.7, label="cosine of read-out directions in the shared latent"); a_.set_title("Which conditions the model represents along the same axes")
    f.tight_layout(); f.savefig(f"{FIG}/geometry.png", dpi=150); plt.close(f)

# baseline properties (cm07)
if os.path.exists(f"{C}/cm07_baseline.json"):
    B = json.load(open(f"{C}/cm07_baseline.json")); ts = list(B["univariate"]); props = list(B["univariate"][ts[0]])
    M = np.array([[B["univariate"][t][p] for t in ts] for p in props])
    f, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = ax[0].imshow(M, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if abs(v) >= .2: ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
    ax[0].set_xticks(range(len(ts))); ax[0].set_xticklabels(ts, rotation=60, ha="right", fontsize=7); ax[0].set_yticks(range(len(props))); ax[0].set_yticklabels(props, fontsize=7)
    plt.colorbar(im, ax=ax[0], shrink=.6, label="Spearman rho"); ax[0].set_title("A  Named properties against each baseline", fontsize=10)
    steps = list(B["blocks"][ts[0]]); x = np.arange(len(ts)); w = .8 / len(steps)
    for i, s_ in enumerate(steps):
        ax[1].bar(x + (i - (len(steps) - 1) / 2) * w, [B["blocks"][t][s_]["r2_over_ceiling"] for t in ts], w, label=s_)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ts, rotation=60, ha="right", fontsize=7); ax[1].set_ylabel("R2 / ceiling (out-of-fold)"); ax[1].legend(fontsize=6)
    ax[1].set_title("B  Cumulative blocks of named properties, and the sequence model", fontsize=10)
    f.tight_layout(); f.savefig(f"{FIG}/baseline.png", dpi=150); plt.close(f)
json.dump(OUT, open(f"{C}/cm08_summary.json", "w"), indent=1)
print(json.dumps(OUT, indent=1)); print("CM08_DONE")
