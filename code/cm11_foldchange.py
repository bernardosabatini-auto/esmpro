"""cm11: predicting each perturbation's pull-down fold change from sequence alone, with no information about the
input / lysate distribution (no abundance, no supernatant or partition targets, no baselines, no measured covariates).

Model (cm03, run s5 "ip_fold_changes"): ESMC-6B layers 50 + 80 mean-pooled + sequence covariates -> shared MLP ->
11 heads, one per pull-down fold change: GA_20 salt 75 / 150 mM, GA_22 and GA_24 KCl 60 / 120 mM (HSPB1); GA_33 heat
37 / 43 vs 35 C for DNAJA1 and DNAJB11; FS76 Mg2+ 2.5 vs 0.25 mM. Out-of-fold, folds grouped by 30 % identity.

Compared, per perturbation, with: the campaign's full multi-task model (also trained on abundance and baselines),
one MLP per perturbation, and ridge on ESMC layer 80. Practical read-outs: Spearman rho; recovery of the strongest
responders (of the measured top / bottom 10 %, the share the prediction places in its own top / bottom 10 %;
10 % expected by chance); sign agreement among the measured strongest 10 % by |fold change|. Paired cluster-bootstrap
95 % interval for fold-change-only minus full model.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pdpipe import plots as pl

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
meta = json.load(open(f"{C}/targets_meta.json")); Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
F = np.load(f"{C}/folds.npz"); has, grp = F["has_features"], F["group"]; n = len(Y)
FC = ["GA20_salt75", "GA20_salt150", "GA22_ip_salt60", "GA22_ip_salt120", "GA24_ip_salt60", "GA24_ip_salt120",
      "GA33_A1_heat37", "GA33_A1_heat43", "GA33_B11_heat37", "GA33_B11_heat43", "FS76_mg"]
NICE = {"GA20_salt75": "HSPB1, salt 75 mM (GA_20)", "GA20_salt150": "HSPB1, salt 150 mM (GA_20)", "GA22_ip_salt60": "HSPB1, KCl 60 (GA_22)",
        "GA22_ip_salt120": "HSPB1, KCl 120 (GA_22)", "GA24_ip_salt60": "HSPB1, KCl 60 (GA_24)", "GA24_ip_salt120": "HSPB1, KCl 120 (GA_24)",
        "GA33_A1_heat37": "DNAJA1, 37 vs 35 C", "GA33_A1_heat43": "DNAJA1, 43 vs 35 C", "GA33_B11_heat37": "DNAJB11, 37 vs 35 C",
        "GA33_B11_heat43": "DNAJB11, 43 vs 35 C", "FS76_mg": "HSPB1, Mg2+ 2.5 vs 0.25 mM"}


def place(oof_path, targets):
    P = np.full((n, len(names)), np.nan); oof = np.load(oof_path)
    rf = oof_path.replace("_oof.npy", "_rows.npy")
    rows = np.load(rf) if os.path.exists(rf) else np.nonzero(has & np.isfinite(Y[:, [names.index(t) for t in targets]]).any(1))[0]
    for j, t in enumerate(targets): P[rows, names.index(t)] = oof[:, j]
    return P


M = {"fold changes only": place(f"{C}/runs/s5/ip_fold_changes_oof.npy", FC), "full multi-task": place(f"{C}/runs/s2/full_oof.npy", names)}
S = np.full((n, len(names)), np.nan)
for t in FC:
    fn = f"{C}/runs/s2/single_{t}_oof.npy"
    if os.path.exists(fn): S[:, names.index(t)] = place(fn, [t])[:, names.index(t)]
M["one MLP per perturbation"] = S
rd = np.load(f"{C}/ridge_oof.npz"); M["ridge layer 80"] = np.stack([rd[f"layer80|{t}"] for t in names], 1)


def metrics(p, y):
    k = np.isfinite(p) & np.isfinite(y); p, y = p[k], y[k]
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    q = int(len(y) * .1); top_y, top_p = set(np.argsort(-y)[:q]), set(np.argsort(-p)[:q]); bot_y, bot_p = set(np.argsort(y)[:q]), set(np.argsort(p)[:q])
    strong = np.argsort(-np.abs(y))[:q]
    return dict(r2=float(r2), rho=float(spearmanr(p, y).correlation), top10_recovered=len(top_y & top_p) / q, bottom10_recovered=len(bot_y & bot_p) / q,
                sign_agree_strongest10=float((np.sign(p[strong]) == np.sign(y[strong] - np.median(y))).mean()) if False else float((np.sign(p[strong] - np.median(p)) == np.sign(y[strong] - np.median(y))).mean()), n=int(k.sum()))


J = {t: {m: metrics(P[:, names.index(t)], Y[:, names.index(t)]) for m, P in M.items()} for t in FC}
for t in FC:
    for m in J[t]: J[t][m]["r2_over_ceiling"] = J[t][m]["r2"] / meta[t]["ceiling"]
# paired cluster bootstrap: fold-changes-only minus full multi-task, mean R2/ceiling over the 11 perturbations
ug, gi = np.unique(grp, return_inverse=True); rng = np.random.default_rng(0); W = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=1000)
def wr2(p, y, w):
    k = np.isfinite(p) & np.isfinite(y); ww = w[gi[k]].astype(float); yy, pp = y[k], p[k]; mu = (ww * yy).sum() / ww.sum()
    return 1 - (ww * (yy - pp) ** 2).sum() / (ww * (yy - mu) ** 2).sum()
def mean_stat(P, w): return np.mean([wr2(P[:, names.index(t)], Y[:, names.index(t)], w) / meta[t]["ceiling"] for t in FC])
boot = {}
for other in ("full multi-task", "one MLP per perturbation", "ridge layer 80"):
    d = [mean_stat(M["fold changes only"], W[r]) - mean_stat(M[other], W[r]) for r in range(1000)]
    boot[other] = dict(diff=float(mean_stat(M["fold changes only"], np.ones(len(ug))) - mean_stat(M[other], np.ones(len(ug)))), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
J["_bootstrap_fold_changes_only_minus"] = boot
json.dump(J, open(f"{C}/cm11_foldchange.json", "w"), indent=1)

print(f"{'perturbation':28} {'ceiling':>7} " + " ".join(f"{m[:14]:>14}" for m in M) + "   rho  top10  bot10  sign")
for t in FC:
    print(f"{NICE[t]:28} {meta[t]['ceiling']:7.2f} " + " ".join(f"{J[t][m]['r2']:14.3f}" for m in M) +
          f"  {J[t]['fold changes only']['rho']:.2f}  {J[t]['fold changes only']['top10_recovered']:.2f}  {J[t]['fold changes only']['bottom10_recovered']:.2f}  {J[t]['fold changes only']['sign_agree_strongest10']:.2f}")
print("fold-changes-only minus:", {k: (round(v["diff"], 3), [round(x, 3) for x in v["ci"]]) for k, v in boot.items()})

# figure: predicted vs measured for all 11 perturbations
f, ax = plt.subplots(3, 4, figsize=(16, 11.5)); P = M["fold changes only"]
for a_, t in zip(ax.flat, FC):
    y = Y[:, names.index(t)]; p = P[:, names.index(t)]
    pl.scatter_reg(a_, p, y, "predicted fold change (sequence only, out-of-fold)", "measured log2 fold change", NICE[t], s=3)
    m_ = J[t]["fold changes only"]
    a_.text(.97, .03, f"R2/ceiling {m_['r2_over_ceiling']:.2f}\nrho {m_['rho']:.2f}\ntop-10% recovered {m_['top10_recovered']:.0%}", transform=a_.transAxes, ha="right", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=.8))
a_ = ax.flat[-1]; xx = np.arange(len(FC)); w = .2
for i, m in enumerate(M):
    a_.bar(xx + (i - 1.5) * w, [J[t][m]["r2_over_ceiling"] for t in FC], w, label=m, color=[pl.HIT, pl.LINE, "#e08e0b", pl.BASE][i])
a_.set_xticks(xx); a_.set_xticklabels([t.replace("GA33_", "").replace("_ip", "") for t in FC], rotation=60, ha="right", fontsize=6.5); a_.set_ylabel("R2 / ceiling"); a_.legend(fontsize=6)
a_.set_title("Four ways to predict each fold change", fontsize=9)
f.tight_layout(); f.savefig(f"{FIG}/fold_changes.png", dpi=140); plt.close(f)
print("CM11_DONE")
