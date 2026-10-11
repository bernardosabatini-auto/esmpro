"""cm24: the figure for the Design A pilot.

Six panels, chosen so that a reader can see the answer rather than take the summary table's word:
  A  validation gate -- the held-out 9ef3 peptide's position against experiment.
  B  the window-level score distribution with every control marked on it.
  C  per-protein interface score vs the measured bound:free partition, with the regression.
  D  the same by decile of the measurement, which is how the proteins were sampled.
  E  real windows vs their own shuffled twins, paired.
  F  cross-validated prediction from each block of features.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C = f"{ROOT}/data/campaign"
FIG = f"{ROOT}/reports/figures/campaign"
os.makedirs(FIG, exist_ok=True)

W = pd.read_csv(f"{C}/pilot_windows.tsv", sep="\t")
D = pd.read_csv(f"{C}/pilot_scores.tsv", sep="\t", index_col=0)
J = json.load(open(f"{C}/cm23_pilot.json"))
V = pd.read_csv(f"{C}/cm22_val.tsv", sep="\t")
done = W.iptm_pep.notna()
win = W[(W.kind == "window") & done]
shuf = W[(W.kind == "shuffled") & done]
ctrl = W[W.kind.isin(["pos_heldout", "pos_intrain", "neg"]) & done]

# Choose the score to display by its SPLIT-HALF RELIABILITY, not by how large its correlation came
# out. Picking the biggest |rho| would select the noisiest feature -- here it picks the one with the
# lowest ceiling of all -- and would make a chance extreme look like the headline.
_R25 = json.load(open(f"{C}/cm25_reliability.json"))["aggregate_reliability"]
BEST = max([c for c in _R25 if c in J["p1"]], key=lambda c: _R25[c]["full"])

fig, ax = plt.subplots(2, 4, figsize=(21.5, 9.8))
fig.subplots_adjust(hspace=0.36, wspace=0.30)

# ---- A: validation gate ----------------------------------------------------
a = ax[0, 0]
v = V.set_index("name")
rows = [("apo ACD dimer\n(fold only)", v.loc["VAL_apo", "monomer_rmsd"], "#888"),
        ("held-out 9ef3 peptide\n(forward threading)", v.loc["VAL_pos_9ef3_ixi9", "pep_rmsd_fwd"], "#c44"),
        ("held-out 9ef3 peptide\n(reverse threading)", v.loc["VAL_pos_9ef3_ixi9", "pep_rmsd_rev"], "#2a7")]
a.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
for i, r in enumerate(rows):
    a.text(r[1] + 0.3, i, f"{r[1]:.2f} A", va="center", fontsize=9)
a.axvline(4, ls="--", c="k", lw=0.8)
a.text(4.1, -0.45, "4 A", fontsize=8)
a.set_xlabel("Ca RMSD to experiment (A)")
a.set_title("A  Validation gate: 9ef3, released 2026-06,\nafter Boltz-2 was trained", fontsize=10)
a.set_xlim(0, max(r[1] for r in rows) * 1.25)

# ---- B: window score distribution + controls --------------------------------
b = ax[0, 1]
b.hist(win.iptm_pep, bins=45, color="#bbb", edgecolor="none", label=f"real windows (n={len(win)})")
b.hist(shuf.iptm_pep, bins=45, histtype="step", color="#444", lw=1.3,
       label=f"shuffled twins (n={len(shuf)})")
cols = {"pos_heldout": "#2a7", "pos_intrain": "#7a2", "neg": "#c44"}
for _, r in ctrl.iterrows():
    b.axvline(r.iptm_pep, color=cols[r.kind], lw=1.4,
              ls="-" if r.kind != "pos_intrain" else "--")
for k, lab in [("pos_heldout", "held-out IxI positives"), ("pos_intrain", "in-training positives"),
               ("neg", "homopolymer negatives")]:
    b.plot([], [], color=cols[k], lw=1.4, label=lab)
b.set_xlabel("interface confidence (ipTM, peptide vs ACD)")
b.set_ylabel("windows")
b.legend(fontsize=7.5, loc="upper left")
b.set_title("B  Where the controls fall among real client windows", fontsize=10)

# ---- C: per-protein score vs measurement ------------------------------------
c = ax[0, 2]
x, y = D[BEST].values, D.part_mean.values
k = np.isfinite(x) & np.isfinite(y)
c.scatter(x[k], y[k], s=26, c=D.decile.values[k], cmap="viridis", alpha=0.8, edgecolor="none")
if k.sum() > 2:
    p = np.polyfit(x[k], y[k], 1)
    xs = np.linspace(x[k].min(), x[k].max(), 50)
    c.plot(xs, np.polyval(p, xs), "k-", lw=1.4)
    j = J["p1"][BEST]
    c.set_title(f"C  rho = {j['rho']:.3f} [{j['lo']:.3f}, {j['hi']:.3f}]   "
                f"(reweighted {j['rho_w']:.3f})\n"
                f"most reliable score; it could have reached "
                f"{np.sqrt(max(_R25[BEST]['full'],0)):.2f}", fontsize=10)
c.set_xlabel(f"predicted interface score  ({BEST})")
c.set_ylabel("measured bound:free partition (z)")

# ---- D: by decile -----------------------------------------------------------
d = ax[0, 3]
g = D.groupby("decile")[BEST]
m, se = g.mean(), g.std() / np.sqrt(g.size())
d.errorbar(m.index, m.values, yerr=se.values, fmt="o-", color="#36c", capsize=3)
d.set_xlabel("decile of the measured partition\n(0 = weakest binders)")
d.set_ylabel(f"mean {BEST}")
d.set_title("D  Does the score rise with measured binding?", fontsize=10)
d.grid(alpha=0.25)

# ---- E: real vs shuffled ----------------------------------------------------
e = ax[1, 0]
m2 = shuf.merge(win, on=["accession", "start"], suffixes=("_s", "_r"))
if len(m2):
    e.scatter(m2.iptm_pep_s, m2.iptm_pep_r, s=20, alpha=0.6, color="#36c", edgecolor="none")
    lo = min(m2.iptm_pep_s.min(), m2.iptm_pep_r.min())
    hi = max(m2.iptm_pep_s.max(), m2.iptm_pep_r.max())
    e.plot([lo, hi], [lo, hi], "k--", lw=1)
    p2 = J.get("p2", {})
    e.set_title(f"E  real vs composition-matched shuffle\n"
                f"mean difference {p2.get('d_iptm', np.nan):+.4f} "
                f"[{p2.get('lo', np.nan):+.4f}, {p2.get('hi', np.nan):+.4f}]", fontsize=10)
e.set_xlabel("ipTM, shuffled window")
e.set_ylabel("ipTM, real window")

# ---- F: blocks --------------------------------------------------------------
f = ax[1, 1]
nm = list(J["p34"].keys())
val = [J["p34"][n]["rho"] for n in nm]
colr = ["#bbb" if "structure" not in n else "#c44" for n in nm]
colr = ["#2a7" if n == "seqmodel" else cc for n, cc in zip(nm, colr)]
f.barh(range(len(nm)), val, color=colr)
f.set_yticks(range(len(nm)))
f.set_yticklabels([n.replace("+", " + ") for n in nm], fontsize=8)
f.invert_yaxis()
f.axvline(0, c="k", lw=0.8)
f.set_xlabel("cross-validated Spearman rho with the measurement")
f.set_title("F  Does structure add over sequence?", fontsize=10)
for i, vv in enumerate(val):
    f.text(vv + 0.01 * np.sign(vv or 1), i, f"{vv:.3f}", va="center", fontsize=8)

# ---- G: test-retest of the window score ------------------------------------
gg = ax[1, 2]
RP = pd.read_csv(f"{C}/rep_pairs.tsv", sep="\t")
R25 = json.load(open(f"{C}/cm25_reliability.json"))
gg.scatter(RP.iptm_pep, RP.iptm_pep_2, s=16, alpha=0.5, color="#a33", edgecolor="none")
gg.plot([0.15, 1.0], [0.15, 1.0], "k--", lw=1)
gg.set_xlabel("ipTM, run 1")
gg.set_ylabel("ipTM, run 2 (same input)")
gg.set_title(f"G  The same window, re-run: r = {R25['iptm_pep']['pearson']:.3f}\n"
             f"{100*(1-R25['iptm_pep']['pearson']):.0f}% of the score is run-to-run noise",
             fontsize=10)

# ---- H: observed vs what the instrument could have shown --------------------
h = ax[1, 3]
ar = R25["aggregate_reliability"]
keys = [k for k in ar if k in J["p1"]]
ceil = [np.sqrt(max(ar[k]["full"], 0)) for k in keys]
obs = [abs(J["p1"][k]["rho"]) for k in keys]
o = np.argsort(ceil)[::-1]
keys = [keys[i] for i in o]; ceil = [ceil[i] for i in o]; obs = [obs[i] for i in o]
yy = np.arange(len(keys))
h.barh(yy, ceil, color="#ddd", label="ceiling (split-half reliability)")
h.barh(yy, obs, height=0.5, color="#36c", label="observed |rho| with the measurement")
h.set_yticks(yy); h.set_yticklabels(keys, fontsize=7.5); h.invert_yaxis()
h.legend(fontsize=7.5, loc="lower right")
h.set_xlabel("|Spearman rho|")
h.set_title("H  Observed signal against what the\nscore could have shown", fontsize=10)

fig.suptitle("Design A pilot: does a predicted HSPB1-ACD / client-window interface "
             "predict measured HSPB1 binding?", fontsize=12.5, y=0.975)
fig.savefig(f"{FIG}/pilot.png", dpi=155, bbox_inches="tight")
print(f"wrote {FIG}/pilot.png   (best score: {BEST})")
print("CM24_DONE")
