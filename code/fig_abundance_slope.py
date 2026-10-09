"""Figure for additional text 9 (ad15): pull-down level against input abundance.
A: binned means (20 bins of input), average of the two baits, 35/37/43 C, with a slope-1 reference.
B: each pull-down run's slope on input against its raw total intensity (load).
C: slope on the typical part of abundance (predicted from PaxDb or sequence) and on the lysate-specific residual."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
GA = f"{ROOT}/data/ga_data"
F = np.load(f"{GA}/ad15_fig.npz"); R = pd.read_csv(f"{GA}/ad15_runs.tsv", sep="\t"); J = json.load(open(f"{GA}/ad15_results.json"))
COL = {35: "#3b6fb6", 37: "#e0a030", 43: "#c0392b"}
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

a = ax[0]
for T in (35, 37, 43):
    c = F[f"curve_avg_{T}"]; s = J[f"slope_avg_{T}"]["slope"][0]
    a.plot(c[:, 0], c[:, 1], "o-", color=COL[T], ms=4, label=f"{T} °C (slope {s:.2f})")
c = F["curve_avg_35"]; x0, y0 = np.median(c[:, 0]), np.median(c[:, 1])
xx = np.array([c[:, 0].min(), c[:, 0].max()])
a.plot(xx, y0 + (xx - x0), "k--", lw=1, label="proportional (slope 1)")
a.set_xlabel("input abundance (log2, run-normalised)"); a.set_ylabel("pull-down level (log2, run-normalised)")
a.set_title("A  Pull-down vs input, 20 bins"); a.legend(fontsize=8, frameon=False)

a = ax[1]
for (ch, T), d in R.groupby(["chap", "temp"]):
    a.scatter(d.total, d.slope, color=COL[int(T)], marker="o" if ch == "21A" else "^", s=22, alpha=.8,
              label=f"{'DNAJA1' if ch == '21A' else 'DNAJB11'} {T} °C")
a.set_xlabel("run load: log2 total raw intensity"); a.set_ylabel("run's slope on input")
a.axvline(J["load"]["input_total"], color="grey", lw=1, ls=":")
a.text(J["load"]["input_total"], a.get_ylim()[0] + 0.01, " input runs", color="grey", fontsize=8)
a.set_title("B  Slope does not follow load"); a.legend(fontsize=7, frameon=False, ncol=2)

a = ax[2]
PR = ["PaxDb, 8 cell lines", "PaxDb, whole organism", "PaxDb, HEK293", "ESMC sequence (out-of-fold)", "all proxies together"]
lab = ["PaxDb 8 cell lines", "PaxDb whole organism", "PaxDb HEK293", "ESMC sequence", "all together"]
for k, (T, dx) in enumerate((("35", -0.12), ("43", 0.12))):
    for i, p in enumerate(PR):
        t = J["typical_vs_specific"][f"avg_{T}_{p}"]
        for part, mk, off in (("typical", "o", -0.04), ("specific", "s", 0.04)):
            v = t[part]
            a.errorbar(i + dx + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color=COL[int(T)], ms=5,
                       mfc="white" if part == "specific" else COL[int(T)], capsize=2,
                       label=f"{T} °C, {'typical' if part == 'typical' else 'lysate-specific'} part" if i == 0 else None)
a.axhline(1, color="k", ls="--", lw=1); a.text(-0.4, 0.985, "proportional capture", ha="left", va="top", fontsize=8)
a.set_xticks(range(len(PR))); a.set_xticklabels(lab, rotation=25, ha="right", fontsize=8)
a.set_ylabel("slope of pull-down level"); a.set_ylim(0, 1.05); a.set_xlim(-0.5, 4.5)
a.set_title("C  Typical vs lysate-specific abundance"); a.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
fig.tight_layout()
os.makedirs(f"{ROOT}/reports/figures", exist_ok=True)
fig.savefig(f"{ROOT}/reports/figures/abundance_slope.png", dpi=170)
print("saved reports/figures/abundance_slope.png")
