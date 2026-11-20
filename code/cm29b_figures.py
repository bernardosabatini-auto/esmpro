"""cm29b: figures for the FS small-molecule report. Reads cm29_fs_drugs.json + cm29_fs_vectors.npz."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C, FIG = f"{ROOT}/data/campaign", f"{ROOT}/reports/figures/fsdrugs"
os.makedirs(FIG, exist_ok=True)
J = json.load(open(f"{C}/cm29_fs_drugs.json"))
V = np.load(f"{C}/cm29_fs_vectors.npz", allow_pickle=True)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GREY, INK, MUTE, PALE = "#9aa0a6", "#202124", "#5f6368", "#e8eaed"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.0,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c7cad1", "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "figure.dpi": 200,
                     "savefig.bbox": "tight", "grid.color": "#e6e8eb", "grid.linewidth": .7})


def save(fig, n):
    fig.savefig(f"{FIG}/{n}", facecolor="white"); plt.close(fig); print("  wrote", n, flush=True)


# ---------------------------------------------------------------- F1: the screen at a glance
ROWS = [("Carbamazepine 10 µM", J["fs73"]["carbamazepine_10"]), ("Carbamazepine 100 µM", J["fs73"]["carbamazepine_100"]),
        ("Lamotrigine 10 µM", J["fs73"]["lamotrigene_10"]), ("Lamotrigine 100 µM", J["fs73"]["lamotrigene_100"]),
        ("Topiramate 10 µM", J["fs73"]["topiramate_10"]), ("Topiramate 100 µM", J["fs73"]["topiramate_100"]),
        ("Lithium 10 mM, 0.25 mM Mg", J["fs76"]["Li_10_vs_0_lowMg"]),
        ("Lithium 10 mM, 2.5 mM Mg", J["fs76"]["Li_10_vs_0_highMg"]),
        ("Magnesium 2.5 vs 0.25 mM", J["fs76"]["Mg_2.5_vs_0.25"])]
HJ = J["fs73_joint"]["hits"]
NHIT = {"Carbamazepine 100 µM": HJ["carbamazepine"]["n_arm_specific"],
        "Lamotrigine 100 µM": HJ["lamotrigene"]["n_arm_specific"],
        "Topiramate 100 µM": HJ["topiramate"]["n_arm_specific"],
        "Lithium 10 mM, 0.25 mM Mg": 0, "Lithium 10 mM, 2.5 mM Mg": 0,
        "Magnesium 2.5 vs 0.25 mM": J["fs76"]["Mg_2.5_vs_0.25"]["n_q05"]}
lab = [f"{r[0]}  ({NHIT[r[0]]} hit{'' if NHIT[r[0]] == 1 else 's'})" if r[0] in NHIT else r[0] for r in ROWS]
tot = [r[1]["spread"] for r in ROWS]
rep = [r[1]["spread"] * np.sqrt(max(r[1]["null_rel95"] or 0, 0)) if r[1]["reliability"] <= (r[1]["null_rel95"] or 0)
       else r[1]["spread"] * np.sqrt(max(r[1]["reliability"], 0)) for r in ROWS]
col = [BLUE] * 6 + [VIOLET, VIOLET, ORANGE]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
y = np.arange(len(ROWS))[::-1]
ax.barh(y, tot, height=.6, color=PALE, label="total spread, noise included")
ax.barh(y, rep, height=.6, color=col, label="reproducible part (bounded by the permutation null)")
for yy, r_ in zip(y, rep):
    ax.annotate(f"{r_:.3f}", (r_, yy), textcoords="offset points", xytext=(5, -2.5),
                fontsize=6.8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(lab)
ax.set_xlabel("spread of the effect across proteins (log2)")
ax.set_title("Every small-molecule arm of the FS pull-downs, on one scale.\n"
             "Only the magnesium axis moves the proteome. Hits are arm-specific proteins\n"
             "at q < 0.05 with |effect| >= 0.5, from the joint model.")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=1)
ax.grid(axis="x"); ax.set_axisbelow(True)
save(fig, "f1_screen_overview.png")

# ---------------------------------------------------------------- F2: dose consistency
HITS = {r["gene"]: r for r in J["fs73_candidates"]["lamotrigene"]}
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), sharex=True, sharey=True)
for ax, (drug, nice) in zip(axes, [("carbamazepine", "Carbamazepine"), ("lamotrigene", "Lamotrigine"),
                                   ("topiramate", "Topiramate")]):
    x, yv = V[f"fs73_{drug}_10"], V[f"fs73_{drug}_100"]
    m = np.isfinite(x) & np.isfinite(yv)
    ax.axhline(0, color="#dfe1e5", lw=.8); ax.axvline(0, color="#dfe1e5", lw=.8)
    ax.plot(x[m], yv[m], ".", ms=1.7, color=GREY, alpha=.35, rasterized=True)
    rows = [r for r in J["fs73_joint"]["hits"][drug]["rows"] if r["arm_specific"]]
    accs = list(V["fs73_acc"])
    OFF = {"DCK": (8, -2), "PIP4K2C": (-56, -4), "PPP4R3A": (-54, 2), "DHFR": (8, -10), "PPP4R2": (8, 2),
           "PTGR1": (8, -8), "GALE": (8, 0), "M6PR": (-40, -10), "FUS": (8, 0), "ADH5": (8, -8)}
    for r_ in rows:
        i = accs.index(r_["acc"])
        ax.plot(x[i], yv[i], "o", ms=6, color=ORANGE, zorder=5)
        ax.annotate(r_["gene"], (x[i], yv[i]), textcoords="offset points",
                    xytext=OFF.get(r_["gene"], (8, -2)), fontsize=6.8, color=ORANGE, fontweight="bold")
    ax.set_title(f"{nice}\n{len(rows)} arm-specific protein(s)")
    ax.set_xlabel("effect at 10 µM (log2)")
    ax.grid(); ax.set_axisbelow(True)
axes[0].set_ylabel("effect at 100 µM (log2)")
fig.suptitle("FS_73: carbamazepine and lamotrigine each move a handful of proteins; topiramate none",
             fontsize=8.5, y=1.08)
save(fig, "f2_dose_consistency.png")

# ---------------------------------------------------------------- F3: the responders, by dose
S = pd.read_csv(f"{ROOT}/data/export/levels/FS73_ip/samples.tsv", sep="\t")
X = pd.read_csv(f"{ROOT}/data/export/levels/FS73_ip/matrix.tsv", sep="\t", index_col=0)
PANELS = [("carbamazepine", "Carbamazepine"), ("lamotrigene", "Lamotrigine")]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, (drug, nice) in zip(axes, PANELS):
    rows = [r for r in J["fs73_joint"]["hits"][drug]["rows"] if r["arm_specific"]]
    for r_, col in zip(rows, [BLUE, ORANGE, AQUA, VIOLET, "#b4495f", GREY]):
        pts = [X.loc[r_["acc"], S.loc[S["cond"] == c, "column"]].mean()
               for c in ("ctl|0", f"{drug}|10", f"{drug}|100")]
        pts = np.array(pts) - pts[0]
        ax.plot([0, 1, 2], pts, "-o", ms=5, lw=1.8, color=col, label=r_["gene"])
    ax.axhline(0, color="#dfe1e5", lw=.9)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "10 µM", "100 µM"])
    ax.set_xlabel(nice.lower())
    ax.set_title(f"{nice}: {len(rows)} arm-specific responders")
    ax.legend(frameon=False, loc="best", fontsize=6.8)
    ax.grid(axis="y"); ax.set_axisbelow(True)
axes[0].set_ylabel("pull-down, relative to control (log2)")
fig.suptitle("Each drug's responders are flat in the other drugs' arms", fontsize=8.5, y=1.06)
save(fig, "f3_drug_hits.png")

# ---------------------------------------------------------------- F4: the shared-control artefact
K = J["fs73_cross_drug"]
fig, ax = plt.subplots(figsize=(4.4, 2.7))
names = list(K); xs = np.arange(len(names))
ax.bar(xs - .19, [K[k]["r_same_samples"] for k in names], width=.36, color=GREY,
       label="both effects from all replicates\n(control samples shared)")
ax.bar(xs + .19, [K[k]["r_disjoint"] for k in names], width=.36, color=BLUE,
       label="effects from disjoint replicates")
ax.axhline(0, color="#dfe1e5", lw=.9)
ax.set_xticks(xs)
ax.set_xticklabels([k.replace("carbamazepine", "carb").replace("lamotrigene", "lamo")
                     .replace("topiramate", "topi").replace("|", "\nvs ") for k in names])
ax.set_ylabel("correlation of two drugs' effects")
ax.set_title("FS_73: drugs appear to agree only because\nthey share the same control samples")
ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, "f4_shared_control.png")

# ---------------------------------------------------------------- F5: is the hit a real partner?
P = J["partition_percentile"]
order = ["HSPB1", "DHFR", "PTGR1", "ADH5", "GALE", "DCK", "PIP4K2C", "PPP4R3A", "PPP4R2",
         "GET4", "UBL4A", "BAG6"]
order = [g for g in order if g in P]
cols = ["GA22_part_base", "GA24_part_base"]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
y = np.arange(len(order))[::-1]
for k, (c, mk, lb) in enumerate(zip(cols, ["o", "s"], ["GA_22 partition", "GA_24 partition"])):
    vals = [P[g].get(c) for g in order]
    ax.plot([v if v is not None else np.nan for v in vals], y, mk, ms=7,
            color=[BLUE, ORANGE][k], alpha=.85, label=lb)
ax.axvline(50, color=GREY, ls="--", lw=1.1)
ax.annotate("median protein", xy=(50, 0.985), xycoords=("data", "axes fraction"),
            fontsize=6.8, color=MUTE, ha="center", va="top")
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlim(0, 100); ax.set_xlabel("percentile of the bound : free partition (100 = most bead-bound)")
ax.set_title("Which responders are real HSPB1 partners?\nOnly DHFR, and the bait itself.")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x"); ax.set_axisbelow(True)
save(fig, "f5_partition.png")
print("CM29B_DONE")
