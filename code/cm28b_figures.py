"""cm28b: figures for the lithium and cofactor report. Reads cm28_lithium_cofactor.json + cm28_vectors.npz."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
C, FIG = f"{ROOT}/data/campaign", f"{ROOT}/reports/figures/arms"
os.makedirs(FIG, exist_ok=True)
J = json.load(open(f"{C}/cm28_lithium_cofactor.json"))
V = np.load(f"{C}/cm28_vectors.npz", allow_pickle=True)
H = np.load(f"{C}/cm28_fs76_halves.npz", allow_pickle=True)   # written by cm28c

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GREY, INK, MUTE = "#9aa0a6", "#202124", "#5f6368"
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.0,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c7cad1", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "figure.dpi": 200,
                     "savefig.bbox": "tight", "grid.color": "#e6e8eb", "grid.linewidth": .7})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)


# ---------------------------------------------------------------- F1: the GA_24 confound
ARM = ["CTRL|0", "NAD|11", "NAD|300", "NADP|0.6", "NADP|15"]
fig, ax = plt.subplots(figsize=(5.4, 2.9))
cols5 = [BLUE, ORANGE, AQUA, VIOLET, GREY]
for k, br in enumerate(sorted(set(H["ga24_br"].tolist()))):
    m = H["ga24_br"] == br
    xs = [H["ga24_pos"][m & (H["ga24_arm"] == a)].mean() for a in ARM]
    ax.plot(xs, range(5), "-o", ms=4.5, lw=1.6, color=cols5[k % 5], label=f"BR {br}", alpha=.9)
ax.set_yticks(range(5)); ax.set_yticklabels(ARM)
ax.set_xlabel("position in the MS acquisition order (0–1)")
ax.set_ylabel("cofactor arm")
ax.set_title("GA_24: cofactor and acquisition order are the same variable\n"
             f"(rank correlation +{J['ga24_design']['r_arm_vs_inj_within_block']:.4f} in every one of "
             f"{J['ga24_design']['n_blocks']} replicate × salt blocks)")
ax.grid(axis="x"); ax.set_axisbelow(True)
ax.annotate(f"CTRL → NADP spans {J['ga24_design']['gap_ctrl_to_nadp']:.2f} of the run;\n"
            f"KCl 0 → 120 spans only {J['ga24_design']['gap_salt0_to_salt120']:.3f}",
            xy=(.02, .97), xycoords="axes fraction", va="top", fontsize=7, color=MUTE)
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, -.45))
save(fig, "f1_ga24_confound.png")

# ---------------------------------------------------------------- F2: attribution
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax, (pk, ok_, key, ttl) in zip(axes, [
        ("attr_pred", "attr_obs", "NADP_vs_CTRL", "NADP vs control\n(spans 0.35 of the run)"),
        ("anchor_pred", "anchor_obs", "salt120_CTRL", "KCl 120 vs 0, control arm\n(spans 0.01 of the run)")]):
    x, y = V[pk], V[ok_]
    m = np.isfinite(x) & np.isfinite(y)
    ax.axhline(0, color="#dfe1e5", lw=.8); ax.axvline(0, color="#dfe1e5", lw=.8)
    ax.plot(x[m], y[m], ".", ms=1.6, color=BLUE, alpha=.28, rasterized=True)
    d = J["ga24_attribution"][key]
    lim = np.nanpercentile(np.abs(x[m]), 99.5) * 1.15
    xs = np.linspace(-lim, lim, 10)
    ax.plot(xs, xs, "--", lw=1.1, color=GREY, label="effect = drift")
    ax.plot(xs, d["slope"] * xs, "-", lw=1.8, color=ORANGE,
            label=(f"fit, slope {d['slope']:.2f}" if abs(d["slope"]) < 3 else "fit (flat)"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(np.nanpercentile(y[m], .3), np.nanpercentile(y[m], 99.7))
    ax.set_xlabel("effect predicted by acquisition drift\n(slope from held-out replicates, log2)")
    ax.set_title(f"{ttl}\nr = {d['r']:+.3f}   drift explains {100 * d['var_explained']:.0f} %")
    ax.legend(frameon=False, loc="upper left")
    if abs(d["slope"]) < 3:
        ax.annotate(f"corrected for noise in the predictor: {d['slope_disattenuated']:.2f}",
                    xy=(.03, .74), xycoords="axes fraction", fontsize=6.8, color=MUTE)
axes[0].set_ylabel("observed effect (log2)")
fig.suptitle("GA_24: the cofactor contrast is drift; a contrast acquired back to back is not",
             fontsize=8.5, y=1.16)
save(fig, "f2_ga24_attribution.png")

# ---------------------------------------------------------------- F3: salt response per arm
rows = J["ga24_salt_by_arm"]
fig, ax = plt.subplots(figsize=(4.4, 3.4))
obs = [r["slope_vs_CTRL"] for r in rows]; exp = [r["slope_if_only_noise"] for r in rows]
lo, hi = min(exp + obs) - .03, 1.03
ax.plot([lo, hi], [lo, hi], "--", lw=1.1, color=GREY)
ax.plot(exp, obs, "o", ms=8, color=BLUE, alpha=.9)
for r in rows:
    dx, dy = {"CTRL|0": (-10, -14), "NAD|11": (6, -11), "NADP|0.6": (6, 4), "NADP|15": (7, -3),
              "NAD|300": (7, -3)}.get(r["arm"], (7, -3))
    ax.annotate(r["arm"], (r["slope_if_only_noise"], r["slope_vs_CTRL"]),
                textcoords="offset points", xytext=(dx, dy), fontsize=7, color=INK)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("slope expected from measurement noise alone\n(r × sd ratio, no change in response)")
ax.set_ylabel("observed slope against the control arm")
ax.set_title("GA_24: no cofactor changes the salt response.\nEvery arm sits on the noise-only line.")
ax.grid(); ax.set_axisbelow(True)
save(fig, "f3_ga24_salt_by_arm.png")

# ---------------------------------------------------------------- F4: FS_76 reproducible spread
B = J["fs76_detection_bound"]
order = ["Li_trend_lowMg", "Li_10_vs_0_lowMg", "Li_10_vs_0_highMg", "Mg_2.5_vs_0.25"]
lab = ["Li trend, 0.25 mM Mg", "Li 10 vs 0, 0.25 mM Mg", "Li 10 vs 0, 2.5 mM Mg", "Mg 2.5 vs 0.25 mM"]
fig, ax = plt.subplots(figsize=(5.0, 2.8))
tot = [J["fs76"][k]["sd"] for k in order]
rep = [B[k]["reproducible_sd_bound"] for k in order]
yy = np.arange(len(order))[::-1]
ax.barh(yy, tot, height=.5, color="#e8eaed", label="total spread (noise included)")
ax.barh(yy, rep, height=.5, color=[BLUE, BLUE, BLUE, ORANGE],
        label="reproducible part (upper bound)")
for y_, r_ in zip(yy, rep):
    ax.annotate(f"{r_:.3f}", (r_, y_), textcoords="offset points", xytext=(5, -2.5), fontsize=7, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(lab)
ax.set_xlabel("spread of the effect across proteins (log2)")
ax.set_title("FS_76: Mg²⁺ moves the whole proteome; lithium moves almost none of it.\n"
             "Same tubes, same proteins, same replicates.")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2)
ax.grid(axis="x"); ax.set_axisbelow(True)
save(fig, "f4_fs76_spread.png")

# ---------------------------------------------------------------- F5: profile noise vs real hits
a_, b_ = H["half_a"], H["half_b"]
gl = H["gene"]
m = np.isfinite(a_) & np.isfinite(b_)
fig, ax = plt.subplots(figsize=(4.3, 4.0))
ax.axhline(0, color="#dfe1e5", lw=.8); ax.axvline(0, color="#dfe1e5", lw=.8)
key = np.isin(gl, ["GET4", "UBL4A", "BAG6"]) & m
lim = max(float(np.nanpercentile(np.abs(np.r_[a_[m], b_[m]]), 99.8)),
          float(np.nanmax(np.abs(np.r_[a_[key], b_[key]]))) if key.any() else 0) * 1.15
ax.plot([-lim, lim], [-lim, lim], "--", lw=1.0, color=GREY)
ax.plot(a_[m], b_[m], ".", ms=1.8, color=GREY, alpha=.35, rasterized=True)
for g, col in (("GET4", BLUE), ("UBL4A", ORANGE), ("BAG6", AQUA)):
    k = np.flatnonzero((gl == g) & m)
    if not len(k): continue
    ax.plot(a_[k], b_[k], "o", ms=7, color=col, zorder=5)
    ax.annotate(g, (a_[k][0], b_[k][0]), textcoords="offset points",
                xytext={"GET4": (10, 6), "UBL4A": (-34, -14), "BAG6": (10, -4)}.get(g, (9, -3)),
                fontsize=7.5, color=col, fontweight="bold")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("lithium trend from replicates 1–2 (log2 per dose step)")
ax.set_ylabel("the same trend from replicates 3–4")
ax.set_title("FS_76 at 0.25 mM Mg²⁺: the proteome-wide lithium\nprofile is noise; three proteins are not")
ax.grid(); ax.set_axisbelow(True)
save(fig, "f5_fs76_halves.png")

print("CM28B_DONE")
