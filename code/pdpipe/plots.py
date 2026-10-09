"""pdpipe.plots: the standard figures. Every function draws on a given Axes so panels compose freely.

  qc_panel        sample correlation heatmap, PCA, detected proteins per sample
  volcano         log2 change vs -log10 p, hits coloured, top hits named
  scatter_reg     any x-y regression: points, OLS line with 95 % band, r, slope, R^2, n
  binned          binned means with SE for a dense scatter (used beside scatter_reg)
  reliability_bar split-half reliability per contrast against its permutation null
  cond_strip      one protein, every sample, grouped by condition
  dose_curve      one protein, every sample against dose, one line per stratum
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import stats as st

HIT = "#c0392b"; BASE = "#9aa3ad"; LINE = "#1f3b73"
PAL = ["#3b6fb6", "#e0a030", "#c0392b", "#2e8b57", "#8e44ad", "#16a085", "#7f8c8d", "#d35400"]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False})


def qc_panel(ds, axes):
    X = ds.X; S = ds.samples
    C = X.dropna().corr().values
    a = axes[0]; im = a.imshow(C, cmap="viridis", vmin=np.percentile(C, 2), vmax=1)
    conds = S["cond"].values; edges = [i for i in range(1, len(conds)) if conds[i] != conds[i - 1]]
    for e in edges: a.axhline(e - .5, color="w", lw=.6); a.axvline(e - .5, color="w", lw=.6)
    mids = [(s + e) / 2 - .5 for s, e in zip([0] + edges, edges + [len(conds)])]
    a.set_xticks(mids); a.set_xticklabels(list(dict.fromkeys(conds)), rotation=90, fontsize=6.5)
    a.set_yticks(mids); a.set_yticklabels(list(dict.fromkeys(conds)), fontsize=6.5)
    plt.colorbar(im, ax=a, fraction=.046, label="Pearson r (complete proteins)"); a.set_title("Sample correlation")
    Z = X.dropna(); Z = Z.sub(Z.mean(1), axis=0).values.T
    U, s, Vt = np.linalg.svd(Z, full_matrices=False); pc = U[:, :2] * s[:2]; ve = s ** 2 / (s ** 2).sum()
    a = axes[1]; cl = list(dict.fromkeys(conds))
    mk = "osD^v<>p*h"
    for i, c in enumerate(cl):
        m = conds == c
        for b in sorted(S.loc[m, "BR"].unique()):
            mm = m & (S["BR"].values == b)
            a.scatter(pc[mm, 0], pc[mm, 1], color=PAL[i % len(PAL)] if len(cl) <= 8 else plt.cm.tab20(i % 20),
                      marker=mk[(b - 1) % len(mk)], s=26, label=c if b == S.loc[m, "BR"].min() else None)
    a.set_xlabel(f"PC1 ({ve[0]:.0%})"); a.set_ylabel(f"PC2 ({ve[1]:.0%})"); a.set_title("PCA (colour = condition, marker = BR)")
    a.legend(fontsize=6, ncol=2 if len(cl) > 7 else 1, loc="best")
    a = axes[2]; det = X.notna().sum(0).values
    a.bar(range(len(det)), det, color=[PAL[cl.index(c) % len(PAL)] if len(cl) <= 8 else plt.cm.tab20(cl.index(c) % 20) for c in conds])
    a.set_ylim(det.min() * .97, det.max() * 1.01); a.set_xticks([]); a.set_xlabel("samples, in design order")
    a.set_ylabel("proteins detected"); a.set_title("Detection per sample")
    return dict(pc_var=ve[:5].tolist())


def volcano(ax, res, title, q=0.05, label_top=8, genes=None):
    x = res["logFC"].values; y = -np.log10(res["p"].values); hit = res["q"].values <= q
    ax.scatter(x[~hit], y[~hit], s=3, color=BASE, alpha=.5, lw=0, rasterized=True)
    ax.scatter(x[hit], y[hit], s=8, color=HIT, lw=0)
    if hit.any() and genes is not None:
        o = np.argsort(res["p"].values)
        for i in [j for j in o if hit[j]][:label_top]:
            ax.annotate(genes[i], (x[i], y[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axvline(0, color="k", lw=.5)
    ax.set_xlabel("log2 change"); ax.set_ylabel("-log10 p"); ax.set_title(f"{title}  ({hit.sum()} at q <= {q})")


def scatter_reg(ax, x, y, xlabel, ylabel, title=None, highlight=None, labels=None, identity=False, alpha=.35, s=4,
                color=BASE, fit=True, robust_lim=True):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=s, color=color, alpha=alpha, lw=0, rasterized=True)
    if highlight is not None:
        h = ok & np.asarray(highlight)
        ax.scatter(x[h], y[h], s=12, color=HIT, lw=0, zorder=3)
        if labels is not None:
            for i in np.nonzero(h)[0][:12]:
                ax.annotate(labels[i], (x[i], y[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    if robust_lim and ok.sum() > 50:
        lo, hi = np.percentile(x[ok], [0.2, 99.8]); pad = (hi - lo) * .08; ax.set_xlim(lo - pad, hi + pad)
        lo, hi = np.percentile(y[ok], [0.2, 99.8]); pad = (hi - lo) * .08; ax.set_ylim(lo - pad, hi + pad)
    r = st.ols(x, y)
    if fit:
        xx = np.linspace(*np.percentile(x[ok], [0.5, 99.5]), 50)
        ax.plot(xx, r["intercept"] + r["slope"] * xx, color=LINE, lw=1.4)
        # 95 % band for the fitted line
        xm = x[ok].mean(); sxx = ((x[ok] - xm) ** 2).sum(); resid = y[ok] - (r["intercept"] + r["slope"] * x[ok])
        se = np.sqrt((resid ** 2).sum() / (ok.sum() - 2) * (1 / ok.sum() + (xx - xm) ** 2 / sxx))
        ax.fill_between(xx, r["intercept"] + r["slope"] * xx - 1.96 * se, r["intercept"] + r["slope"] * xx + 1.96 * se,
                        color=LINE, alpha=.18, lw=0)
    if identity:
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k:", lw=.8)
    ax.axhline(0, color="k", lw=.3); ax.axvline(0, color="k", lw=.3)
    ax.text(.03, .97, f"r = {r['r']:.2f}   R² = {r['r2']:.1%}\nslope = {r['slope']:.2f} [{r['slope_ci'][0]:.2f}, {r['slope_ci'][1]:.2f}]\nn = {r['n']}",
            transform=ax.transAxes, va="top", fontsize=7, bbox=dict(fc="white", ec="none", alpha=.8))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    return r


def binned(ax, x, y, nb=15, color=LINE, label=None):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    e = np.unique(np.percentile(x[ok], np.linspace(0, 100, nb + 1))); b = np.clip(np.digitize(x[ok], e[1:-1]), 0, len(e) - 2)
    mx = np.array([x[ok][b == k].mean() for k in range(len(e) - 1)]); my = np.array([y[ok][b == k].mean() for k in range(len(e) - 1)])
    se = np.array([y[ok][b == k].std() / np.sqrt(max((b == k).sum(), 1)) for k in range(len(e) - 1)])
    ax.errorbar(mx, my, yerr=1.96 * se, fmt="o-", color=color, ms=3.5, lw=1, capsize=1.5, label=label)


def reliability_bar(ax, rel, title="Split-half reliability"):
    """rel: {contrast name: split_half() output}."""
    names = list(rel); v = [rel[k]["reliability"] for k in names]
    ax.bar(range(len(names)), v, color=[HIT if rel[k]["reliability"] > rel[k].get("null_rel95", 1) else BASE for k in names])
    for i, k in enumerate(names):
        if "null_rel95" in rel[k]:
            ax.plot([i - .4, i + .4], [rel[k]["null_rel95"]] * 2, "k-", lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=.5); ax.set_ylabel("reliable share of the spread\n(full data, Spearman-Brown)")
    ax.set_title(title)


def cond_strip(ax, ds, acc, order=None, xt=None, title=None):
    S = ds.samples; order = order or ds.cond_order()
    for i, c in enumerate(order):
        cols = S.loc[S["cond"] == c, "column"]; v = ds.X.loc[acc, cols].values.astype(float)
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=12, color=PAL[i % len(PAL)])
        ax.plot([i - .25, i + .25], [np.nanmean(v)] * 2, "k-", lw=1.2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xt or order, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("log2 intensity"); ax.set_title(title or ds.proteins.loc[acc, "gene"], fontsize=8.5)


def dose_curve(ax, ds, acc, dose, strata, zero=None, title=None):
    S = ds.samples; zero = zero if zero is not None else S.loc[S[dose] > 0, dose].min() / 3
    for i, (lev, g) in enumerate(S.groupby(strata, sort=True)):
        d = np.where(g[dose] > 0, g[dose], zero); v = ds.X.loc[acc, g["column"]].values.astype(float)
        ax.scatter(d * (1 + .04 * (i - .5)), v, s=10, color=PAL[i], alpha=.7)
        m = pd.Series(v).groupby(d).mean()
        ax.plot(m.index, m.values, "-", color=PAL[i], lw=1.3, label=f"{strata} {lev}")
    ax.set_xscale("log"); ax.set_xlabel(f"{dose} (0 drawn at {zero:g})"); ax.set_ylabel("log2 intensity")
    ax.set_title(title or ds.proteins.loc[acc, "gene"], fontsize=8.5)
