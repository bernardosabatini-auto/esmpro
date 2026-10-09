"""pdpipe.cross: line up several screens on shared proteins (UniProt accession) and compare them.

profile(name, ...) returns one Series per screen (a control level or a contrast); align() joins them;
the scatter matrix uses plots.scatter_reg so every comparison carries its regression line, r and slope.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from . import plots as pl


def align(series):
    """series: {label: pd.Series indexed by accession}. Inner join on accession."""
    return pd.concat(series, axis=1, join="inner").dropna()


def scatter_matrix(df, fig_path, title=None, size=3.3):
    cols = list(df.columns); n = len(cols)
    f, ax = plt.subplots(n - 1, n - 1, figsize=(size * (n - 1), size * (n - 1)), squeeze=False)
    out = {}
    for i in range(1, n):
        for j in range(n - 1):
            a = ax[i - 1, j]
            if j >= i: a.axis("off"); continue
            r = pl.scatter_reg(a, df[cols[j]].values, df[cols[i]].values, cols[j], cols[i], s=3)
            out[f"{cols[i]} ~ {cols[j]}"] = dict(r=r["r"], slope=r["slope"], n=r["n"])
    if title: f.suptitle(title, fontweight="bold")
    f.tight_layout(); f.savefig(fig_path, dpi=140); plt.close(f)
    return out
