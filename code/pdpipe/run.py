"""pdpipe.run: the standard first pass over one screen.

  python -m pdpipe.run fs73          (from code/)

1. Load the screen (configs/<name>.yaml) and save it in standard form to data/fs_data/<NAME>/.
2. QC figure: sample correlation, PCA, detection per sample.
3. Decide whether biological replicates behave as batches: the share of within-condition variance
   that BR explains, against the share expected by chance. If it is clearly above chance the
   model carries a BR block term.
4. Moderated contrasts from the config; volcano grid; cross-check against the workbook's own logFC.
5. Split-half reliability of every contrast with a permutation null.
Writes contrasts.tsv and run.json beside the dataset, figures to reports/figures/<NAME>/.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from . import io, stats as st, plots as pl


def br_share(ds):
    """Share of within-condition variance explained by BR, averaged over complete proteins, and its null value."""
    X = ds.X.dropna(); S = ds.samples
    W = X - st.cond_means(X, S)[S["cond"]].values                  # within-condition residuals
    ssw = (W ** 2).sum(1)
    sb = sum(len(g) * W[g["column"]].mean(1) ** 2 for _, g in S.groupby("BR"))
    nb = S["BR"].nunique(); dfw = len(S) - S["cond"].nunique()
    return float((sb / ssw).mean()), (nb - 1) / dfw


def first_pass(name, n_perm=20):
    cfg = io.load_config(name); ds = io.load(cfg)
    out = f"{io.ROOT}/data/fs_data/{ds.name}"; fig = f"{io.ROOT}/reports/figures/{ds.name}"
    os.makedirs(fig, exist_ok=True); io.save(ds, out)
    R = dict(n_proteins=int(len(ds.X)), n_samples=int(ds.X.shape[1]), n_complete=int(ds.X.notna().all(1).sum()),
             samples_per_condition=ds.samples.groupby("cond", sort=False).size().to_dict())
    print(f"{ds.name}: {R['n_proteins']} proteins x {R['n_samples']} samples; complete {R['n_complete']}", flush=True)

    f, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    R["qc"] = pl.qc_panel(ds, ax); f.tight_layout(); f.savefig(f"{fig}/qc.png", dpi=160); plt.close(f)

    share, null = br_share(ds); R["br_share"] = dict(observed=share, chance=null)
    block = share > 1.5 * null
    print(f"BR explains {share:.1%} of within-condition variance (chance {null:.1%}) -> block term {'ON' if block else 'off'}", flush=True)
    D, names = st.design(ds.samples, block=block)
    fit = st.ebayes(st.lmfit(ds.X, D)); R.update(block=bool(block), d0=fit["d0"], s02=fit["s02"])

    res = {}; tab = []
    for cn, (a, b) in cfg["contrasts"].items():
        c = np.zeros(len(names)); c[names.index(a)] = 1; c[names.index(b)] = -1
        r = st.contrast(fit, c, cn); res[cn] = r
        tab.append(r.add_prefix(f"{cn}:"))
    T = pd.concat(tab, axis=1); T.insert(0, "gene", ds.proteins["gene"]); T.to_csv(f"{out}/contrasts.tsv", sep="\t")

    n = len(res); nc = min(n, 5); nr = int(np.ceil(n / nc))
    f, ax = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.9 * nr), squeeze=False)
    for i, (cn, r) in enumerate(res.items()):
        pl.volcano(ax.flat[i], r, cn, genes=ds.proteins["gene"].values)
    for a in ax.flat[n:]: a.axis("off")
    f.tight_layout(); f.savefig(f"{fig}/volcano.png", dpi=160); plt.close(f)

    # cross-check against the workbook's own estimates (proteoDA workbooks only)
    f, ax = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.9 * nr), squeeze=False); R["published_check"] = {}
    for i, (cn, r) in enumerate(res.items()):
        if ds.published is None or "published_logfc" not in cfg: ax.flat[i].axis("off"); continue
        col = cfg["published_logfc"].format(name=cn)
        if col not in ds.published: ax.flat[i].axis("off"); continue
        pub = pd.to_numeric(ds.published[col], errors="coerce")
        o = pl.scatter_reg(ax.flat[i], pub.values, r["logFC"].values, "workbook logFC", "pdpipe logFC", cn, identity=True)
        pq = pd.to_numeric(ds.published[col.replace("logFC", "adjpval")], errors="coerce")
        R["published_check"][cn] = dict(r=o["r"], slope=o["slope"], hits_workbook=int((pq < .05).sum()), hits_pdpipe=int((r["q"] < .05).sum()),
                                        hits_both=int(((pq < .05) & (r["q"] < .05)).sum()))
    for a in ax.flat[n:]: a.axis("off")
    if R["published_check"]: f.tight_layout(); f.savefig(f"{fig}/published_check.png", dpi=150)
    plt.close(f)

    rel = {}
    for cn, (a, b) in cfg["contrasts"].items():
        rel[cn] = st.split_half(ds.X, ds.samples, {a: 1, b: -1}, n_perm=n_perm)
        print(f"  {cn:<18} hits {int((res[cn]['q'] < .05).sum()):>4}   split-half r {rel[cn]['r_half']:+.3f}  reliability {rel[cn]['reliability']:+.3f}  (null 95% {rel[cn]['null_rel95']:+.3f})", flush=True)
    R["reliability"] = rel
    f, ax = plt.subplots(figsize=(max(4, .55 * n + 1.5), 3.4)); pl.reliability_bar(ax, rel, f"{ds.name}: reproducibility of each contrast")
    f.tight_layout(); f.savefig(f"{fig}/reliability.png", dpi=160); plt.close(f)
    json.dump(R, open(f"{out}/run.json", "w"), indent=1, default=float)
    return ds, fit, names, res, R


if __name__ == "__main__":
    first_pass(sys.argv[1])
