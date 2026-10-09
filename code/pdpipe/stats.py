"""pdpipe.stats: per-protein linear models, moderated tests, and split-half reliability.

Model. For each protein, log2 intensity = design @ beta + noise, fitted on the samples where the
protein was detected (missing values are left out, never imputed). The default design is one mean
per condition ("cell means"), optionally plus a sum-to-zero block term for biological replicate
(BR) when replicates were processed as batches across conditions.

Moderation. Per-protein variances are shrunk toward a common prior (Smyth 2004, the method inside
limma): s2_post = (d0 s0^2 + df s2) / (d0 + df), with d0 and s0^2 estimated from all proteins.
Contrasts give moderated t statistics with df + d0 degrees of freedom; several contrasts at once
give a moderated F test. P values are adjusted by Benjamini-Hochberg.

Reliability. A contrast is computed separately in two disjoint halves of the biological
replicates, and the two profiles are correlated across proteins. Their correlation r is the
reproducible share of the half-data spread; the Spearman-Brown value 2r/(1+r) is that share for
the full data. It is the ceiling on how much of a contrast's variance across proteins any
explanation (annotation, sequence model, another experiment) can account for. A permutation null
(condition labels shuffled within each BR) shows what r looks like with no real effect.
"""
from itertools import combinations
import numpy as np, pandas as pd
from scipy import special, stats


def design(samples, block=False):
    conds = list(dict.fromkeys(samples["cond"]))
    D = (samples["cond"].values[:, None] == np.array(conds)[None]).astype(float)
    names = list(conds)
    if block:
        brs = sorted(samples["BR"].unique())
        for b in brs[1:]:   # sum-to-zero coding: condition columns stay the mean over blocks
            D = np.c_[D, (samples["BR"] == b).astype(float) - (samples["BR"] == brs[0]).astype(float)]
            names.append(f"BR{b}")
    return D, names


def lmfit(X, D):
    """X: proteins x samples DataFrame (NaN allowed). Fits each NA pattern once."""
    Y = X.values; n, m = Y.shape; p = D.shape[1]
    beta = np.full((n, p), np.nan); V = np.full((n, p, p), np.nan); s2 = np.full(n, np.nan); df = np.zeros(n)
    obs = np.isfinite(Y)
    pats, inv = np.unique(obs, axis=0, return_inverse=True)
    for k, pat in enumerate(pats):
        rows = np.nonzero(inv.ravel() == k)[0]
        Dm = D[pat]
        est = np.abs(Dm).sum(0) > 0                       # columns estimable from the observed samples
        Dm = Dm[:, est]
        if Dm.shape[1] == 0: continue                     # protein never detected
        r = np.linalg.matrix_rank(Dm)
        if r < Dm.shape[1]: continue
        Vm = np.linalg.inv(Dm.T @ Dm)
        B = Y[np.ix_(rows, np.nonzero(pat)[0])] @ Dm @ Vm   # (rows, p_est)
        res = Y[np.ix_(rows, np.nonzero(pat)[0])] - B @ Dm.T
        d = pat.sum() - r
        beta[np.ix_(rows, np.nonzero(est)[0])] = B
        V[np.ix_(rows, np.nonzero(est)[0], np.nonzero(est)[0])] = Vm
        df[rows] = d
        if d > 0: s2[rows] = (res ** 2).sum(1) / d
    return dict(beta=beta, V=V, s2=s2, df=df, index=X.index)


def _trigamma_inv(x):
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if np.max(-dif / y) < 1e-8: break
    return y


def ebayes(fit):
    s2, df = fit["s2"], fit["df"]
    ok = np.isfinite(s2) & (s2 > 0) & (df > 0)
    z = np.log(s2[ok]); h = df[ok] / 2
    e = z - special.digamma(h) + np.log(h)
    emean = e.mean(); evar = e.var(ddof=1) - special.polygamma(1, h).mean()
    if evar > 0:
        d0 = 2 * _trigamma_inv(evar); s02 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
        post = (d0 * s02 + df * np.nan_to_num(s2)) / (d0 + df)
    else:
        d0, s02 = np.inf, np.exp(emean); post = np.full_like(s2, s02)
    post[df <= 0] = np.nan if not np.isfinite(d0) else s02
    fit.update(d0=float(d0), s02=float(s02), s2_post=post)
    return fit


def contrast(fit, c, name="contrast"):
    """c: weight vector over design columns. Returns a DataFrame: logFC, se, t, p, q, df."""
    c = np.asarray(c, float); use = c != 0
    b = fit["beta"]; ok = np.isfinite(b[:, use]).all(1)
    est = np.where(ok, np.nan_to_num(b) @ c, np.nan)
    v = np.einsum("i,nij,j->n", c[use], np.nan_to_num(fit["V"][:, use][:, :, use]), c[use])
    se = np.sqrt(fit["s2_post"] * v); dft = fit["df"] + fit["d0"]
    t = est / se; p = 2 * stats.t.sf(np.abs(t), np.minimum(dft, 1e6))
    out = pd.DataFrame(dict(logFC=est, se=se, t=t, p=p, df=dft), index=fit["index"])
    out["q"] = bh(out["p"].values)
    return out


def ftest(fit, C):
    """C: (p, r) matrix of contrasts tested jointly. Returns DataFrame F, p, q."""
    C = np.asarray(C, float); use = (C != 0).any(1); r = C.shape[1]
    b = fit["beta"][:, use]; ok = np.isfinite(b).all(1) & np.isfinite(fit["s2_post"])
    Cu = C[use]; Vu = fit["V"][:, use][:, :, use]
    F = np.full(len(b), np.nan)
    if ok.any():
        cb = np.nan_to_num(b[ok]) @ Cu                                   # (k, r)
        M = np.einsum("ia,nij,jb->nab", Cu, Vu[ok], Cu)                   # (k, r, r)
        F[ok] = np.einsum("na,na->n", cb, np.linalg.solve(M, cb[..., None])[..., 0]) / (r * fit["s2_post"][ok])
    p = stats.f.sf(F, r, np.minimum(fit["df"] + fit["d0"], 1e6))
    out = pd.DataFrame(dict(F=F, p=p), index=fit["index"]); out["q"] = bh(out["p"].values)
    return out


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.any():
        ps = p[ok]; o = np.argsort(ps); n = len(ps)
        qs = np.minimum.accumulate((ps[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        tmp = np.empty(n); tmp[o] = np.minimum(qs, 1); q[ok] = tmp
    return q


def cond_means(X, samples):
    return pd.DataFrame({c: X[g["column"]].mean(1) for c, g in samples.groupby("cond", sort=False)})


def _effect(X, samples, weights, keep):
    tot = 0
    for c, w in weights.items():
        cols = samples.loc[keep & (samples["cond"] == c), "column"]
        if len(cols) == 0: return None
        tot = tot + w * X[cols].mean(1)
    return tot


def _splits(brs):
    brs = sorted(brs); k = len(brs) // 2; seen = set(); out = []
    for a in combinations(brs, k):
        b = tuple(x for x in brs if x not in a)
        key = frozenset([a, b])
        if key in seen: continue
        seen.add(key); out.append(set(a))
    return out


def _relabelings(S, n_perm, rng):
    """Label vectors with condition labels permuted within each BR, excluding the real labelling and
    (for two conditions) its global swap, which give the same |r|. Exact enumeration when small."""
    from itertools import product, permutations
    groups = [g.index.values for _, g in S.groupby("BR")]
    orig = S["cond"].values; conds = list(dict.fromkeys(orig))
    swap = np.array([conds[1 - conds.index(c)] for c in orig]) if len(conds) == 2 else None
    from math import factorial
    from collections import Counter
    def n_distinct(g):
        c = Counter(orig[g]); t = factorial(len(g))
        for v in c.values(): t //= factorial(v)
        return t
    total = np.prod([float(n_distinct(g)) for g in groups])
    def ok(lab): return not (lab == orig).all() and not (swap is not None and (lab == swap).all())
    out = []
    if total <= 4 * n_perm:
        per = [sorted(set(permutations(orig[g]))) for g in groups]
        for combo in product(*per):
            lab = orig.copy()
            for g, p in zip(groups, combo): lab[g] = p
            if ok(lab): out.append(lab)
        if len(out) > n_perm: out = [out[i] for i in rng.choice(len(out), n_perm, replace=False)]
    else:
        seen = set()
        while len(out) < n_perm:
            lab = orig.copy()
            for g in groups: lab[g] = rng.permutation(orig[g])
            if ok(lab) and lab.tobytes() not in seen: seen.add(lab.tobytes()); out.append(lab)
    return out


def split_half(X, samples, weights, n_perm=0, seed=0, adjust=None):
    """weights: {condition key: weight}. Halves are sets of BR numbers, so a BR processed as a batch
    stays together. Returns r per split, the mean half r, the Spearman-Brown full-data reliability,
    and (if n_perm) a relabelling null: condition labels permuted within each BR, never the real
    labelling. With few BRs the null is small (4 BRs and two conditions give 7 relabellings) and
    is conservative when a real effect exists, because partial relabellings keep part of it.

    adjust: optional nuisance correction, adjust(X, S_fit) -> apply(X, S_apply) -> adjusted X.
    It is CROSS-FITTED: each half is corrected with parameters learned from the other half only.
    Learning a correction from the same samples it is applied to shrinks each half's own noise but
    not the noise the halves share through the full-data condition means, and so manufactures
    split-half agreement (seen on FS73: drug-vs-drug contrasts reached reliability 0.47)."""
    inv = samples["cond"].isin(list(weights))
    S = samples[inv].reset_index(drop=True)

    def mean_r(S_):
        rs = []
        for A in _splits(S_["BR"].unique()):
            inA = S_["BR"].isin(A)
            if adjust is None:
                a = _effect(X, S_, weights, inA); b = _effect(X, S_, weights, ~inA)
            else:
                SA, SB = S_[inA], S_[~inA]
                XA = adjust(X, SB.reset_index(drop=True))(X, SA.reset_index(drop=True))
                XB = adjust(X, SA.reset_index(drop=True))(X, SB.reset_index(drop=True))
                a = _effect(XA, SA, weights, np.ones(len(SA), bool)); b = _effect(XB, SB, weights, np.ones(len(SB), bool))
            if a is None or b is None: continue
            ok = np.isfinite(a) & np.isfinite(b)
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return rs

    rs = mean_r(S); r = float(np.mean(rs))
    out = dict(r_splits=[float(x) for x in rs], r_half=r, reliability=float(2 * r / (1 + r)) if r > -1 else np.nan)
    if n_perm:
        rng = np.random.default_rng(seed); null = []
        for lab in _relabelings(S, n_perm, rng):
            P = S.copy(); P["cond"] = lab; v = mean_r(P)
            if v: null.append(np.mean(v))
        sb = [2 * x / (1 + x) for x in null]
        out.update(null=[float(x) for x in null], n_null=len(null), null_rel95=float(np.percentile(sb, 95)) if sb else np.nan,
                   p_perm=float((1 + sum(x >= r for x in null)) / (1 + len(null))))
    return out


def ols(x, y):
    """Slope, intercept, r, R2, n and a 95 % CI on the slope for a scatter regression."""
    ok = np.isfinite(x) & np.isfinite(y); x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    res = stats.linregress(x, y); tq = stats.t.ppf(0.975, len(x) - 2)
    return dict(slope=res.slope, intercept=res.intercept, r=res.rvalue, r2=res.rvalue ** 2, n=int(len(x)),
                slope_ci=(res.slope - tq * res.stderr, res.slope + tq * res.stderr), p=res.pvalue)


def n_factors(X, samples, n_perm=10, seed=0):
    """Parallel analysis on within-condition residuals of complete proteins: keep the leading
    components whose singular values exceed the 95th percentile of those obtained after permuting
    each protein's residuals independently within condition (which destroys sample-level structure)."""
    Xc = X.dropna(); W = (Xc - cond_means(Xc, samples)[samples["cond"]].values).values
    s = np.linalg.svd(W, compute_uv=False); rng = np.random.default_rng(seed); null = []
    groups = [np.nonzero((samples["cond"] == c).values)[0] for c in samples["cond"].unique()]
    for _ in range(n_perm):
        P = W.copy()
        for g in groups:
            idx = rng.permuted(np.tile(g, (len(P), 1)), axis=1); P[:, g] = np.take_along_axis(W, idx, 1)
        null.append(np.linalg.svd(P, compute_uv=False))
    thr = np.percentile(null, 95, axis=0); k = 0
    while k < len(s) and s[k] > thr[k]: k += 1
    return k, (s ** 2 / (s ** 2).sum()), (thr ** 2 / (s ** 2).sum())


def remove_factors(X, samples, k):
    """Remove k sample-level nuisance profiles. The profiles U (proteins x k) are the leading
    components of within-condition residuals, so they hold no condition effect by construction.
    Each sample's loading on them is then measured on its full profile (around the grand mean) and
    subtracted, which also removes the nuisance that happens to sit in condition means. A true
    effect is lost only to the extent that its profile lies along U; `aligned` reports that share
    for the condition means. Complete proteins only."""
    Xc = X.dropna(); W = (Xc - cond_means(Xc, samples)[samples["cond"]].values).values
    U = np.linalg.svd(W, full_matrices=False)[0][:, :k]
    mu = Xc.mean(1).values[:, None]; Z = Xc.values - mu
    A = U.T @ Z                                    # k x samples: each sample's loading
    adj = Xc - U @ A
    M = cond_means(Xc, samples); Mc = (M.sub(M.mean(1), axis=0)).values
    aligned = float(((U @ (U.T @ Mc)) ** 2).sum() / (Mc ** 2).sum())
    return adj, dict(k=k, loadings=A, profiles=U, aligned=aligned)


def factor_adjuster(k):
    """Cross-fit helper for split_half: learn k nuisance profiles from S_fit's within-condition residuals."""
    def fit(X, S_fit):
        Xf = X[S_fit["column"]]; ok = Xf.notna().all(1)
        W = (Xf[ok] - cond_means(Xf[ok], S_fit)[S_fit["cond"]].values).values
        U = np.linalg.svd(W, full_matrices=False)[0][:, :k]
        def apply(X2, S2):
            Z = X2.loc[ok[ok].index, S2["column"]]
            Zc = Z.values - np.nanmean(Z.values, 1, keepdims=True)
            full = np.isfinite(Zc).all(1)                  # loadings from proteins complete in both halves
            A = U[full].T @ Zc[full]
            return Z - U @ A
        return apply
    return fit


def covariate_adjuster(h):
    """Cross-fit helper: per-protein slope on a sample covariate h (Series indexed by column, e.g. the
    bait's own level), learned within condition on S_fit, then removed from the applied samples."""
    def fit(X, S_fit):
        Xf = X[S_fit["column"]]; hf = h[S_fit["column"]].values
        hc = hf - pd.Series(hf).groupby(S_fit["cond"].values).transform("mean").values
        Xc = Xf - cond_means(Xf, S_fit)[S_fit["cond"]].values
        m = Xc.notna().values; num = np.nansum(Xc.values * hc, 1); den = (m * hc ** 2).sum(1)
        b = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), 0.0), index=X.index)
        def apply(X2, S2):
            hh = h[S2["column"]].values
            return X2[S2["column"]] - np.outer(b.values, hh - hf.mean())
        return apply
    return fit


def cross_concordance(X, samples, wx, wy):
    """Correlation between two contrasts measured on disjoint biological replicates: x from one BR half,
    y from the other, both ways, averaged over all balanced splits. Two contrasts that share samples (for
    example two doses against the same control) are otherwise correlated by the shared noise alone.
    Returns the raw r, the r corrected for each contrast's half-data reliability, and one split's vectors for plotting."""
    S = samples[samples["cond"].isin(list(wx) + list(wy))].reset_index(drop=True)
    rs, ex = [], None
    for A in _splits(S["BR"].unique()):
        inA = S["BR"].isin(A)
        for p, q in ((inA, ~inA), (~inA, inA)):
            x = _effect(X, S, wx, p); y = _effect(X, S, wy, q)
            if x is None or y is None: continue
            ok = np.isfinite(x) & np.isfinite(y); rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
            if ex is None: ex = (x, y)
    rx = split_half(X, samples, wx)["r_half"]; ry = split_half(X, samples, wy)["r_half"]
    r = float(np.mean(rs))
    return dict(r=r, r_disattenuated=float(r / np.sqrt(rx * ry)) if rx > 0 and ry > 0 else np.nan,
                rel_half_x=rx, rel_half_y=ry, example=ex)


def leave_one_br_out(X, samples, weights, accs):
    """Effect of a contrast for the named proteins, recomputed leaving out each BR in turn.
    A hit carried by one sample flips or collapses when that BR is left out."""
    S = samples[samples["cond"].isin(list(weights))]
    out = {}
    for b in sorted(S["BR"].unique()):
        e = _effect(X.loc[accs], S, weights, S["BR"] != b)
        if e is not None: out[f"without BR{b}"] = e
    return pd.DataFrame(out)


def top_hit_replication(X, samples, weights, n_top=10, block=False):
    """Cross-validated replication of the strongest hits: rank proteins by moderated t in one BR half,
    take the top n, and measure their effect in the other half, signed by the direction found in the first.
    Returns per split the mean signed held-out effect of the top n and of all proteins (the baseline)."""
    S = samples[samples["cond"].isin(list(weights))].reset_index(drop=True)
    rows = []
    for A in _splits(S["BR"].unique()):
        inA = S["BR"].isin(A)
        for p, q in ((inA, ~inA), (~inA, inA)):
            Sp = S[p].reset_index(drop=True); Xp = X[Sp["column"]]
            D, names = design(Sp, block=block); fit = ebayes(lmfit(Xp, D))
            c = np.zeros(len(names))
            for k, w in weights.items(): c[names.index(k)] = w
            r = contrast(fit, c)
            top = r["p"].nsmallest(n_top).index; sgn = np.sign(r.loc[top, "logFC"])
            held = _effect(X, S, weights, q)
            rows.append(dict(train_effect=float((r.loc[top, "logFC"] * sgn).mean()), heldout_effect=float((held[top] * sgn).mean()),
                             heldout_same_sign=float((np.sign(held[top]) == sgn).mean()), top=list(top)))
    return rows


def trend_weights(levels, x):
    """Contrast weights over cell means that give the least-squares slope of the means on x."""
    x = np.asarray(x, float); xc = x - x.mean()
    return dict(zip(levels, xc / (xc ** 2).sum()))
