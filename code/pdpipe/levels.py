"""pdpipe.levels: the implementation of each analysis level. Driven by pdpipe.study; see that module
for the level list, the capability gates and the CLI.

Every function has the same shape:

    level_<name>(cfg, caps, out, state) -> state

`cfg` is the study config, `caps` the capability dict, `out` the study's output directory, and `state`
carries loaded objects between levels so a full run parses each fraction once. Each level also writes its
own outputs to `out`, so a level can be re-run alone (`--levels complex`) and will reload what it needs.

The analysis logic is the campaign's, generalised: the only thing that changes between studies is which
fractions exist and what the conditions are called.
"""
import os, json, subprocess, itertools
import numpy as np, pandas as pd

from . import io, stats as st, partition as pt, plots as pl
from .study import DERIVED, LEVELS

ROOT = io.ROOT
CAMP = f"{ROOT}/data/campaign"
LAYER_IDX = {l: i for i, l in enumerate(range(0, 81, 10))}       # layers.npy is (n, 9, 2560)


# ----------------------------------------------------------------------------- shared helpers
def _conds(ds, cfg):
    """Condition keys kept for this study (drug arms and failed blocks are excluded in the config)."""
    keep = ds.samples["cond"].unique().tolist()
    inc, exc = cfg.get("conditions_include"), cfg.get("conditions_exclude")
    if inc:
        keep = [c for c in keep if any(__import__("re").search(p, c) for p in inc)]
    if exc:
        keep = [c for c in keep if not any(__import__("re").search(p, c) for p in exc)]
    return keep


def _fit(ds, keep):
    sub = ds.subset(ds.samples["cond"].isin(keep).values)
    from .run import br_share
    share, null = br_share(sub)
    block = share > 1.5 * null
    D, names = st.design(sub.samples, block=block)
    return sub, st.ebayes(st.lmfit(sub.X, D)), names, block


def fold_blocks(pairs, k, outer, n, min_members=10):
    B = []
    for f in np.unique(outer):
        i = np.nonzero(k & (outer == f))[0]
        if len(i) < 3:
            continue
        A = pairs[np.ix_(i, i)].astype(np.float64); c = A.sum(1)
        if (c > 0).sum() >= min_members:
            B.append((i, A, c))
    return B


def partner_mean(v, B, n):
    pm = np.full(n, np.nan)
    for i, A, c in B:
        ok = c > 0; m = np.full(len(i), np.nan); m[ok] = (A @ v[i])[ok] / c[ok]; pm[i] = m
    return pm


def cboot_r(a, b, g, nb=400, seed=0):
    ug, gi = np.unique(g, return_inverse=True); o = []
    for s in range(nb):
        w = np.random.default_rng(seed + s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
        den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
        o.append((w * (a - ma) * (b - mb)).sum() / den if den > 0 else np.nan)
    o = np.array(o); o = o[np.isfinite(o)]
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))] if len(o) else [np.nan, np.nan]


def cboot_diff(a1, b1, a2, b2, g, nb=400, seed=0):
    """paired: both correlations on the SAME resampled proteins."""
    ug, gi = np.unique(g, return_inverse=True); o = []
    for s in range(nb):
        w = np.random.default_rng(seed + s).multinomial(len(ug), np.ones(len(ug)) / len(ug))[gi].astype(float)
        rs = []
        for a, b in ((a1, b1), (a2, b2)):
            ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
            den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
            rs.append((w * (a - ma) * (b - mb)).sum() / den if den > 0 else np.nan)
        o.append(rs[0] - rs[1])
    o = np.array(o); o = o[np.isfinite(o)]
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))] if len(o) else [np.nan, np.nan]


def _device(torch):
    """A CUDA device only if it can actually run a kernel.

    `torch.cuda.is_available()` is not enough on this cluster: the RTX PRO 6000 Blackwell cards are
    sm_120, and a torch build without sm_120 kernels still reports the device as available, then fails
    at the first operation with "no kernel image is available for execution on the device". Probe it.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = (torch.ones(8, device="cuda") * 2).sum().item()
        return "cuda"
    except Exception as e:
        print(f"  (GPU present but unusable -- {str(e).splitlines()[0]}; falling back to CPU. "
              f"The `boltz` env has a CUDA 13 torch that supports these cards.)", flush=True)
        return "cpu"


def _deciles(v, q=10):
    ok = np.isfinite(v); D = np.zeros((len(v), q + 1))
    if ok.sum() > q:
        e = np.quantile(v[ok], np.linspace(0, 1, q + 1)); e[0] -= 1e-9; e[-1] += 1e-9
        D[np.nonzero(ok)[0], np.clip(np.digitize(v[ok], e[1:-1]), 0, q - 1)] = 1.0
    D[~ok, q] = 1.0
    return D


def project_out(v, k, X):
    b = np.linalg.lstsq(X[k], v[k], rcond=None)[0]
    o = np.full_like(v, np.nan); o[k] = v[k] - X[k] @ b
    return o


def _load_targets(out):
    Z = np.load(f"{out}/targets.npz", allow_pickle=True)
    meta = json.load(open(f"{out}/targets_meta.json"))
    U = pd.read_csv(f"{out}/proteins.tsv", sep="\t")["accession"].astype(str).tolist()
    return list(Z["names"].astype(str)), Z["Y"].astype(np.float64), meta, U


# ----------------------------------------------------------------------------- levels
def level_load(cfg, caps, out, state):
    """Parse every declared fraction, save it in standard form, and report its replicate structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .run import br_share
    ds, qc = {}, {}
    for key, name in cfg["fractions"].items():
        d = io.load(io.load_config(name))
        if cfg.get("exclude_samples"):
            d = d.subset(~d.samples["column"].isin(cfg["exclude_samples"]).values)
        io.save(d, f"{out}/{key}")
        share, null = br_share(d)
        qc[key] = dict(config=name, n_proteins=int(len(d.X)), n_samples=int(d.X.shape[1]),
                       n_conditions=int(d.samples["cond"].nunique()),
                       br_share=float(share), br_chance=float(null), block_term=bool(share > 1.5 * null))
        print(f"  {key:<6} {name:<8} {len(d.X):>6} proteins x {d.X.shape[1]:>3} samples, "
              f"{d.samples['cond'].nunique()} conditions; BR explains {share:.1%} (chance {null:.1%})", flush=True)
        f, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
        try:
            pl.qc_panel(d, ax)
            f.tight_layout(); f.savefig(f"{out}/qc_{key}.png", dpi=150)
        except Exception as e:                        # QC plotting must never block the pipeline
            print(f"  (qc panel for {key} skipped: {e})")
        plt.close(f)
        ds[key] = d
    json.dump(qc, open(f"{out}/qc.json", "w"), indent=1)
    state["ds"] = ds
    return state


def level_targets(cfg, caps, out, state):
    """Per-fraction baselines and contrasts, every derived fraction the capabilities allow, each with
    its split-half ceiling. This is the only level that knows what the experiment measured."""
    ds = state.get("ds")
    if ds is None:
        ds = {k: _reload(f"{out}/{k}") for k in cfg["fractions"]}
        state["ds"] = ds
    ipcfg = io.load_config(cfg["fractions"]["ip"])
    contrasts = {k: v for k, v in (ipcfg.get("contrasts") or {}).items()
                 if not cfg.get("contrasts") or k in cfg["contrasts"]}
    ref = cfg.get("reference")

    # derived fractions, built from matched reactions of the same tube
    for d, (a, b, flag, desc) in DERIVED.items():
        if caps[flag]:
            ds[d] = pt.pair(ds[a], ds[b], name=f"{cfg['name']}_{d}")
            print(f"  derived {d:<7} {ds[d].X.shape[0]:>6} proteins x {ds[d].X.shape[1]:>3} reactions  ({desc})", flush=True)

    U = sorted(set().union(*[set(map(str, d.X.index)) for d in ds.values()]))
    idx = {a: i for i, a in enumerate(U)}
    Y, META = {}, {}

    def put(name, s, meta):
        v = np.full(len(U), np.nan)
        s = s.dropna(); s.index = [str(a).split(";")[0] for a in s.index]
        s = s[~s.index.duplicated()]
        for acc, x in s.items():
            if acc in idx:
                v[idx[acc]] = x
        Y[name] = v; META[name] = {**meta, "n": int(np.isfinite(v).sum())}
        print(f"  {name:<24} n {META[name]['n']:>6}  sd {np.nanstd(v):.3f}  ceiling {meta['ceiling']:.3f}", flush=True)

    for key, d in ds.items():
        keep = _conds(d, cfg)
        if not keep:
            raise SystemExit(
                f"\n  conditions_include / conditions_exclude removed every condition from `{key}`.\n"
                f"  Conditions present: {sorted(d.samples['cond'].unique())}\n"
                f"  conditions_include = {cfg.get('conditions_include')}\n"
                f"  conditions_exclude = {cfg.get('conditions_exclude')}\n"
                f"  The condition key is the fraction config's `condition:` fields joined by '|', so check\n"
                f"  the order and the number of fields before writing the regex.")
        if ref is not None and ref not in keep:
            print(f"  WARNING: reference {ref!r} is not among {key}'s kept conditions {keep}; "
                  f"using {keep[0]!r} for this fraction.")
        r = ref if ref in keep else keep[0]
        sub, fit, names, block = _fit(d, keep)
        cm = st.cond_means(sub.X, sub.samples)
        sh = st.split_half(sub.X, sub.samples, {r: 1})
        put(f"{key}_base", cm[r] - np.nanmedian(cm[r]),
            dict(kind="baseline" if key in ("ip",) else ("abundance" if key in ("sup", "input") else "derived"),
                 fraction=key, what=f"{key} level at {r}", reference=r,
                 ceiling=float(sh["reliability"]), block_term=bool(block)))
        for cn, (ca, cb) in contrasts.items():
            if ca not in names or cb not in names:
                print(f"  (skip {key}_{cn}: condition missing from this fraction)"); continue
            c = np.zeros(len(names)); c[names.index(ca)] = 1; c[names.index(cb)] = -1
            res = st.contrast(fit, c, cn)
            shc = st.split_half(sub.X, sub.samples, {ca: 1, cb: -1})
            put(f"{key}_{cn}", res["logFC"],
                dict(kind="effect", fraction=key, what=f"{cn} in {key}", contrast=[ca, cb],
                     ceiling=float(shc["reliability"]), block_term=bool(block)))

    names = list(Y)
    np.savez(f"{out}/targets.npz", names=np.array(names), Y=np.column_stack([Y[t] for t in names]).astype(np.float32))
    json.dump(META, open(f"{out}/targets_meta.json", "w"), indent=1, default=float)
    pd.DataFrame({"accession": U}).to_csv(f"{out}/proteins.tsv", sep="\t", index=False)
    print(f"\n  {len(names)} targets over {len(U)} proteins -> {out}/targets.npz", flush=True)
    return state


def _reload(d):
    X = pd.read_csv(f"{d}/matrix.tsv", sep="\t", index_col=0)
    S = pd.read_csv(f"{d}/samples.tsv", sep="\t")
    P = pd.read_csv(f"{d}/proteins.tsv", sep="\t", index_col=0)
    return io.Dataset(os.path.basename(d), X, S, P, None, json.load(open(f"{d}/meta.json")))


def level_folds(cfg, caps, out, state):
    """MMseqs2 clustering at 30 % identity, then five outer blocks assigned whole clusters at a time,
    so no protein shares a block boundary with a close relative."""
    U = pd.read_csv(f"{out}/proteins.tsv", sep="\t")["accession"].astype(str).tolist()
    ann = io.uniprot_annotation()
    seq = ann["sequence"].reindex(U) if "sequence" in ann else pd.Series(index=U, dtype=object)
    have = seq.notna()
    print(f"  {have.sum()}/{len(U)} proteins have a sequence", flush=True)
    tmp = f"{out}/_mmseqs"; os.makedirs(tmp, exist_ok=True)
    fa = f"{tmp}/in.fasta"
    with open(fa, "w") as fh:
        for a in np.array(U)[have.values]:
            fh.write(f">{a}\n{seq[a]}\n")
    mm = os.environ.get("MMSEQS_BIN", "mmseqs")
    cmd = [mm, "easy-cluster", fa, f"{tmp}/clu", f"{tmp}/tmp", "--min-seq-id", "0.3", "-c", "0.5",
           "--cov-mode", "1", "-v", "1"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        rep = pd.read_csv(f"{tmp}/clu_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
        g = dict(zip(rep["member"].astype(str), rep["rep"].astype(str)))
    except Exception as e:
        print(f"  MMseqs2 unavailable or failed ({e}); falling back to one cluster per protein.")
        print("  WARNING: without clustering, homologs can straddle blocks and accuracy will be optimistic.")
        g = {a: a for a in U}
    group = np.array([g.get(a, a) for a in U])
    ug = pd.unique(group)
    rng = np.random.default_rng(0); rng.shuffle(ug)
    assign = {c: i % 5 for i, c in enumerate(ug)}
    outer = np.array([assign[c] for c in group])
    np.savez(f"{out}/folds.npz", outer=outer, group=group)
    print(f"  {len(ug)} clusters -> 5 blocks of {np.bincount(outer)}", flush=True)
    return state


def level_embed(cfg, caps, out, state):
    """Reuse the campaign's cached ESMC-6B states where possible; embed only what is new."""
    U = pd.read_csv(f"{out}/proteins.tsv", sep="\t")["accession"].astype(str).tolist()
    cu = pd.read_csv(f"{CAMP}/union.tsv", sep="\t")["accession"].astype(str).tolist()
    pos = {a: i for i, a in enumerate(cu)}
    hit = [a for a in U if a in pos]
    miss = [a for a in U if a not in pos]
    print(f"  {len(hit)}/{len(U)} proteins already embedded in the campaign cache; {len(miss)} new", flush=True)
    if miss:
        ann = io.uniprot_annotation()
        fa = f"{out}/missing.fasta"
        with open(fa, "w") as fh:
            for a in miss:
                s = ann["sequence"].get(a) if "sequence" in ann else None
                if isinstance(s, str) and s:
                    fh.write(f">{a}\n{s}\n")
        print(f"  wrote {fa}. To embed them:\n"
              f"    python code/cm01_embed.py --fasta {fa} --out {out}/new_features.npz\n"
              f"  (one RTX card, roughly a minute per 1,600 proteins). Re-run this level afterwards.")
    L = np.load(f"{CAMP}/layers.npy", mmap_mode="r")
    rows = np.array([pos[a] for a in hit])
    np.savez(f"{out}/emb.npz", accession=np.array(hit), rows=rows)
    print(f"  embedding index written ({L.shape[1]} layers x {L.shape[2]} dims per protein)", flush=True)
    return state


def level_predict(cfg, caps, out, state):
    """Out-of-fold ridge on the ESMC-6B states, scored against each target's own ceiling. The campaign's
    multi-task MLP beat this by +0.028 of ceiling; it is a separate, much larger job (cm03) and is wired
    in behind options.mlp."""
    names, Y, META, U = _load_targets(out)
    F = np.load(f"{out}/folds.npz"); outer = F["outer"]
    E = np.load(f"{out}/emb.npz", allow_pickle=True)
    acc, rows = list(E["accession"].astype(str)), E["rows"]
    L = np.load(f"{CAMP}/layers.npy", mmap_mode="r")
    use = [LAYER_IDX[l] for l in cfg["options"].get("layers", [50, 80])]
    X = np.asarray(L[rows][:, use, :], dtype=np.float32).reshape(len(rows), -1)
    # Feature standardisation uses every protein, test folds included. That is label-free -- it moves no
    # information about the measurement across the fold boundary -- so it does not inflate the held-out
    # score. Anything that touches y (alpha selection below) is kept strictly inside the training folds.
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ai = {a: i for i, a in enumerate(U)}
    sel = np.array([ai[a] for a in acc])
    Yc, oc = Y[sel], outer[sel]

    import torch
    dev = _device(torch)
    print(f"  ridge on {dev}: {X.shape[0]} proteins x {X.shape[1]} features", flush=True)
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
    grp_study = np.load(f"{out}/folds.npz")["group"][sel]
    ug = {g: i for i, g in enumerate(pd.unique(grp_study))}
    inner = np.array([ug[g] % 2 for g in grp_study])          # inner split held out by CLUSTER

    def _ridge(tr_mask, y_np, alphas):
        """Eigendecompose the training Gram once; every alpha is then a diagonal solve."""
        A = Xt[torch.as_tensor(tr_mask, device=dev)]
        b = torch.as_tensor(y_np[tr_mask], device=dev, dtype=torch.float32)
        mu = b.mean(); b = b - mu
        w, V = torch.linalg.eigh((A.T @ A).double())
        VtAtb = V.T @ (A.T @ b).double()
        return {al: (V @ (VtAtb / (w + al))).float() for al in alphas}, float(mu)

    P = np.full_like(Yc, np.nan); R = {}
    for j, t in enumerate(names):
        y = Yc[:, j]; k = np.isfinite(y)
        if k.sum() < 100:
            continue
        for f in range(5):
            tr, te = k & (oc != f), k & (oc == f)
            if tr.sum() < 50 or te.sum() < 5:
                continue
            # alpha is chosen on an inner split held out by sequence cluster, NEVER on training error:
            # training error falls monotonically as alpha shrinks, so selecting on it always returns the
            # least regularised model and silently overfits.
            itr, iva = tr & (inner == 0), tr & (inner == 1)
            if itr.sum() > 50 and iva.sum() > 20:
                Wd, mu = _ridge(itr, y, ALPHAS)
                Xv = Xt[torch.as_tensor(iva, device=dev)]
                yv = torch.as_tensor(y[iva], device=dev, dtype=torch.float32)
                best = min(ALPHAS, key=lambda al: float(((Xv @ Wd[al] + mu - yv) ** 2).mean()))
            else:
                best = 1e4
            Wf, muf = _ridge(tr, y, (best,))
            P[te, j] = (Xt[torch.as_tensor(te, device=dev)] @ Wf[best]).cpu().numpy() + muf
        q = k & np.isfinite(P[:, j])
        ss = ((y[q] - y[q].mean()) ** 2).sum()
        r2 = 1 - ((y[q] - P[q, j]) ** 2).sum() / ss if ss > 0 else np.nan
        cl = META[t]["ceiling"]
        R[t] = dict(r2=float(r2), ceiling=float(cl), r2_over_ceiling=float(r2 / cl) if cl > 0 else np.nan,
                    n=int(q.sum()))
        print(f"  {t:<24} R2 {r2:+.3f}  ceiling {cl:.3f}  R2/ceiling {R[t]['r2_over_ceiling']:+.3f}", flush=True)
    m = np.mean([v["r2_over_ceiling"] for v in R.values() if np.isfinite(v["r2_over_ceiling"])])
    print(f"\n  mean R2/ceiling over {len(R)} targets: {m:.3f}   "
          f"(campaign ridge 0.245, multi-task MLP 0.273)", flush=True)
    np.savez(f"{out}/oof.npz", names=np.array(names), accession=np.array(acc), P=P.astype(np.float32))
    json.dump(dict(per_target=R, mean_r2_over_ceiling=float(m)), open(f"{out}/predict.json", "w"), indent=1, default=float)
    if cfg["options"].get("mlp"):
        print("\n  options.mlp is set: run the campaign's multi-task MLP with\n"
              f"    python code/cm_launch.py --sweep s1\n  after pointing cm02/cm03 at {out}.")
    return state


def level_properties(cfg, caps, out, state):
    """What measured covariates say, and what sequence adds over them. Abundance comes from this study's
    own input or supernatant when they exist, and from PaxDb otherwise -- recorded, because the campaign
    found abundance to be the strongest single predictor of a baseline pull-down."""
    names, Y, META, U = _load_targets(out)
    src = caps["abundance_source"]
    ab = np.full(len(U), np.nan); ab_target = None
    for key in ("input", "sup"):
        t = f"{key}_base"
        if t in names:
            ab, ab_target = Y[:, names.index(t)], t; break
    else:
        np_ = f"{CAMP}/named_properties.tsv"
        if os.path.exists(np_):
            NP = pd.read_csv(np_, sep="\t", index_col=0).reindex(U)
            ab = NP["abundance_paxdb"].values.astype(float)
    ann = io.uniprot_annotation()
    L = np.log10(pd.to_numeric(ann["length"], errors="coerce").reindex(U).values.astype(float) + 1)
    B = np.column_stack([ab, L, np.isfinite(ab).astype(float), np.ones(len(U))])
    B = np.where(np.isfinite(B), B, 0.0)
    F = np.load(f"{out}/folds.npz"); outer = F["outer"]
    R, skipped = {}, []
    for j, t in enumerate(names):
        # The abundance covariate IS one of the targets when it comes from this study's own input or
        # supernatant. Scoring that target against it is circular -- it predicts itself and returns
        # R2/ceiling slightly above 1, which is the tell. Exclude it rather than report it.
        if t == ab_target:
            skipped.append(t)
            print(f"  {t:<24} SKIPPED: this target is the abundance covariate itself (would be circular)", flush=True)
            continue
        y = Y[:, j]; k = np.isfinite(y)
        if k.sum() < 100:
            continue
        p = np.full(len(U), np.nan)
        for f in range(5):
            tr, te = k & (outer != f), k & (outer == f)
            if tr.sum() < 50 or te.sum() < 5:
                continue
            w = np.linalg.lstsq(B[tr], y[tr], rcond=None)[0]; p[te] = B[te] @ w
        q = k & np.isfinite(p); ss = ((y[q] - y[q].mean()) ** 2).sum()
        R[t] = float(1 - ((y[q] - p[q]) ** 2).sum() / ss) / META[t]["ceiling"] if ss > 0 else np.nan
        print(f"  {t:<24} abundance+length R2/ceiling {R[t]:+.3f}", flush=True)
    if any(np.isfinite(v) and v > 1.02 for v in R.values()):
        print("  WARNING: an R2/ceiling above 1 means the covariate and the target share a measurement. "
              "Check what `abundance_source` is.")
    json.dump(dict(abundance_source=src, abundance_target=ab_target, skipped_as_circular=skipped,
                   per_target=R), open(f"{out}/properties.json", "w"), indent=1, default=float)
    print(f"\n  abundance source: {src}", flush=True)
    return state


def _complex_adjacency(U):
    """Complex Portal co-membership over this study's proteins."""
    import re, csv as _csv
    p = f"{ROOT}/data/external/complexes/complexportal_9606.tsv"
    pos = {a: i for i, a in enumerate(U)}
    n = len(U); pairs = np.zeros((n, n), dtype=bool); ncx = 0
    if not os.path.exists(p):
        print(f"  Complex Portal file missing ({p}); complex levels cannot run.")
        return pairs, 0
    for r in _csv.reader(open(p), delimiter="\t"):
        if r[0].startswith("#"):
            continue
        mem = set(re.findall(r"([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d(?:[A-Z][A-Z0-9]{2}\d){1,2})(?:-\d+)?\(", r[18]))
        idx = sorted({pos[m] for m in mem if m in pos})
        if len(idx) >= 2:
            pairs[np.ix_(idx, idx)] = True; ncx += 1
    np.fill_diagonal(pairs, False)
    return pairs, ncx


def level_complex(cfg, caps, out, state):
    """The complex effect: do the subunits of one complex miss the sequence prediction together?
    (Report section 3c.) Correlation between a protein's residual and the mean residual of its
    complex-mates in the same held-out block, against a within-block permutation null."""
    names, Y, META, U = _load_targets(out)
    O = np.load(f"{out}/oof.npz", allow_pickle=True)
    acc = list(O["accession"].astype(str)); Pm = O["P"].astype(np.float64)
    ai = {a: i for i, a in enumerate(U)}
    P = np.full_like(Y, np.nan); P[[ai[a] for a in acc]] = Pm
    F = np.load(f"{out}/folds.npz"); outer, grp = F["outer"], F["group"]
    pairs, ncx = _complex_adjacency(U)
    in_cx = pairs.any(1); n = len(U)
    print(f"  {ncx} complexes with >=2 measured members; {in_cx.sum()} proteins in at least one", flush=True)
    if in_cx.sum() < 50:
        print("  too few proteins in annotated complexes; skipping.")
        return state
    J = {}
    for j, t in enumerate(names):
        y, p = Y[:, j], P[:, j]
        k = np.isfinite(y) & np.isfinite(p) & in_cx
        if k.sum() < 150:
            continue
        res = y - p; B = fold_blocks(pairs, k, outer, n)
        if not B:
            continue
        pm = partner_mean(res, B, n); q = k & np.isfinite(pm)
        if q.sum() < 50:
            continue
        r = float(np.corrcoef(res[q], pm[q])[0, 1])
        rng = np.random.default_rng(0); null = []
        for _ in range(200):
            vp = res.copy()
            for i, _a, _c in B:
                vp[i] = res[rng.permutation(i)]
            pp = partner_mean(vp, B, n); z = k & np.isfinite(pp)
            null.append(np.corrcoef(vp[z], pp[z])[0, 1])
        null = np.array(null)
        J[t] = dict(kind=META[t]["kind"], fraction=META[t].get("fraction"), r=r, n=int(q.sum()),
                    null_mean=float(null.mean()), z=float((r - null.mean()) / (null.std() + 1e-9)),
                    ci=cboot_r(res[q], pm[q], grp[q]))
        print(f"  {t:<24} complex effect r {r:+.3f} [{J[t]['ci'][0]:+.3f},{J[t]['ci'][1]:+.3f}]  "
              f"null {null.mean():+.3f}  z {J[t]['z']:5.1f}  n {J[t]['n']}", flush=True)
    if J:
        print(f"\n  mean over {len(J)} targets: r = {np.mean([v['r'] for v in J.values()]):.3f}   "
              f"(campaign: 0.336)", flush=True)
    json.dump(J, open(f"{out}/complex.json", "w"), indent=1, default=float)
    return state


def level_decompose(cfg, caps, out, state):
    """What the complex effect is made of (report section 3g). Needs at least two fractions."""
    names, Y, META, U = _load_targets(out)
    O = np.load(f"{out}/oof.npz", allow_pickle=True)
    acc = list(O["accession"].astype(str))
    ai = {a: i for i, a in enumerate(U)}
    P = np.full_like(Y, np.nan); P[[ai[a] for a in acc]] = O["P"].astype(np.float64)
    F = np.load(f"{out}/folds.npz"); outer, grp = F["outer"], F["group"]
    pairs, _ = _complex_adjacency(U)
    in_cx = pairs.any(1); n = len(U)
    J = {}

    def resid(t):
        j = names.index(t); return Y[:, j] - P[:, j], np.isfinite(Y[:, j]) & np.isfinite(P[:, j]) & in_cx

    # ---- 1. the measurement-stage gradient
    # The point of section 3g: a stage the sample never reached cannot have created the effect.
    STAGE_ORDER = [("input", "pre-IP lysate (never over the beads)"),
                   ("sup", "unbound supernatant (never over the beads)"),
                   ("mock", "BAIT-FREE IP (over the beads, no bait)"),
                   ("ip", "pull-down (over the beads, with bait)"),
                   ("part", "partition (lysate divided out)"),
                   ("enrich", "enrichment over input"),
                   ("spec", "specific binding (IP - bait-free IP)")]
    suffixes = sorted({t.split("_", 1)[1] for t in names if "_" in t})
    J["stage"] = {}
    for suf in suffixes:
        have = [(k, d) for k, d in STAGE_ORDER if f"{k}_{suf}" in names]
        if len(have) < 2:
            continue
        R, K = {}, None
        for k, _d in have:
            R[k], kk = resid(f"{k}_{suf}")
            K = kk if K is None else (K & kk)
        B = fold_blocks(pairs, K, outer, n)
        if not B:
            continue
        pm = {k: partner_mean(R[k], B, n) for k in R}
        q = K & np.all([np.isfinite(pm[k]) for k in pm], 0)
        if q.sum() < 100:
            continue
        row = {}
        for k, d in have:
            rr = float(np.corrcoef(R[k][q], pm[k][q])[0, 1])
            cl = META[f"{k}_{suf}"]["ceiling"]
            row[k] = dict(r=rr, ci=cboot_r(R[k][q], pm[k][q], grp[q]), ceiling=float(cl),
                          r_disattenuated=float(rr / cl) if cl > 0 else np.nan, what=d)
        for (a, _), (b, _) in itertools.combinations(have, 2):
            row[f"d_{a}_minus_{b}"] = cboot_diff(R[a][q], pm[a][q], R[b][q], pm[b][q], grp[q])
        row["n"] = int(q.sum())
        J["stage"][suf] = row
        print(f"  [{suf}] n={row['n']}  " + "  ".join(
            f"{k} {row[k]['r']:.3f}(dis {row[k]['r_disattenuated']:.3f})" for k, _ in have), flush=True)
        if "mock" in row and "ip" in row:
            d = row.get("d_mock_minus_ip")
            print(f"      DIRECT capture control: the bait-free IP gives {row['mock']['r']:.3f} against the "
                  f"pull-down's {row['ip']['r']:.3f} (difference CI {d[0]:+.3f},{d[1]:+.3f}).")
            print("      A large bait-free value means capture itself creates complex-level structure; a "
                  "small one means the bait does.")

    # ---- 2. abundance: flexible control, and abundance-discordant partners
    # The covariate must not be a copy of the target. If the abundance source is the same fraction as the
    # target -- or a component of it, since part = ip - sup, enrich = ip - input, spec = ip - mock -- then
    # the residual and the covariate carry the same measurement noise. Projecting it out then strips shared
    # noise from both the residual and the partner mean, and the control reads "it is co-abundance" when it
    # is not. Those targets are rerun against PaxDb, which is an independent measurement, and are dropped
    # from the mean when PaxDb is unavailable rather than quietly contributing a biased number.
    COMPONENTS = {"part": ("ip", "sup"), "enrich": ("ip", "input"), "spec": ("ip", "mock")}

    def _self_ref(t, src):
        f = t.split("_", 1)[0]
        if src is None or src == "paxdb":
            return None
        if f == src:
            return f"the target IS the abundance source ({src}_base)"
        if src in COMPONENTS.get(f, ()):
            return f"{f} is derived from {src}, so it carries the abundance source's noise"
        return None

    ab, ab_src = None, None
    for key in ("input", "sup"):
        if f"{key}_base" in names:
            ab, ab_src = Y[:, names.index(f"{key}_base")], key
            break
    ab_pax = None
    if os.path.exists(f"{CAMP}/named_properties.tsv"):
        _p = pd.read_csv(f"{CAMP}/named_properties.tsv", sep="\t", index_col=0).reindex(U)
        if "abundance_paxdb" in _p:
            ab_pax = _p["abundance_paxdb"].values.astype(float)
    if ab is None:
        ab, ab_src = ab_pax, "paxdb"

    def _ab_blocks(a):
        """Linear and decile design blocks, and the different-quintile adjacency, for one covariate."""
        FLEX = np.column_stack([_deciles(a), np.ones(n)])
        LIN = np.column_stack([np.where(np.isfinite(a), a, 0.0), np.isfinite(a).astype(float), np.ones(n)])
        qn = np.full(n, -1); ok = np.isfinite(a)
        if ok.sum() > 5:
            e = np.quantile(a[ok], [.2, .4, .6, .8]); qn[ok] = np.digitize(a[ok], e)
        dadj = pairs & ~(qn[:, None] == qn[None, :]) & (qn[:, None] >= 0) & (qn[None, :] >= 0)
        return LIN, FLEX, dadj

    if ab is not None and np.isfinite(ab).sum() > 200:
        BLK = {ab_src: _ab_blocks(ab)}
        if ab_pax is not None and np.isfinite(ab_pax).sum() > 200 and "paxdb" not in BLK:
            BLK["paxdb"] = _ab_blocks(ab_pax)
        print(f"  abundance covariate: {ab_src}"
              + ("" if "paxdb" not in BLK or ab_src == "paxdb"
                 else "  (PaxDb held in reserve for self-referential targets)"), flush=True)
        J["abundance_source"] = ab_src
        J["abundance"] = {}
        for t in names:
            res, k = resid(t)
            if k.sum() < 150:
                continue
            why, src = _self_ref(t, ab_src), ab_src
            if why is not None:
                if "paxdb" in BLK:
                    src = "paxdb"
                else:
                    J["abundance"][t] = dict(raw=float("nan"), self_referential=why, excluded=True)
                    print(f"  {t:<24} EXCLUDED from the abundance control: {why}; no independent "
                          f"covariate available", flush=True)
                    continue
            LIN, FLEX, diff_adj = BLK[src]
            row = {"covariate": src}
            if why is not None:
                row["self_referential_against_default"] = why
            for lab, X in (("raw", None), ("linear", LIN), ("flexible", FLEX)):
                v = res if X is None else project_out(res, k, X)
                B = fold_blocks(pairs, k, outer, n)
                if not B:
                    break
                pm = partner_mean(v, B, n); q = k & np.isfinite(pm) & np.isfinite(v)
                if q.sum() < 50:
                    break
                row[lab] = float(np.corrcoef(v[q], pm[q])[0, 1])
            Bd = fold_blocks(diff_adj, k, outer, n)
            if Bd:
                pmd = partner_mean(res, Bd, n); qd = k & np.isfinite(pmd)
                if qd.sum() >= 50:
                    row["diff_quintile"] = float(np.corrcoef(res[qd], pmd[qd])[0, 1])
                    row["n_diff"] = int(qd.sum())
            if len(row) > 1:
                J["abundance"][t] = row
                mark = "" if why is None else f"  [vs PaxDb: {t.split('_', 1)[0]} shares {ab_src}]"
                print(f"  {t:<24} raw {row.get('raw', np.nan):+.3f}  linear {row.get('linear', np.nan):+.3f}  "
                      f"flexible {row.get('flexible', np.nan):+.3f}  diff-quintile "
                      f"{row.get('diff_quintile', np.nan):+.3f}{mark}", flush=True)
        A = {t: r for t, r in J["abundance"].items() if not r.get("excluded")}
        if A:
            def mn(key):
                v = [r[key] for r in A.values() if key in r and np.isfinite(r[key])]
                return float(np.mean(v)) if v else np.nan
            J["abundance_mean"] = {k: mn(k) for k in ("raw", "linear", "flexible", "diff_quintile")}
            J["abundance_mean"]["n_targets"] = len(A)
            J["abundance_mean"]["n_excluded"] = len(J["abundance"]) - len(A)
            print(f"\n  MEAN over {len(A)} targets  raw {J['abundance_mean']['raw']:.3f}  "
                  f"linear {J['abundance_mean']['linear']:.3f}  "
                  f"flexible {J['abundance_mean']['flexible']:.3f}  "
                  f"diff-quintile {J['abundance_mean']['diff_quintile']:.3f}", flush=True)
            if J["abundance_mean"]["n_excluded"]:
                print(f"  ({J['abundance_mean']['n_excluded']} target(s) excluded as self-referential.)",
                      flush=True)
            print("  (campaign: 0.336 / 0.326 / 0.304 / 0.214 -- a large drop at diff-quintile means the "
                  "effect is co-abundance.)", flush=True)
    json.dump(J, open(f"{out}/decompose.json", "w"), indent=1, default=float)
    return state


def level_structure(cfg, caps, out, state):
    """The three structural tiers. Opt-in; see the campaign report sections 3d-3f for why."""
    print("  The structural tiers are campaign scripts and are not parameterised by study config.")
    print("  All three returned nulls on 33 targets, and 3g explained why. To run them anyway:")
    print("    3d  AlphaFold monomer features  : python code/cm13_structure.py && python code/cm14_struct_test.py")
    print("    3e  experimental interfaces     : bash code/fetch_pdb_assemblies.sh && python code/cm18_interface.py")
    print("    3f  Boltz-2 co-folding          : python code/cm20_pilot_select.py && sbatch slurm/boltz_pilot.sbatch")
    print("  Cost: 3d ~1 h CPU; 3e ~11 GB download + 17 min CPU; 3f ~5 GPU-h for 200 proteins.")
    return state


def level_report(cfg, caps, out, state):
    """Assemble what ran into one markdown document, and render it if pandoc is available."""
    name = cfg["name"]
    L = [f"---", f'title: "{cfg.get("title", name)}"', f'date: "{pd.Timestamp.today():%d %B %Y}"',
         "geometry: margin=2.1cm", "fontsize: 10pt", "---", "",
         "# What this study measured", ""]
    L.append(f"Fractions: " + ", ".join(f"`{k}` ({v})" for k, v in cfg["fractions"].items()) + ".")
    L.append("")
    L.append(f"Abundance axis: {caps['abundance_source']}.")
    L.append("")
    if not caps["has_mock"]:
        L.append("No bait-free IP was measured. That is the one fraction that would separate genuine binding "
                 "from bead carry-over, and would test the capture-nonlinearity hypothesis of report "
                 "section 3g directly rather than by inference.")
        L.append("")
    for fn, title in (("qc.json", "Replicate structure"), ("predict.json", "Sequence prediction"),
                      ("properties.json", "Measured covariates"), ("complex.json", "The complex effect"),
                      ("decompose.json", "What the complex effect is made of")):
        p = f"{out}/{fn}"
        if not os.path.exists(p):
            continue
        L += [f"# {title}", "", "```", json.dumps(json.load(open(p)), indent=1)[:4000], "```", ""]
    md = f"{out}/report.md"
    open(md, "w").write("\n".join(L))
    try:
        subprocess.run(["pandoc", md, "-o", f"{out}/report.pdf", "--pdf-engine=pdflatex"],
                       check=True, capture_output=True)
        print(f"  wrote {out}/report.pdf")
    except Exception as e:
        print(f"  wrote {md} (pandoc not run: {e})")
    return state
