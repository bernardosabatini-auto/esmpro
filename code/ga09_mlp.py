"""ga09: ensemble-MLP readout and per-sample likelihoods on the GA_33 fold changes.

Same protocol as pd09 on the salt data, with what that run taught built in: the full
standardised embedding (whitened PCs always lost), no residual-on-ridge form (it never moved off
the ridge solution), and a slower learning rate in the grid (the 75 mM networks peaked at epoch
1-2). The outer folds are scikit-learn's GroupKFold over the exact rows ga05 scored
(folds_ga05.npz), so ridge here and in ga05 differ only in how the penalty is chosen -- here on the
inner split, like every MLP candidate.

Per-sample noise model for a 6-vs-6 replicate contrast (one per co-chaperone, two for averaged
targets): each replicate run gets its own noise level, times a smooth (quadratic) dependence on
the protein's abundance, fitted on outer-training proteins only from leave-one-out scatter within
each group. The fold change is then the precision-weighted difference of group means, and for an
averaged target the two co-chaperones are combined by precision as well. Measurement noise is
2 % of the variance of the heat response but 25-50 % for the STAU effect, so this is where the
per-sample likelihood can act.

  python ga09_mlp.py --layer 80 --targets temp_avg_43v35,temp_21A_43v35,temp_24_43v35,temp_avg_37v35
  python ga09_mlp.py --layer 50 --targets stau_avg_35,stau_avg_37,stau_avg_43
"""
import os, sys, json, time, argparse, itertools, math
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pd_readout import ridge_fit, ridge_pred, r2, inner_split, train_mlp, boot_diff

D = f"{ROOT}/data/ga_data"
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--members", type=int, default=16)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--combine", default="mean", choices=("mean", "median"))
ap.add_argument("--var-cap", type=float, default=None, help="cap on the het head's variance, units of var(y)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
rng = np.random.default_rng(0)
T0 = time.time()

blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
FO = np.load(f"{D}/folds_ga05.npz", allow_pickle=True)
EMB = np.load(f"{D}/emb_L{a.layer}.npy")
groups = FO["groups"]
CEIL = {k: v["ceiling"] for k, v in json.load(open(f"{D}/ga05_results.json")).items() if not k.startswith("_")}
print(f"GA_33 readout | ESMC layer {a.layer} | {a.members}-member ensembles ({a.combine}) | "
      f"het variance cap {a.var_cap} | device {dev}", flush=True)


def pairs_of(name):
    """(treated, control) replicate matrices behind a target, one pair per co-chaperone."""
    kind, chap, rest = name.split("_", 2)
    chaps = ("21A", "24") if chap == "avg" else (chap,)
    if kind == "stau":
        return [(blk[f"br_{c}_10_{rest}"], blk[f"br_{c}_0_{rest}"]) for c in chaps]
    t = rest.replace("v35", "")
    return [(blk[f"br_{c}_0_{t}"], blk[f"br_{c}_0_35"]) for c in chaps]


def noise_model(pairs, rows_fit, rows_out):
    """Fit per-run log-variance + quadratic abundance on rows_fit; return (y_hat, v) on rows_out."""
    mats = [M for pr in pairs for M in pr]                      # 2 or 4 groups of 6 runs
    with np.errstate(invalid="ignore"):
        ab = np.nanmean(np.hstack(mats), 1)
    am, asd = np.nanmean(ab[rows_fit]), np.nanstd(ab[rows_fit])
    z = np.where(np.isfinite(ab), (ab - am) / asd, 0.0)
    nrun = 6 * len(mats)
    Xs, ys = [], []
    for gi, M in enumerate(mats):
        Mf = M[rows_fit]
        for j in range(6):
            with np.errstate(invalid="ignore"):
                r = Mf[:, j] - np.nanmean(np.delete(Mf, j, 1), 1)
            ok = np.isfinite(r)
            X = np.zeros((ok.sum(), nrun + 2))
            X[:, gi * 6 + j] = 1
            X[:, nrun], X[:, nrun + 1] = z[rows_fit][ok], z[rows_fit][ok] ** 2
            Xs.append(X); ys.append(np.log(r[ok] ** 2 + 1e-8))
    beta = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]
    zo = z[rows_out]

    def w_of(gi, M):
        lv = np.stack([beta[gi * 6 + j] + beta[nrun] * zo + beta[nrun + 1] * zo ** 2 for j in range(6)], 1)
        lv = lv + 1.27 - math.log(6 / 5)                       # E log chi2_1; LOO over 5 others
        return np.exp(-lv) * np.isfinite(M[rows_out])

    est, var = [], []
    for pi, (B, A) in enumerate(pairs):
        wb, wa = w_of(2 * pi, B), w_of(2 * pi + 1, A)
        sb, sa = wb.sum(1), wa.sum(1)
        mb = np.nansum(np.nan_to_num(B[rows_out]) * wb, 1) / np.maximum(sb, 1e-12)
        ma = np.nansum(np.nan_to_num(A[rows_out]) * wa, 1) / np.maximum(sa, 1e-12)
        est.append(mb - ma)
        var.append(1 / np.maximum(sb, 1e-12) + 1 / np.maximum(sa, 1e-12))
    est, var = np.stack(est, 1), np.stack(var, 1)
    yhat = (est / var).sum(1) / (1 / var).sum(1)
    run_sd = np.exp(0.5 * (beta[:nrun] + 1.27 - math.log(6 / 5)))
    return yhat, 1 / (1 / var).sum(1), run_sd


GRID = [dict(hidden=h, drop=d, wd=w, lr=lr) for h, d, w, lr in
        itertools.product(((256,), (512, 128), (1024, 256)), (0.2, 0.4), (1e-2, 1e-1), (1e-3, 3e-4))]
if a.quick:
    GRID = GRID[:2]
ALPHAS = (1e3, 3e3, 1e4, 3e4, 1e5)


def cname(c):
    return f"h{'-'.join(map(str, c['hidden'])):<9} drop {c['drop']:.1f} wd {c['wd']:.0e} lr {c['lr']:.0e}"


results = {"layer": a.layer, "members": a.members, "grid": [cname(c) for c in GRID]}
for name in a.targets.split(","):
    t0 = time.time()
    rows, fid, Y = FO[f"rows_{name}"], FO[f"fold_{name}"], FO[f"y_{name}"].astype(np.float64)
    pairs = pairs_of(name)
    G = groups[rows]
    X = torch.as_tensor(EMB[rows], device=dev)
    nf = 2 if a.quick else 5
    prep = []
    for f in range(nf):
        tr, te = np.nonzero(fid != f)[0], np.nonzero(fid == f)[0]
        isv = inner_split(G[tr], 100 + f)
        itr, iva = tr[~isv], tr[isv]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)
        ym, ys = yt[itr].mean(), yt[itr].std()
        best = (-1e9, None)
        for al in ALPHAS:
            wm = ridge_fit(Z[itr], (yt[itr] - ym) / ys, al)
            v = r2(yt[iva].cpu(), (ridge_pred(Z[iva], wm) * ys + ym).cpu())
            if v > best[0]:
                best = (v, al)
        wm = ridge_fit(Z[tr], (yt[tr] - ym) / ys, best[1])
        rte = (ridge_pred(Z[te], wm) * ys + ym).cpu().numpy()
        yhat, vv, run_sd = noise_model(pairs, rows[tr], rows[itr])
        prep.append(dict(tr=tr, te=te, itr=itr, iva=iva, Z=Z, yt=yt, ym=ym, ys=ys, alpha=best[1],
                         ridge_te=rte, yhat=torch.as_tensor(yhat, device=dev, dtype=torch.float32),
                         v=torch.as_tensor(vv, device=dev, dtype=torch.float32) / ys ** 2, run_sd=run_sd))
    mask = np.zeros(len(Y), bool)
    rid = np.full(len(Y), np.nan)
    for p in prep:
        mask[p["te"]] = True
        rid[p["te"]] = p["ridge_te"]
    Ym = Y[mask]
    r2r = r2(Ym, rid[mask])
    ceil = CEIL.get(name, float("nan"))
    vfrac = float(np.mean([float(p["v"].mean()) for p in prep]))
    yc = float(np.mean([np.corrcoef(p["yhat"].cpu().numpy(), Y[p["itr"]])[0, 1] for p in prep]))
    print(f"\n{'='*104}\n{name}  n={len(Y)}  ceiling {100*ceil:.1f}%  ridge R2 {100*r2r:.2f}%  "
          f"alphas {[p['alpha'] for p in prep]}", flush=True)
    print(f"  per-run noise sd (fold 1): {np.round(prep[0]['run_sd'], 3).tolist()}", flush=True)
    print(f"  measurement variance = {100*vfrac:.1f}% of var(y);  corr(weighted, plain fold change) {yc:.4f}", flush=True)

    def run(cfg, loss):
        oof, vals, eps, taus = np.full(len(Y), np.nan), [], [], []
        for p in prep:
            ztr = ((p["yhat"] if loss != "mse" else p["yt"][p["itr"]]) - p["ym"]) / p["ys"]
            zva = (p["yt"][p["iva"]] - p["ym"]) / p["ys"]
            pva, pte, inf = train_mlp(cfg, loss, p["Z"][p["itr"]], ztr, p["v"], p["Z"][p["iva"]], zva,
                                      p["Z"][p["te"]], members=a.members, epochs=(3 if a.quick else a.epochs),
                                      patience=(2 if a.quick else a.patience),
                                      combine=a.combine, var_cap=a.var_cap)
            vals.append(r2(zva.cpu(), pva.cpu()))
            oof[p["te"]] = (pte * p["ys"] + p["ym"]).cpu().numpy()
            eps.append(inf["epochs"]); taus.append(inf["tau2"])
        return oof, float(np.mean(vals)), float(np.mean(eps)), float(np.mean(taus))

    print(f"  {'configuration':<40}{'loss':>6}{'inner val':>11}{'outer R2':>10}{'vs ridge':>10}{'epoch':>7}{'s':>5}", flush=True)
    allm = []
    for c in GRID:
        t1 = time.time()
        oof, val, ep, _ = run(c, "mse")
        o = r2(Ym, oof[mask])
        allm.append(dict(cfg=c, name=cname(c), loss="mse", val=val, outer=o, epochs=ep, oof=oof))
        print(f"  {cname(c):<40}{'mse':>6}{100*val:>10.2f}%{100*o:>9.2f}%{100*(o-r2r):>+9.2f}{ep:>7.0f}{time.time()-t1:>5.0f}", flush=True)
    top = sorted(allm, key=lambda d: -d["val"])[: a.top]
    for d0 in top:
        for loss in ("naive", "rep", "het"):
            t1 = time.time()
            oof, val, ep, tau = run(d0["cfg"], loss)
            o = r2(Ym, oof[mask])
            allm.append(dict(cfg=d0["cfg"], name=d0["name"], loss=loss, val=val, outer=o, epochs=ep, oof=oof,
                             tau2=tau, vs_mse=o - d0["outer"]))
            ex = f"   tau2 {tau:.3f}" if loss == "rep" else ""
            print(f"  {d0['name']:<40}{loss:>6}{100*val:>10.2f}%{100*o:>9.2f}%{100*(o-r2r):>+9.2f}{ep:>7.0f}"
                  f"{time.time()-t1:>5.0f}   ({100*(o-d0['outer']):+.2f} vs its mse){ex}", flush=True)
    sel = max(allm, key=lambda d: d["val"])
    lo, hi = boot_diff(Ym, sel["oof"][mask], rid[mask], a.boot, rng)
    bl = 0.5 * sel["oof"] + 0.5 * rid
    ob = r2(Ym, bl[mask])
    blo, bhi = boot_diff(Ym, bl[mask], rid[mask], a.boot, rng)
    print(f"\n  SELECTED on inner val: {sel['name']}  loss {sel['loss']}", flush=True)
    print(f"    outer R2 {100*sel['outer']:.2f}%  vs ridge {100*r2r:.2f}%  ->  {100*(sel['outer']-r2r):+.2f} points "
          f"[{100*lo:+.2f},{100*hi:+.2f}]   ({100*sel['outer']/ceil:.0f}% of ceiling)", flush=True)
    print(f"    50/50 with ridge (fixed weight): {100*ob:.2f}%  {100*(ob-r2r):+.2f} points [{100*blo:+.2f},{100*bhi:+.2f}]", flush=True)
    results[name] = dict(n=int(len(Y)), ceiling=ceil, ridge_r2=r2r, ridge_alphas=[p["alpha"] for p in prep],
                         meas_var_frac=vfrac, yhat_corr=yc,
                         models=[{k: v for k, v in d.items() if k not in ("oof", "cfg")} for d in allm],
                         selected=dict(name=sel["name"], loss=sel["loss"], val=sel["val"], outer=sel["outer"],
                                       gain=sel["outer"] - r2r, gain_ci=[lo, hi]),
                         blend=dict(outer=ob, gain=ob - r2r, gain_ci=[blo, bhi]),
                         minutes=(time.time() - t0) / 60)
    np.save(f"{D}/ga09_oof_{name}{a.tag}.npy", np.stack([Y, rid, sel["oof"], bl]))

if dev == "cuda":
    print(f"\npeak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
print(f"total {(time.time()-T0)/60:.1f} min", flush=True)
if not a.quick:
    json.dump(results, open(f"{D}/ga09_results{a.tag}.json", "w"), indent=1)
print("GA09_DONE", flush=True)
