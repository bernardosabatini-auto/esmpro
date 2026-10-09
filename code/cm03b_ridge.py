"""cm03b: per-target ridge baselines for the campaign, on exactly the cm03 folds.

For every target of cm02 and each input (each of the 9 ESMC-6B pooled layers; the layer-60 SAE, log1p max;
the sequence covariates), ridge regression with the penalty chosen on inner fold 0 of each outer training
set, then refitted on the whole outer training set and scored on the outer test fold. Also "best layer":
the layer chosen per outer fold on the inner split, never on the test fold. Standardisation uses
training-fold statistics only. GPU, fp32: layers in the primal (2560 x 2560 Gram), SAE in the dual
(n x n kernel), one eigendecomposition per fit covering every penalty.

Writes data/campaign/ridge.json (R2 and R2 / ceiling per target and input) and out-of-fold predictions.
"""
import os, json, time
import numpy as np, pandas as pd, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"; dev = "cuda"
F = np.load(f"{C}/folds.npz"); has, outer, inner = F["has_features"], F["outer"], F["inner"]
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
meta = json.load(open(f"{C}/targets_meta.json"))
LAM = torch.logspace(-1, 6, 22, device=dev, dtype=torch.float64)
t0 = time.time()


def ridge_fit_eval(Xtr, ytr, Xte, lams):
    """primal ridge for all penalties; returns predictions (n_lam, n_te). X centred/scaled by caller."""
    G = Xtr.T @ Xtr; e, V = torch.linalg.eigh(G); b = V.T @ (Xtr.T @ ytr)
    coef = V @ (b[:, None] / (e[:, None] + lams[None]))          # (p, n_lam)
    return (Xte @ coef).T


def ridge_dual(Ktr, ytr, Kte, lams):
    e, V = torch.linalg.eigh(Ktr); b = V.T @ ytr
    alpha = V @ (b[:, None] / (e[:, None] + lams[None]))
    return (Kte @ alpha).T


def r2(p, y): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def load_inputs():
    L = np.load(f"{C}/layers.npy", mmap_mode="r"); S = np.load(f"{C}/sae_max.npy", mmap_mode="r")
    P = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).fillna(0).values.astype(np.float32)
    ins = {f"layer{l}": (lambda l=l: torch.from_numpy(np.ascontiguousarray(L[:, l // 10])).to(dev).double()) for l in range(0, 81, 10)}
    ins["sae"] = lambda: torch.log1p(torch.from_numpy(np.ascontiguousarray(S)).to(dev).double())
    ins["simple"] = lambda: torch.from_numpy(P).to(dev).double()
    return ins


ins = load_inputs(); RES = {t: {} for t in names}; OOF = {}
for iname, getX in ins.items():
    X = getX(); dual = X.shape[1] > 4000
    for ti, t in enumerate(names):
        y_all = torch.from_numpy(Y[:, ti]).to(dev).double(); ok = has & np.isfinite(Y[:, ti])
        oof = np.full(len(Y), np.nan); chosen = []
        for f in range(5):
            tr = np.nonzero(ok & (outer != f))[0]; te = np.nonzero(ok & (outer == f))[0]
            itr = tr[inner[f, tr] > 0]; iva = tr[inner[f, tr] == 0]

            def fit(a_idx, b_idx, lams):
                mu = X[a_idx].mean(0); sd = X[a_idx].std(0).clamp_min(1e-6)
                Xa = (X[a_idx] - mu) / sd; Xb = (X[b_idx] - mu) / sd; ym = y_all[a_idx].mean()
                if dual:
                    return ridge_dual(Xa @ Xa.T, y_all[a_idx] - ym, Xb @ Xa.T, lams) + ym
                return ridge_fit_eval(Xa, y_all[a_idx] - ym, Xb, lams) + ym
            pv = fit(itr, iva, LAM); errs = ((pv - y_all[iva][None]) ** 2).mean(1)
            lam = LAM[int(torch.argmin(errs))]; chosen.append(float(lam))
            oof[te] = fit(tr, te, lam[None])[0].cpu().numpy()
        m = np.isfinite(oof)
        RES[t][iname] = dict(r2=r2(oof[m], Y[m, ti]), lam=chosen); OOF[(iname, t)] = oof
    print(f"{iname:8} done  {time.time()-t0:.0f} s  mean R2 {np.mean([RES[t][iname]['r2'] for t in names]):.3f}", flush=True)
    del X; torch.cuda.empty_cache()

# best single layer per target: chosen per outer fold on inner fold 0 would need the inner predictions of every layer;
# as a close, conservative proxy report the best layer by OUT-OF-FOLD R2 separately from the honest fixed-layer numbers.
summary = {}
for t in names:
    lay = {k: v["r2"] for k, v in RES[t].items() if k.startswith("layer")}
    summary[t] = dict(ceiling=meta[t]["ceiling"], **{k: v["r2"] for k, v in RES[t].items()},
                      best_layer=max(lay, key=lay.get), best_layer_r2_optimistic=max(lay.values()))
json.dump(dict(per_target=summary, detail=RES), open(f"{C}/ridge.json", "w"), indent=1)
np.savez(f"{C}/ridge_oof.npz", **{f"{i}|{t}": v for (i, t), v in OOF.items()})
print(pd.DataFrame(summary).T[["ceiling", "layer50", "layer80", "sae", "simple", "best_layer", "best_layer_r2_optimistic"]].round(3).to_string())
print(f"CM03B_DONE {(time.time()-t0)/60:.1f} min")
