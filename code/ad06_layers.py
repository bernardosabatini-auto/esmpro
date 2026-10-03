"""ad06: does reading several ESMC layers at once predict better than the best single layer?

The best layer differed by response (layer 50 for salt and staurosporine, the final layer for heat and
the co-chaperone preference). For eight responses, on their usual cluster-grouped outer folds:
  * each of the nine pooled layers (0, 10, ..., 80) alone, ridge;
  * "best single layer": the layer chosen on an inner split of each training fold (never on the test fold);
  * stacking: one ridge per layer, combined with non-negative weights learned on inner out-of-fold
    predictions of the training fold;
  * concatenation: all nine layers in one ridge, each layer standardised and scaled by 1/sqrt(9) so
    each contributes equally (dual form, n x n kernels).
Every penalty is chosen on the inner split. Ridge in the primal uses per-fold Gram matrices, so the
inner models cost one subtraction each; the concatenation uses the dual. GPU (torch), fp32 solves.
"""
import os, sys, json, csv, time
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pd_readout import group_kfold
PD, GA = f"{ROOT}/data/pd_data", f"{ROOT}/data/ga_data"
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
ALPHAS = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]   # finer: staurosporine R2 swings from -20 % to +6 % across this range
T0 = time.time()


def keys(fn):
    return [l[1:].strip().split()[0].split("|")[0] for l in open(fn) if l[0] == ">"]


old = np.load(f"{PD}/embeddings_layers.npz", allow_pickle=True)
new = np.load(f"{GA}/embeddings_layers_new.npz", allow_pickle=True)
LAY = [int(x) for x in old["layers"]]
po, pn = old["pool"], new["pool"]
src = {k: ("o", i) for i, k in enumerate(keys(f"{PD}/sequences.fasta"))}
src.update({k: ("n", i) for i, k in enumerate(keys(f"{GA}/new_sequences.fasta"))})


def emb(accs):
    return np.stack([(po if src[a][0] == "o" else pn)[src[a][1]] for a in accs]).astype(np.float32)   # (n, 9, 2560)


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


pb, pacc = resolved(PD); gb, gacc = resolved(GA)
FP = np.load(f"{PD}/folds.npz", allow_pickle=True); FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True); SP = np.load(f"{GA}/spec.npz", allow_pickle=True)
TG = {}
for c, nm in (("s75", "salt 75"), ("s150", "salt 150")):
    r = FP[f"rows_{c}"]; TG[nm] = (pacc[r], pb[f"y_{c}"].astype(float)[r], FP[f"fold_{c}"], FP["groups"][r])
for t, nm in (("temp_avg_43v35", "heat 43"), ("temp_avg_37v35", "heat 37"), ("stau_avg_35", "stau 35"), ("stau_avg_37", "stau 37"), ("stau_avg_43", "stau 43")):
    r = FG[f"rows_{t}"]; TG[nm] = (gacc[r], FG[f"y_{t}"].astype(float), FG[f"fold_{t}"], FG["groups"][r])
r = SP["rows_spec_avg"]; TG["DNAJB11 pref"] = (gacc[r], SP["y_spec_avg"].astype(float), SP["fold_spec_avg"], SP["groups"][r])


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ridge_from_gram(XtX, Xty, al):
    d = XtX.shape[0]
    return torch.linalg.solve(XtX + al * torch.eye(d, device=dev), Xty)


def nnls(P, y, iters=2000):
    """Non-negative least squares for a handful of columns, projected gradient (with intercept)."""
    P = torch.as_tensor(P, device=dev, dtype=torch.float64); y = torch.as_tensor(y, device=dev, dtype=torch.float64)
    Pm, ym = P.mean(0), y.mean(); Pc, yc = P - Pm, y - ym
    L = torch.linalg.matrix_norm(Pc.T @ Pc, 2) + 1e-9
    w = torch.full((P.shape[1],), 1.0 / P.shape[1], device=dev, dtype=torch.float64)
    for _ in range(iters):
        w = torch.clamp(w - (Pc.T @ (Pc @ w - yc)) / L, min=0)
    return w.cpu().numpy(), float((ym - Pm @ w).cpu())


out = {}
print(f"{'response':<14}" + "".join(f"{'L'+str(l):>7}" for l in LAY) + f"{'chosen 1':>10}{'stacked':>9}{'concat':>8}   stacked - chosen [95% CI]   mean stack weights", flush=True)
for nm, (accs, y, fid, G) in TG.items():
    X = emb(accs)                                              # (n, 9, d)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(y, device=dev, dtype=torch.float32)
    n, nl, d = X.shape
    P_layer = np.zeros((nl, n)); P_choice = np.zeros(n); P_stack = np.zeros(n); P_cat = np.zeros(n); W = []
    for f in range(5):
        tr = np.nonzero(fid != f)[0]; te = np.nonzero(fid == f)[0]
        ifid = group_kfold(G[tr], 5)
        ym = y[tr].mean()
        inner_pred = np.zeros((nl, len(ALPHAS), len(tr)))
        best_al = []
        for l in range(nl):
            Xl = Xt[tr, l]; yl = yt[tr] - ym
            grams = [(Xl[ifid == k].T @ Xl[ifid == k], Xl[ifid == k].T @ yl[ifid == k]) for k in range(5)]
            G_all = sum(g[0] for g in grams); b_all = sum(g[1] for g in grams)
            for ai, al in enumerate(ALPHAS):
                for k in range(5):
                    w = ridge_from_gram(G_all - grams[k][0], b_all - grams[k][1], al)
                    inner_pred[l, ai, ifid == k] = (Xl[ifid == k] @ w).cpu().numpy() + ym
            sc = [r2(y[tr], inner_pred[l, ai]) for ai in range(len(ALPHAS))]
            ai = int(np.argmax(sc)); best_al.append((ai, sc[ai]))
            w = ridge_from_gram(G_all, b_all, ALPHAS[ai])
            P_layer[l, te] = (Xt[te, l] @ w).cpu().numpy() + ym
        # chosen single layer (inner-split choice)
        lbest = int(np.argmax([s for _, s in best_al])); P_choice[te] = P_layer[lbest, te]
        # stacking on inner OOF predictions
        Pin = np.stack([inner_pred[l, best_al[l][0]] for l in range(nl)], 1)
        w, b0 = nnls(Pin, y[tr]); W.append(w)
        P_stack[te] = P_layer[:, te].T @ w + b0
        # concatenation, dual: K = sum over layers / nl
        Xc = Xt[:, :, :].reshape(n, nl * d) / np.sqrt(nl)
        K = Xc[tr] @ Xc[tr].T
        bestc = (-9, None)
        for al in ALPHAS:
            pin = np.zeros(len(tr))
            for k in range(5):
                a_ = torch.as_tensor(ifid != k, device=dev); b_ = torch.as_tensor(ifid == k, device=dev)
                yk = yt[tr][a_] - yt[tr][a_].mean()
                c = torch.linalg.solve(K[a_][:, a_] + al * torch.eye(int(a_.sum()), device=dev), yk)
                pin[ifid == k] = (K[b_][:, a_] @ c).cpu().numpy() + float(yt[tr][a_].mean())
            v = r2(y[tr], pin)
            if v > bestc[0]: bestc = (v, al)
        c = torch.linalg.solve(K + bestc[1] * torch.eye(len(tr), device=dev), yt[tr] - ym)
        P_cat[te] = ((Xc[te] @ Xc[tr].T) @ c).cpu().numpy() + ym
        del K, Xc
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    d_ = []
    for _ in range(1000):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); yy = y[s]
        d_.append((((yy - P_choice[s]) ** 2).sum() - ((yy - P_stack[s]) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
    lo, hi = np.percentile(d_, [2.5, 97.5])
    Wm = np.mean(W, 0)
    rec = dict(layers={int(l): r2(y, P_layer[i]) for i, l in enumerate(LAY)}, chosen=r2(y, P_choice), stacked=r2(y, P_stack), concat=r2(y, P_cat),
               stacked_minus_chosen=[r2(y, P_stack) - r2(y, P_choice), float(lo), float(hi)], weights={int(l): float(Wm[i]) for i, l in enumerate(LAY)})
    out[nm] = rec
    top = sorted(rec["weights"].items(), key=lambda z: -z[1])[:3]
    print(f"{nm:<14}" + "".join(f"{100*rec['layers'][l]:>6.1f}%" for l in LAY) + f"{100*rec['chosen']:>9.1f}%{100*rec['stacked']:>8.1f}%{100*rec['concat']:>7.1f}%"
          f"   {100*rec['stacked_minus_chosen'][0]:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]   " + ", ".join(f"L{l} {w:.2f}" for l, w in top), flush=True)
json.dump(out, open(f"{GA}/ad06_results.json", "w"), indent=1)
print(f"\n{(time.time()-T0)/60:.1f} min\nAD06_DONE", flush=True)
