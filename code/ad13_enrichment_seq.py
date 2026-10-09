"""ad13: sequence against the input-lysate quantities (GA_33), on the GPU, with the ad06 machinery.

Targets (biological-replicate means; contaminants excluded):
  input abundance                              the ad08 reference lysate level
  enrichment 35, DNAJA1 / DNAJB11 / average    35 C control pull-down minus input (ad09)
  heat 43 raw / lysate-like / selective, heat 37 selective   the ad11 decomposition (disjoint replicates)
  enrichment 35 ..., abundance removed         residual of enrichment on input (binding at fixed abundance)
Folds: 5-fold grouped by 30 % sequence cluster (pd_readout.group_kfold), the same rule for every target.
Per-layer ridge (penalty on an inner split), the inner-chosen single layer, and the NNLS stack of the
nine layers, as in ad06 (the concatenated-layer model is left out).
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


blk, gacc = resolved(GA)
EN = np.load(f"{GA}/enrichment.npz", allow_pickle=True); SH = np.load(f"{GA}/selective_heat.npz")
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
inp, con = EN["input"], EN["contaminant"].astype(bool)
ok4 = lambda c: np.isfinite(blk[f"br_{c}_0_35"]).sum(1) >= 4
e = {c: np.where(ok4(c), EN[f"e_{c}_0_35"], np.nan) for c in ("21A", "24")}
inrows = np.zeros(len(gacc), bool); inrows[FG["rows_temp_avg_43v35"]] = True
inrows37 = np.zeros(len(gacc), bool); inrows37[FG["rows_temp_avg_37v35"]] = True
RAW = {"input abundance": (inp, None), "enrichment 35 DNAJA1": (e["21A"], None), "enrichment 35 DNAJB11": (e["24"], None),
       "enrichment 35 avg": ((e["21A"] + e["24"]) / 2, None),
       "heat 43 raw": (SH["raw43"], inrows), "heat 43 lysate-like": (SH["fit43"], inrows), "heat 43 selective": (SH["sel43"], inrows),
       "heat 37 selective": (SH["sel37"], inrows37)}


def detrend(y):
    """Enrichment at fixed lysate abundance: residual of a linear fit on input."""
    m = np.isfinite(y) & np.isfinite(inp) & ~con
    b = np.polyfit(inp[m], y[m], 1)
    return np.where(m, y - np.polyval(b, inp), np.nan)


for c, nm in (("21A", "DNAJA1"), ("24", "DNAJB11")):
    RAW[f"enrichment 35 {nm}, abundance removed"] = (detrend(e[c]), None)
RAW["enrichment 35 avg, abundance removed"] = (detrend((e["21A"] + e["24"]) / 2), None)
only = sys.argv[1].split("|") if len(sys.argv) > 1 else None
TG = {}
for nm, (y, m0) in RAW.items():
    if only and nm not in only: continue
    m = np.isfinite(y) & ~con & (m0 if m0 is not None else True)
    r = np.nonzero(m)[0]
    G = FG["groups"][r]
    TG[nm] = (gacc[r], y[r].astype(float), group_kfold(G, 5), G)
print({k: len(v[1]) for k, v in TG.items()}, flush=True)


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
print(f"{'response':<14}" + "".join(f"{'L'+str(l):>7}" for l in LAY) + f"{'chosen 1':>10}{'stacked':>9}   stacked - chosen [95% CI]   mean stack weights", flush=True)
for nm, (accs, y, fid, G) in TG.items():
    X = emb(accs)                                              # (n, 9, d)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xt = torch.as_tensor(X, device=dev)
    yt = torch.as_tensor(y, device=dev, dtype=torch.float32)
    n, nl, d = X.shape
    P_layer = np.zeros((nl, n)); P_choice = np.zeros(n); P_stack = np.zeros(n); W = []
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
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    d_ = []
    for _ in range(1000):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); yy = y[s]
        d_.append((((yy - P_choice[s]) ** 2).sum() - ((yy - P_stack[s]) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
    lo, hi = np.percentile(d_, [2.5, 97.5])
    Wm = np.mean(W, 0)
    rec = dict(layers={int(l): r2(y, P_layer[i]) for i, l in enumerate(LAY)}, chosen=r2(y, P_choice), stacked=r2(y, P_stack),
               stacked_minus_chosen=[r2(y, P_stack) - r2(y, P_choice), float(lo), float(hi)], weights={int(l): float(Wm[i]) for i, l in enumerate(LAY)})
    out[nm] = rec
    np.savez(f"{GA}/ad13_oof_{nm.replace(' ', '_').replace(',', '')}.npz", accession=accs, y=y, pred=P_stack)   # out-of-fold stacked predictions
    top = sorted(rec["weights"].items(), key=lambda z: -z[1])[:3]
    print(f"{nm:<14}" + "".join(f"{100*rec['layers'][l]:>6.1f}%" for l in LAY) + f"{100*rec['chosen']:>9.1f}%{100*rec['stacked']:>8.1f}%"
          f"   {100*rec['stacked_minus_chosen'][0]:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]   " + ", ".join(f"L{l} {w:.2f}" for l, w in top), flush=True)
fn = f"{GA}/ad13_results.json"
prev = json.load(open(fn)) if os.path.exists(fn) else {}
prev.update(out); json.dump(prev, open(fn, "w"), indent=1)
print(f"\n{(time.time()-T0)/60:.1f} min\nAD13_DONE", flush=True)
