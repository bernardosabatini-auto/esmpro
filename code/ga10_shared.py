"""ga10: is there one nonlinear sequence model behind every temperature, with only the linear
readout changing?

Two families of GA_33 targets, each fitted on the proteins kept for all of its targets:
  stau   staurosporine vs control at 35, 37 and 43 C (co-chaperone averaged), ESMC layer 50
  heat   37 vs 35 C and 43 vs 35 C (co-chaperone averaged), ESMC final layer

Per outer fold (5 folds, whole sequence clusters held out), all with the same fixed
architecture (16-member ensemble, 2560-512-128, GELU, dropout 0.4, weight decay 1e-2, lr 3e-4,
median over members). A fixed configuration is used deliberately: on the staurosporine targets
inner-split model selection is anti-informative (pd/ga09), and a fixed model keeps every
comparison like for like.

  ridge      ridge on the embedding, one per target (penalty on the inner split)
  separate   one network per target
  shared     one network, K linear output heads trained jointly
  transfer   the network trained on target A, frozen; its last hidden layer (16 members x 128 =
             2048 features) read out by a NEW ridge for target B. A->B against B's own network
             answers "same nonlinear model, different linear readout" directly.

Out-of-fold predictions are saved for the per-protein and per-family analysis (ga11).
"""
import os, sys, json, time, argparse
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
sys.path.insert(0, f"{ROOT}/code")
from pd_readout import ridge_fit, ridge_pred, r2, inner_split, train_ens, ens_hidden, group_kfold

D = f"{ROOT}/data/ga_data"
ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=("stau", "heat"))
ap.add_argument("--members", type=int, default=16)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
T0 = time.time()
FAM = {"stau": (["stau_avg_35", "stau_avg_37", "stau_avg_43"], 50),
       "heat": (["temp_avg_37v35", "temp_avg_43v35"], 80)}
TG, LAYER = FAM[a.family]
CFG = dict(hidden=(512, 128), drop=0.4, wd=1e-2, lr=3e-4)
ALPHAS = (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5)

FO = np.load(f"{D}/folds_ga05.npz", allow_pickle=True)
rows = sorted(set.intersection(*[set(FO[f"rows_{t}"].tolist()) for t in TG]))
rows = np.array(rows)
LK = {t: dict(zip(FO[f"rows_{t}"].tolist(), FO[f"y_{t}"].tolist())) for t in TG}
Y = np.array([[LK[t][r] for t in TG] for r in rows])
G = FO["groups"][rows]
fid = group_kfold(G, 5)
X = torch.as_tensor(np.load(f"{D}/emb_L{LAYER}.npy")[rows], device=dev)
K, n = len(TG), len(rows)
print(f"{a.family}: {K} targets {TG} | {n} proteins kept for all | layer {LAYER} | {a.members} members | {dev}", flush=True)
print(f"  target correlations:\n{np.round(np.corrcoef(Y.T), 3)}", flush=True)

P = {k: np.full((n, K), np.nan) for k in ("ridge", "separate", "shared")}
PT = np.full((K, n, K), np.nan)                     # transfer: trunk from A, readout for B
folds = range(2 if a.quick else 5)


def ridge_cv(Ztr_, ytr_, Zva_, yva_):
    best = (-1e9, None)
    for al in ALPHAS:
        wm = ridge_fit(Ztr_, ytr_, al)
        v = r2(yva_.cpu(), ridge_pred(Zva_, wm).cpu())
        if v > best[0]:
            best = (v, al)
    return best[1]


for f in folds:
    t0 = time.time()
    tr, te = np.nonzero(fid != f)[0], np.nonzero(fid == f)[0]
    isv = inner_split(G[tr], 100 + f)
    itr, iva = tr[~isv], tr[isv]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    Yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)
    ym, ys = Yt[itr].mean(0), Yt[itr].std(0)
    Ys = (Yt - ym) / ys
    ep = dict(members=a.members, epochs=(3 if a.quick else a.epochs), patience=(2 if a.quick else a.patience))
    # ridge, per target
    for k in range(K):
        al = ridge_cv(Z[itr], Ys[itr, k], Z[iva], Ys[iva, k])
        wm = ridge_fit(Z[tr], Ys[tr, k], al)
        P["ridge"][te, k] = (ridge_pred(Z[te], wm) * ys[k] + ym[k]).cpu().numpy()
    # shared trunk, K heads
    net = train_ens(CFG, Z[itr], Ys[itr], Z[iva], Ys[iva], **ep)
    with torch.no_grad():
        P["shared"][te] = (net(Z[te]).median(0).values * ys + ym).cpu().numpy()
    # one network per target, then its frozen trunk read out for every target
    for A in range(K):
        net = train_ens(CFG, Z[itr], Ys[itr, A:A + 1], Z[iva], Ys[iva, A:A + 1], **ep)
        with torch.no_grad():
            P["separate"][te, A] = (net(Z[te])[..., 0].median(0).values * ys[A] + ym[A]).cpu().numpy()
            H = ens_hidden(net, Z).permute(1, 0, 2).reshape(n, -1)       # (n, E*h)
        hm, hs = H[tr].mean(0), H[tr].std(0) + 1e-6
        H = (H - hm) / hs
        for B in range(K):
            al = ridge_cv(H[itr], Ys[itr, B], H[iva], Ys[iva, B])
            wm = ridge_fit(H[tr], Ys[tr, B], al)
            PT[A, te, B] = (ridge_pred(H[te], wm) * ys[B] + ym[B]).cpu().numpy()
    print(f"  fold {f}: {time.time()-t0:.0f}s", flush=True)

mask = ~np.isnan(P["ridge"][:, 0])
R = lambda p, k: r2(Y[mask, k], p[mask, k])
out = dict(family=a.family, targets=TG, layer=LAYER, n=int(n), cfg={k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()})
print(f"\n  variance explained (out of fold)", flush=True)
print(f"  {'target':<16}{'ridge':>8}{'separate':>10}{'shared':>9}{'sep+ridge':>11}{'shared+ridge':>14}", flush=True)
for k, t in enumerate(TG):
    row = dict(ridge=R(P["ridge"], k), separate=R(P["separate"], k), shared=R(P["shared"], k),
               sep_blend=R(0.5 * P["separate"] + 0.5 * P["ridge"], k),
               shared_blend=R(0.5 * P["shared"] + 0.5 * P["ridge"], k))
    out[t] = row
    print(f"  {t:<16}" + "".join(f"{100*row[c]:>{w}.2f}%" for c, w in
          (("ridge", 7), ("separate", 9), ("shared", 8), ("sep_blend", 10), ("shared_blend", 13))), flush=True)
print(f"\n  transfer: trunk trained on the ROW target, new ridge readout for the COLUMN target", flush=True)
print("  " + " " * 16 + "".join(f"{t:>16}" for t in TG), flush=True)
TM = np.array([[R(PT[A], B) for B in range(K)] for A in range(K)])
for A, t in enumerate(TG):
    print(f"  {t:<16}" + "".join(f"{100*TM[A, B]:>15.2f}%" for B in range(K)), flush=True)
out["transfer"] = TM.tolist()
if not a.quick:
    np.savez(f"{D}/ga10_{a.family}.npz", rows=rows, Y=Y, fold=fid, targets=np.array(TG),
             ridge=P["ridge"], separate=P["separate"], shared=P["shared"], transfer=PT)
    json.dump(out, open(f"{D}/ga10_{a.family}.json", "w"), indent=1)
if dev == "cuda":
    print(f"\npeak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
print(f"total {(time.time()-T0)/60:.1f} min\nGA10_DONE", flush=True)
