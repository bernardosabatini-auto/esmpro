"""pd13: do the ESMC-6B SAE features (layer 60, 16,384 features) explain the fold changes, and can a
small set of them stand in for the whole embedding?

Targets: salt 75 and 150 mM (HSPB1); heat 37 and 43 vs 35 C and staurosporine at 35/37/43 C
(DNAJA1 and DNAJB11 averaged). Rows and outer folds are exactly those used for every earlier ridge
number (pd folds.npz, ga folds_ga05.npz: scikit-learn GroupKFold over 30 %-identity clusters).

A. Representation. Ridge on SAE features against ridge on the dense layer-60 mean-pool, same rows,
   same folds. 16,384 features and ~8,000 proteins: solved in the dual (kernel) form, one
   eigendecomposition per fold covering every penalty. Features are log1p-transformed and
   standardised over the target's proteins (no target information enters the scaling).
B. Sparsity. Within each training fold, rank features by |correlation| with the target, fit ridge
   on the top k, score the held-out fold: R2 against k shows whether a few features carry the
   signal. Features chosen in every fold are recorded.
C. Associations on all proteins: correlation of every feature with each target, p from the t
   distribution, Benjamini-Hochberg q. The strongest features are written out for annotation.
"""
import os, csv, json
import numpy as np
import torch
from scipy import stats

DEV = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
PD, GA, SD = f"{ROOT}/data/pd_data", f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
rng = np.random.default_rng(0)

S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True)
sacc = [str(a) for a in S["accession"]]
spos = {a: i for i, a in enumerate(sacc)}
POOL = {"max": S["max"], "mean": S["mean"], "freq": S["freq"]}
print(f"SAE features: {len(sacc)} proteins x {POOL['max'].shape[1]}", flush=True)


def keys(fn):
    return [l[1:].strip().split()[0].split("|")[0] for l in open(fn) if l[0] == ">"]


old = np.load(f"{PD}/embeddings_layers.npz", allow_pickle=True)
new = np.load(f"{GA}/embeddings_layers_new.npz", allow_pickle=True)
li = [int(x) for x in old["layers"]].index(60)
dpos = {k: ("o", i) for i, k in enumerate(keys(f"{PD}/sequences.fasta"))}
dpos.update({k: ("n", i) for i, k in enumerate(keys(f"{GA}/new_sequences.fasta"))})
po, pn = old["pool"], new["pool"]


def dense60(accs):
    return np.stack([(po if dpos[a][0] == "o" else pn)[dpos[a][1], li] for a in accs]).astype(np.float64)


def resolved(d):
    blk = np.load(f"{d}/blocks.npz", allow_pickle=True)
    st = {r["sheet_accession"]: r["resolved_accession"] for r in csv.DictReader(open(f"{d}/sequences_status.tsv"), delimiter="\t")}
    return blk, np.array([st.get(str(a), str(a)) for a in blk["accession"]])


pblk, pacc = resolved(PD)
gblk, gacc = resolved(GA)
FP = np.load(f"{PD}/folds.npz", allow_pickle=True)
FG = np.load(f"{GA}/folds_ga05.npz", allow_pickle=True)
T = {}
for c, nm in (("s75", "salt_75"), ("s150", "salt_150")):
    r = FP[f"rows_{c}"]
    T[nm] = (pacc[r], pblk[f"y_{c}"].astype(float)[r], FP[f"fold_{c}"])
for t in ("temp_avg_43v35", "temp_avg_37v35", "stau_avg_35", "stau_avg_37", "stau_avg_43"):
    r = FG[f"rows_{t}"]
    T[t] = (gacc[r], FG[f"y_{t}"].astype(float), FG[f"fold_{t}"])


def prep(M):
    X = np.log1p(np.asarray(M, np.float64))
    X -= X.mean(0)
    sd = X.std(0)
    keep = sd > 0
    X = X[:, keep] / sd[keep]
    return X, np.nonzero(keep)[0]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


ALPHAS = np.array([1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])


def kernel_ridge_cv(X, y, fid):
    """Ridge in dual form; every penalty from one eigendecomposition per fold. Returns best R2, alpha, OOF."""
    Xg = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    K = (Xg @ Xg.T).double()                       # products in fp32, eigensystem in fp64
    del Xg
    yg = torch.as_tensor(y, dtype=torch.float64, device=DEV)
    P = np.zeros((len(ALPHAS), len(y)))
    for f in np.unique(fid):
        tr = torch.as_tensor(fid != f, device=DEV); te = torch.as_tensor(fid == f, device=DEV)
        ym = yg[tr].mean()
        w, V = torch.linalg.eigh(K[tr][:, tr])
        Vy = V.T @ (yg[tr] - ym)
        Kte = K[te][:, tr] @ V
        for ai, al in enumerate(ALPHAS):
            P[ai, fid == f] = (Kte @ (Vy / (w + al)) + ym).cpu().numpy()
    sc = [r2(y, p) for p in P]
    b = int(np.argmax(sc))
    return sc[b], float(ALPHAS[b]), P[b]


def cluster_boot(y, p1, p0, G, nb=1000):
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    d = []
    for _ in range(nb):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); yy = y[s]
        d.append((((yy - p0[s]) ** 2).sum() - ((yy - p1[s]) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


cl_pd = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{PD}/clusters.tsv"), delimiter="\t")}
cl_ga = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{GA}/clusters.tsv"), delimiter="\t")}
KS = (1, 3, 10, 30, 100, 300, 1000)
out = {}
print(f"\nA. REPRESENTATION (percent of variance explained, held-out clusters)", flush=True)
print(f"{'target':<16}{'n':>6}{'dense L60':>11}{'SAE max':>9}{'SAE mean':>10}{'SAE freq':>10}{'SAE max - dense':>24}", flush=True)
for t, (accs, y, fid) in T.items():
    ix = np.array([spos[a] for a in accs])
    G = np.array([(cl_pd if t.startswith("salt") else cl_ga).get(a, a) for a in accs])
    Xd = dense60(accs); Xd -= Xd.mean(0); Xd /= Xd.std(0) + 1e-9
    rd, ad, pd_ = kernel_ridge_cv(Xd, y, fid)
    rec = dict(n=int(len(y)), dense=dict(r2=rd, alpha=ad))
    for pool in ("max", "mean", "freq"):
        X, kept = prep(POOL[pool][ix])
        rr, aa, pp = kernel_ridge_cv(X, y, fid)
        rec[pool] = dict(r2=rr, alpha=aa, n_features=int(len(kept)))
        if pool == "max":
            Xm, kept_m, pm = X, kept, pp
    rec["max_minus_dense_ci"] = cluster_boot(y, pm, pd_, G)
    print(f"{t:<16}{len(y):>6}{100*rd:>10.2f}%{100*rec['max']['r2']:>8.2f}%{100*rec['mean']['r2']:>9.2f}%{100*rec['freq']['r2']:>9.2f}%"
          f"{100*(rec['max']['r2']-rd):>+12.2f} [{100*rec['max_minus_dense_ci'][0]:+.2f},{100*rec['max_minus_dense_ci'][1]:+.2f}]", flush=True)
    # B. sparsity, on the max-pooled features, selection inside each training fold
    np.save(f"{SD}/pd13_oof_{t}.npy", np.stack([y, pd_, pm]))
    yr = np.zeros((len(KS), len(y)))
    sel_count = np.zeros(Xm.shape[1])
    for f in np.unique(fid):
        tr, te = fid != f, fid == f
        yc = y[tr] - y[tr].mean()
        Xt = Xm[tr] - Xm[tr].mean(0)
        cor = (Xt.T @ yc) / (np.sqrt((Xt ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
        order = np.argsort(-np.abs(cor))
        sel_count[order[:30]] += 1
        # penalty per k chosen on 15 % of the training fold's clusters, then refit on the whole fold
        ug = np.array(sorted(set(G[tr])))
        np.random.default_rng(100 + int(f)).shuffle(ug)
        vg = set(ug[: int(0.15 * len(ug))])
        iv = np.array([g in vg for g in G[tr]])
        for ki, k in enumerate(KS):
            S_ = order[:k]
            A = Xt[:, S_]
            bestv, bestal = -1e9, None
            for al in (1e-1, 1e1, 1e2, 1e3, 1e4):
                Ai, yi = A[~iv] - A[~iv].mean(0), yc[~iv] - yc[~iv].mean()
                w = np.linalg.solve(Ai.T @ Ai + al * np.eye(k), Ai.T @ yi)
                v = r2(yc[iv], (A[iv] - A[~iv].mean(0)) @ w + yc[~iv].mean())
                if v > bestv:
                    bestv, bestal = v, al
            w = np.linalg.solve(A.T @ A + bestal * np.eye(k), A.T @ yc)
            yr[ki, te] = (Xm[te][:, S_] - Xm[tr][:, S_].mean(0)) @ w + y[tr].mean()
    rec["sparse"] = {int(k): float(r2(y, yr[ki])) for ki, k in enumerate(KS)}
    rec["stable_top30"] = [int(kept_m[j]) for j in np.nonzero(sel_count == len(np.unique(fid)))[0]]
    # C. associations on all proteins
    yc = y - y.mean()
    cor = (Xm.T @ yc) / (np.sqrt((Xm ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    tt = cor * np.sqrt((len(y) - 2) / np.maximum(1 - cor ** 2, 1e-12))
    p = 2 * stats.t.sf(np.abs(tt), len(y) - 2)
    o = np.argsort(p)
    q = np.empty_like(p); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    act = POOL["max"][ix][:, kept_m].astype(np.float32) > 0
    top = np.argsort(-np.abs(cor))[:300]
    rec["n_fdr05"] = int((q < 0.05).sum())
    rec["top_features"] = [dict(feature=int(kept_m[j]), r=float(cor[j]), q=float(q[j]),
                                n_active=int(act[:, j].sum()),
                                mean_y_active=float(y[act[:, j]].mean()) if act[:, j].any() else None,
                                mean_y_inactive=float(y[~act[:, j]].mean()) if (~act[:, j]).any() else None)
                           for j in top]
    out[t] = rec
print(f"\nB. SPARSITY: R2 from the top-k features chosen inside each training fold (max-pooled)", flush=True)
print(f"{'target':<16}" + "".join(f"{'k='+str(k):>9}" for k in KS) + f"{'all':>9}{'stable in 5/5':>15}{'FDR<0.05':>10}", flush=True)
for t, rec in out.items():
    print(f"{t:<16}" + "".join(f"{100*rec['sparse'][k]:>8.2f}%" for k in KS) + f"{100*rec['max']['r2']:>8.2f}%"
          f"{len(rec['stable_top30']):>15}{rec['n_fdr05']:>10}", flush=True)
json.dump(out, open(f"{SD}/pd13_results.json", "w"), indent=1)
want = set()
for t, rec in out.items():
    want |= {d["feature"] for d in rec["top_features"][:100]}
    want |= set(rec["stable_top30"])
base = rng.choice(POOL["max"].shape[1], 500, replace=False)
open(f"{SD}/features_to_annotate.txt", "w").write("\n".join(str(f) for f in sorted(want)) + "\n")
open(f"{SD}/features_baseline_sample.txt", "w").write("\n".join(str(int(f)) for f in sorted(base)) + "\n")
print(f"\n{len(want)} features to annotate; 500 random baseline features", flush=True)
print("PD13_DONE", flush=True)
