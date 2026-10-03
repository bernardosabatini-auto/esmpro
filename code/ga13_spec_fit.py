"""ga13: what predicts which co-chaperone a protein prefers, and how that preference changes with heat?

Targets from ga12 (bait specificity DNAJB11 minus DNAJA1; its change from 35 to 43 C), on the
scikit-learn cluster folds written there. Ridge in dual form on the GPU (fp32 products, fp64
eigensystems), every penalty from one eigendecomposition per fold, for:
  composition   20 amino-acid fractions + log length
  ESMC L50, L80 mean-pooled embeddings (merged salt-pass + new proteins)
  SAE max, mean the layer-60 sparse-autoencoder features (pd12), log1p, standardised
Then, for the SAE features: correlations on all proteins, BH q, and the top 150 per target for
annotation and interpretation (ga14).
"""
import os, json
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D, SD = f"{ROOT}/data/ga_data", f"{ROOT}/data/sae"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
SP = np.load(f"{D}/spec.npz", allow_pickle=True)
acc = [str(a) for a in SP["accession"]]
S = np.load(f"{SD}/sae_l60.npz", allow_pickle=True)
spos = {str(a): i for i, a in enumerate(S["accession"])}
E50, E80 = np.load(f"{D}/emb_L50.npy"), np.load(f"{D}/emb_L80.npy")
seq, cur = {}, None
for ln in open(f"{D}/sequences.fasta"):
    if ln[0] == ">":
        cur = ln[1:].strip().split()[0].split("|")[0]; seq[cur] = []
    else:
        seq[cur].append(ln.strip())
AA = "ACDEFGHIKLMNPQRSTVWY"
TARGETS = ("spec_avg", "spec_35", "spec_43", "dheat_43")
ALPHAS = np.array([1e-1, 1, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])


def stdz(X):
    X = np.asarray(X, np.float64); X = X - X.mean(0); sd = X.std(0); k = sd > 0
    return X[:, k] / sd[k], np.nonzero(k)[0]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def kr(X, y, fid):
    Xg = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    K = (Xg @ Xg.T).double(); del Xg
    yg = torch.as_tensor(y, dtype=torch.float64, device=DEV)
    P = np.zeros((len(ALPHAS), len(y)))
    for f in np.unique(fid):
        tr = torch.as_tensor(fid != f, device=DEV); te = torch.as_tensor(fid == f, device=DEV)
        ym = yg[tr].mean()
        w, V = torch.linalg.eigh(K[tr][:, tr])
        Vy = V.T @ (yg[tr] - ym); Kte = K[te][:, tr] @ V
        for ai, al in enumerate(ALPHAS):
            P[ai, fid == f] = (Kte @ (Vy / (w + al)) + ym).cpu().numpy()
    sc = [r2(y, p) for p in P]; b = int(np.argmax(sc))
    return sc[b], P[b]


def cboot(y, p1, p0, G, nb=1000):
    ug, inv = np.unique(G, return_inverse=True); mem = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    d = []
    for _ in range(nb):
        s = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))]); yy = y[s]
        d.append((((yy - p0[s]) ** 2).sum() - ((yy - p1[s]) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


summ = json.load(open(f"{D}/ga12_summary.json"))
out = {}
print(f"{'target':<10}{'ceiling':>9}{'composition':>13}{'ESMC L50':>10}{'ESMC L80':>10}{'SAE max':>9}{'SAE mean':>10}   SAE mean - L50", flush=True)
for t in TARGETS:
    rows, fid, y = SP[f"rows_{t}"], SP[f"fold_{t}"], SP[f"y_{t}"].astype(np.float64)
    a_ = [acc[r] for r in rows]; G = SP["groups"][rows]
    comp = np.array([[s.count(c) / max(len(s), 1) for c in AA] + [np.log10(max(len(s), 1))] for s in ("".join(seq.get(x, [])) for x in a_)])
    ix = np.array([spos[x] for x in a_])
    res, preds = {}, {}
    for nm, X in (("composition", comp), ("ESMC L50", E50[rows]), ("ESMC L80", E80[rows]),
                  ("SAE max", np.log1p(S["max"][ix].astype(np.float64))), ("SAE mean", np.log1p(S["mean"][ix].astype(np.float64)))):
        Xs, _ = stdz(X)
        res[nm], preds[nm] = kr(Xs, y, fid)
    ci = cboot(y, preds["SAE mean"], preds["ESMC L50"], G)
    print(f"{t:<10}{100*summ[t]['reliability']:>8.1f}%" + "".join(f"{100*res[k]:>{w}.1f}%" for k, w in
          (("composition", 12), ("ESMC L50", 9), ("ESMC L80", 9), ("SAE max", 8), ("SAE mean", 9)))
          + f"   {100*(res['SAE mean']-res['ESMC L50']):+.1f} [{100*ci[0]:+.1f},{100*ci[1]:+.1f}]", flush=True)
    # SAE associations (max-pooled, as in pd13/pd15)
    Xm, kept = stdz(np.log1p(S["max"][ix].astype(np.float64)))
    yc = y - y.mean()
    cor = (Xm.T @ yc) / (np.sqrt((Xm ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    from scipy import stats
    tt = cor * np.sqrt((len(y) - 2) / np.maximum(1 - cor ** 2, 1e-12)); p = 2 * stats.t.sf(np.abs(tt), len(y) - 2)
    o = np.argsort(p); q = np.empty_like(p); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    top = np.argsort(-np.abs(cor))[:150]
    out[t] = dict(ceiling=summ[t]["reliability"], r2=res, sae_mean_minus_l50_ci=ci, n=int(len(y)), n_fdr05=int((q < 0.05).sum()),
                  top_features=[dict(feature=int(kept[j]), r=float(cor[j]), q=float(q[j])) for j in top])
    np.save(f"{D}/ga13_oof_{t}.npy", np.stack([y] + [preds[k] for k in ("composition", "ESMC L50", "ESMC L80", "SAE max", "SAE mean")]))
json.dump(out, open(f"{D}/ga13_results.json", "w"), indent=1)
want = sorted({d["feature"] for v in out.values() for d in v["top_features"]})
open(f"{SD}/features_spec.txt", "w").write("\n".join(map(str, want)) + "\n")
print(f"\n{len(want)} features to annotate\nGA13_DONE", flush=True)
