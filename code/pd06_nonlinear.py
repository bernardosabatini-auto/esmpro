"""pd06: nonlinear and joint models for the salt-dependent fold change.

Stage 5 (pd05) established the linear picture: on cluster-grouped folds, baseline abundance
alone reaches r ~ 0.40-0.42, the ESMC mean-pool reaches r ~ 0.40-0.42, and the two together
reach ~0.56, all against a measurement ceiling of 0.993-0.996. Ridge beat k-NN in embedding
space by ~0.11, so the signal is not pure family lookup. Two questions remain:

  1. Is the embedding -> fold-change map nonlinear? Gradient-boosted trees and an MLP against
     the same ridge on the same rows answers that directly.
  2. Does fitting the two doses jointly help? y_75 and y_150 correlate at r = 0.913, so they
     are two noisy reads of one underlying salt sensitivity. Two ways to use that:
       - a shared-trunk multi-output model, and
       - a reparameterisation into s = (y_75 + y_150)/2, the dose-averaged sensitivity, and
         d = y_150 - y_75, the dose shape. s has less noise than either target, so if the
         limit is measurement noise rather than learnability, s is easier to predict and
         reconstructing y_75 = s - d/2 beats fitting y_75 on its own.

Every model here runs on the SAME row subset (proteins passing the 7-of-10 rule in both
comparisons) and the SAME cluster-grouped folds, so differences are about the model.

The XIC column is excluded throughout: it is a total across all runs, so with the baseline it
reconstructs the condition level and leaks the target. See the note in pd05_fit.py.
"""
import os, csv, json, argparse, time, copy
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/pd_data"
ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--pca", type=int, default=256)
a = ap.parse_args()
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- data
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
emb = np.load(f"{D}/embeddings.npz", allow_pickle=True)
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
pos = {str(s): i for i, s in enumerate(emb["accession"])}
ei = np.array([pos[resolved.get(s, s)] for s in acc])
MEAN = emb["mean_pool"][ei]
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])

with np.errstate(invalid="ignore"):
    base = np.nanmean(blk["br_base"], 1)
npep = np.log10(np.maximum(blk["npeptides"], 1))


def clean(M):
    M = np.asarray(M, float).copy()
    for j in range(M.shape[1]):
        bad = ~np.isfinite(M[:, j])
        if bad.any():
            M[bad, j] = np.nanmedian(M[~bad, j]) if (~bad).any() else 0.0
    return M


ABUND = clean(np.column_stack([base, npep]))
y75, y150 = blk["y_s75"].astype(float), blk["y_s150"].astype(float)
keep = (blk["keep_s75"].astype(bool) & blk["keep_s150"].astype(bool)
        & np.isfinite(y75) & np.isfinite(y150))
M, A, G = MEAN[keep], ABUND[keep], groups[keep]
Y = np.column_stack([y75[keep], y150[keep]])
S = Y.mean(1)                 # dose-averaged salt sensitivity (the low-noise target)
Dd = Y[:, 1] - Y[:, 0]        # dose shape, 150 mM minus 75 mM
n = keep.sum()
print(f"{n} proteins pass the 7-of-10 rule in BOTH comparisons | "
      f"{len(set(G))} clusters | corr(y75,y150) = {stats.pearsonr(Y[:,0],Y[:,1])[0]:+.3f}", flush=True)
print(f"sd  y75 {Y[:,0].std():.3f}   y150 {Y[:,1].std():.3f}   mean S {S.std():.3f}   "
      f"shape D {Dd.std():.3f}", flush=True)

ceil = {}
try:
    p5 = json.load(open(f"{D}/pd05_results.json"))
    ceil = {c: p5[c]["ceiling"]["max_r"] for c in ("s75", "s150")}
    print(f"ceilings from pd05: 75 mM {ceil['s75']:.3f}   150 mM {ceil['s150']:.3f}", flush=True)
except Exception as e:
    print(f"(no pd05 ceilings: {e})", flush=True)

folds = list(GroupKFold(n_splits=a.folds).split(M, S, G))
for tr, te in folds:
    assert not (set(G[tr]) & set(G[te])), "cluster appears in both train and test"


def boot_ci(y, p, nb):
    i = np.arange(len(y))
    bs = [stats.pearsonr(y[s], p[s])[0] for s in (rng.choice(i, len(i), replace=True) for _ in range(nb))]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def report(name, y, p, extra=""):
    r = stats.pearsonr(y, p)[0]
    rho = stats.spearmanr(y, p)[0]
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    lo, hi = boot_ci(y, p, a.boot)
    print(f"    {name:34s} r = {r:+.3f} [{lo:+.3f},{hi:+.3f}]  rho {rho:+.3f}  R2 {r2:+.3f} {extra}",
          flush=True)
    return dict(r=float(r), lo=lo, hi=hi, rho=float(rho), r2=float(r2))


# ---------------------------------------------------------------- estimators
def oof(make, X, y):
    """Out-of-fold predictions. `make` returns a fresh fitted-from-scratch estimator."""
    p = np.zeros((len(y),) + y.shape[1:])
    for tr, te in folds:
        m = make()
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def ridge(alpha, pca=None):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, svd_solver="randomized", random_state=0))
    return lambda: make_pipeline(*steps, Ridge(alpha=alpha))


def ridge_best(X, y, alphas=(10, 100, 1e3, 1e4, 1e5)):
    out = (-9, None, None)
    for al in alphas:
        p = oof(ridge(al), X, y)
        r = stats.pearsonr(y, p)[0] if y.ndim == 1 else np.mean(
            [stats.pearsonr(y[:, j], p[:, j])[0] for j in range(y.shape[1])])
        if r > out[0]:
            out = (r, al, p)
    return out


def gb(lr=0.06, leaves=31, pca=None):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, svd_solver="randomized", random_state=0))
    return lambda: make_pipeline(*steps, HistGradientBoostingRegressor(
        max_iter=500, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=0))


class EarlyMLP:
    """Torch MLP early-stopped on an inner validation split carved out BY CLUSTER.

    Two deliberate choices. First, the inner split holds out whole sequence clusters: a random
    inner split would put near-identical paralogs on both sides of it and choose a stopping
    point that does not transfer to the outer, cluster-grouped test fold. Second, it runs on the
    GPU -- the whole design is 7k x 2.6k floats (72 MB), so every tensor lives on the card and
    an epoch is a few milliseconds. The job has to hold a GPU anyway (the only account that can
    submit here requires one), so the MLP may as well use it; gradient boosting stays on the
    16 CPU cores alongside.
    """

    def __init__(self, hidden=(512, 128), wd=1e-3, lr=1e-3, drop=0.1,
                 max_epochs=400, patience=30, batch=128, seed=0):
        self.h, self.wd, self.lr, self.drop = hidden, wd, lr, drop
        self.max_epochs, self.patience, self.batch, self.seed = max_epochs, patience, batch, seed

    def fit(self, X, y, gr):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        y2 = y[:, None] if y.ndim == 1 else y
        self.sc = StandardScaler().fit(X)
        self.ym, self.ys = y2.mean(0), y2.std(0) + 1e-9
        Xz = torch.as_tensor(self.sc.transform(X), dtype=torch.float32, device=dev)
        yz = torch.as_tensor((y2 - self.ym) / self.ys, dtype=torch.float32, device=dev)

        g = np.asarray(gr)
        ug = np.array(sorted(set(g)))
        np.random.default_rng(self.seed).shuffle(ug)
        vg = set(ug[: max(1, int(0.15 * len(ug)))])
        v = torch.as_tensor(np.array([x in vg for x in g]), device=dev)
        Xt, yt, Xv, yv = Xz[~v], yz[~v], Xz[v], yz[v]

        layers, d = [], X.shape[1]
        for hdim in self.h:
            layers += [torch.nn.Linear(d, hdim), torch.nn.ReLU(), torch.nn.Dropout(self.drop)]
            d = hdim
        layers.append(torch.nn.Linear(d, yz.shape[1]))
        net = torch.nn.Sequential(*layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, self.best_ep, bad, bw, self.hist = -9.0, 0, 0, None, []
        nt = Xt.shape[0]
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.randperm(nt, device=dev)
            for i in range(0, nt, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xv)
                # Pearson per output, averaged -- the same quantity the outer score reports
                a_, b_ = pv - pv.mean(0), yv - yv.mean(0)
                r = float((( a_ * b_).sum(0) / (a_.norm(dim=0) * b_.norm(dim=0) + 1e-9)).mean())
            self.hist.append(r)
            if r > best + 1e-4:
                best, self.best_ep, bad = r, ep, 0
                bw = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(bw)
        net.eval()
        self.net, self.dev, self.best_r, self.flat = net, dev, best, (y.ndim == 1)
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            p = self.net(torch.as_tensor(self.sc.transform(X), dtype=torch.float32,
                                         device=self.dev)).cpu().numpy()
        p = p * self.ys + self.ym
        return p[:, 0] if self.flat else p


def oof_mlp(X, y, **kw):
    p, eps = np.zeros((len(y),) + y.shape[1:]), []
    for tr, te in folds:
        m = EarlyMLP(**kw).fit(X[tr], y[tr], G[tr])
        p[te] = m.predict(X[te])
        eps.append(m.best_ep)
    return p, eps


# ---------------------------------------------------------------- run
out = {"n": int(n), "clusters": len(set(G)), "ceilings": ceil,
       "r_y75_y150": float(stats.pearsonr(Y[:, 0], Y[:, 1])[0]),
       "sd": dict(y75=float(Y[:, 0].std()), y150=float(Y[:, 1].std()),
                  mean=float(S.std()), shape=float(Dd.std()))}
FS = {"ESMC": M, "ESMC+abundance": np.hstack([M, A]), "abundance": A}
t0 = time.time()

for ti, (tname, yt) in enumerate((("75 mM", Y[:, 0]), ("150 mM", Y[:, 1]),
                                  ("dose-averaged mean", S))):
    print(f"\n{'='*86}\n{tname}   (sd {yt.std():.3f})", flush=True)
    res = {}
    for fname, X in FS.items():
        print(f"  features: {fname}  ({X.shape[1]}d)", flush=True)
        r, al, p = ridge_best(X, yt)
        res[f"ridge | {fname}"] = report("ridge (linear reference)", yt, p, f"alpha {al:g}")
        p = oof(gb(pca=a.pca if X.shape[1] > 64 else None), X, yt)
        res[f"gb | {fname}"] = report(f"gradient boosting", yt, p)
        p, eps = oof_mlp(X, yt)
        res[f"mlp | {fname}"] = report("MLP 512-128", yt, p, f"epochs {eps}")
    out[tname] = res

# joint: multi-output, and the dose reparameterisation
print(f"\n{'='*86}\nJOINT MODELS — does fitting the two doses together help?", flush=True)
X = np.hstack([M, A])
jr = {}
r, al, pj = ridge_best(X, Y)
for j, nm in enumerate(("75 mM", "150 mM")):
    jr[f"multi-output ridge | {nm}"] = report(f"multi-output ridge -> {nm}", Y[:, j], pj[:, j], f"alpha {al:g}")
pj, eps = oof_mlp(X, Y)
for j, nm in enumerate(("75 mM", "150 mM")):
    jr[f"multi-output MLP | {nm}"] = report(f"multi-output MLP -> {nm}", Y[:, j], pj[:, j], f"epochs {eps}")

_, als, ps = ridge_best(X, S)
_, ald, pd_ = ridge_best(X, Dd)
rec = {"75 mM": ps - pd_ / 2, "150 mM": ps + pd_ / 2}
print(f"    (mean target r {stats.pearsonr(S,ps)[0]:+.3f} alpha {als:g};  "
      f"shape target r {stats.pearsonr(Dd,pd_)[0]:+.3f} alpha {ald:g})", flush=True)
jr["dose-shape | mean"] = report("mean+shape -> reconstructed 75", Y[:, 0], rec["75 mM"])
jr["dose-shape | 150"] = report("mean+shape -> reconstructed 150", Y[:, 1], rec["150 mM"])
out["joint"] = jr

# shuffled-label control through the full nonlinear path
sh = rng.permutation(n)
p = oof(gb(pca=a.pca), np.hstack([M, A]), S[sh])
out["control_shuffled"] = report("shuffled labels (GB, must be ~0)", S[sh], p)

json.dump(out, open(f"{D}/pd06_results.json", "w"), indent=1)
print(f"\n{(time.time()-t0)/60:.1f} min\nwrote {D}/pd06_results.json\nPD06_DONE", flush=True)
