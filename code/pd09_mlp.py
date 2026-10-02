"""pd09: make an MLP readout beat ridge, and replace the replicate mean with a per-sample model.

Ridge on the ESMC embedding (layer 50 of 81) explains 16.0 % / 18.2 % of the variance in the log2
fold change at 75 / 150 mM. One MLP tried in pd06 lost to it, early-stopping at epoch 2-5 in four
of five folds -- it overfit almost at once. Two levers here.

1. Regularise the readout properly. Every configuration is a 16-member ensemble trained in one
   vectorised pass (stacked weights, batched matmul, a separate minibatch order per member), so
   the ensemble costs about what one model does. Swept: PCA-whitened vs full standardised input,
   width/depth, dropout, weight decay, and a RESIDUAL-ON-RIDGE form whose last layer starts at
   zero, so the model begins exactly at the linear solution and keeps only what the nonlinear term
   earns on held-out clusters. The ridge offset seen in training is cross-fitted (out-of-fold
   within the training set); an in-sample ridge would hand the MLP shrunken residuals.

2. Use the individual samples instead of the mean. Each block is 10 runs (2 technical groups x 5
   biological replicates). A Gaussian model of every run, with a per-protein level profiled out,
   reduces exactly to weighted regression on a precision-weighted fold change y_hat with a known
   per-protein measurement variance v_p -- so the likelihood of all 20 samples is implemented
   without blowing the data up 20-fold. v_p comes from a variance model fitted on TRAINING
   proteins only: a per-run noise level (one 75 mM run is 1.7x noisier than the best) times a
   smooth dependence on abundance (replicate scatter is 4x larger in the lowest quintile). The
   fold change is formed within each technical group and then combined, because the technical
   groups carry a weak protein-specific batch offset that is partly shared across conditions and
   would otherwise stop cancelling once runs are weighted unequally.

   Three ways to let that variance discount samples:
     naive  weight 1 / v_p                    measurement noise only
     rep    NLL with variance tau^2 + v_p     tau^2 = learned model error; the correct likelihood
     het    beta-NLL, variance s(x)^2 + v_p   the network predicts its own per-protein uncertainty
   Model error is ~85 % of the variance and measurement noise ~1 %, so under the correct
   likelihood the weights come out nearly uniform; 'naive' is included to show what ignoring
   model error does.

Selection is nested: an inner split that holds out 15 % of each outer training fold's CLUSTERS
drives early stopping and chooses the configuration; the outer cluster-grouped test folds
(identical to pd05/pd07b) are touched once. Every model is scored on the same standard target.
"""
import os, csv, json, time, argparse, itertools, math
import numpy as np, torch
import torch.nn.functional as F
# No scikit-learn: this runs in the env whose torch has Blackwell (sm_120) kernels. The OUTER
# folds are scikit-learn's GroupKFold, written to disk by pd_folds.py, so they are identical to
# the folds every ridge number in pd05/pd07b was scored on. The inner cross-fit below only needs
# to be grouped, not to match anything, and uses the same greedy rule locally.
FOLDS = np.load(f"{os.environ.get('ESM_PROAE_ROOT', '/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae')}"
                f"/data/pd_data/folds.npz")


def group_kfold(G, n=5):
    """Largest groups first, each to the currently lightest fold (GroupKFold's rule)."""
    ug, inv = np.unique(G, return_inverse=True)
    cnt = np.bincount(inv)
    order = np.argsort(-cnt, kind="stable")
    load, g2f = np.zeros(n), np.zeros(len(ug), int)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    fid = g2f[inv]
    return [(np.nonzero(fid != f)[0], np.nonzero(fid == f)[0]) for f in range(n)]

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae")
D = f"{ROOT}/data/pd_data"
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=50)
ap.add_argument("--members", type=int, default=16)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--top", type=int, default=3, help="configs carried into the loss comparison")
ap.add_argument("--quick", action="store_true", help="smoke test: 2 folds, 2 configs, few epochs")
ap.add_argument("--out", default=f"{D}/pd09_results.json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
rng = np.random.default_rng(0)
T0 = time.time()

# ------------------------------------------------------------------ data
blk = np.load(f"{D}/blocks.npz", allow_pickle=True)
L = np.load(f"{D}/embeddings_layers.npz", allow_pickle=True)
LAY = [int(x) for x in L["layers"]]
acc = np.array([str(x) for x in blk["accession"]])
resolved = {r["sheet_accession"]: r["resolved_accession"]
            for r in csv.DictReader(open(f"{D}/sequences_status.tsv"), delimiter="\t")}
fa = [ln[1:].strip().split()[0].split("|")[0] for ln in open(f"{D}/sequences.fasta") if ln[0] == ">"]
pos = {s: i for i, s in enumerate(fa)}
EMB = L["pool"][np.array([pos[resolved.get(s, s)] for s in acc]), LAY.index(a.layer)].astype(np.float32)
cl = {r["accession"]: r["cluster"] for r in csv.DictReader(open(f"{D}/clusters.tsv"), delimiter="\t")}
groups = np.array([cl.get(resolved.get(s, s), s) for s in acc])
with np.errstate(invalid="ignore"):
    ABUND = np.nanmean(blk["raw_base"], 1)            # used ONLY inside the noise model
print(f"{len(acc)} proteins | ESMC layer {a.layer} | {EMB.shape[1]}d | device {dev} | "
      f"{a.members}-member ensembles", flush=True)


# ------------------------------------------------------------------ measurement model
def noise_model(tr, cond):
    """Per-run log-variance plus a quadratic in abundance, fitted on training proteins only.
    Returns per-protein precision-weighted fold change y_hat and its variance v (training rows)."""
    Rb, Rc = blk["raw_base"][tr], blk[f"raw_{cond}"][tr]
    ab = ABUND[tr]
    am, asd = np.nanmean(ab), np.nanstd(ab)
    z = (ab - am) / asd
    rows, ys = [], []
    for bi, R in enumerate((Rb, Rc)):
        for j in range(10):
            oth = np.delete(R, j, 1)
            with np.errstate(invalid="ignore"):
                r = R[:, j] - np.nanmean(oth, 1)
            ok = np.isfinite(r) & np.isfinite(z)
            X = np.zeros((ok.sum(), 22))
            X[:, bi * 10 + j] = 1
            X[:, 20], X[:, 21] = z[ok], z[ok] ** 2
            rows.append(X); ys.append(np.log(r[ok] ** 2 + 1e-8))
    X, ylog = np.vstack(rows), np.concatenate(ys)
    beta = np.linalg.lstsq(X, ylog, rcond=None)[0]
    # E[log chi2_1] = -1.27; a leave-one-out residual over ~9 others is inflated by ~10/9
    def logv(bi, j, zz):
        return beta[bi * 10 + j] + beta[20] * zz + beta[21] * zz ** 2 + 1.27 - math.log(10 / 9)
    zz = np.where(np.isfinite(z), z, 0.0)
    est, var = [], []
    for t in (slice(0, 5), slice(5, 10)):
        js = range(t.start, t.stop)
        Wb = np.stack([np.exp(-logv(0, j, zz)) for j in js], 1) * np.isfinite(Rb[:, t])
        Wc = np.stack([np.exp(-logv(1, j, zz)) for j in js], 1) * np.isfinite(Rc[:, t])
        sb, sc = Wb.sum(1), Wc.sum(1)
        mb = np.where(sb > 0, np.nansum(np.nan_to_num(Rb[:, t]) * Wb, 1) / np.maximum(sb, 1e-12), np.nan)
        mc = np.where(sc > 0, np.nansum(np.nan_to_num(Rc[:, t]) * Wc, 1) / np.maximum(sc, 1e-12), np.nan)
        est.append(mc - mb)
        var.append(np.where((sb > 0) & (sc > 0), 1 / np.maximum(sb, 1e-12) + 1 / np.maximum(sc, 1e-12), np.inf))
    est, var = np.stack(est, 1), np.stack(var, 1)
    w = np.where(np.isfinite(est) & np.isfinite(var), 1 / var, 0.0)
    yhat = np.nansum(np.nan_to_num(est) * w, 1) / np.maximum(w.sum(1), 1e-12)
    v = 1 / np.maximum(w.sum(1), 1e-12)
    run_sd = np.exp(0.5 * (beta[:20] + 1.27 - math.log(10 / 9)))
    return yhat, v, dict(run_sd=run_sd.tolist(), abund_beta=beta[20:].tolist())


# ------------------------------------------------------------------ ridge (GPU, closed form)
def ridge_fit(X, y, alpha):
    Xd, yd = X.double(), y.double()
    mu = yd.mean()
    A = Xd.T @ Xd + alpha * torch.eye(Xd.shape[1], device=dev, dtype=torch.float64)
    w = torch.linalg.solve(A, Xd.T @ (yd - mu))
    return w.float(), mu.float()


def ridge_pred(X, wm):
    return X @ wm[0] + wm[1]


ALPHAS = (300., 1e3, 3e3, 1e4, 3e4, 1e5)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ------------------------------------------------------------------ vectorised ensemble MLP
class Ens(torch.nn.Module):
    def __init__(s, E, din, hidden, dout, drop, zero_last):
        super().__init__()
        dims = [din, *hidden, dout]
        s.W, s.b = torch.nn.ParameterList(), torch.nn.ParameterList()
        for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
            bd = 1 / math.sqrt(m)
            w = torch.empty(E, m, n).uniform_(-bd, bd) * (0 if (zero_last and i == len(dims) - 2) else 1)
            s.W.append(torch.nn.Parameter(w))
            s.b.append(torch.nn.Parameter(torch.zeros(E, 1, n)))
        s.drop, s.pin = drop, drop / 3

    def forward(s, x):               # x (E,B,d) in training, (N,d) broadcast to all members in eval
        if s.training and s.pin > 0:
            x = F.dropout(x, s.pin)
        for i, (w, b) in enumerate(zip(s.W, s.b)):
            x = torch.matmul(x, w) + b
            if i < len(s.W) - 1:
                x = F.dropout(F.gelu(x), s.drop, s.training)
        return x                      # (E,B,dout)


def train_mlp(cfg, loss, Xtr, ytr, otr, vtr, Xva, yva, ova, Xte, ote):
    """Returns ensemble-mean predictions on val and test (original target units handled outside)."""
    E = a.members
    het = loss == "het"
    multi = ytr.dim() == 2
    k = ytr.shape[1] if multi else 1
    net = Ens(E, Xtr.shape[1], cfg["hidden"], (k * 2 if het else k), cfg["drop"], cfg["resid"]).to(dev)
    logtau = torch.nn.Parameter(torch.zeros(k, device=dev))
    # the learned model-error variance is a likelihood parameter, not a weight: no decay on it
    opt = torch.optim.AdamW([{"params": list(net.parameters()), "weight_decay": cfg["wd"]},
                             {"params": [logtau], "weight_decay": 0.0}], lr=a.lr)
    yt = ytr if multi else ytr[:, None]
    ot = otr if multi else otr[:, None]
    vt = vtr if multi else vtr[:, None]
    yv = yva if multi else yva[:, None]
    ov = ova if multi else ova[:, None]
    n = yt.shape[0]
    wnaive = (1 / vt) / (1 / vt).mean(0, keepdim=True)
    best = torch.full((E,), -1e9, device=dev)
    bad = torch.zeros(E, device=dev)
    state = [p.detach().clone() for p in net.parameters()]
    ep_best = torch.zeros(E, device=dev)
    for ep in range(a.epochs):
        net.train()
        perm = torch.argsort(torch.rand(E, n, device=dev), 1)
        for i in range(0, n, a.batch):
            idx = perm[:, i:i + a.batch]
            out = net(Xtr[idx])
            mu = out[..., :k] + ot[idx]
            yb = yt[idx]
            if loss == "mse":
                l = ((mu - yb) ** 2).mean()
            elif loss == "naive":
                l = (wnaive[idx] * (mu - yb) ** 2).mean()
            elif loss == "rep":
                var = torch.exp(logtau) + vt[idx]
                l = (0.5 * (mu - yb) ** 2 / var + 0.5 * torch.log(var)).mean()
            else:  # het, beta-NLL with beta = 0.5
                var = F.softplus(out[..., k:]) + 1e-3 + vt[idx]
                l = ((0.5 * (mu - yb) ** 2 / var + 0.5 * torch.log(var)) * var.detach() ** 0.5).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xva)[..., :k] + ov                                      # (E,nv,k)
            sse = ((pv - yv) ** 2).sum((1, 2))
            sst = ((yv - yv.mean(0)) ** 2).sum()
            sc = 1 - sse / sst
            imp = sc > best + 1e-5
            best = torch.where(imp, sc, best)
            ep_best = torch.where(imp, torch.full_like(ep_best, ep), ep_best)
            bad = torch.where(imp, torch.zeros_like(bad), bad + 1)
            for sp, p in zip(state, net.parameters()):
                m = imp.view(-1, *([1] * (p.dim() - 1)))
                sp.copy_(torch.where(m, p.detach(), sp))
        if bool((bad >= a.patience).all()):
            break
    with torch.no_grad():
        for sp, p in zip(state, net.parameters()):
            p.copy_(sp)
        net.eval()
        pva = (net(Xva)[..., :k] + ov).mean(0)
        pte = (net(Xte)[..., :k] + (ote if multi else ote[:, None])).mean(0)
    out = dict(epochs=float(ep_best.mean()), member_val=float(best.mean()), tau2=torch.exp(logtau).tolist())
    return (pva if multi else pva[:, 0]), (pte if multi else pte[:, 0]), out


# ------------------------------------------------------------------ configurations
GRID = [dict(inp=i, hidden=h, drop=d, wd=w, resid=r)
        for i, h, d, w, r in itertools.product(("pca256", "full"), ((256,), (1024,), (512, 128)),
                                               (0.3, 0.6), (1e-2, 1e-1), (False, True))]
if a.quick:
    GRID = GRID[:1] + GRID[-1:]


def cname(c):
    return (f"{c['inp']:<6} h{'-'.join(map(str, c['hidden'])):<8} drop {c['drop']:.1f} wd {c['wd']:.0e}"
            f"{'  +ridge' if c['resid'] else ''}")


# ------------------------------------------------------------------ per-fold preparation
def prepare(k, Y, targets_cond):
    """Everything that depends on the outer fold but not on the MLP configuration."""
    G = groups[k]
    cond_ = targets_cond[0]
    assert (FOLDS[f"rows_{cond_}"] == np.nonzero(k)[0]).all(), "fold file is for different rows"
    fid = FOLDS[f"fold_{cond_}"]
    folds = [(np.nonzero(fid != f)[0], np.nonzero(fid == f)[0]) for f in range(5)]
    if a.quick:
        folds = folds[:2]
    Xall = torch.as_tensor(EMB[k], device=dev)
    prep = []
    for fi, (tr, te) in enumerate(folds):
        assert not set(G[tr]) & set(G[te])
        ug = np.array(sorted(set(G[tr])))
        np.random.default_rng(100 + fi).shuffle(ug)
        vg = set(ug[: int(0.15 * len(ug))])
        isv = np.array([g in vg for g in G[tr]])
        itr, iva = tr[~isv], tr[isv]
        Xo = Xall[tr]
        mu, sd = Xo.mean(0), Xo.std(0) + 1e-6
        Z = (Xall - mu) / sd
        U, S, V = torch.linalg.svd(Z[tr], full_matrices=False)
        P = (Z @ V[:256].T) / (S[:256] / math.sqrt(len(tr) - 1))           # whitened PCs
        feats = {"full": Z, "pca256": P}
        # targets, standardised on the inner-training rows
        yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)
        ym, ys = yt[itr].mean(0), yt[itr].std(0)
        # measurement model on training proteins only; fold change and its variance
        yh, vv = [], []
        nm = None
        for j, cond in enumerate(targets_cond):
            yhat, v, nm = noise_model(k.nonzero()[0][tr], cond)
            yh.append(yhat); vv.append(v)
        yhat = torch.as_tensor(np.stack(yh, 1), device=dev, dtype=torch.float32)
        vtr = torch.as_tensor(np.stack(vv, 1), device=dev, dtype=torch.float32)
        if Y.ndim == 1:
            yhat, vtr = yhat[:, 0], vtr[:, 0]
        # map: tr-ordered arrays -> inner-train rows
        pos_tr = {r: i for i, r in enumerate(tr)}
        sel = torch.as_tensor([pos_tr[r] for r in itr], device=dev)
        # ridge: alpha by inner val (fit on inner-train), then refit on all of outer-train
        bestal, bestsc, rv = None, -1e9, None
        for al in ALPHAS:
            wm = ridge_fit(Z[itr], (yt[itr] - ym) / ys if Y.ndim == 1 else ((yt[itr] - ym) / ys)[:, 0], al) \
                if Y.ndim == 1 else None
            if Y.ndim == 1:
                pv = ridge_pred(Z[iva], wm) * ys + ym
                sc = r2(yt[iva], pv)
                if sc > bestsc:
                    bestal, bestsc, rv = al, sc, pv
        prep.append(dict(tr=tr, te=te, itr=itr, iva=iva, feats=feats, ym=ym, ys=ys, yt=yt,
                         yhat=yhat[sel], v=vtr[sel] / (ys ** 2), alpha=bestal, ridge_val=rv,
                         noise=nm, G=G))
    return prep


def ridge_offsets(p, X, multi_cols=None):
    """Cross-fitted ridge offsets in standardised units: OOF inside outer-train, full fit for test."""
    tr, te, itr, iva, ym, ys, yt = p["tr"], p["te"], p["itr"], p["iva"], p["ym"], p["ys"], p["yt"]
    ycols = [None] if multi_cols is None else list(range(multi_cols))
    offs_tr, offs_va, offs_te = [], [], []
    for c in ycols:
        yy = yt if c is None else yt[:, c]
        mm = ym if c is None else ym[c]
        ss = ys if c is None else ys[c]
        al = p["alpha"] if c is None else p["alpha_mt"][c]
        o = torch.zeros(len(yy), device=dev)
        Gt = p["G"][tr]
        for a_, b_ in group_kfold(Gt, 5):
            wm = ridge_fit(X[tr[a_]], (yy[tr[a_]] - mm) / ss, al)
            o[tr[b_]] = ridge_pred(X[tr[b_]], wm)
        wm = ridge_fit(X[tr], (yy[tr] - mm) / ss, al)
        o[te] = ridge_pred(X[te], wm)
        offs_tr.append(o[itr]); offs_va.append(o[iva]); offs_te.append(o[te])
    if multi_cols is None:
        return offs_tr[0], offs_va[0], offs_te[0]
    return torch.stack(offs_tr, 1), torch.stack(offs_va, 1), torch.stack(offs_te, 1)


def run_cfg(prep, Y, cfg, loss, multi=False):
    n = len(Y)
    oof = np.full(Y.shape, np.nan)
    vals, info = [], []
    for p in prep:
        X = p["feats"][cfg["inp"]]
        Xs = p["feats"]["full"]
        ym, ys, yt = p["ym"], p["ys"], p["yt"]
        ztr = (yt[p["itr"]] - ym) / ys
        zva = (yt[p["iva"]] - ym) / ys
        if loss in ("naive", "rep", "het"):
            ztr = (p["yhat"] - ym) / ys
        if cfg["resid"]:
            if "roffs" not in p:
                p["roffs"] = ridge_offsets(p, Xs)
            otr, ova, ote = p["roffs"]
        else:
            sh = (len(p["itr"]), Y.shape[1]) if multi else (len(p["itr"]),)
            otr = torch.zeros(sh, device=dev)
            ova = torch.zeros(((len(p["iva"]), Y.shape[1]) if multi else (len(p["iva"]),)), device=dev)
            ote = torch.zeros(((len(p["te"]), Y.shape[1]) if multi else (len(p["te"]),)), device=dev)
        pva, pte, inf = train_mlp(cfg, loss, X[p["itr"]], ztr, otr, p["v"], X[p["iva"]], zva, ova,
                                  X[p["te"]], ote)
        pva = pva * ys + ym
        pte = pte * ys + ym
        vals.append(r2(yt[p["iva"]], pva))
        oof[p["te"]] = pte.cpu().numpy()
        info.append(inf)
    return oof, float(np.mean(vals)), info


def boot_diff(y, p1, p0, nb):
    i = np.arange(len(y))
    d = []
    for _ in range(nb):
        s = rng.choice(i, len(i), replace=True)
        yy = y[s]
        sst = ((yy - yy.mean()) ** 2).sum()
        d.append((((yy - p0[s]) ** 2).sum() - ((yy - p1[s]) ** 2).sum()) / sst)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ridge_oof(prep, Y):
    oof = np.full(len(Y), np.nan)
    for p in prep:
        X, ym, ys, yt = p["feats"]["full"], p["ym"], p["ys"], p["yt"]
        wm = ridge_fit(X[p["tr"]], (yt[p["tr"]] - ym) / ys, p["alpha"])
        oof[p["te"]] = (ridge_pred(X[p["te"]], wm) * ys + ym).cpu().numpy()
    return oof


# ------------------------------------------------------------------ main
results = {"layer": a.layer, "members": a.members, "grid": [cname(c) for c in GRID]}
for cond, label in (("s75", "75 mM"), ("s150", "150 mM")):
    y = blk[f"y_{cond}"].astype(float)
    k = blk[f"keep_{cond}"].astype(bool) & np.isfinite(y)
    Y = y[k]
    t0 = time.time()
    prep = prepare(k, Y, [cond])
    mask = np.zeros(len(Y), bool)
    for p in prep:
        mask[p["te"]] = True
    Ym = Y[mask]
    rid = ridge_oof(prep, Y)
    r2_ridge = r2(torch.tensor(Ym), torch.tensor(rid[mask]))
    print(f"\n{'='*100}\n{label}  n={len(Y)}  ridge (alpha by inner val) R2 = {r2_ridge:+.4f}   "
          f"alphas {[p['alpha'] for p in prep]}   prep {time.time()-t0:.0f}s", flush=True)
    nz = prep[0]["noise"]
    print(f"  noise model (fold 1): per-run sd base {np.round(nz['run_sd'][:10],3).tolist()}", flush=True)
    print(f"                        per-run sd cond {np.round(nz['run_sd'][10:],3).tolist()}", flush=True)
    yh_corr = [float(np.corrcoef(p['yhat'].cpu().numpy(), p['yt'][p['itr']].cpu().numpy())[0, 1]) for p in prep]
    vfrac = [float(p['v'].mean()) for p in prep]
    print(f"  corr(weighted fold change, plain fold change) {np.mean(yh_corr):.4f};  "
          f"mean measurement variance = {100*np.mean(vfrac):.2f}% of var(y)", flush=True)

    # stage A: architecture and regularisation, plain MSE
    print(f"\n  STAGE A — {len(GRID)} configurations, MSE loss, selected on inner held-out clusters", flush=True)
    print(f"  {'configuration':<46}{'inner val R2':>13}{'outer R2':>10}{'vs ridge':>10}{'epochs':>8}{'s':>6}", flush=True)
    stA = []
    for c in GRID:
        t1 = time.time()
        oof, val, info = run_cfg(prep, Y, c, "mse")
        o = r2(torch.tensor(Ym), torch.tensor(oof[mask]))
        stA.append(dict(cfg=c, name=cname(c), val=val, outer=o, epochs=float(np.mean([i["epochs"] for i in info])),
                        oof=oof))
        print(f"  {cname(c):<46}{val:>+13.4f}{o:>+10.4f}{o-r2_ridge:>+10.4f}{stA[-1]['epochs']:>8.0f}"
              f"{time.time()-t1:>6.0f}", flush=True)
    stA.sort(key=lambda d: -d["val"])
    top = stA[: a.top]

    # stage B: per-sample likelihoods on the top configurations
    print(f"\n  STAGE B — per-sample losses on the top {a.top} configurations", flush=True)
    stB = []
    for d0 in top:
        for loss in ("naive", "rep", "het"):
            t1 = time.time()
            oof, val, info = run_cfg(prep, Y, d0["cfg"], loss)
            o = r2(torch.tensor(Ym), torch.tensor(oof[mask]))
            tau = np.mean([i["tau2"][0] for i in info]) if loss == "rep" else float("nan")
            stB.append(dict(cfg=d0["cfg"], name=d0["name"], loss=loss, val=val, outer=o, oof=oof, tau2=tau))
            extra = "" if np.isnan(tau) else f"   learned model error tau2 = {tau:.3f} of var(y)"
            print(f"  {d0['name']:<46}{loss:>6}{val:>+11.4f}{o:>+10.4f}{o-r2_ridge:>+10.4f}"
                  f"{time.time()-t1:>6.0f}s{extra}", flush=True)

    # the selected model: best inner val across stages A and B
    allm = [dict(d, loss="mse") for d in stA] + stB
    sel = max(allm, key=lambda d: d["val"])
    lo, hi = boot_diff(Ym, sel["oof"][mask], rid[mask], a.boot)
    # blend with ridge, weight chosen on inner val is not available for OOF ensembles cheaply ->
    # report the fixed 50/50 blend as a descriptive extra, clearly labelled
    blend = 0.5 * sel["oof"] + 0.5 * rid
    ob = r2(torch.tensor(Ym), torch.tensor(blend[mask]))
    blo, bhi = boot_diff(Ym, blend[mask], rid[mask], a.boot)
    print(f"\n  SELECTED on inner val: {sel['name']}  loss {sel['loss']}", flush=True)
    print(f"    outer R2 {sel['outer']:+.4f}  vs ridge {r2_ridge:+.4f}  ->  {100*(sel['outer']-r2_ridge):+.2f} points "
          f"[{100*lo:+.2f},{100*hi:+.2f}]", flush=True)
    print(f"    50/50 blend with ridge (fixed weight, descriptive): R2 {ob:+.4f}  "
          f"{100*(ob-r2_ridge):+.2f} points [{100*blo:+.2f},{100*bhi:+.2f}]", flush=True)
    results[cond] = dict(
        n=int(len(Y)), ridge_r2=r2_ridge, ridge_alphas=[p["alpha"] for p in prep],
        yhat_corr=float(np.mean(yh_corr)), meas_var_frac=float(np.mean(vfrac)),
        noise_run_sd=nz["run_sd"], noise_abund_beta=nz["abund_beta"],
        stageA=[{k_: v for k_, v in d.items() if k_ != "oof"} for d in stA],
        stageB=[{k_: v for k_, v in d.items() if k_ != "oof"} for d in stB],
        selected=dict(name=sel["name"], loss=sel["loss"], val=sel["val"], outer=sel["outer"],
                      gain=sel["outer"] - r2_ridge, gain_ci=[lo, hi]),
        blend=dict(outer=ob, gain=ob - r2_ridge, gain_ci=[blo, bhi]))
    np.save(f"{D}/pd09_oof_{cond}.npy", np.stack([Y, rid, sel["oof"]]))

if dev == "cuda":
    print(f"\npeak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
print(f"total {(time.time()-T0)/60:.1f} min", flush=True)
if not a.quick:
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)
print("PD09_DONE", flush=True)
