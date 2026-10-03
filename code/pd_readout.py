"""Shared GPU readout pieces: closed-form ridge, a vectorised ensemble MLP with per-member early
stopping, and the per-sample losses. Lifted from pd09_mlp.py (whose results stand as committed) so
that the salt and GA_33 readouts run identical model code. Needs only numpy and torch, so it runs
in the env whose torch has Blackwell (sm_120) kernels.

Losses, all in standardised target units:
  mse    plain squared error on the replicate-mean fold change
  naive  squared error weighted by 1 / measurement variance (measurement noise only)
  rep    Gaussian NLL, variance = learned model error tau^2 + measurement variance
  het    beta-NLL (beta = 0.5), variance = network-predicted s(x)^2 + measurement variance
"""
import math
import numpy as np, torch
import torch.nn.functional as F


def ridge_fit(X, y, alpha):
    Xd, yd = X.double(), y.double()
    mu = yd.mean()
    A = Xd.T @ Xd + alpha * torch.eye(Xd.shape[1], device=X.device, dtype=torch.float64)
    return torch.linalg.solve(A, Xd.T @ (yd - mu)).float(), mu.float()


def ridge_pred(X, wm):
    return X @ wm[0] + wm[1]


def r2(y, p):
    y, p = torch.as_tensor(y), torch.as_tensor(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def inner_split(G, seed, frac=0.15):
    """Hold out `frac` of the clusters: the stopping/selection split, grouped like the outer one."""
    ug = np.array(sorted(set(G)))
    np.random.default_rng(seed).shuffle(ug)
    vg = set(ug[: int(frac * len(ug))])
    return np.array([g in vg for g in G])


class Ens(torch.nn.Module):
    """E independent MLPs as stacked weights; one batched matmul trains all of them."""

    def __init__(s, E, din, hidden, dout, drop):
        super().__init__()
        dims = [din, *hidden, dout]
        s.W, s.b = torch.nn.ParameterList(), torch.nn.ParameterList()
        for m, n in zip(dims[:-1], dims[1:]):
            bd = 1 / math.sqrt(m)
            s.W.append(torch.nn.Parameter(torch.empty(E, m, n).uniform_(-bd, bd)))
            s.b.append(torch.nn.Parameter(torch.zeros(E, 1, n)))
        s.drop, s.pin = drop, drop / 3

    def forward(s, x):                 # (E,B,d) in training; (N,d) broadcasts to every member
        if s.training and s.pin > 0:
            x = F.dropout(x, s.pin)
        for i, (w, b) in enumerate(zip(s.W, s.b)):
            x = torch.matmul(x, w) + b
            if i < len(s.W) - 1:
                x = F.dropout(F.gelu(x), s.drop, s.training)
        return x


def train_mlp(cfg, loss, Xtr, ytr, vtr, Xva, yva, Xte, members=16, epochs=400, patience=25, batch=256):
    """Single-target ensemble. y in standardised units, v the measurement variance in the same
    units. Each member keeps its own best-on-validation weights; the ensemble mean is returned."""
    dev = Xtr.device
    E, het = members, loss == "het"
    net = Ens(E, Xtr.shape[1], cfg["hidden"], 2 if het else 1, cfg["drop"]).to(dev)
    logtau = torch.nn.Parameter(torch.zeros(1, device=dev))
    opt = torch.optim.AdamW([{"params": list(net.parameters()), "weight_decay": cfg["wd"]},
                             {"params": [logtau], "weight_decay": 0.0}], lr=cfg.get("lr", 1e-3))
    yt, vt, yv = ytr[:, None], vtr[:, None], yva[:, None]
    wn = (1 / vt) / (1 / vt).mean()
    n = len(yt)
    best = torch.full((E,), -1e9, device=dev)
    bad = torch.zeros(E, device=dev)
    epb = torch.zeros(E, device=dev)
    state = [p.detach().clone() for p in net.parameters()]
    sst = ((yv - yv.mean()) ** 2).sum()
    for ep in range(epochs):
        net.train()
        perm = torch.argsort(torch.rand(E, n, device=dev), 1)
        for i in range(0, n, batch):
            idx = perm[:, i:i + batch]
            out = net(Xtr[idx])
            mu, yb = out[..., :1], yt[idx]
            if loss == "mse":
                l = ((mu - yb) ** 2).mean()
            elif loss == "naive":
                l = (wn[idx] * (mu - yb) ** 2).mean()
            elif loss == "rep":
                var = torch.exp(logtau) + vt[idx]
                l = (0.5 * (mu - yb) ** 2 / var + 0.5 * torch.log(var)).mean()
            else:
                var = F.softplus(out[..., 1:]) + 1e-3 + vt[idx]
                l = ((0.5 * (mu - yb) ** 2 / var + 0.5 * torch.log(var)) * var.detach() ** 0.5).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sc = 1 - ((net(Xva)[..., :1] - yv) ** 2).sum((1, 2)) / sst
            imp = sc > best + 1e-5
            best = torch.where(imp, sc, best)
            epb = torch.where(imp, torch.full_like(epb, ep), epb)
            bad = torch.where(imp, torch.zeros_like(bad), bad + 1)
            for sp, p in zip(state, net.parameters()):
                sp.copy_(torch.where(imp.view(-1, *([1] * (p.dim() - 1))), p.detach(), sp))
        if bool((bad >= patience).all()):
            break
    with torch.no_grad():
        for sp, p in zip(state, net.parameters()):
            p.copy_(sp)
        net.eval()
        pva = net(Xva)[..., 0].mean(0)
        pte = net(Xte)[..., 0].mean(0)
    return pva, pte, dict(epochs=float(epb.mean()), tau2=float(torch.exp(logtau)))


def boot_diff(y, p1, p0, nb, rng):
    """Paired bootstrap interval on R2(p1) - R2(p0) over proteins."""
    i = np.arange(len(y))
    d = []
    for _ in range(nb):
        s = rng.choice(i, len(i), replace=True)
        yy = y[s]
        d.append((((yy - p0[s]) ** 2).sum() - ((yy - p1[s]) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
