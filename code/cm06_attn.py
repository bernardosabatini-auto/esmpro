"""cm06: the campaign multi-task MLP with a learned attention-pooling front end over residues.

Same targets, folds, protocol and population training as cm03 (vectorised members, inner-fold early stopping,
top-member ensemble on the outer test fold). The difference is the input: besides the mean-pooled layers, each
protein is read residue by residue from the final ESMC-6B layer (cm01b), and the model learns which residues
matter. Gated attention (Ilse et al. 2018), H heads:
    u_i = W_v x_i                         value, d_v
    a_ih = softmax_i( w_h . (tanh(W_a x_i) * sigmoid(W_g x_i)) )   within the protein
    pooled_h = sum_i a_ih u_i             concatenated over heads -> joins the mean-pooled branch -> trunk -> heads
Residues of a batch are flattened (no padding); the softmax runs per protein with scatter max / sum. Batches
are packed by a residue budget so long proteins do not blow up memory. All residue states (7.36 M x 2560 fp16,
37 GB) sit on one 96 GB RTX card, gathered by index; nothing crosses the PCIe bus in the loop.

  python cm06_attn.py --spec specs/<name>.json --out data/campaign/runs/<name>
"""
import os, json, time, math, argparse, itertools
import numpy as np, torch, pandas as pd

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--top", type=int, default=4)
ap.add_argument("--residues", default=f"{C}/residues"); ap.add_argument("--save-attn", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); spec = json.load(open(a.spec)); dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True; t_start = time.time()

F = np.load(f"{C}/folds.npz"); has, outer, inner = F["has_features"], F["outer"], F["inner"]
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names = list(Tz["names"].astype(str)); Y = Tz["Y"]
meta = json.load(open(f"{C}/targets_meta.json"))
U = pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str).tolist()
RX = np.load(f"{a.residues}/res_index.npz", allow_pickle=True); racc = {s: i for i, s in enumerate(RX["accession"].astype(str))}
have_res = np.array([u in racc for u in U])
keep = has & have_res & np.isfinite(Y).any(1); rows = np.nonzero(keep)[0]; n = len(rows)
rstart = torch.as_tensor([RX["start"][racc[U[i]]] for i in rows], device=dev); rlen = torch.as_tensor([RX["length"][racc[U[i]]] for i in rows], device=dev)
t0 = time.time()
RES = torch.from_numpy(np.load(f"{a.residues}/res_L80.npy", mmap_mode="r")[:]).to(dev)       # (R, 2560) fp16, 37 GB
print(f"residues on GPU: {RES.shape[0]/1e6:.2f}M x {RES.shape[1]}  {torch.cuda.memory_allocated()/1e9:.1f} GB  ({time.time()-t0:.0f} s)", flush=True)
LAY = spec.get("layers", list(range(0, 81, 10)))
XL = torch.from_numpy(np.load(f"{C}/layers.npy", mmap_mode="r")[rows][:, [l // 10 for l in LAY]]).to(dev)
Yt = torch.from_numpy(Y[rows]).to(dev); Mt = torch.isfinite(Yt); Yt = torch.nan_to_num(Yt); T = Y.shape[1]
outer_r, inner_r = outer[rows], inner[:, rows]
# per-feature residue normalisation (global over a sample of residues; no target information)
samp = RES[torch.randint(0, RES.shape[0], (200000,), device=dev)].float(); RMU, RSD = samp.mean(0), samp.std(0).clamp_min(1e-3); del samp


def batches(idx, budget, gen):
    """shuffle proteins, then pack consecutive ones until the residue budget is reached (a protein longer than the
    budget is cropped to a random window of `budget` residues during training)"""
    idx = idx[torch.randperm(len(idx), generator=gen, device=dev)]; L = rlen[idx].clamp_max(budget).cpu().numpy()
    out, cur, s = [], [], 0
    for j, l in enumerate(L):
        if cur and s + l > budget: out.append(idx[cur]); cur, s = [], 0
        cur.append(j); s += l
    if cur: out.append(idx[cur])
    return out


def gather(bidx, budget, train, gen):
    """flatten the residues of the proteins in bidx: returns (Rb, 2560) bf16 standardised and segment ids"""
    L = rlen[bidx]; st = rstart[bidx]
    if train:   # random crop for proteins longer than the budget
        off = (torch.rand(len(bidx), generator=gen, device=dev) * (L - budget).clamp_min(0)).long(); L = L.clamp_max(budget); st = st + off
    seg = torch.repeat_interleave(torch.arange(len(bidx), device=dev), L)
    pos = torch.arange(int(L.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(L, 0) - L, L)
    x = RES[st[seg] + pos]
    return ((x.float() - RMU) / RSD).to(torch.bfloat16), seg


class Pop:
    def __init__(s, g, M, gen):
        s.g, s.M = g, M; P = {}
        def lin(i, o): return (torch.randn(M, i, o, generator=gen, device=dev) / math.sqrt(i)).requires_grad_()
        def zeros(*sh): return torch.zeros(M, *sh, device=dev).requires_grad_()
        H, dv, da = g["heads"], g["dv"], g["da"]
        P["Wv"], P["Wa"], P["Wg"] = lin(2560, dv), lin(2560, da), lin(2560, da); P["wh"] = lin(da, H)
        P["WL"] = (torch.randn(M, len(LAY), 2560, g["p"], generator=gen, device=dev) / math.sqrt(2560 * len(LAY))).requires_grad_(); P["bL"] = zeros(len(LAY) * g["p"])
        d_in = H * dv + len(LAY) * g["p"]
        P["W1"], P["b1"] = lin(d_in, g["h"]), zeros(g["h"]); P["W2"], P["b2"] = lin(g["h"], g["k"]), zeros(g["k"]); P["WH"], P["bH"] = lin(g["k"], T), zeros(T)
        s.P = P; s.m = {k: torch.zeros_like(v) for k, v in P.items()}; s.v = {k: torch.zeros_like(v) for k, v in P.items()}; s.t = 0

    def forward(s, bidx, train, hp, gen, budget):
        P, M = s.P, s.M; B = len(bidx); H = s.g["heads"]
        x, seg = gather(bidx, budget, train, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            u = torch.einsum("rd,mdv->mrv", x, P["Wv"])                                         # (M, R, dv)
            gte = torch.tanh(torch.einsum("rd,mda->mra", x, P["Wa"])) * torch.sigmoid(torch.einsum("rd,mda->mra", x, P["Wg"]))
            sc = torch.einsum("mra,mah->mrh", gte, P["wh"]).float()                              # (M, R, H)
            mx = torch.full((M, B, H), -1e30, device=dev).scatter_reduce(1, seg.view(1, -1, 1).expand(M, -1, H), sc, "amax", include_self=True)
            e = torch.exp(sc - mx[:, seg]); den = torch.zeros(M, B, H, device=dev).index_add_(1, seg, e)
            att = e / den[:, seg]                                                                # (M, R, H)
            uf = u.float(); pooled = torch.cat([torch.zeros(M, B, uf.shape[-1], device=dev).index_add_(1, seg, att[..., h:h + 1] * uf)
                                                for h in range(H)], -1)                      # one head at a time: M x R x dv, not M x R x H x dv
            xl = ((XL[bidx].float() - s.norm["muL"]) / s.norm["sdL"]).to(torch.bfloat16)
            hl = torch.einsum("bld,mldp->mblp", xl, P["WL"]).reshape(M, B, -1) + P["bL"][:, None]
            hcat = torch.nn.functional.gelu(torch.cat([pooled, hl.float()], -1))
            if train: hcat = hcat * (torch.rand(hcat.shape, generator=gen, device=dev) >= hp["drop"].view(M, 1, 1)) / (1 - hp["drop"].view(M, 1, 1))
            h1 = torch.nn.functional.gelu(torch.bmm(hcat, P["W1"]) + P["b1"][:, None])
            if train: h1 = h1 * (torch.rand(h1.shape, generator=gen, device=dev) >= hp["drop"].view(M, 1, 1)) / (1 - hp["drop"].view(M, 1, 1))
            z = torch.bmm(h1, P["W2"]) + P["b2"][:, None]; yhat = torch.bmm(z, P["WH"]) + P["bH"][:, None]
        return yhat.float(), z.float(), (att, seg)

    def step(s, loss, lr, wd):
        s.t += 1; grads = torch.autograd.grad(loss, list(s.P.values())); b1, b2 = .9, .999
        with torch.no_grad():
            for (k, p), gr in zip(s.P.items(), grads):
                sh = (s.M,) + (1,) * (p.dim() - 1)
                s.m[k].mul_(b1).add_(gr, alpha=1 - b1); s.v[k].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh, vh = s.m[k] / (1 - b1 ** s.t), s.v[k] / (1 - b2 ** s.t)
                p.sub_(lr.view(sh) * (mh / (vh.sqrt() + 1e-8) + (0 if k.startswith("b") else wd.view(sh) * p)))


def masked_loss(yhat, y, m):
    se = ((yhat - y[None]) ** 2) * m[None]; return (se.sum(1) / m.sum(0).clamp_min(1)).mean(-1)


def r2(pred, y, m):
    out = []
    for t in range(y.shape[1]):
        k = m[:, t] & np.isfinite(pred[:, t])
        out.append(float(1 - ((y[k, t] - pred[k, t]) ** 2).sum() / ((y[k, t] - y[k, t].mean()) ** 2).sum()) if k.sum() > 20 else float("nan"))
    return np.array(out)


def evaluate(pop, idx, hp, gen, budget):
    ys = []
    for b in batches(idx, budget, gen):                # eval: whole proteins, same residue budget as training
        y_, _, _ = pop.forward(b, False, hp, gen, budget * 1000); ys.append((b, y_))
    out = torch.zeros(pop.M, n, T, device=dev)
    for b, y_ in ys: out[:, b] = y_
    return out[:, idx]


def run(g, fold, seed):
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    te = torch.as_tensor(np.nonzero(outer_r == fold)[0], device=dev); va = torch.as_tensor(np.nonzero(inner_r[fold] == 0)[0], device=dev)
    tr = torch.as_tensor(np.nonzero((outer_r != fold) & (inner_r[fold] > 0))[0], device=dev)
    xs = XL[tr].float(); norm = dict(muL=xs.mean(0), sdL=xs.std(0).clamp_min(1e-3)); del xs
    ym = torch.stack([Yt[tr, t][Mt[tr, t]].mean() for t in range(T)]); ysd = torch.stack([Yt[tr, t][Mt[tr, t]].std() for t in range(T)]); Ys = (Yt - ym) / ysd
    hpg = list(itertools.product(g["lr"], g["wd"], g["drop"], range(g.get("seeds", 1)))); M = len(hpg)
    hp = {k: torch.tensor([h[i] for h in hpg], device=dev, dtype=torch.float32) for i, k in enumerate(["lr", "wd", "drop"])}
    pop = Pop(g, M, gen); pop.norm = norm; E = g["epochs"]; budget = g["budget"]
    best = torch.full((M,), float("inf"), device=dev); snap = {k: v.detach().clone() for k, v in pop.P.items()}
    for ep in range(E):
        lr_ep = hp["lr"] * 0.5 * (1 + math.cos(math.pi * ep / E))
        for b in batches(tr, budget, gen):
            yhat, _, _ = pop.forward(b, True, hp, gen, budget); pop.step(masked_loss(yhat, Ys[b], Mt[b]).sum(), lr_ep, hp["wd"])
        with torch.no_grad():
            lv = masked_loss(evaluate(pop, va, hp, gen, budget), Ys[va], Mt[va]); better = lv < best; best = torch.where(better, lv, best)
            for k in snap: snap[k] = torch.where(better.view((M,) + (1,) * (snap[k].dim() - 1)), pop.P[k].detach(), snap[k])
        if ep % 10 == 0: print(f"    fold {fold} epoch {ep} val {best.min().item():.4f}  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    pop.P = snap
    with torch.no_grad():
        order = torch.argsort(best)[: a.top]; yt = evaluate(pop, te, hp, gen, budget)
        pred = (yt[order].mean(0) * ysd + ym).cpu().numpy()
    return te.cpu().numpy(), pred, [dict(lr=h[0], wd=h[1], drop=h[2], seed=h[3], val=float(best[i])) for i, h in enumerate(hpg)]


ceil = np.array([meta[t]["ceiling"] for t in names]); Ynp = Y[rows]; Mnp = np.isfinite(Ynp)
for gi, g in enumerate(spec["groups"]):
    g = {**spec.get("defaults", {}), **g}; gname = g["name"]; t0 = time.time(); oof = np.full((n, T), np.nan); mem = []
    torch.cuda.reset_peak_memory_stats()
    for f in [int(x) for x in a.folds.split(",")]:
        te, pred, members = run(g, f, 100 * gi + f); oof[te] = pred; mem.append(members)
    R = r2(oof, Ynp, Mnp)
    res = dict(group=gname, config=g, r2=dict(zip(names, R.tolist())), r2_over_ceiling=dict(zip(names, (R / ceil).tolist())), mean_r2=float(np.nanmean(R)),
               mean_r2_over_ceiling=float(np.nanmean(R / ceil)), seconds=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9, members=mem)
    json.dump(res, open(f"{a.out}/{gname}.json", "w"), default=float); np.save(f"{a.out}/{gname}_oof.npy", oof.astype(np.float32))
    print(f"[{gname}] mean R2 {res['mean_r2']:.3f} (R2/ceiling {res['mean_r2_over_ceiling']:.3f})  {res['seconds']/60:.1f} min  peak {res['peak_gb']:.1f} GB", flush=True)
np.save(f"{a.out}/rows_attn.npy", rows); print("CM06_DONE", flush=True)
