"""cm03: one multi-task MLP over every HSP pull-down target, trained as a vectorised population.

Inputs per protein (data/campaign, cm02b): ESMC-6B mean-pooled hidden states (any subset of layers 0..80),
the layer-60 SAE (log1p max activation), and sequence covariates. Outputs: all 33 targets of cm02 at once
(baselines, abundance, bound:free partition, enrichment over input, salt / temperature / Mg effects),
masked where a protein was not measured.

Architecture (one member):
  layer branch  each selected layer: LayerNorm-free standardisation (train-fold stats) -> Linear(2560, p)
                -> concatenated (L*p)                                     [mode "proj"]
                or a softmax-weighted mix of the layers -> Linear(2560, p)  [mode "mix"]
  SAE branch    log1p(max) standardised -> input dropout -> Linear(16384, s)
  simple        25 covariates -> Linear(25, 16)
  trunk         concat -> GELU -> dropout -> Linear(h) -> GELU -> dropout -> Linear(k)  = latent z
  heads         Linear(k, 33): one linear read-out per target from the SHARED latent
Loss: masked MSE on targets standardised with training-fold statistics, averaged per target, then over
targets (optionally weighted).

Efficiency: a group of M members with the same tensor shapes is trained as ONE batched computation
(every weight has a leading member axis; einsum/bmm; a vectorised AdamW with per-member lr / weight decay /
dropout). All inputs live on the GPU in fp16; minibatches are index gathers; compute in bf16 autocast with
fp32 master weights. One process fills an RTX card; no data loader, no host round trips in the loop.

Protocol, per outer fold (grouped by 30 % identity cluster, cm02b): train on outer-train minus inner fold 0,
early-stop each member on inner fold 0, rank members by inner validation loss, predict the outer test fold
with the mean of the best `--top` members. Selection never sees the test fold. Writes out-of-fold
predictions for every member's group-best ensemble and per-target R2 against the split-half ceiling.

  python cm03_mlp.py --spec specs/<name>.json --out data/campaign/runs/<name>
"""
import os, json, time, math, argparse, itertools
import numpy as np, torch

ROOT = os.environ.get("ESM_PROAE_ROOT", "/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae"); C = f"{ROOT}/data/campaign"
ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--top", type=int, default=8)
ap.add_argument("--save-latent", action="store_true", help="per-fold latent of the best member for all proteins")
ap.add_argument("--save-model", action="store_true", help="per-fold parameters of the best member")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
spec = json.load(open(a.spec)); dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
t_start = time.time()

# ---------------------------------------------------------------- data, once, on the GPU
F = np.load(f"{C}/folds.npz"); has = F["has_features"]; outer, inner = F["outer"], F["inner"]
Tz = np.load(f"{C}/targets.npz", allow_pickle=True); names_all = list(Tz["names"].astype(str)); Yall = Tz["Y"]
meta = json.load(open(f"{C}/targets_meta.json"))
tnames = spec.get("targets", names_all); tix = [names_all.index(t) for t in tnames]
Y = Yall[:, tix]; keep = has & np.isfinite(Y).any(1)
rows = np.nonzero(keep)[0]; n = len(rows)
LAY = spec.get("layers", [50]); use_sae = spec.get("sae", True); use_simple = spec.get("simple", True)
XL = torch.from_numpy(np.load(f"{C}/layers.npy", mmap_mode="r")[rows][:, [l // 10 for l in LAY]]).to(dev) if LAY else None   # (n, L, 2560) fp16
MP = spec.get("maskpool", [])               # cm15: layer-80 states pooled over structural masks, one extra slot each
if MP:
    mm = json.load(open(f"{C}/maskpool_meta.json"))["masks"]
    XM = torch.from_numpy(np.load(f"{C}/maskpool.npy", mmap_mode="r")[rows][:, [mm.index(m) for m in MP]]).to(dev)
    XL = XM if XL is None else torch.cat([XL, XM], 1)
    LAY = list(LAY) + [f"mask:{m}" for m in MP]             # only len(LAY) is used from here on
XS = torch.log1p(torch.from_numpy(np.load(f"{C}/sae_max.npy", mmap_mode="r")[rows]).to(dev).float()).half() if use_sae else None
import pandas as pd
SIM = pd.read_csv(f"{C}/simple.tsv", sep="\t", index_col=0).iloc[rows].fillna(0).values.astype(np.float32)
COV = spec.get("covariates", [])            # measured per-protein properties (cm07 named_properties.tsv), with missingness flags
if COV:
    NP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).iloc[rows][COV].values.astype(np.float32)
    miss = np.isnan(NP); NP = np.where(miss, np.nanmedian(NP, 0), NP)       # median fill; flags carry the missingness
    SIM = np.hstack([SIM, NP, miss.astype(np.float32)])
    assert not (set(COV) & {"abundance_here"} and "GA33_input" in tnames), "abundance_here IS the GA33_input target: drop the target"
if spec.get("struct"):                      # cm13 AlphaFold structural features, same median-fill + flag treatment
    SD = pd.read_csv(f"{C}/struct.tsv", sep="\t", index_col=0).reindex(pd.read_csv(f"{C}/union.tsv", sep="\t")["accession"].astype(str))
    SD = SD.drop(columns=[c for c in ("seq_match", "seq_len_fasta") if c in SD.columns]).iloc[rows].values.astype(np.float32)
    sm = np.isnan(SD); SD = np.where(sm, np.nanmedian(SD, 0), SD)
    SIM = np.hstack([SIM, SD, sm.any(1, keepdims=True).astype(np.float32)])
XP = torch.from_numpy(SIM).to(dev) if use_simple else None
Yt = torch.from_numpy(Y[rows]).to(dev); Mt = torch.isfinite(Yt); Yt = torch.nan_to_num(Yt); MASK = Mt.clone()
if spec.get("residualize"):
    NPP = pd.read_csv(f"{C}/named_properties.tsv", sep="\t", index_col=0).iloc[rows]
    PRED_NAMES = names_all + list(NPP.columns); PRED = np.hstack([Yall[rows], NPP.values]).astype(np.float32)
outer_r, inner_r = outer[rows], inner[:, rows]
T = Y.shape[1]
print(f"{n} proteins, {T} targets, layers {LAY}, sae {use_sae}, simple {use_simple}, covariates {COV}; data on GPU "
      f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)


def stats(x, idx):
    """per-feature mean/std over training rows (float32)"""
    xs = x[idx].float(); mu = xs.mean(0); sd = xs.std(0).clamp_min(1e-3); return mu, sd


# ---------------------------------------------------------------- vectorised population
class Pop:
    def __init__(s, g, M, n_targets, gen):
        s.g = g; s.M = M; P = {}
        def lin(i, o, scale=1.0): return (torch.randn(M, i, o, generator=gen, device=dev) * scale / math.sqrt(i)).requires_grad_()
        def zeros(*shape): return torch.zeros(M, *shape, device=dev).requires_grad_()
        d_in = 0
        if LAY:
            if g["mode"] == "proj":
                P["WL"] = (torch.randn(M, len(LAY), 2560, g["p"], generator=gen, device=dev) / math.sqrt(2560 * len(LAY))).requires_grad_()
                P["bL"] = zeros(len(LAY) * g["p"]); d_in += len(LAY) * g["p"]
            else:
                P["aL"] = zeros(len(LAY)); P["WL"] = lin(2560, g["p"]); P["bL"] = zeros(g["p"]); d_in += g["p"]
        if use_sae: P["WS"] = lin(16384, g["s"]); P["bS"] = zeros(g["s"]); d_in += g["s"]
        if use_simple: P["WP"] = lin(SIM.shape[1], 16); P["bP"] = zeros(16); d_in += 16
        P["W1"] = lin(d_in, g["h"]); P["b1"] = zeros(g["h"])
        P["W2"] = lin(g["h"], g["k"]); P["b2"] = zeros(g["k"])
        P["WH"] = lin(g["k"], n_targets); P["bH"] = zeros(n_targets)
        s.P = P; s.m = {k: torch.zeros_like(v) for k, v in P.items()}; s.v = {k: torch.zeros_like(v) for k, v in P.items()}; s.t = 0

    def forward(s, idx, norm, train, hp, gen):
        P, M = s.P, s.M; B = len(idx); parts = []
        def drop(x, rate):   # rate: (M,) per-member dropout
            if not train: return x
            keep = (torch.rand(x.shape, generator=gen, device=dev) >= rate.view(M, *[1] * (x.dim() - 1))).to(x.dtype)
            return x * keep / (1 - rate.view(M, *[1] * (x.dim() - 1)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if LAY:
                xl = ((XL[idx].float() - norm["muL"]) / norm["sdL"]).to(torch.bfloat16)          # (B, L, 2560)
                if s.g["mode"] == "proj":
                    h = torch.einsum("bld,mldp->mblp", xl, P["WL"]).reshape(M, B, -1) + P["bL"][:, None]
                else:
                    w = torch.softmax(P["aL"], -1); xm = torch.einsum("bld,ml->mbd", xl, w)
                    h = torch.bmm(xm, P["WL"]) + P["bL"][:, None]
                parts.append(h)
            if use_sae:
                xs = ((XS[idx].float() - norm["muS"]) / norm["sdS"]).to(torch.bfloat16)
                if train:   # per-member input dropout needs a per-member copy of the batch
                    xs = drop(xs.unsqueeze(0).expand(M, -1, -1), hp["in_drop"])
                    parts.append(torch.bmm(xs, P["WS"].to(torch.bfloat16)) + P["bS"][:, None])
                else:       # evaluation: one shared input, no M-fold copy
                    parts.append(torch.einsum("bf,mfs->mbs", xs, P["WS"].to(torch.bfloat16)) + P["bS"][:, None])
            if use_simple:
                xp = ((XP[idx] - norm["muP"]) / norm["sdP"]).unsqueeze(0).expand(M, -1, -1)
                parts.append(torch.bmm(xp, P["WP"]) + P["bP"][:, None])
            x = torch.nn.functional.gelu(torch.cat(parts, -1)); x = drop(x, hp["drop"])
            x = torch.nn.functional.gelu(torch.bmm(x, P["W1"]) + P["b1"][:, None]); x = drop(x, hp["drop"])
            z = torch.bmm(x, P["W2"]) + P["b2"][:, None]
            yhat = torch.bmm(z, P["WH"]) + P["bH"][:, None]
        return yhat.float(), z.float()

    def step(s, loss, lr, wd):
        s.t += 1; grads = torch.autograd.grad(loss, list(s.P.values()))
        b1, b2, eps = .9, .999, 1e-8
        with torch.no_grad():
            for (k, p), gr in zip(s.P.items(), grads):
                sh = (s.M,) + (1,) * (p.dim() - 1)
                s.m[k].mul_(b1).add_(gr, alpha=1 - b1); s.v[k].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = s.m[k] / (1 - b1 ** s.t); vh = s.v[k] / (1 - b2 ** s.t)
                decay = 0 if k.startswith("b") or k == "aL" else wd.view(sh)
                p.sub_(lr.view(sh) * (mh / (vh.sqrt() + eps) + decay * p))


def forward_chunked(pop, idx, norm, hp, gen, chunk=2048):
    ys, zs = [], []
    for i in range(0, len(idx), chunk):
        y_, z_ = pop.forward(idx[i:i + chunk], norm, False, hp, gen); ys.append(y_); zs.append(z_)
    return torch.cat(ys, 1), torch.cat(zs, 1)


def masked_loss(yhat, y, m, w):
    """yhat (M,B,T); per-member mean over targets of per-target MSE on observed entries"""
    se = ((yhat - y[None]) ** 2) * m[None]; cnt = m.sum(0).clamp_min(1)
    return ((se.sum(1) / cnt) * w).sum(-1) / w.sum()          # (M,)


def r2(pred, y, m):
    out = []
    for t in range(y.shape[1]):
        k = m[:, t] & np.isfinite(pred[:, t])
        if k.sum() < 20: out.append(float("nan")); continue
        yy, pp = y[k, t], pred[k, t]; out.append(float(1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()))
    return np.array(out)


# ---------------------------------------------------------------- one group = one architecture shape, M members
def run_group(g, fold, gen_seed):
    gen = torch.Generator(device=dev); gen.manual_seed(gen_seed)
    te = np.nonzero(outer_r == fold)[0]; va = np.nonzero(inner_r[fold] == 0)[0]; tr = np.nonzero((outer_r != fold) & (inner_r[fold] > 0))[0]
    norm = {}
    if LAY: norm["muL"], norm["sdL"] = stats(XL, tr)
    if use_sae: norm["muS"], norm["sdS"] = stats(XS, tr)
    if use_simple: norm["muP"], norm["sdP"] = stats(XP, tr)
    # optional: residualise targets on measured predictors, cross-fitted (OLS on this fold's training rows only)
    if spec.get("residualize"):
        Yres = Yt.clone()
        for t, preds in spec["residualize"].items():
            ti = tnames.index(t); X = torch.as_tensor(PRED[:, [PRED_NAMES.index(p) for p in preds]], device=dev)
            okp = torch.isfinite(X).all(1); Xf = torch.nan_to_num(X)
            fit_rows = torch.as_tensor(tr, device=dev); fit_rows = fit_rows[Mt[fit_rows, ti] & okp[fit_rows]]
            A = torch.cat([Xf[fit_rows], torch.ones(len(fit_rows), 1, device=dev)], 1)
            beta = torch.linalg.lstsq(A, Yt[fit_rows, ti:ti + 1]).solution
            Yres[:, ti] = Yt[:, ti] - (torch.cat([Xf, torch.ones(n, 1, device=dev)], 1) @ beta)[:, 0]
            MASK[:, ti] = Mt[:, ti] & okp                   # proteins without their predictors drop out of this target
        Yuse, Muse = Yres, MASK
    else:
        Yuse, Muse = Yt, Mt
    # targets standardised on training rows
    ym = torch.stack([(Yuse[tr, t][Muse[tr, t]]).mean() for t in range(T)]); ys = torch.stack([(Yuse[tr, t][Muse[tr, t]]).std() for t in range(T)])
    Ys = (Yuse - ym) / ys
    hpg = list(itertools.product(g["lr"], g["wd"], g["drop"], g["in_drop"], range(g.get("seeds", 1))))
    M = len(hpg); hp = {k: torch.tensor([h[i] for h in hpg], device=dev, dtype=torch.float32) for i, k in enumerate(["lr", "wd", "drop", "in_drop"])}
    w = torch.tensor(g.get("weights", [1.0] * T), device=dev)
    pop = Pop(g, M, T, gen)
    trt = torch.as_tensor(tr, device=dev); bs = g.get("batch", 512); steps_ep = max(1, len(tr) // bs); E = g["epochs"]
    best = torch.full((M,), float("inf"), device=dev); best_ep = torch.zeros(M, device=dev)
    snap = {k: v.detach().clone() for k, v in pop.P.items()}
    vat = torch.as_tensor(va, device=dev)
    for ep in range(E):
        lr_ep = hp["lr"] * 0.5 * (1 + math.cos(math.pi * ep / E))
        perm = trt[torch.randperm(len(tr), generator=gen, device=dev)]
        for i in range(steps_ep):
            idx = perm[i * bs:(i + 1) * bs]
            yhat, _ = pop.forward(idx, norm, True, hp, gen)
            loss = masked_loss(yhat, Ys[idx], Muse[idx], w)
            pop.step(loss.sum(), lr_ep, hp["wd"])
        with torch.no_grad():
            yv, _ = forward_chunked(pop, vat, norm, hp, gen); lv = masked_loss(yv, Ys[vat], Muse[vat], w)
            better = lv < best; best = torch.where(better, lv, best); best_ep = torch.where(better, torch.full_like(best_ep, ep), best_ep)
            for k in snap:
                sh = (M,) + (1,) * (snap[k].dim() - 1); snap[k] = torch.where(better.view(sh), pop.P[k].detach(), snap[k])
    pop.P = {k: v for k, v in snap.items()}
    with torch.no_grad():
        order = torch.argsort(best)[: a.top]
        yt, zt = forward_chunked(pop, torch.as_tensor(te, device=dev), norm, hp, gen)
        yv, _ = forward_chunked(pop, vat, norm, hp, gen)
        pred = (yt[order].mean(0) * ys + ym).cpu().numpy(); predv = (yv[order].mean(0) * ys + ym).cpu().numpy()
        z = None
        if a.save_latent:   # latent of the best member for EVERY protein, from this fold's model (fold models are not aligned)
            _, zall = forward_chunked(pop, torch.arange(n, device=dev), norm, hp, gen); z = zall[order[0]].cpu().numpy()
        head = None
        if a.save_model:    # full parameters of the best member, plus the normalisation, for attribution
            head = {k: v[order[0]].float().cpu().numpy() for k, v in pop.P.items()}
            head.update({f"norm_{k}": v.cpu().numpy() for k, v in norm.items()}, ym=ym.cpu().numpy(), ys=ys.cpu().numpy())
    members = [dict(lr=h[0], wd=h[1], drop=h[2], in_drop=h[3], seed=h[4], val_loss=float(best[i]), best_epoch=int(best_ep[i])) for i, h in enumerate(hpg)]
    truth = (Yuse.cpu().numpy(), Muse.cpu().numpy()) if spec.get("residualize") else None
    return te, pred, va, predv, members, order.cpu().numpy(), z, head, truth


# ---------------------------------------------------------------- run every group of the spec
folds = [int(f) for f in a.folds.split(",")]
Ynp = Y[rows]; Mnp = np.isfinite(Ynp)
ceil = np.array([meta[t]["ceiling"] for t in tnames])
summary = []
for gi, g in enumerate(spec["groups"]):
    g = {**spec.get("defaults", {}), **g}; gname = g.get("name", f"g{gi}")
    t0 = time.time(); oof = np.full((n, T), np.nan); val_rows = []; YTRUE = Ynp.copy(); MTRUE = Mnp.copy()
    torch.cuda.reset_peak_memory_stats()
    zs = np.full((len(folds), n, g["k"]), np.nan, np.float32) if a.save_latent else None; heads = {}
    for f in folds:
        te, pred, va, predv, members, order, z, head, truth = run_group(g, f, 1000 * gi + f)
        if truth is not None: YTRUE[te] = truth[0][te]; MTRUE[te] = truth[1][te]
        oof[te] = pred; val_rows.append(dict(fold=f, r2_val=r2(predv, Ynp[va], Mnp[va]).tolist(), members=members, top=order.tolist()))
        if zs is not None: zs[folds.index(f)] = z
        if head is not None: heads[f] = head
    sec = time.time() - t0
    R = r2(oof, YTRUE, MTRUE)
    if spec.get("residualize"): np.save(f"{a.out}/{gname}_truth.npy", np.where(MTRUE, YTRUE, np.nan).astype(np.float32))
    res = dict(group=gname, config={k: v for k, v in g.items() if k != "weights"}, r2=dict(zip(tnames, R.tolist())),
               r2_over_ceiling=dict(zip(tnames, (R / ceil).tolist())), mean_r2=float(np.nanmean(R)), mean_r2_over_ceiling=float(np.nanmean(R / ceil)),
               seconds=sec, peak_gb=torch.cuda.max_memory_allocated() / 1e9, n_members=len(val_rows[0]["members"]), folds=val_rows)
    json.dump(res, open(f"{a.out}/{gname}.json", "w"), default=float)
    np.save(f"{a.out}/{gname}_oof.npy", oof.astype(np.float32)); np.save(f"{a.out}/{gname}_rows.npy", rows)   # rows differ by spec: never share
    if zs is not None: np.save(f"{a.out}/{gname}_latent.npy", zs)
    if heads: np.savez(f"{a.out}/{gname}_model.npz", **{f"f{f}_{k}": v for f, h in heads.items() for k, v in h.items()})
    summary.append(dict(group=gname, mean_r2=res["mean_r2"], mean_r2_over_ceiling=res["mean_r2_over_ceiling"], seconds=sec, peak_gb=res["peak_gb"]))
    print(f"[{gname}] mean R2 {res['mean_r2']:.3f}  (R2/ceiling {res['mean_r2_over_ceiling']:.3f})  {sec/60:.1f} min  "
          f"{res['n_members']} members x {len(folds)} folds  peak {res['peak_gb']:.1f} GB", flush=True)
np.save(f"{a.out}/rows.npy", rows); json.dump(dict(targets=tnames, summary=summary, minutes=(time.time() - t_start) / 60), open(f"{a.out}/summary.json", "w"), indent=1)
print("CM03_DONE", flush=True)
