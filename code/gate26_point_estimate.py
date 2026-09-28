"""Gate 26: is the single-draw TM penalising us for being generative, and how much latent
error can the frozen decoder absorb?

Two experiments in one job.

A. Point estimators. Draw K latents per protein and score four readouts against the reference:
     single      one draw (what every number in this project reports)
     medoid      the draw closest to the others in latent space (selection)
     mean_ode    layer_norm(mean of the K ODE endpoints)          <- conditional mean, K x ODE cost
     mean_1step  layer_norm(mean of K one-step estimates x0+v(x0,0))  <- conditional mean, K forward passes
   If the metric rewards a point estimate, mean_* beats single. layer_norm is the projection onto
   the latent manifold that the true latents satisfy exactly (CLAUDE.md section 7).

B. Latent-error transfer. Perturb the TRUE latent by sigma, project, decode, and score. Gives the
   TM-versus-latent-error curve, so "close a 0.02 TM gap" becomes a concrete latent-accuracy target.

  python gate26_point_estimate.py --ckpt last_pf_840M_p128x8_long512_1p3M.ckpt --h5 dataset_casp_esmc.h5 \
      --names-file notes/casp_domains_le256.txt --k 8
"""
import os, sys, json, argparse, time, tempfile, shutil
import numpy as np, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
from gate6_fape_train import ProteinDatasetFAPE, H5_PATH, PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--cfg-w", type=float, default=2.0); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--bs", type=int, default=10); ap.add_argument("--h5", default=None)
ap.add_argument("--names-file", default=None); ap.add_argument("--offset", type=int, default=1000)
ap.add_argument("--sigmas", default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
ap.add_argument("--skip-transfer", action="store_true")
a = ap.parse_args(); dev = torch.device("cuda")
D = PROJECT / "data/phase1_dataset"; ck = D / a.ckpt

# ---- checkpoint ----
st = None
if a.ckpt.endswith(".ckpt"):
    st = torch.load(str(ck), weights_only=False, map_location="cpu")
    arch, weights, meta = st["arch"], st["ema"], {"epoch": st["epoch"], "tm": st["best_tm"]}
    mtype = st.get("model"); ex = dict(st.get("extra_arch") or {})
else:
    meta = json.loads(open(str(ck) + ".meta.json").read())
    arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond") if k in meta}
    weights = torch.load(str(ck), weights_only=True); mtype = meta.get("model")
    ex = {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact", "pair_fused") if k in meta}
sample_fn, pair_net = G.sample, False
if mtype in ("PairFlowNet", "_Net"):
    import gate10_pair_flow as G10
    G10.install(); ex.pop("recycle", None); ex.pop("p_rec", None); ex.pop("rec_every", None)
    net = G10.PairFlowNet(**arch, **ex).to(dev); sample_fn, pair_net = G10.sample_pair, True
    print(f"pair-track model {ex}", flush=True)
else:
    net = G.LatentFlowNet(**arch).to(dev); print("pair-free model", flush=True)
net.load_state_dict(G.extend_pos_table(G.adapt_state_dict(weights, net.state_dict().keys()), net.state_dict())); net.eval()
print(f"checkpoint {a.ckpt}: epoch {meta['epoch']}, train-time TM {meta['tm']:.3f}", flush=True)

# ---- data ----
h5p = a.h5 if (a.h5 and os.path.isabs(a.h5)) else (str(D / a.h5) if a.h5 else None)
h5p = h5p or (st.get("h5_path") if st else meta.get("h5_path")) or str(D / "dataset_100k_esmc.h5")
vf = ProteinDatasetFAPE(h5p, "val", max_len=G.MAX_LEN)
torch.manual_seed(42); idx = torch.randperm(len(vf))[a.offset:a.offset + a.n].tolist()
if a.names_file:
    p = a.names_file if os.path.isabs(a.names_file) else str(PROJECT / a.names_file)
    want = [l.strip() for l in open(p) if l.strip()]; pos = {n: i for i, n in enumerate(vf.names)}
    idx = [pos[n] for n in want if n in pos]
names = [vf.names[i] for i in idx]; N = len(idx)
print(f"{N} proteins from {os.path.basename(h5p)}, K={a.k}, w={a.cfg_w}, {a.steps} steps", flush=True)
dec = G.load_decoder(dev)
work = tempfile.mkdtemp(prefix="g26_"); dirs = {}
for d in ("truth", "single", "medoid", "mean_ode", "mean_1step", "oracle"):
    dirs[d] = f"{work}/{d}"; os.makedirs(dirs[d])
sig = [float(x) for x in a.sigmas.split(",")]
if not a.skip_transfer:
    for s in sig:
        dirs[f"sig{s}"] = f"{work}/sig{s}"; os.makedirs(dirs[f"sig{s}"])

per_sample_tm = {}; zmse_acc = {}; est_zmse = {}
for s0 in range(0, N, a.bs):
    sub = idx[s0:s0 + a.bs]; nb = names[s0:s0 + a.bs]
    esm, z_true, ca_true, mask, _ = next(iter(DataLoader(Subset(vf, sub), batch_size=len(sub))))
    esm, z_true, mask = esm.to(dev), z_true.to(dev), mask.to(dev)
    B, L = mask.shape
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pr = net.compute_pair(esm, mask) if pair_net else None
        zs = []
        for k in range(a.k):
            g = torch.Generator(device=dev).manual_seed(1234 + k)
            zs.append(sample_fn(net, esm, mask, a.steps, a.cfg_w, gen=g).float())
        Z = torch.stack(zs)                                  # (K,B,L,8)
        # one-step conditional mean: x0 + v(x0, t=0), averaged over K noise draws
        one = []
        for k in range(a.k):
            g = torch.Generator(device=dev).manual_seed(4321 + k)
            x0 = torch.randn(B, L, G.D_LAT, device=dev, generator=g)
            t0 = torch.zeros(B, device=dev)
            v = net(x0, t0, esm, mask, None, None, pair=pr) if pair_net else net(x0, t0, esm, mask, None, None)
            one.append((x0 + v.float()))
        O = torch.stack(one)
    m = mask.unsqueeze(-1)
    est = {}
    est["single"] = Z[0]
    est["mean_ode"] = F.layer_norm(Z.mean(0), (G.D_LAT,)) * m
    est["mean_1step"] = F.layer_norm(O.mean(0), (G.D_LAT,)) * m
    # medoid in latent space: the draw with the smallest mean distance to the others
    flat = (Z * m).flatten(2)
    dmat = torch.cdist(flat.permute(1, 0, 2), flat.permute(1, 0, 2))     # (B,K,K)
    est["medoid"] = Z[dmat.mean(-1).argmin(-1), torch.arange(B, device=dev)]
    for kk, v in est.items():
        est_zmse.setdefault(kk, []).append(float((((v.detach() - z_true) ** 2) * m).sum() / (m.sum() * G.D_LAT)))
    cas = {kk: dec(v.float(), mask).float().detach().cpu() for kk, v in est.items()}
    ca_k = [dec(Z[k].float(), mask).float().detach().cpu() for k in range(a.k)]
    for i, n in enumerate(nb):
        Ln = int(mask[i].sum())
        _write_pseudo_backbone_pdb(ca_true[i, :Ln].numpy(), f"{dirs['truth']}/{n}.pdb")
        for kk in cas: _write_pseudo_backbone_pdb(cas[kk][i, :Ln].numpy(), f"{dirs[kk]}/{n}.pdb")
        for k in range(a.k):
            os.makedirs(f"{work}/s{k}", exist_ok=True)
            _write_pseudo_backbone_pdb(ca_k[k][i, :Ln].numpy(), f"{work}/s{k}/{n}.pdb")
    if not a.skip_transfer:
        for s in sig:
            g = torch.Generator(device=dev).manual_seed(7)
            zn = z_true if s == 0 else F.layer_norm(z_true + s * torch.randn(z_true.shape, device=dev, generator=g), (G.D_LAT,))
            zmse_acc.setdefault(s, []).append(float((((zn - z_true) ** 2) * m).sum() / (m.sum() * G.D_LAT)))
            can = dec((zn * m).float(), mask).float().detach().cpu()
            for i, n in enumerate(nb):
                Ln = int(mask[i].sum()); _write_pseudo_backbone_pdb(can[i, :Ln].numpy(), f"{dirs[f'sig{s}']}/{n}.pdb")
    print(f"  {min(s0+a.bs, N)}/{N}", flush=True)

def mean_tm(d, tag):
    tm = _foldseek_tm(d, dirs["truth"], f"{work}/w_{tag}")
    v = [tm.get(n, 0.0) for n in names]
    return float(np.mean(v)), float(np.mean([x > 0.5 for x in v])), float(np.mean([n in tm for n in names])), v

out = {"ckpt": a.ckpt, "n": N, "k": a.k, "cfg_w": a.cfg_w, "steps": a.steps, "estimators": {}, "transfer": {}}
print(f"\n== A. point estimators ({N} proteins, K={a.k}, w={a.cfg_w})")
print(f"{'estimator':>12s} {'TM':>7s} {'TM>0.5':>7s} {'cov':>5s}")
sample_tms = []
for k in range(a.k):
    _, _, _, v = mean_tm(f"{work}/s{k}", f"s{k}"); sample_tms.append(v)
S = np.array(sample_tms)                                     # (K,N)
for tag in ("single", "medoid", "mean_ode", "mean_1step"):
    tm, frac, cov, v = mean_tm(dirs[tag], tag)
    out["estimators"][tag] = {"tm": tm, "tm_frac": frac, "coverage": cov, "per_protein": v}
    print(f"{tag:>12s} {tm:7.4f} {frac:7.2f} {cov:5.2f}", flush=True)
out["estimators"]["mean_of_K_draws"] = {"tm": float(S.mean())}
out["estimators"]["best_of_K"] = {"tm": float(S.max(0).mean())}
out["estimators"]["within_protein_sd"] = float(S.std(0, ddof=1).mean())
out["model_z_mse"] = {k: float(np.mean(v)) for k, v in est_zmse.items()}
print(f"{'mean of draws':>12s} {S.mean():7.4f}")
print(f"{'best-of-K':>12s} {S.max(0).mean():7.4f}   (oracle)")
print(f"within-protein sd across the {a.k} draws: {S.std(0, ddof=1).mean():.4f}")

if not a.skip_transfer:
    print(f"\n== B. latent-error transfer (true latent + sigma, projected, decoded)")
    print(f"{'sigma':>6s} {'z_mse':>8s} {'TM':>7s} {'TM>0.5':>7s}")
    for s in sig:
        tm, frac, cov, v = mean_tm(dirs[f"sig{s}"], f"sig{s}")
        zm = float(np.mean(zmse_acc[s]))   # measured mean squared latent error after the manifold projection
        out["transfer"][str(s)] = {"tm": tm, "tm_frac": frac, "z_mse": zm, "coverage": cov}
        print(f"{s:6.2f} {zm:8.3f} {tm:7.4f} {frac:7.2f}", flush=True)
json.dump(out, open(PROJECT / "notes" / f"gate26_{a.ckpt.replace('.ckpt','').replace('.pt','')}_{'named' if a.names_file else 'off'+str(a.offset)}.json", "w"), indent=1)
shutil.rmtree(work, ignore_errors=True)
print("ALLDONE")
