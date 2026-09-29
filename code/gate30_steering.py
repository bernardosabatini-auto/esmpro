"""Gate 30: steer the generative model. Three mechanisms, none needing retraining.

gate28 showed the DATA latent is not a space you can walk in: interpolating or perturbing a
valid latent leaves protein space (Ca-Ca 2.7 A, clashes up 30x), while sampling the model's own
prior gives well-formed novel structures. The conclusion is not "we cannot navigate", it is
"navigate through the flow, not around it". This tests three ways to do that.

  noise_interp  interpolate the INITIAL NOISE of the ODE, not the output latent. Every point on
                the path is a legitimate sample, so every point should decode to a real protein.
  guide_rg      classifier-style guidance: at each step form the one-step estimate
                x1_hat = x_t + (1-t) v, decode it WITH gradient (the ProteinAE decoder is
                differentiable, gate19), and push x1_hat down the gradient of an objective on
                the decoded coordinates. Objective: hit a target radius of gyration.
  guide_contact same, objective = bring a chosen residue pair within 8 A (a design constraint).
  inpaint       fix a contiguous motif to a real protein's latent and sample the rest, the
                flow-matching form of inpainting: at time t the fixed positions are held at
                (1-t) * noise + t * z_motif.

For each we report whether the objective actually moved AND whether the structure is still
protein-like, because a steering knob that destroys geometry is worthless.

  python gate30_steering.py --ckpt last_pf_840M_p128x8_long512_1p3M.ckpt --n 32
"""
import os, sys, json, argparse, tempfile, shutil
import numpy as np, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
from gate6_fape_train import ProteinDatasetFAPE, PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=32); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--h5", default="dataset_exp_val_esmc.h5"); ap.add_argument("--names-file", default="notes/exp_val_names.txt")
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--guide-scales", default="0,2,10,40")
ap.add_argument("--guide-norm", action="store_true", help="normalise the guidance gradient to unit RMS per sample, so a scale means the same displacement whatever the objective (a single-pair contact concentrates its gradient on 2 of ~150 residues; Rg spreads it over all)")
ap.add_argument("--rg-target-frac", type=float, default=0.80, help="target Rg as a fraction of the unguided sample's Rg")
ap.add_argument("--alphas", default="0.25,0.5,0.75")
ap.add_argument("--motif-frac", type=float, default=0.3)
ap.add_argument("--out-dir", default="gate30")
a = ap.parse_args(); dev = torch.device("cuda")
D = PROJECT / "data/phase1_dataset"; OUT = str(PROJECT / "notes" / a.out_dir); os.makedirs(OUT, exist_ok=True)

# ---- model ----
ck = D / a.ckpt
st = torch.load(str(ck), weights_only=False, map_location="cpu") if a.ckpt.endswith(".ckpt") else None
if st is not None:
    arch, weights, mtype, ex = st["arch"], st["ema"], st.get("model"), dict(st.get("extra_arch") or {})
else:
    meta = json.loads(open(str(ck) + ".meta.json").read())
    arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond") if k in meta}
    weights = torch.load(str(ck), weights_only=True); mtype = meta.get("model")
    ex = {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact", "pair_fused") if k in meta}
pair_net = mtype in ("PairFlowNet", "_Net")
if pair_net:
    import gate10_pair_flow as G10
    G10.install(); net = G10.PairFlowNet(**arch, **ex).to(dev)
else:
    net = G.LatentFlowNet(**arch).to(dev)
net.load_state_dict(G.extend_pos_table(G.adapt_state_dict(weights, net.state_dict().keys()), net.state_dict())); net.eval()
for p in net.parameters(): p.requires_grad_(False)
dec = G.load_decoder(dev)
for p in dec.parameters(): p.requires_grad_(False)
print(f"{a.ckpt}: pair={pair_net}", flush=True)

vf = ProteinDatasetFAPE(str(D / a.h5), "val", max_len=G.MAX_LEN)
want = [l.strip() for l in open(PROJECT / a.names_file) if l.strip()]
pos = {n: i for i, n in enumerate(vf.names)}
idx = [pos[n] for n in want if n in pos]
idx = [i for i in idx if int(vf.h5["val"][vf.names[i]].attrs["n_residues"]) <= a.max_len][: a.n]
print(f"{len(idx)} proteins <= {a.max_len} residues", flush=True)

def proj(z, mask): return F.layer_norm(z, (G.D_LAT,)) * mask.unsqueeze(-1)

def geom(ca):
    n = len(ca)
    d1 = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    dm = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    iu = np.triu_indices(n, k=3)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=-1) if n > 3 else np.array([0.0])
    d4 = np.linalg.norm(ca[4:] - ca[:-4], axis=-1) if n > 4 else np.array([0.0])
    m = min(len(d3), len(d4))
    helix = float(((np.abs(d3[:m] - 5.1) < 1.0) & (np.abs(d4[:m] - 6.2) < 1.2)).mean()) if m else 0.0
    return {"ca_ca": float(d1.mean()), "ca_ca_sd": float(d1.std()),
            "in_range": float(((d1 > 3.7) & (d1 < 3.9)).mean()),
            "clash": float((dm[iu] < 4.0).mean()), "rg": rg, "helix": helix}

def rg_of(ca, mask):
    """Differentiable radius of gyration per sample. ca (B,L,3), mask (B,L)."""
    m = mask.unsqueeze(-1).float(); n = m.sum(1).clamp(min=1.0)
    c = (ca * m).sum(1, keepdim=True) / n.unsqueeze(1)
    return torch.sqrt((((ca - c) ** 2).sum(-1, keepdim=True) * m).sum(1) / n).squeeze(-1)

@torch.no_grad()
def _v(x, t, esm, mask, x_sc, drop_all):
    B = mask.shape[0]
    d = torch.ones(B, dtype=torch.bool, device=dev) if drop_all else None
    return net(x, t, esm, mask, d, x_sc)

def sample(esm, mask, x0=None, gen=None, guide=None, scale=0.0, fixed=None, drop_all=True):
    """Euler ODE from the model's prior. guide: callable(ca, mask) -> scalar objective to MINIMISE,
    applied to the decoded one-step estimate with gradient. fixed: (z_target, keep_mask)."""
    B, L = mask.shape
    x = torch.randn(B, L, G.D_LAT, device=dev, generator=gen) if x0 is None else x0.clone()
    eps_fix = torch.randn(B, L, G.D_LAT, device=dev, generator=gen) if fixed is not None else None
    x_sc = None
    ts = torch.linspace(0, 1, a.steps + 1, device=dev)
    for i in range(a.steps):
        t = ts[i].expand(B); dt = ts[i + 1] - ts[i]
        if fixed is not None:                       # inpainting: hold the motif on its own path
            z_t, keep = fixed
            # one noise draw for the motif's whole trajectory: re-drawing it each step made the
            # fixed region incoherent from step to step (6 % designable in the first attempt)
            x = torch.where(keep.unsqueeze(-1), (1 - ts[i]) * eps_fix + ts[i] * z_t, x)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v = _v(x, t, esm, mask, x_sc, drop_all).float()
        if guide is not None and scale > 0:
            x1 = (x + (1 - ts[i]) * v).detach().requires_grad_(True)
            with torch.enable_grad():
                ca = dec(proj(x1, mask).float(), mask)
                obj = guide(ca, mask).sum()
                g, = torch.autograd.grad(obj, x1)
            if a.guide_norm:                         # equalise objectives: unit-RMS displacement per step
                rms = g.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-8).view(-1, 1, 1)
                g = g / rms
            v = v - scale * g * (1 - ts[i])          # push the endpoint estimate downhill
        if getattr(net, "self_cond", False): x_sc = x + (1 - ts[i]) * v
        x = x + v * dt
    return proj(x, mask)

def slerp(u, w, al):
    un = F.normalize(u.flatten(1), dim=-1); wn = F.normalize(w.flatten(1), dim=-1)
    dot = (un * wn).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    om = torch.acos(dot); so = torch.sin(om)
    out = (torch.sin((1 - al) * om) / so) * u.flatten(1) + (torch.sin(al * om) / so) * w.flatten(1)
    return out.view_as(u)

alphas = [float(x) for x in a.alphas.split(",")]
scales = [float(x) for x in a.guide_scales.split(",")]
tags = ["prior"] + [f"noiseinterp{al:g}" for al in alphas] + [f"rg_s{s:g}" for s in scales] + \
       [f"contact_s{s:g}" for s in scales] + ["inpaint"]
for t in tags: os.makedirs(f"{OUT}/{t}", exist_ok=True)
os.makedirs(f"{OUT}/motif_src", exist_ok=True)
rows = {t: [] for t in tags}; extra = {t: [] for t in tags}

for s0 in range(0, len(idx), a.bs):
    sub = idx[s0:s0 + a.bs]
    esm, z_true, ca_true, mask, _ = next(iter(DataLoader(Subset(vf, sub), batch_size=len(sub))))
    esm, z_true, mask = esm.to(dev), z_true.to(dev), mask.to(dev)
    nm = [vf.names[i] for i in sub]; B, L = mask.shape
    g1 = torch.Generator(device=dev).manual_seed(11 + s0); g2 = torch.Generator(device=dev).manual_seed(999 - s0)
    x0a = torch.randn(B, L, G.D_LAT, device=dev, generator=g1)
    x0b = torch.randn(B, L, G.D_LAT, device=dev, generator=g2)

    z_prior = sample(esm, mask, x0=x0a)
    ca_prior = dec(z_prior.float(), mask).float().detach().cpu()
    rg0 = rg_of(dec(z_prior.float(), mask).float(), mask).detach()

    out = {"prior": (z_prior, mask)}
    for al in alphas:
        out[f"noiseinterp{al:g}"] = (sample(esm, mask, x0=slerp(x0a, x0b, al)), mask)
    tgt = a.rg_target_frac * rg0
    out.update({f"rg_s{s:g}": (sample(esm, mask, x0=x0a, guide=lambda ca, m: (rg_of(ca, m) - tgt) ** 2, scale=s), mask) for s in scales})
    # contact objective: first and last quarter-point residues within 8 A
    ia = (mask.sum(1) // 4).long(); ib = (3 * mask.sum(1) // 4).long()
    def contact_obj(ca, m):
        bi = torch.arange(ca.shape[0], device=ca.device)
        d = torch.norm(ca[bi, ia] - ca[bi, ib], dim=-1)
        return F.relu(d - 8.0) ** 2
    out.update({f"contact_s{s:g}": (sample(esm, mask, x0=x0a, guide=contact_obj, scale=s), mask) for s in scales})
    # inpaint: hold a contiguous motif of the real latent
    keep = torch.zeros_like(mask)
    for i in range(B):
        Ln = int(mask[i].sum()); k = max(8, int(a.motif_frac * Ln)); st_ = (Ln - k) // 2
        keep[i, st_:st_ + k] = True
    out["inpaint"] = (sample(esm, mask, x0=x0a, fixed=(proj(z_true, mask), keep)), mask)

    for t, (z, mk) in out.items():
        ca = dec(z.float(), mk).float().detach().cpu()
        for i, n in enumerate(nm):
            Ln = int(mk[i].sum())
            if Ln < 30: continue
            c = ca[i, :Ln].numpy()
            _write_pseudo_backbone_pdb(c, f"{OUT}/{t}/{n}.pdb")
            gm = geom(c); gm["name"] = n
            bi = int(ia[i]); bj = int(ib[i])
            gm["contact_d"] = float(np.linalg.norm(c[bi] - c[bj])) if max(bi, bj) < Ln else -1.0
            gm["rg_target"] = float(tgt[i]); gm["rg0"] = float(rg0[i])
            rows[t].append(gm)
    for i, n in enumerate(nm):
        Ln = int(mask[i].sum()); _write_pseudo_backbone_pdb(ca_true[i, :Ln].numpy(), f"{OUT}/motif_src/{n}.pdb")
    print(f"  {min(s0+a.bs, len(idx))}/{len(idx)}", flush=True)

W = tempfile.mkdtemp(prefix="g30_")
tm_src = {t: _foldseek_tm(f"{OUT}/{t}", f"{OUT}/motif_src", f"{W}/{t}") for t in tags}
shutil.rmtree(W, ignore_errors=True)
print(f"\n{'mode':>16s} {'n':>4s} {'CaCa':>6s} {'in-range':>8s} {'clash':>6s} {'Rg':>6s} {'Rg/target':>9s} {'helix':>6s} {'contact A':>9s} {'TM to real':>10s}")
summary = {}
for t in tags:
    R = rows[t]
    if not R: continue
    f = lambda k: float(np.mean([r[k] for r in R]))
    tmv = float(np.mean([tm_src[t].get(r["name"], 0.0) for r in R]))
    summary[t] = {"n": len(R), "ca_ca": f("ca_ca"), "in_range": f("in_range"), "clash": f("clash"),
                  "rg": f("rg"), "rg_over_target": float(np.mean([r["rg"] / max(r["rg_target"], 1e-6) for r in R])),
                  "helix": f("helix"), "contact_d": f("contact_d"), "tm_to_real": tmv}
    s = summary[t]
    print(f"{t:>16s} {s['n']:4d} {s['ca_ca']:6.2f} {s['in_range']:8.2f} {s['clash']:6.3f} {s['rg']:6.1f} "
          f"{s['rg_over_target']:9.2f} {s['helix']:6.2f} {s['contact_d']:9.1f} {s['tm_to_real']:10.3f}", flush=True)
json.dump({"ckpt": a.ckpt, "summary": summary, "per_protein": rows}, open(PROJECT / "notes" / "gate30_steering.json", "w"), indent=1)
print("ALLDONE")
