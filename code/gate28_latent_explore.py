"""Gate 28: move around in the 8-dim latent and ask whether what comes out is a protein.

Four ways of producing a latent, all decoded by the frozen ProteinAE decoder:
  real     the true latent of a held-out experimental structure (positive control)
  uncond   sample the flow with the conditioning DROPPED (the CFG null token and null pair),
           i.e. the model's unconditional prior over structures at a chosen length
  interp   spherical interpolation between the true latents of two real proteins of the same
           length, re-projected onto the manifold at each step
  noise    a real latent plus sigma * noise, re-projected

Then measure whether the decoded Ca trace is protein-like, WITHOUT any reference:
  consecutive Ca-Ca distance (truth 3.80 +- 0.03 A) and the fraction inside [3.7, 3.9]
  clashes: residue pairs more than 2 apart in sequence closer than 4 A
  radius of gyration against the 2.2 * N^0.38 expectation for a compact globule
  a Ca-only secondary-structure proxy: helix if d(i,i+3) ~ 5.1 and d(i,i+4) ~ 6.2, strand if
  d(i,i+3) ~ 10.0 and d(i,i+4) ~ 13.0; "coil fraction" high means an unstructured blob
  nearest-neighbour TM to the 626 real held-out structures (is it even in protein space)

Writes every generated structure to notes/gate28_<tag>/<name>.pdb so the designability loop
(inverse fold -> ESMFold2 -> TM back) can run on them in gate29.

  python gate28_latent_explore.py --ckpt last_pf_840M_p128x8_long512_1p3M.ckpt --n 60
"""
import os, sys, json, argparse, shutil, tempfile
import numpy as np, torch, torch.nn.functional as F, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
from gate6_fape_train import ProteinDatasetFAPE, PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=60, help="proteins per mode")
ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg-w", type=float, default=2.0)
ap.add_argument("--h5", default="dataset_exp_val_esmc.h5"); ap.add_argument("--names-file", default="notes/exp_val_names.txt")
ap.add_argument("--sigmas", default="0.2,0.4"); ap.add_argument("--interp-alphas", default="0.25,0.5,0.75")
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--out-dir", default=None)
a = ap.parse_args(); dev = torch.device("cuda")
D = PROJECT / "data/phase1_dataset"; OUT = a.out_dir or str(PROJECT / "notes" / "gate28")
os.makedirs(OUT, exist_ok=True)

# ---------------- model ----------------
ck = D / a.ckpt; st = None
if a.ckpt.endswith(".ckpt"):
    st = torch.load(str(ck), weights_only=False, map_location="cpu")
    arch, weights = st["arch"], st["ema"]; mtype = st.get("model"); ex = dict(st.get("extra_arch") or {})
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
dec = G.load_decoder(dev)
print(f"{a.ckpt}: pair={pair_net} {ex if pair_net else ''}", flush=True)

# ---------------- data ----------------
vf = ProteinDatasetFAPE(str(D / a.h5), "val", max_len=G.MAX_LEN)
want = [l.strip() for l in open(PROJECT / a.names_file) if l.strip()]
pos = {n: i for i, n in enumerate(vf.names)}
idx = [pos[n] for n in want if n in pos][: a.n * 2]
print(f"{len(idx)} reference proteins from {a.h5}", flush=True)

def proj(z, mask):
    return F.layer_norm(z, (G.D_LAT,)) * mask.unsqueeze(-1)

@torch.no_grad()
def sample_uncond(esm, mask, n_steps, gen):
    """Euler ODE with the conditioning permanently dropped: the model's own prior."""
    B, L = mask.shape
    x = torch.randn(B, L, G.D_LAT, device=dev, generator=gen); x_sc = None
    ts = torch.linspace(0, 1, n_steps + 1, device=dev)
    ones = torch.ones(B, dtype=torch.bool, device=dev)
    for i in range(n_steps):
        t = ts[i].expand(B); dt = ts[i + 1] - ts[i]
        v = net(x, t, esm, mask, ones, x_sc)
        if getattr(net, "self_cond", False): x_sc = x + (1 - ts[i]) * v
        x = x + v * dt
    return proj(x, mask)

def slerp(a_, b_, w):
    """Per-residue spherical interpolation; the latents live on a sphere of radius sqrt(8)."""
    an = F.normalize(a_, dim=-1); bn = F.normalize(b_, dim=-1)
    dot = (an * bn).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    om = torch.acos(dot); so = torch.sin(om)
    return (torch.sin((1 - w) * om) / so) * a_ + (torch.sin(w * om) / so) * b_

# ---------------- geometry ----------------
def geom(ca):
    """ca: (n,3) numpy. Reference-free protein-likeness."""
    n = len(ca)
    d1 = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    dm = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    iu = np.triu_indices(n, k=3)
    clash = float((dm[iu] < 4.0).mean())
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
    rg_exp = 2.2 * n ** 0.38
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=-1) if n > 3 else np.array([0.0])
    d4 = np.linalg.norm(ca[4:] - ca[:-4], axis=-1) if n > 4 else np.array([0.0])
    m = min(len(d3), len(d4))
    helix = float(((np.abs(d3[:m] - 5.1) < 1.0) & (np.abs(d4[:m] - 6.2) < 1.2)).mean()) if m else 0.0
    strand = float(((d3[:m] > 9.0) & (d4[:m] > 11.5)).mean()) if m else 0.0
    return {"n": n, "ca_ca_mean": float(d1.mean()), "ca_ca_sd": float(d1.std()),
            "ca_ca_in_range": float(((d1 > 3.7) & (d1 < 3.9)).mean()),
            "clash_frac": clash, "rg": rg, "rg_over_expected": rg / rg_exp,
            "helix_frac": helix, "strand_frac": strand, "coil_frac": 1.0 - helix - strand}

# ---------------- generate ----------------
modes = {}
sig = [float(x) for x in a.sigmas.split(",")]; alphas = [float(x) for x in a.interp_alphas.split(",")]
tags = ["real", "uncond"] + [f"noise{s:g}" for s in sig] + [f"interp{al:g}" for al in alphas]
for t in tags: os.makedirs(f"{OUT}/{t}", exist_ok=True)
rows = {t: [] for t in tags}; seqs = {t: {} for t in tags}; lat = {t: {} for t in tags}
done = 0
for s0 in range(0, min(len(idx), a.n * 2), a.bs):
    sub = idx[s0:s0 + a.bs]
    if len(sub) < 2: break
    esm, z_true, ca_true, mask, _ = next(iter(DataLoader(Subset(vf, sub), batch_size=len(sub))))
    esm, z_true, mask = esm.to(dev), z_true.to(dev), mask.to(dev)
    nm = [vf.names[i] for i in sub]
    sq = [str(vf.h5["val"][n].attrs["sequence"])[:G.MAX_LEN] for n in nm]
    gen = torch.Generator(device=dev).manual_seed(7 + s0)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = {"real": proj(z_true, mask)}
        out["uncond"] = sample_uncond(esm, mask, a.steps, gen)
        for s in sig:
            out[f"noise{s:g}"] = proj(z_true + s * torch.randn(z_true.shape, device=dev, generator=gen), mask)
        perm = torch.roll(torch.arange(len(sub), device=dev), 1)      # pair each protein with its neighbour
        for al in alphas:
            zi = slerp(z_true.float(), z_true[perm].float(), al)
            mm = mask & mask[perm]                                     # only the shared length
            out[f"interp{al:g}"] = proj(zi, mm)
        cas = {t: dec(v.float(), mask if not t.startswith("interp") else (mask & mask[perm])).float().cpu() for t, v in out.items()}
    for t, ca in cas.items():
        mk = (mask & mask[perm]) if t.startswith("interp") else mask
        for i, n in enumerate(nm):
            Ln = int(mk[i].sum())
            if Ln < 30: continue
            c = ca[i, :Ln].numpy()
            _write_pseudo_backbone_pdb(c, f"{OUT}/{t}/{n}.pdb")
            lat[t][n] = out[t][i, :Ln].float().cpu().numpy().astype(np.float32)   # the latent that produced it, for the designability loop
            g = geom(c); g["name"] = n; rows[t].append(g)
            seqs[t][n] = sq[i][:Ln]
    done += len(sub)
    print(f"  {done} proteins", flush=True)
    if done >= a.n: break

# ---------------- nearest real neighbour ----------------
W = tempfile.mkdtemp(prefix="g28_")
nn_tm = {}
for t in tags:
    tm = _foldseek_tm(f"{OUT}/{t}", f"{OUT}/real", f"{W}/{t}")
    nn_tm[t] = tm
shutil.rmtree(W, ignore_errors=True)

print(f"\n{'mode':>10s} {'n':>4s} {'CaCa':>6s} {'sd':>5s} {'in3.7-3.9':>9s} {'clash':>6s} {'Rg/exp':>7s} {'helix':>6s} {'strand':>6s} {'TM to its own real':>18s}")
summary = {}
for t in tags:
    R = rows[t]
    if not R: continue
    f = lambda k: float(np.mean([r[k] for r in R]))
    tmv = [nn_tm[t].get(r["name"], 0.0) for r in R]
    summary[t] = {"n": len(R), "ca_ca_mean": f("ca_ca_mean"), "ca_ca_sd": f("ca_ca_sd"),
                  "ca_ca_in_range": f("ca_ca_in_range"), "clash_frac": f("clash_frac"),
                  "rg_over_expected": f("rg_over_expected"), "helix_frac": f("helix_frac"),
                  "strand_frac": f("strand_frac"), "tm_to_source": float(np.mean(tmv)),
                  "tm_cov": float(np.mean([r["name"] in nn_tm[t] for r in R]))}
    s = summary[t]
    print(f"{t:>10s} {s['n']:4d} {s['ca_ca_mean']:6.2f} {s['ca_ca_sd']:5.2f} {s['ca_ca_in_range']:9.2f} "
          f"{s['clash_frac']:6.3f} {s['rg_over_expected']:7.2f} {s['helix_frac']:6.2f} {s['strand_frac']:6.2f} {s['tm_to_source']:18.3f}", flush=True)
for t in tags:
    if lat[t]: np.savez_compressed(f"{OUT}/{t}/latents.npz", **lat[t])
json.dump({"ckpt": a.ckpt, "summary": summary, "per_protein": rows,
           "seqs": {t: seqs[t] for t in tags}},
          open(PROJECT / "notes" / "gate28_latent_explore.json", "w"), indent=1)
print(f"\nstructures in {OUT}/<mode>/; run gate29 for the designability loop")
print("ALLDONE")
