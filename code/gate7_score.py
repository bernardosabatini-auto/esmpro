"""Score a Gate 7 latent flow checkpoint on the SAME validation proteins as
gate6_score_checkpoint.py (seed-42 permutation of the val split), so numbers
compare directly with the inherited 0.427 / 32% / 11.56 A. Sweeps the guidance
weight and reports best-of-K sampling. GPU.

  python gate7_score.py --ckpt best_lf_base.pt --n 100 --cfg-w 1,1.5,2,3,4 --k 8
"""
import os, sys, json, argparse, time
import numpy as np, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
from gate6_fape_train import ProteinDatasetFAPE, H5_PATH, PROJECT
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--cfg-w", type=str, default="1,1.5,2,3,4"); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--bs", type=int, default=20)
a = ap.parse_args()
dev = torch.device("cuda")
ck = PROJECT / "data/phase1_dataset" / a.ckpt
meta = json.loads(open(str(ck) + ".meta.json").read())
arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond")}
net = G.LatentFlowNet(**arch).to(dev); net.load_state_dict(torch.load(str(ck), weights_only=True)); net.eval()
print(f"checkpoint {a.ckpt}: epoch {meta['epoch']}, train-time TM {meta['tm']:.3f} at w={meta['cfg_w']}, arch {arch}")

# identical protein selection to gate6_score_checkpoint.py
vf = ProteinDatasetFAPE(H5_PATH, "val")
torch.manual_seed(42); idx = torch.randperm(len(vf))[:a.n].tolist()
class Holder: pass
val = Holder(); N = len(idx)
val.esm = torch.empty(N, G.MAX_LEN, G.D_ESM, dtype=torch.float16); val.z = torch.empty(N, G.MAX_LEN, G.D_LAT)
val.mask = torch.empty(N, G.MAX_LEN, dtype=torch.bool); val.ca = torch.empty(N, G.MAX_LEN, 3); k = 0
for esm, z, ca, mask, _ in DataLoader(Subset(vf, idx), batch_size=20, num_workers=2):
    b = esm.shape[0]; val.esm[k:k+b] = esm.half(); val.z[k:k+b] = z; val.ca[k:k+b] = ca; val.mask[k:k+b] = mask; k += b
val.__len__ = lambda: N
G.RamSplit.__len__  # (unused) keep import honest
dec = G.load_decoder(dev)
print(f"{N} val proteins (gate set), {a.steps} ODE steps, K={a.k}")
print(f"{'w':>4s} {'TM':>6s} {'TM>0.5':>7s} {'TM>0.3':>7s} {'RMSD':>6s} {'best-of-K':>9s} {'cov':>5s} {'s':>4s}")
rows = {}
for w in [float(x) for x in a.cfg_w.split(",")]:
    t0 = time.perf_counter()
    r = G.evaluate_structures(net, dec, val, dev, n=N, bs=a.bs, n_steps=a.steps, cfg_w=w, n_samples=a.k)
    rows[w] = r
    print(f"{w:4.1f} {r['tm']:6.3f} {r['tm_frac']:7.2f} {'':7s} {r['rmsd']:6.2f} {r['tm_best_of_k']:9.3f} "
          f"{r['coverage']:5.2f} {time.perf_counter()-t0:4.0f}", flush=True)
json.dump({"ckpt": a.ckpt, "meta": meta, "n": N, "k": a.k, "steps": a.steps,
           "rows": {str(w): r for w, r in rows.items()}},
          open(PROJECT / "notes" / f"tm_{a.ckpt.replace('.pt', '')}_sweep.json", "w"), indent=2)
