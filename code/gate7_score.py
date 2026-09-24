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
ap.add_argument("--h5", default=None, help="score against this HDF5 (val split) instead of the checkpoint's own data file; its esm2_emb must match the checkpoint's conditioner")
ap.add_argument("--names-file", default=None, help="score exactly these val protein names (one per line) instead of the permutation slice")
ap.add_argument("--offset", type=int, default=0,
                help="skip this many proteins of the seed-42 permutation; 0 = the gate set, which "
                     "overlaps the selection set. Use >=1000 for a held-out, selection-free score.")
a = ap.parse_args()
dev = torch.device("cuda")
ck = PROJECT / "data/phase1_dataset" / a.ckpt
if a.ckpt.endswith(".ckpt"):          # last_<label>.ckpt: full state, use the EMA weights
    st = torch.load(str(ck), weights_only=False, map_location="cpu")
    arch, weights = st["arch"], st["ema"]
    meta = {"epoch": st["epoch"], "tm": st["best_tm"], "cfg_w": None,
            "esm": "online" if st.get("mix_logits") is not None or st.get("online") else "stored",
            "layer_mix": st.get("mix_logits") is not None}
else:                                  # best_<label>.pt + sidecar
    meta = json.loads(open(str(ck) + ".meta.json").read())
    arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond") if k in meta}
    weights = torch.load(str(ck), weights_only=True)
if meta.get("model") == "PairFlowNet" or (a.ckpt.endswith(".ckpt") and st.get("model") == "PairFlowNet"):
    import gate10_pair_flow as G10
    G10.install()
    ex = meta.get("extra_arch") or st.get("extra_arch") if a.ckpt.endswith(".ckpt") else {k: meta[k] for k in ("d_pair", "n_pair_blocks", "pair_contact") if k in meta}
    net = G10.PairFlowNet(**arch, **ex).to(dev); print("pair-track model", ex)
else:
    net = G.LatentFlowNet(**arch).to(dev)
net.load_state_dict(weights); net.eval()
online = meta.get("esm", "stored") == "online"
embed = None
if online:
    import os as _o
    kind = meta.get("esm_kind") or (st.get("esm_kind") if a.ckpt.endswith(".ckpt") else None) or "esm2"
    path = meta.get("esm_path") or (st.get("esm_path") if a.ckpt.endswith(".ckpt") else None) or \
           _o.environ.get("ESM2_PATH", str(PROJECT / "data" / "esm2" / "esm2_t33_650M_UR50D"))
    embed = G.OnlineESM(path, layer_mix=bool(meta.get("layer_mix", False)), device=dev, kind=kind)
    ml = meta.get("mix_logits") if not a.ckpt.endswith(".ckpt") else st.get("mix_logits")
    if ml is not None:
        embed.mix_logits.data.copy_(torch.as_tensor(ml, device=dev))
    print(f"online {kind} from {path}, d_cond {embed.d_cond}, layer mix {embed.layer_mix}")
print(f"checkpoint {a.ckpt}: epoch {meta['epoch']}, train-time TM {meta['tm']:.3f} at w={meta['cfg_w']}, arch {arch}")

# identical protein selection to gate6_score_checkpoint.py
h5p = a.h5 or meta.get("h5_path") or (st.get("h5_path") if a.ckpt.endswith(".ckpt") else None) or os.environ.get("SCORE_H5", "")
vf = ProteinDatasetFAPE(h5p or H5_PATH, "val")
if h5p: print(f"embeddings/latents from {h5p}")
torch.manual_seed(42); idx = torch.randperm(len(vf))[a.offset:a.offset + a.n].tolist()
if a.names_file:
    want = [l.strip() for l in open(a.names_file) if l.strip()]; pos = {n: i for i, n in enumerate(vf.names)}
    idx = [pos[n] for n in want if n in pos]; print(f"scoring {len(idx)} named proteins from {a.names_file}")
class Holder:
    def __len__(self): return self.z.shape[0]
val = Holder(); N = len(idx)
d_emb = int(vf.h5["val"][vf.names[0]]["esm2_emb"].shape[1])
val.esm = torch.empty(N, G.MAX_LEN, d_emb, dtype=torch.float16); val.z = torch.empty(N, G.MAX_LEN, G.D_LAT)
val.mask = torch.empty(N, G.MAX_LEN, dtype=torch.bool); val.ca = torch.empty(N, G.MAX_LEN, 3); k = 0
for esm, z, ca, mask, _ in DataLoader(Subset(vf, idx), batch_size=20, num_workers=2):
    b = esm.shape[0]; val.esm[k:k+b] = esm.half(); val.z[k:k+b] = z; val.ca[k:k+b] = ca; val.mask[k:k+b] = mask; k += b
val.online = online
val.seqs = [str(vf.h5["val"][vf.names[i]].attrs["sequence"])[:G.MAX_LEN] for i in idx] if online else None
val.cond = lambda i: ([val.seqs[j] for j in i.tolist()] if online else val.esm[i])
dec = G.load_decoder(dev)
print(f"{N} val proteins ({'named list' if a.names_file else ('gate set' if a.offset == 0 else f'held-out, offset {a.offset}')}), {a.steps} ODE steps, K={a.k}")
print(f"{'w':>4s} {'TM':>6s} {'TM>0.5':>7s} {'TM>0.3':>7s} {'RMSD':>6s} {'best-of-K':>9s} {'cov':>5s} {'s':>4s}")
rows = {}
for w in [float(x) for x in a.cfg_w.split(",")]:
    t0 = time.perf_counter()
    r = G.evaluate_structures(net, dec, val, dev, n=N, bs=a.bs, n_steps=a.steps, cfg_w=w, n_samples=a.k, embed=embed)
    rows[w] = r
    print(f"{w:4.1f} {r['tm']:6.3f} {r['tm_frac']:7.2f} {'':7s} {r['rmsd']:6.2f} {r['tm_best_of_k']:9.3f} "
          f"{r['coverage']:5.2f} {time.perf_counter()-t0:4.0f}", flush=True)
json.dump({"ckpt": a.ckpt, "meta": meta, "n": N, "k": a.k, "steps": a.steps,
           "rows": {str(w): r for w, r in rows.items()}},
          open(PROJECT / "notes" / f"tm_{a.ckpt.replace('.pt', '').replace('.ckpt', '')}_sweep_{'named_' + os.path.basename(a.names_file).replace('.txt','') if a.names_file else 'off' + str(a.offset)}.json", "w"), indent=2)
