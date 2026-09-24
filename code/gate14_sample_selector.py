"""Gate 14: training-free sample selection for best-of-K inference.

Best-of-8 is worth +0.03-0.06 TM but needs the answer to pick the sample.
This measures how much of that oracle gap three signals that need NO answer
and NO training recover, per protein, on any evaluation set:

  agree    mean Kabsch-superposed TM-like score of a sample to the other K-1
           samples (consensus / medoid).  Higher = more self-consistent.
  cycle    re-encode the decoded backbone with the frozen ProteinAE encoder
           and compare with the sampled latent (negative masked MSE).
           Higher = the sample decodes to a structure the encoder maps back
           to it, i.e. it sits on the latent manifold.
  dspread  decode the same latent twice (the decoder is a 3-step flow from
           noise) and score the two decodes against each other (TM-like).
           Higher = the decoder is confident about that latent.
  combo    mean of the three within-protein ranks.

Reports, per set: mean TM of the first sample, mean over all K samples
(= expected single-shot), oracle best-of-K, each selector's pick, the
fraction of the oracle gap recovered, and the mean within-protein Spearman
correlation of each signal with the true TM. Foldseek TM (exhaustive) per
sample, coverage printed.

  python gate14_sample_selector.py --ckpt best_pf_459M_esmc_afdb.pt --offset 1000 --n 100 --k 8
  python gate14_sample_selector.py --ckpt best_pf_459M_esmc_afdb.pt --h5 .../dataset_casp_esmc.h5 --names-file notes/casp_domains_le256.txt
"""
import os, sys, json, time, argparse, tempfile, shutil
import numpy as np, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G
from gate6_fape_train import ProteinDatasetFAPE, H5_PATH, PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--cfg-w", type=float, default=2.0); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--bs", type=int, default=20); ap.add_argument("--h5", default=None); ap.add_argument("--names-file", default=None)
ap.add_argument("--offset", type=int, default=1000); ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = torch.device("cuda")
ck = PROJECT / "data/phase1_dataset" / a.ckpt
meta = json.loads(open(str(ck) + ".meta.json").read())
arch = {k: meta[k] for k in ("d_model", "n_layers", "n_heads", "dropout", "self_cond", "d_cond") if k in meta}
weights = torch.load(str(ck), weights_only=True)
if meta.get("model") == "PairFlowNet":
    import gate10_pair_flow as G10; G10.install()
    net = G10.PairFlowNet(**arch, **meta.get("extra_arch", {})).to(dev)
else:
    net = G.LatentFlowNet(**arch).to(dev)
net.load_state_dict(weights); net.eval()
assert meta.get("esm", "stored") == "stored", "online-ESM checkpoints not supported here; use a stored-embedding one"
h5p = a.h5 or meta.get("h5_path") or H5_PATH
vf = ProteinDatasetFAPE(h5p, "val", max_len=G.MAX_LEN)
torch.manual_seed(42); idx = torch.randperm(len(vf))[a.offset:a.offset + a.n].tolist()
if a.names_file:
    want = [l.strip() for l in open(a.names_file) if l.strip()]; pos = {n: i for i, n in enumerate(vf.names)}
    idx = [pos[n] for n in want if n in pos]
N = len(idx); pnames = [vf.names[i] for i in idx]
d_emb = int(vf.h5["val"][vf.names[0]]["esm2_emb"].shape[1])
assert int(arch.get("d_cond", 1280)) == d_emb, f"conditioner mismatch {arch.get('d_cond')} vs {d_emb}"
esm_all = torch.empty(N, G.MAX_LEN, d_emb, dtype=torch.float16); z_all = torch.empty(N, G.MAX_LEN, G.D_LAT)
mask_all = torch.empty(N, G.MAX_LEN, dtype=torch.bool); ca_all = torch.empty(N, G.MAX_LEN, 3); kk = 0
for esm, z, ca, mask, _ in DataLoader(Subset(vf, idx), batch_size=20, num_workers=2):
    b = esm.shape[0]; esm_all[kk:kk+b] = esm.half(); z_all[kk:kk+b] = z; ca_all[kk:kk+b] = ca; mask_all[kk:kk+b] = mask; kk += b
dec = G.load_decoder(dev); ae = dec.ae_model; fm = dec.fm
print(f"{a.ckpt}: {N} proteins from {os.path.basename(h5p)} ({'named' if a.names_file else f'offset {a.offset}'}), K={a.k}, w={a.cfg_w}, {a.steps} steps", flush=True)

def kabsch_tm(P, Q, mask):
    """Batched TM-like score after Kabsch (RMSD-optimal) superposition. P,Q (M,N,3), mask (M,N) -> (M,)."""
    m = mask.unsqueeze(-1).double(); L = m.sum(1)                      # (M,1)
    p = P.double() * m; q = Q.double() * m
    p = p - p.sum(1, keepdim=True) / L.unsqueeze(-1); q = q - q.sum(1, keepdim=True) / L.unsqueeze(-1)
    p = p * m; q = q * m
    H = p.transpose(1, 2) @ q                                          # (M,3,3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    pr = (R @ p.transpose(1, 2)).transpose(1, 2)
    dist2 = ((pr - q) ** 2).sum(-1)                                    # (M,N)
    d0 = (1.24 * (L.squeeze(-1) - 15).clamp(min=1) ** (1 / 3) - 1.8).clamp(min=0.5)
    tm = (1 / (1 + dist2 / d0.unsqueeze(-1) ** 2) * mask.double()).sum(1) / L.squeeze(-1)
    return tm.float()

@torch.no_grad()
def reencode(bb_nm, mask):
    """bb_nm: decoded backbone (B,N,4,3) in nm. Returns encoder latent (B,N,8)."""
    B, Nn = mask.shape
    x_1 = bb_nm.reshape(B, Nn * 4, 3)
    cmf = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, Nn * 4)
    x_1 = fm._mask_and_zero_com(x_1, cmf)
    batch = {"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": Nn}
    return ae.encoder(batch)["single_repr"].float()

work = tempfile.mkdtemp(prefix="sel_"); K = a.k
for k in range(K): os.makedirs(f"{work}/gt{k}"); os.makedirs(f"{work}/pr{k}")
sig = {s: np.zeros((N, K)) for s in ("agree", "cycle", "dspread")}
gen = torch.Generator(device="cuda").manual_seed(0); t0 = time.perf_counter()
with torch.no_grad():
    for s in range(0, N, a.bs):
        sl = slice(s, min(s + a.bs, N)); B = sl.stop - sl.start
        esm = esm_all[sl].to(dev); mask = mask_all[sl].to(dev); Lr = mask.sum(1)
        cas, zs = [], []
        for k in range(K):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = G.sample(net, esm, mask, a.steps, a.cfg_w, gen)
            z = z.float()
            ca1, bb1 = dec(z, mask, return_backbone=True)             # A and A
            ca2 = dec(z, mask)                                         # second decode, different decoder noise
            z_re = reencode(bb1 / 10.0, mask)
            m = mask.unsqueeze(-1).float()
            sig["cycle"][sl, k] = (-((z_re - z) ** 2 * m).sum((1, 2)) / (m.sum((1, 2)) * G.D_LAT)).cpu().numpy()
            sig["dspread"][sl, k] = kabsch_tm(ca1, ca2, mask).cpu().numpy()
            cas.append(ca1); zs.append(z)
            for b in range(B):
                L_ = int(Lr[b]); nm = f"p{s+b:05d}"
                _write_pseudo_backbone_pdb(ca_all[s+b, :L_].numpy(), f"{work}/gt{k}/{nm}.pdb")
                _write_pseudo_backbone_pdb(ca1[b, :L_].cpu().numpy(), f"{work}/pr{k}/{nm}.pdb")
        for k in range(K):
            others = [kabsch_tm(cas[k], cas[j], mask) for j in range(K) if j != k]
            sig["agree"][sl, k] = torch.stack(others, 0).mean(0).cpu().numpy()
        print(f"  {sl.stop}/{N}  {time.perf_counter()-t0:.0f}s", flush=True)
names = [f"p{i:05d}" for i in range(N)]
tm = np.full((N, K), np.nan)
for k in range(K):
    d = _foldseek_tm(f"{work}/pr{k}", f"{work}/gt{k}", f"{work}/tm{k}")
    for i, nm in enumerate(names):
        if nm in d: tm[i, k] = d[nm]
shutil.rmtree(work, ignore_errors=True)
cov = np.isfinite(tm).mean(); tm0 = np.nan_to_num(tm, nan=0.0)

def rank(x):  # within-protein ranks, higher signal -> higher rank
    return np.argsort(np.argsort(x, 1), 1).astype(float)
sig["combo"] = (rank(sig["agree"]) + rank(sig["cycle"]) + rank(sig["dspread"])) / 3
from scipy.stats import spearmanr
res = {"first": float(tm0[:, 0].mean()), "mean_sample": float(tm0.mean()), "oracle": float(tm0.max(1).mean()),
       "frac_gt05_first": float((tm0[:, 0] > 0.5).mean()), "frac_gt05_oracle": float((tm0.max(1) > 0.5).mean()), "coverage": float(cov)}
gap = res["oracle"] - res["mean_sample"]
print(f"\nfirst {res['first']:.3f}  mean-of-{K} {res['mean_sample']:.3f}  oracle {res['oracle']:.3f}  (gap {gap:.3f}); TM>0.5 first {res['frac_gt05_first']:.2f} oracle {res['frac_gt05_oracle']:.2f}; coverage {cov:.2f}")
print(f"{'selector':9s} {'TM':>6s} {'TM>0.5':>7s} {'gap rec.':>8s} {'spearman':>8s}")
for sname, S in sig.items():
    pick = S.argmax(1); chosen = tm0[np.arange(N), pick]
    rho = np.nanmean([spearmanr(S[i], tm0[i]).correlation for i in range(N) if np.std(tm0[i]) > 1e-6])
    res[sname] = {"tm": float(chosen.mean()), "frac_gt05": float((chosen > 0.5).mean()), "gap_recovered": float((chosen.mean() - res["mean_sample"]) / max(gap, 1e-9)), "spearman": float(rho)}
    print(f"{sname:9s} {chosen.mean():6.3f} {(chosen > 0.5).mean():7.2f} {res[sname]['gap_recovered']:8.2f} {rho:8.2f}", flush=True)
# a "top-2 by combo then agree" style check: does picking the best of the top-half by combo help?
tag = a.tag or (os.path.basename(a.names_file).replace(".txt", "") if a.names_file else f"off{a.offset}")
json.dump({"ckpt": a.ckpt, "set": tag, "n": N, "k": K, "cfg_w": a.cfg_w, "steps": a.steps, "summary": res, "names": pnames,
           "tm": tm.tolist(), "signals": {k: v.tolist() for k, v in sig.items()}},
          open(PROJECT / "notes" / f"gate14_selector_{a.ckpt.replace('.pt','')}_{tag}.json", "w"), indent=1)
print(f"done {time.perf_counter()-t0:.0f}s")
