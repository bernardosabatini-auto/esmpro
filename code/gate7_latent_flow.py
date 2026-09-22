"""Gate 7: sequence-conditioned latent flow model.

p(z | sequence) as flow matching over the ProteinAE per-residue 8-dim latent,
conditioned on frozen ESM-2 embeddings. The frozen decoder is used ONLY at
evaluation time. Motivation (docs/H200_SETUP.md, 2026-09-22): the latent
encodes absolute orientation and the decoder basin is narrow, so p(z|seq) is
multimodal; a point regressor with FAPE through the decoder fights that, a
generative model absorbs it and also gives ensembles and guidance for free.

Usage (see slurm/train_latent_flow.sbatch):
  python gate7_latent_flow.py --label lf_base --d-model 512 --n-layers 12 \
      --epochs 100 --eval-every 2 --eval-n 100
"""
import os as _os
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)
import os, sys, time, json, math, copy, argparse, tempfile, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset

os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, "."); sys.path.insert(0, ROOT + "/code")
from gate6_fape_train import (ProteinDatasetFAPE, H5_PATH, PROJECT, fape_loss,
                              _write_pseudo_backbone_pdb, _foldseek_tm)
from gate6_corrected_eval import kabsch_rmsd

D_LAT, D_ESM, MAX_LEN = 8, 1280, 256
CKPT_DIR = PROJECT / "data" / "phase1_dataset"


# ---------------------------------------------------------------------------
# Data: read the HDF5 once, keep everything in host RAM (train ESM in fp16 is
# 52 GB for 79,653 proteins; the H200 nodes have 1.5 TB).
# ---------------------------------------------------------------------------
class RamSplit:
    def __init__(self, split, n=0, workers=8, keep_ca=False, seed=42):
        ds = ProteinDatasetFAPE(H5_PATH, split)
        torch.manual_seed(seed)
        idx = torch.randperm(len(ds))[:n].tolist() if n else list(range(len(ds)))
        self.names = [ds.names[i] for i in idx]
        N = len(idx)
        self.esm = torch.empty(N, MAX_LEN, D_ESM, dtype=torch.float16)
        self.z = torch.empty(N, MAX_LEN, D_LAT)
        self.mask = torch.empty(N, MAX_LEN, dtype=torch.bool)
        self.ca = torch.empty(N, MAX_LEN, 3) if keep_ca else None
        loader = DataLoader(Subset(ds, idx), batch_size=64, num_workers=workers)
        t0, k = time.perf_counter(), 0
        for esm, z, ca, mask, _ in loader:
            b = esm.shape[0]
            self.esm[k:k+b] = esm.half(); self.z[k:k+b] = z; self.mask[k:k+b] = mask
            if keep_ca: self.ca[k:k+b] = ca
            k += b
        print(f"  cached {split}: {N} proteins, {self.esm.numel()*2/1e9:.1f} GB, "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    def __len__(self): return self.z.shape[0]

    def batches(self, bs, shuffle, gen=None):
        N = len(self); order = torch.randperm(N, generator=gen) if shuffle else torch.arange(N)
        for s in range(0, N, bs):
            i = order[s:s+bs]
            yield self.esm[i], self.z[i], self.mask[i], (self.ca[i] if self.ca is not None else None), i


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with a learned relative-position bias and
    key padding, via scaled_dot_product_attention with a float mask."""
    def __init__(self, d, n_heads, rel_pos=32, dropout=0.0):
        super().__init__()
        self.h, self.dh, self.rel = n_heads, d // n_heads, rel_pos
        self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)
        self.bias = nn.Embedding(2 * rel_pos + 1, n_heads)
        nn.init.zeros_(self.bias.weight)
        self.dropout = dropout

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B, H, L, dh)
        pos = torch.arange(L, device=x.device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.rel, self.rel) + self.rel
        bias = self.bias(rel).permute(2, 0, 1).unsqueeze(0)          # (1, H, L, L)
        pad = torch.zeros(B, 1, 1, L, device=x.device, dtype=bias.dtype)
        pad = pad.masked_fill(~mask[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias + pad).to(q.dtype),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero modulation from a global
    conditioning vector (time + pooled sequence condition)."""
    def __init__(self, d, n_heads, rel_pos, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = Attention(d, n_heads, rel_pos, dropout)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, mask):
        s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + s1) + b1, mask)
        x = x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)
        return x


class LatentFlowNet(nn.Module):
    def __init__(self, d_model=512, n_layers=12, n_heads=8, dropout=0.0,
                 rel_pos=32, self_cond=True):
        super().__init__()
        self.self_cond = self_cond
        self.in_proj = nn.Linear(D_LAT * (2 if self_cond else 1), d_model)
        self.cond_norm = nn.LayerNorm(D_ESM)
        self.cond_proj = nn.Linear(D_ESM, d_model)
        self.null_cond = nn.Parameter(torch.zeros(1, 1, d_model))   # CFG "no sequence"
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, rel_pos, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out_proj = nn.Linear(d_model, D_LAT)
        nn.init.zeros_(self.out_ada[1].weight); nn.init.zeros_(self.out_ada[1].bias)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        self.d_model = d_model

    def forward(self, x_t, t, esm, mask, cond_drop=None, x_sc=None):
        """x_t: (B,L,8) noisy latent; t: (B,) in [0,1]; esm: (B,L,1280);
        cond_drop: (B,) bool, True = replace the sequence condition with the
        null token; x_sc: (B,L,8) self-conditioning estimate of x1 or None.
        Returns the predicted velocity v = x1 - x0, (B,L,8)."""
        B, L, _ = x_t.shape
        if self.self_cond:
            x_sc = torch.zeros_like(x_t) if x_sc is None else x_sc
            x_in = torch.cat([x_t, x_sc], dim=-1)
        else:
            x_in = x_t
        c_tok = self.cond_proj(self.cond_norm(esm.float()))
        if cond_drop is not None and cond_drop.any():
            c_tok = torch.where(cond_drop[:, None, None], self.null_cond.expand(B, L, -1), c_tok)
        h = self.in_proj(x_in) + c_tok + self.pos.weight[:L][None]
        m = mask.unsqueeze(-1).float()
        c_pool = (c_tok * m).sum(1) / m.sum(1).clamp(min=1.0)
        c = self.t_mlp(timestep_embedding(t, self.d_model)) + c_pool
        for blk in self.blocks:
            h = blk(h, c, mask)
        s, b = self.out_ada(c).unsqueeze(1).chunk(2, dim=-1)
        return self.out_proj(self.out_norm(h) * (1 + s) + b)


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------
def fm_loss(net, z1, esm, mask, p_drop=0.1, p_sc=0.5, t_dist="logit-normal"):
    B = z1.shape[0]; dev = z1.device
    x0 = torch.randn_like(z1)
    if t_dist == "logit-normal":
        t = torch.sigmoid(torch.randn(B, device=dev))
    else:
        t = torch.rand(B, device=dev)
    tt = t[:, None, None]
    x_t = (1 - tt) * x0 + tt * z1
    v_target = z1 - x0
    drop = torch.rand(B, device=dev) < p_drop
    x_sc = None
    if net.self_cond and torch.rand(()) < p_sc:
        with torch.no_grad():
            v0 = net(x_t, t, esm, mask, drop, None)
            x_sc = (x_t + (1 - tt) * v0).detach()
    v = net(x_t, t, esm, mask, drop, x_sc)
    m = mask.unsqueeze(-1).float()
    return (((v - v_target) ** 2) * m).sum() / (m.sum() * D_LAT).clamp(min=1.0)


@torch.no_grad()
def sample(net, esm, mask, n_steps=50, cfg_w=1.0, gen=None, project=True):
    """Euler ODE from noise to latent with classifier-free guidance.
    Returns (B,L,8). project=True re-applies the per-residue LayerNorm the
    true latents satisfy exactly (CLAUDE.md section 7)."""
    B, L = mask.shape; dev = esm.device
    x = torch.randn(B, L, D_LAT, device=dev, generator=gen)
    x_sc = None
    ts = torch.linspace(0, 1, n_steps + 1, device=dev)
    for i in range(n_steps):
        t = ts[i].expand(B); dt = (ts[i + 1] - ts[i])
        v = net(x, t, esm, mask, None, x_sc)
        if cfg_w != 1.0:
            v_u = net(x, t, esm, mask, torch.ones(B, dtype=torch.bool, device=dev), x_sc)
            v = v_u + cfg_w * (v - v_u)
        if net.self_cond:
            x_sc = x + (1 - ts[i]) * v
        x = x + v * dt
    if project:
        x = F.layer_norm(x, (D_LAT,))
    return x * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Structure evaluation through the frozen decoder
# ---------------------------------------------------------------------------
def load_decoder(device, n_steps=3):
    import lightning as Lt, hydra
    from proteinfoundation.proteinflow.proteinae import ProteinAE
    from gate6_fape_train import DifferentiableDecoder
    with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config",
                                     version_base=hydra.__version__):
        hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True,
                                        weights_only=False).eval().to(device)
    for p in ae.parameters():
        p.requires_grad = False
    return DifferentiableDecoder(ae, n_steps=n_steps).to(device)


@torch.no_grad()
def evaluate_structures(net, dec, val, device, n=100, bs=20, n_steps=50, cfg_w=1.0,
                        n_samples=1, seed=0):
    """Sample n_samples latents per protein, decode, TM via Foldseek.
    Returns dict with mean TM (first sample), best-of-K TM, TM>0.5 fraction,
    RMSD, latent error to the stored z, and coverage. Never raises."""
    net.eval()
    work = tempfile.mkdtemp(prefix="lfeval_")
    try:
        gen = torch.Generator(device=device.type).manual_seed(seed)
        n = min(n, len(val))
        tm_by_k = [{} for _ in range(n_samples)]
        rm, zerr, fape = [], [], []
        for s in range(0, n, bs):
            esm = val.esm[s:s+bs].to(device); mask = val.mask[s:s+bs].to(device)
            ca = val.ca[s:s+bs]; z_true = val.z[s:s+bs].to(device)
            for k in range(n_samples):
                gt_d, pr_d = os.path.join(work, f"gt{k}"), os.path.join(work, f"pr{k}")
                os.makedirs(gt_d, exist_ok=True); os.makedirs(pr_d, exist_ok=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    z = sample(net, esm, mask, n_steps, cfg_w, gen)
                    pred = dec(z.float(), mask).float()
                if k == 0:
                    m = mask.unsqueeze(-1).float()
                    zerr.append((((z.float() - z_true) ** 2 * m).sum() / (m.sum() * D_LAT)).item())
                    rm += kabsch_rmsd(pred.cpu(), ca, mask.cpu())
                    fape.append(fape_loss(pred.cpu(), ca, mask.cpu(), fix_ends=True).item())
                pred = pred.cpu().numpy()
                for b in range(pred.shape[0]):
                    L_ = int(mask[b].sum()); nm = f"p{s+b:05d}"
                    _write_pseudo_backbone_pdb(ca[b, :L_].numpy(), os.path.join(gt_d, f"{nm}.pdb"))
                    _write_pseudo_backbone_pdb(pred[b, :L_], os.path.join(pr_d, f"{nm}.pdb"))
        names = [f"p{i:05d}" for i in range(n)]
        for k in range(n_samples):
            tm_by_k[k] = _foldseek_tm(os.path.join(work, f"pr{k}"), os.path.join(work, f"gt{k}"),
                                      os.path.join(work, f"tm{k}"))
        meas = [tm_by_k[0][x] for x in names if x in tm_by_k[0]]
        best = [max(tm_by_k[k].get(x, 0.0) for k in range(n_samples)) for x in names]
        out = {"tm": float(np.mean(meas)) if meas else float("nan"),
               "tm_frac": float(np.mean([tm_by_k[0].get(x, 0.0) > 0.5 for x in names])),
               "tm_best_of_k": float(np.mean(best)), "k": n_samples,
               "coverage": len(meas) / n, "rmsd": float(np.nanmean(rm)),
               "fape": float(np.mean(fape)), "z_mse": float(np.mean(zerr)), "cfg_w": cfg_w}
        return out
    except Exception as e:
        print(f"    [structure eval failed: {e}]", flush=True)
        return {"tm": float("nan"), "tm_frac": float("nan"), "coverage": 0.0, "cfg_w": cfg_w}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        net.train()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="lf_base")
    p.add_argument("--n-train", type=int, default=0); p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--d-model", type=int, default=512); p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=8); p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-self-cond", action="store_true")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--p-drop", type=float, default=0.1, help="condition dropout for CFG")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=2); p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-w", type=str, default="1.0,2.0", help="guidance weights to evaluate")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="tiny CPU/GPU run, no eval")
    return p.parse_args()


def main(a):
    torch.manual_seed(0); np.random.seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60 + "\nGate 7: sequence-conditioned latent flow\n" + "=" * 60, flush=True)
    arch = {"d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
            "dropout": a.dropout, "self_cond": not a.no_self_cond}
    net = LatentFlowNet(**arch).to(device)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters(): p.requires_grad = False
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  {a.label}: {n_params/1e6:.1f}M params, arch {arch}", flush=True)

    print("Caching data in RAM...", flush=True)
    train = RamSplit("train", a.n_train, a.workers)
    val = RamSplit("val", a.n_val, a.workers, keep_ca=True)
    print(f"  train {len(train)}  val {len(val)}  batch {a.batch_size}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / a.batch_size); total = steps_per_epoch * a.epochs
    def lr_at(step):
        if step < a.warmup: return step / max(a.warmup, 1)
        pr = (step - a.warmup) / max(total - a.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(pr, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    dec = None if a.smoke else load_decoder(device)
    cfg_ws = [float(x) for x in a.cfg_w.split(",")]

    start, best_tm, best_ep, history, step = 0, -1.0, 0, [], 0
    gen = torch.Generator().manual_seed(0)
    if a.resume:
        st = torch.load(str(CKPT_DIR / a.resume), weights_only=False, map_location=device)
        if st["arch"] != arch: raise ValueError(f"arch mismatch: {st['arch']} vs {arch}")
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start, step = st["epoch"], st["step"]
        best_tm, best_ep, history = st["best_tm"], st["best_ep"], st["history"]
        gen.set_state(st["gen"]); torch.set_rng_state(st["torch_rng"])
        print(f"  RESUMED at epoch {start}, step {step}, best TM {best_tm:.3f}", flush=True)

    for epoch in range(start, a.epochs):
        t0 = time.perf_counter(); net.train(); tl, nb = 0.0, 0
        for esm, z, mask, _, _ in train.batches(a.batch_size, True, gen):
            esm, z, mask = esm.to(device, non_blocking=True), z.to(device), mask.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = fm_loss(net, z, esm, mask, a.p_drop)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - a.ema)
            tl += loss.item(); nb += 1
            if nb == 1 or nb % 200 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(f"    batch {nb}: loss={loss.item():.4f} gnorm={gn:.2f} lr={sched.get_last_lr()[0]:.2e} "
                      f"peak GPU={mem:.1f} GB", flush=True)
            if a.smoke and nb >= 3: break
        train_loss = tl / max(nb, 1)

        # validation flow loss with the EMA weights
        ema.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for esm, z, mask, _, _ in val.batches(a.batch_size, False):
                esm, z, mask = esm.to(device), z.to(device), mask.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    torch.manual_seed(1234 + vn)          # same noise/t each epoch
                    vl += fm_loss(ema, z, esm, mask, p_drop=0.0, p_sc=0.0).item()
                vn += 1
                if a.smoke and vn >= 2: break
        val_loss = vl / max(vn, 1)
        rec = {"epoch": epoch + 1, "train": train_loss, "val": val_loss, "step": step}
        el = time.perf_counter() - t0
        print(f"  {epoch+1:4d}  train {train_loss:.4f}  val {val_loss:.4f}  {el:6.0f}s", flush=True)

        tm_for_select = float("nan")
        if dec is not None and (epoch + 1) % a.eval_every == 0:
            for w in cfg_ws:
                t1 = time.perf_counter()
                r = evaluate_structures(ema, dec, val, device, n=a.eval_n, n_steps=a.sample_steps,
                                        cfg_w=w, n_samples=1)
                print(f"        w={w:.1f}: TM {r['tm']:.3f}  TM>0.5 {r['tm_frac']:.2f}  "
                      f"RMSD {r.get('rmsd', float('nan')):.2f}  Ca-FAPE {r.get('fape', float('nan')):.3f}  "
                      f"z_mse {r.get('z_mse', float('nan')):.3f}  coverage {r['coverage']:.2f}  "
                      f"[{time.perf_counter()-t1:.0f}s]", flush=True)
                rec[f"eval_w{w}"] = r
                if not np.isnan(r["tm"]) and r["tm"] > tm_for_select:
                    tm_for_select = r["tm"]; tm_for_select_w = w
            if not np.isnan(tm_for_select) and tm_for_select > best_tm:
                best_tm, best_ep = tm_for_select, epoch + 1
                torch.save(ema.state_dict(), str(CKPT_DIR / f"best_{a.label}.pt"))
                meta = {"epoch": epoch + 1, "tm": best_tm, "cfg_w": tm_for_select_w, **arch,
                        "model": "LatentFlowNet"}
                json.dump(meta, open(str(CKPT_DIR / f"best_{a.label}.pt.meta.json"), "w"), indent=2)
                print(f"        new best TM {best_tm:.3f} (w={tm_for_select_w}) -> best_{a.label}.pt", flush=True)
        history.append(rec)

        last = CKPT_DIR / f"last_{a.label}.ckpt"
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1, "step": step,
                    "best_tm": best_tm, "best_ep": best_ep, "history": history, "arch": arch,
                    "gen": gen.get_state(), "torch_rng": torch.get_rng_state()}, str(last) + ".tmp")
        os.replace(str(last) + ".tmp", str(last))
        json.dump({"label": a.label, "arch": arch, "n_params": n_params, "config": vars(a),
                   "best_tm": best_tm, "best_ep": best_ep, "history": history},
                  open(PROJECT / "notes" / f"gate7_{a.label}_results.json", "w"), indent=2)
        if best_ep > 0 and (epoch + 1) - best_ep >= a.patience * a.eval_every:
            print(f"  Early stopping on TM (best epoch {best_ep}, TM {best_tm:.3f})"); break
        if a.smoke: break
    print(f"\nBest TM {best_tm:.3f} at epoch {best_ep}")


if __name__ == "__main__":
    main(parse_args())
