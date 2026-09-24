"""Where does a training step of the 459M pair-flow model spend its time?
Real batches from the real ragged cache (budget batching), timed phase by phase with
CUDA synchronisation, plus a torch.profiler kernel table for one step.
  STEP_BUDGET=48 python step_profile.py --d-pair 128 --n-pair-blocks 8 --d-model 1024 --n-layers 24 --n-heads 16
"""
import os, sys, time, argparse, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G, gate10_pair_flow as G10
ap = argparse.ArgumentParser(); ap.add_argument("--d-pair", type=int, default=128); ap.add_argument("--n-pair-blocks", type=int, default=8)
ap.add_argument("--d-model", type=int, default=1024); ap.add_argument("--n-layers", type=int, default=24); ap.add_argument("--n-heads", type=int, default=16)
ap.add_argument("--budget", type=int, default=int(os.environ.get("STEP_BUDGET", "48"))); ap.add_argument("--n-steps", type=int, default=12)
ap.add_argument("--recycle", action="store_true")
ap.add_argument("--fused", action="store_true")
ap.add_argument("--h5", default=None, help="data file for the batches (default dataset_100k_esmc.h5); use a long shard with ESM_PROAE_MAX_LEN=512")
a = ap.parse_args(); dev = torch.device("cuda")
train = G.RamSplit("train", 6000, 4, h5_path=a.h5 or str(G.PROJECT / "data/phase1_dataset/dataset_100k_esmc.h5"))
print(f"window MAX_LEN={G.MAX_LEN}, data {os.path.basename(a.h5) if a.h5 else 'dataset_100k_esmc.h5'}", flush=True)
if a.recycle:
    import gate16_recycle_flow as G16; G16.install(); Net = G16.RecFlowNet; loss_fn = G16.fm_loss_rec
else:
    G10.install(); Net = G10.PairFlowNet; loss_fn = G10.fm_loss_pair
net = Net(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_cond=2560, d_pair=a.d_pair, n_pair_blocks=a.n_pair_blocks, pair_fused=a.fused).to(dev)
print(f"{sum(p.numel() for p in net.parameters())/1e6:.0f}M params; budget {a.budget} x 256^2, recycle {a.recycle}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
gen = torch.Generator().manual_seed(0); plan = train.batch_plan(a.budget, True, gen, budget_cap=3)
def sync(): torch.cuda.synchronize(); return time.perf_counter()
tot = {k: 0.0 for k in ("assemble", "h2d", "fwd", "bwd", "opt")}; n = 0; shapes = []
for bi in plan[: a.n_steps + 2]:
    t0 = sync(); c, z, mask, _, _ = train.batch_from(bi); t1 = sync()
    z, mask = z.to(dev), mask.to(dev); esm = c.to(dev); t2 = sync()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): loss = loss_fn(net, z, esm, mask, 0.1)
    t3 = sync(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); t4 = sync(); opt.step(); t5 = sync()
    if n >= 2:   # skip warm-up / compile steps
        for k, d in zip(tot, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)): tot[k] += d
        shapes.append((int(z.shape[0]), int(z.shape[1])))
    n += 1
m = n - 2; step = sum(tot.values()) / m
R = int(os.environ.get("REPEAT_COPIES", "1"))
print(f"REPEAT_COPIES={R}: samples/step = {R} x proteins; per step ({m} steps, shapes B,L = {shapes[:4]}...): " + "  ".join(f"{k} {v/m:.3f}s" for k, v in tot.items()) + f"  | total {step:.3f}s", flush=True)
tok = sum(B * L for B, L in shapes) / m
P = sum(p.numel() for p in net.parameters())
flops = 6 * P * tok * 1.5   # fwd+bwd of the DiT plus the no-grad self-cond forward (pair track and attention extra)
tok = tok * R
print(f"tokens/step {tok:.0f} (x{R} copies); DiT-only FLOP estimate {flops/1e12:.0f} TFLOP/step -> {flops/step/1e12:.0f} TFLOPS achieved (RTX Pro 6000 bf16 dense peak ~ 500-ish; H200 ~990)")
print(f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB  (PAD8={G.PAD8} DIT_COMPILE={G.DIT_COMPILE} fused={a.fused} PAIR_CKPT={os.environ.get('PAIR_CKPT','1')})", flush=True)
from torch.nn.attention import sdpa_kernel, SDPBackend
Lq = int(shapes[0][1]); q = torch.randn(8, 16, Lq, 64, device=dev, dtype=torch.bfloat16); bias = torch.randn(8, 16, Lq, Lq, device=dev, dtype=torch.bfloat16)
for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
    try:
        with sdpa_kernel([be]): torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=bias); ok = "ok"
    except Exception as e: ok = "unavailable: " + str(e).splitlines()[0][:70]
    print(f"SDPA backend {be.name} with bf16 bias at L={Lq}: {ok}")
from torch.profiler import profile, ProfilerActivity
c, z, mask, _, _ = train.batch_from(plan[3]); z, mask, esm = z.to(dev), mask.to(dev), c.to(dev)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): loss = loss_fn(net, z, esm, mask, 0.1)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=70))
