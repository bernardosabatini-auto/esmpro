"""Micro-benchmark: where does the pair track spend its time?"""
import os, sys, time, torch
sys.path.insert(0, os.environ["ESM_PROAE_ROOT"] + "/code"); sys.argv = ["x"]
import gate7_latent_flow as G, gate10_pair_flow as G10
dev = torch.device("cuda"); B, L = 128, int(os.environ.get("PROF_L", "160"))   # 160 ~ a typical bucketed batch length
esm = torch.randn(B, L, 1280, device=dev, dtype=torch.float16); mask = torch.ones(B, L, dtype=torch.bool, device=dev)
mask[:, L-8:] = False; z = torch.randn(B, L, 8, device=dev)
def bench(fn, n=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
base = G.LatentFlowNet(d_model=768, n_layers=16, n_heads=12).to(dev)
pair = G10.PairFlowNet(d_model=768, n_layers=16, n_heads=12, d_pair=64, n_pair_blocks=6).to(dev)
opt_b = torch.optim.AdamW(base.parameters()); opt_p = torch.optim.AdamW(pair.parameters())
def step_base():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): l = G.fm_loss(base, z, esm, mask)
    opt_b.zero_grad(); l.backward(); opt_b.step()
def step_pair():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): l = G10.fm_loss_pair(pair, z, esm, mask)
    opt_p.zero_grad(); l.backward(); opt_p.step()
print(f"train step  no-pair: {bench(step_base):.3f}s   pair: {bench(step_pair):.3f}s", flush=True)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    def pair_fwd():
        p = pair.compute_pair(esm, mask); p.sum().backward()
    print(f"pair track fwd+bwd only (checkpointed): {bench(pair_fwd):.3f}s", flush=True)
    with torch.no_grad():
        print(f"pair track fwd only: {bench(lambda: pair.compute_pair(esm, mask)):.3f}s", flush=True)
        p = pair.compute_pair(esm, mask)
        pm = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).to(torch.bfloat16); pb = p.to(torch.bfloat16)
        blk = pair.pair.blocks[0]
        print(f"one PairBlock fwd: {bench(lambda: blk(pb, pm)):.3f}s;  tri_out alone: {bench(lambda: blk.tri_out(pb, pm)):.3f}s;  transition alone: {bench(lambda: blk.trans(blk.norm(pb))):.3f}s", flush=True)
        a = torch.randn(B, L, L, 64, device=dev, dtype=torch.bfloat16)
        print(f"raw einsum bikd,bjkd->bijd: {bench(lambda: torch.einsum('bikd,bjkd->bijd', a, a)):.3f}s;  as bmm: {bench(lambda: torch.matmul(a.permute(0,3,1,2), a.permute(0,3,2,1))):.3f}s", flush=True)
        t = torch.zeros(B, 1, device=dev)
        print(f"DiT fwd with pair bias: {bench(lambda: pair(z, t[:,0], esm, mask, pair=p)):.3f}s   no-pair DiT fwd: {bench(lambda: base(z, t[:,0], esm, mask)):.3f}s", flush=True)
print(f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
