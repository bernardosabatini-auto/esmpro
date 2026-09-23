"""Load ESMC-6B on a GPU via transformers, forward a batch, report timing/memory."""
import torch, time, os, sys
sys.path.insert(0, os.environ["ESM_PROAE_ROOT"] + "/code")
import gate7_latent_flow as G
t0 = time.time()
emb = G.OnlineESM(os.environ["ESM_PROAE_ROOT"] + "/data/esmc6b", device="cuda", kind="esmc")
print(f"loaded ESMC in {time.time()-t0:.0f}s, d_cond {emb.d_cond}, weights {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
import h5py
h = h5py.File(G.H5_PATH, "r"); names = list(h["val"].keys())[:64]
seqs = [str(h["val"][n].attrs["sequence"]) for n in names]
for bs in (16, 64):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = emb(seqs[:bs])
    torch.cuda.synchronize()
    print(f"batch {bs}: {time.time()-t0:.2f}s, out {tuple(out.shape)} {out.dtype}, finite {torch.isfinite(out).all().item()}, "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB, token norms {out[0, :3].float().norm(dim=-1).tolist()}", flush=True)
# sanity: padded positions beyond each sequence should be ignored downstream; check first real vs padded token norms
L0 = len(seqs[0]); print(f"seq0 len {L0}: norm at last real {out[0, L0-1].float().norm():.1f}, first pad {out[0, L0].float().norm():.1f}")
