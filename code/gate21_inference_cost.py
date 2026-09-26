"""Inference cost per protein (batch 1) of our model at several lengths on this GPU: full pipeline =
ESMC-6B embedding + latent flow sampling (n_steps Euler steps, CFG = 2 trunk forwards per step, pair
track once) + 3-step decoder. Compare with ESMFold2-Fast timed by gate12 on the same GPU class.
  ESM_PROAE_MAX_LEN=512 python gate21_inference_cost.py --ckpt last_pf_459M_p128x8_long512_scratch.ckpt
"""
import os, sys, time, json, argparse, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G
from gate6_fape_train import PROJECT
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, default=25); ap.add_argument("--cfg-w", type=float, default=2.0)
ap.add_argument("--lengths", default="64,128,256,384,512"); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args(); dev = torch.device("cuda")
ck = PROJECT / "data/phase1_dataset" / a.ckpt
st = torch.load(str(ck), weights_only=False, map_location="cpu")
arch = st["arch"]; ex = dict(st.get("extra_arch") or {}); ex.pop("recycle", None); ex.pop("p_rec", None); ex.pop("rec_every", None)
import gate10_pair_flow as G10; G10.install()
net = G10.PairFlowNet(**arch, **ex).to(dev); net.load_state_dict(G.adapt_state_dict(st["ema"], net.state_dict().keys())); net.eval()
dec = G.load_decoder(dev)
esm = G.OnlineESM(str(PROJECT / "data/esmc6b"), device=dev, kind="esmc")
n_params = sum(p.numel() for p in net.parameters())
print(f"{a.ckpt}: {n_params/1e6:.0f}M params, {a.steps} steps, w={a.cfg_w} -> {2*a.steps} trunk forwards per protein; GPU {torch.cuda.get_device_name()}", flush=True)
import random; random.seed(0)
AA = "ACDEFGHIKLMNPQRSTVWY"
def sync(): torch.cuda.synchronize(); return time.perf_counter()
rows = {}
with torch.no_grad():
    for L in [int(x) for x in a.lengths.split(",")]:
        seq = "".join(random.choice(AA) for _ in range(L)); mask = torch.ones(1, L, dtype=torch.bool, device=dev)
        for it in range(a.reps + 1):
            t0 = sync()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e = esm([seq], L=L)
            t1 = sync()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = G.sample(net, e, mask, a.steps, a.cfg_w)
            t2 = sync()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ca = dec(z.float(), mask)
            t3 = sync()
            if it == 0: continue          # warm-up / compile
            r = rows.setdefault(L, {"esmc": 0.0, "flow": 0.0, "decoder": 0.0, "total": 0.0})
            for k, v in (("esmc", t1 - t0), ("flow", t2 - t1), ("decoder", t3 - t2), ("total", t3 - t0)): r[k] += v / a.reps
        r = rows[L]; print(f"L={L:4d}: ESMC {r['esmc']*1000:6.0f} ms  flow {r['flow']*1000:6.0f} ms  decoder {r['decoder']*1000:5.0f} ms  total {r['total']:.2f} s", flush=True)
print(f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
json.dump({"ckpt": a.ckpt, "gpu": torch.cuda.get_device_name(), "steps": a.steps, "cfg_w": a.cfg_w, "params": n_params, "rows": rows},
          open(PROJECT / "notes" / f"gate21_inference_cost_{torch.cuda.get_device_name().replace(' ', '_')}.json", "w"), indent=1)
