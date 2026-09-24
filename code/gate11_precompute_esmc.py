"""Gate 11: precompute ESMC-6B last-layer embeddings for the 100k dataset.

Writes data/phase1_dataset/dataset_100k_esmc.h5 with the SAME group layout as
dataset_100k.h5 (split/name/{esm2_emb, z, ca_coords} + sequence attr) but with
`esm2_emb` holding the ESMC-6B embedding (n, 2560) fp16, uncompressed, so the
existing stored-mode trainer works unchanged via --h5-path. ~105 GB for train;
GPU time ~25 min on one RTX (0.94 s per 64 sequences).

  python gate11_precompute_esmc.py --splits train,val --bs 64
"""
import os, sys, time, argparse, h5py, numpy as np, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
import gate7_latent_flow as G
ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="train,val"); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default=str(G.PROJECT / "data/phase1_dataset/dataset_100k_esmc.h5"))
ap.add_argument("--esm-path", default=str(G.PROJECT / "data/esmc6b"))
ap.add_argument("--src", default=str(G.H5_PATH), help="source HDF5 with split groups holding z, ca_coords and a sequence attr")
ap.add_argument("--kind", default="esmc", help="esmc (default) or esm2: which language model to embed with")
a = ap.parse_args()
if a.kind == "esm2" and a.esm_path.endswith("esmc6b"): a.esm_path = str(G.PROJECT / "data/esm2/esm2_t33_650M_UR50D")
emb = G.OnlineESM(a.esm_path, device="cuda", kind=a.kind)
src = h5py.File(a.src, "r"); out = h5py.File(a.out, "a")
t0 = time.perf_counter()
for split in a.splits.split(","):
    g_out = out.require_group(split); names = list(src[split].keys())
    todo = [n for n in names if n not in g_out]
    print(f"{split}: {len(names)} proteins, {len(todo)} to do", flush=True)
    for s in range(0, len(todo), a.bs):
        batch = todo[s:s + a.bs]
        seqs = [str(src[split][n].attrs["sequence"]) for n in batch]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e = emb(seqs).float().cpu().numpy().astype(np.float16)          # (B, 256, 2560)
        for i, n in enumerate(batch):
            gi = src[split][n]; L = gi["ca_coords"].shape[0]
            g = g_out.create_group(n)
            g.create_dataset("esm2_emb", data=e[i, :L])                       # ESMC embedding, same key for the loader
            g.create_dataset("z", data=gi["z"][:]); g.create_dataset("ca_coords", data=gi["ca_coords"][:])
            if "backbone" in gi: g.create_dataset("backbone", data=gi["backbone"][:])
            for k, v in gi.attrs.items(): g.attrs[k] = v
            g.attrs["embedding"] = ("ESMC-6B last hidden state" if a.kind == "esmc" else "ESM-2 650M layer 33") + ", bf16 autocast, stored fp16"
        if (s // a.bs) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {split} {s+len(batch)}/{len(todo)}  {el:.0f}s  eta {el/(s+len(batch))*(len(todo)-s-len(batch))/60:.0f} min", flush=True)
        out.flush()
out.attrs["source"] = f"{a.kind} embeddings ({a.esm_path}) of {a.src} sequences"
out.close(); print(f"done {time.perf_counter()-t0:.0f}s -> {a.out}")
