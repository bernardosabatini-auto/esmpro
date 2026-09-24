"""Hold out a validation set of long proteins: move --n proteins (deterministic) from the LAST
AFDB long shard into data/phase1_dataset/dataset_long_val{,_esmc}.h5 as split 'val' (with an
empty 'train' group so the file can serve as --h5-path). Groups are deleted from the shard files.
  python gate18_make_long_val.py --n 1000
"""
import os, argparse, h5py, numpy as np
ROOT = os.environ["ESM_PROAE_ROOT"]; D = f"{ROOT}/data/phase1_dataset"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--shard", default="dataset_afdb_long_3")
a = ap.parse_args()
src = h5py.File(f"{D}/{a.shard}.h5", "a"); srce = h5py.File(f"{D}/{a.shard}_esmc.h5", "a")
names = sorted(src["train"].keys()); rng = np.random.default_rng(42); pick = sorted(rng.choice(names, a.n, replace=False).tolist())
missing = [n for n in pick if n not in srce["train"]]; assert not missing, f"{len(missing)} picked proteins lack embeddings"
out = h5py.File(f"{D}/dataset_long_val.h5", "w"); oute = h5py.File(f"{D}/dataset_long_val_esmc.h5", "w")
for f in (out, oute): f.create_group("train"); f.create_group("val")
for n in pick:
    src.copy(src["train"][n], out["val"], name=n); srce.copy(srce["train"][n], oute["val"], name=n)
    del src["train"][n]; del srce["train"][n]
out.attrs["source"] = f"{a.n} AFDB long (257-512) proteins held out from {a.shard}, seed 42"; oute.attrs["source"] = out.attrs["source"]
for f in (out, oute, src, srce): f.close()
open(f"{ROOT}/notes/long_val_names.txt", "w").write("\n".join(pick) + "\n")
print(f"moved {len(pick)} proteins to dataset_long_val{{,_esmc}}.h5; shard now {len(h5py.File(f'{D}/{a.shard}.h5','r')['train'])} proteins")
