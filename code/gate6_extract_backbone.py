"""
Extract true backbone atoms (N, CA, C) for true-FAPE frames.

The dataset stores alpha carbons only, so FAPE currently builds pseudo-frames
from neighbouring CAs ~3.8 A apart across flexible dihedrals. True frames use
N-CA-C within one rigid residue and are far more stable.

Verified: re-processing a PDB through ProteinProcessor + canonicalize_pyg_data
reproduces the stored ca_coords to 0.00e+00, so fresh N/C land in exactly the
same frame as the stored CAs. Writes a side file keyed by structure name; the
main dataset is untouched.

CPU only, niced, safe to run alongside GPU training.
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
import os, sys, time, argparse
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, ROOT + "/ProteinAE_v1")
sys.path.insert(0, ROOT)

PROJECT = Path(ROOT)
H5_IN = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"
H5_OUT = PROJECT / "data" / "phase1_dataset" / "backbone_100k.h5"
STRUCT = PROJECT / "data" / "phase1_dataset" / "structures"
MAX_LEN = 256

_proc = None


def _init():
    global _proc
    os.chdir(ROOT + "/ProteinAE_v1")
    from proteinfoundation.autoencode import ProteinProcessor
    _proc = ProteinProcessor()


def _one(name):
    """Return (name, (n,3,3) float32 array of N/CA/C) or (name, None)."""
    global _proc
    try:
        from gate6_100k import canonicalize_pyg_data
        ic, _ = canonicalize_pyg_data(_proc.process_pdb(STRUCT / f"{name}.pdb"))
        c = ic.coords.numpy()[:MAX_LEN, [0, 1, 2], :].astype(np.float32)
        if not np.isfinite(c).all():
            return name, None
        return name, c
    except Exception:
        return name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    h5 = h5py.File(str(H5_IN), "r")
    todo = []
    for split in ["train", "val", "test"]:
        for nm in h5[split].keys():
            todo.append((split, nm, h5[split][nm]["ca_coords"].shape[0]))
    h5.close()
    if args.limit:
        todo = todo[:args.limit]
    print(f"structures to extract: {len(todo)}  workers: {args.workers}", flush=True)

    out = h5py.File(str(H5_OUT), "w")
    for s in ["train", "val", "test"]:
        out.create_group(s)

    by_name = {nm: (sp, n) for sp, nm, n in todo}
    names = [nm for _, nm, _ in todo]
    t0, done, failed, mismatch = time.perf_counter(), 0, [], []

    with Pool(args.workers, initializer=_init) as pool:
        for nm, arr in pool.imap_unordered(_one, names, chunksize=64):
            sp, n_expected = by_name[nm]
            if arr is None:
                failed.append(nm)
            elif arr.shape[0] != n_expected:
                mismatch.append(nm)          # residue count must match ca_coords
            else:
                out[sp].create_dataset(nm, data=arr, compression="gzip",
                                       compression_opts=4)
            done += 1
            if done % 5000 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(names)}  {el:6.0f}s  "
                      f"eta {el/done*(len(names)-done):5.0f}s  "
                      f"failed {len(failed)} mismatch {len(mismatch)}", flush=True)

    out.attrs["n_written"] = done - len(failed) - len(mismatch)
    out.attrs["atom_order"] = "N,CA,C"
    out.close()
    print(f"\ndone in {time.perf_counter()-t0:.0f}s")
    print(f"  written  : {done - len(failed) - len(mismatch)}")
    print(f"  failed   : {len(failed)}")
    print(f"  mismatch : {len(mismatch)}  (residue count != stored ca_coords)")
    if failed[:3]:
        print(f"  e.g. failed: {failed[:3]}")
    if mismatch[:3]:
        print(f"  e.g. mismatch: {mismatch[:3]}")


if __name__ == "__main__":
    main()
