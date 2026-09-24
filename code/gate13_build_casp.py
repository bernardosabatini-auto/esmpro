"""Gate 13: experimental benchmark from CASP15/CASP16 single-domain targets.

Reads the official CASP evaluation-unit PDBs (data/casp/casp15_domains,
data/casp/casp16_domains; experimental coordinates, one domain per file),
keeps the residues that have N, CA, C and O, maps MSE->M, drops domains with
non-standard residues or outside [32, --max-len], PCA-canonicalises like the
training data, encodes the ProteinAE latent (control: true latent through the
decoder) and writes data/phase1_dataset/dataset_casp.h5 with the training
layout (val/<name>/{z, ca_coords, backbone}, sequence attr) so that
gate11_precompute_esmc.py (embeddings) and gate7_score.py (--names-file)
work unchanged. No ESM embeddings here: this step is CPU-only.

  python gate13_build_casp.py --device cpu
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/ProteinAE_v1"); sys.path.insert(0, ROOT + "/code")
from gate8_build_afdb import three_to_one
ap = argparse.ArgumentParser()
ap.add_argument("--dirs", default=f"{ROOT}/data/casp/casp15_domains,{ROOT}/data/casp/casp16_domains")
ap.add_argument("--out", default=f"{ROOT}/data/phase1_dataset/dataset_casp.h5")
ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--min-len", type=int, default=32)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

def parse(path):
    """Residues in order of first appearance with all four backbone atoms; MSE as M.
    Returns (seq, (n,37,3) coords in the ProteinProcessor layout, n_dropped, resolution)."""
    order, atoms, res = [], {}, None
    for line in open(path):
        if line.startswith("REMARK") and "RESOLUTION RANGE HIGH" in line:
            try: res = float(line.split(":")[1])
            except ValueError: pass
        if line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] == "MSE"):
            key = (line[21], line[22:27]); nm = line[12:16].strip()
            if key not in atoms: atoms[key] = {"res": line[17:20]}; order.append(key)
            if nm in ("N", "CA", "C", "O") and nm not in atoms[key]:
                atoms[key][nm] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        elif line.startswith("ENDMDL"):
            break
    keep = [k for k in order if all(x in atoms[k] for x in ("N", "CA", "C", "O"))]
    seq = "".join("M" if atoms[k]["res"] == "MSE" else three_to_one.get(atoms[k]["res"], "X") for k in keep)
    coords = np.full((len(keep), 37, 3), 1e-5, dtype=np.float32)
    for i, k in enumerate(keep):
        at = atoms[k]; coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 4] = at["N"], at["CA"], at["C"], at["O"]
    return seq, coords, len(order) - len(keep), res

from canonicalize import canonicalize_pyg_data
from torch_geometric.data import Data
items, skipped = [], {}
for d in a.dirs.split(","):
    casp = os.path.basename(d.rstrip("/")).split("_")[0]
    for f in sorted(glob.glob(f"{d}/*.pdb")):
        name = os.path.basename(f)[:-4]
        seq, c, dropped, res = parse(f)
        n = len(seq)
        if "X" in seq: skipped[name] = "nonstandard"; continue
        if not (a.min_len <= n <= a.max_len): skipped[name] = f"length {n}"; continue
        item, _ = canonicalize_pyg_data(Data(coords=torch.from_numpy(c)))
        cc = item.coords.numpy().astype(np.float32)
        if cc.shape[0] != n or not np.isfinite(cc[:, :3]).all(): skipped[name] = "canonicalise"; continue
        items.append({"id": f"{casp}_{name}", "casp": casp, "seq": seq, "coords": cc, "dropped": dropped, "resolution": res, "file": f})
# CASP16 ships the same evaluation unit under T0xxx/T1xxx/T2xxx prefixes (phase 0/1/2): keep one copy per sequence,
# preferring the T1 name, so that no domain is counted twice.
by_seq = {}
for it in sorted(items, key=lambda it: (0 if it["id"].split("_")[1].startswith("T1") else 1, it["id"])):
    if it["seq"] in by_seq: skipped[it["id"]] = f"duplicate of {by_seq[it['seq']]['id']}"
    else: by_seq[it["seq"]] = it
items = sorted(by_seq.values(), key=lambda it: it["id"])
print(f"{len(items)} unique domains kept, {len(skipped)} skipped: {json.dumps(skipped)}", flush=True)
print("lengths:", sorted(len(it["seq"]) for it in items))

# --- encode the ProteinAE latent exactly as gate8 does -------------------------------------
os.chdir(ROOT + "/ProteinAE_v1")
import hydra
from einops import rearrange
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
dev = torch.device(a.device)
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False, map_location=dev).eval().to(dev)
fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0); BB = ProteinProcessor.BACKBONE_ATOM_INDICES

@torch.no_grad()
def encode(batch_items):
    datas = [Data(coords=torch.from_numpy(it["coords"]), id=it["id"]) for it in batch_items]
    batch = next(iter(DensePaddingDataLoader(datas, batch_size=len(datas))))
    x_1 = rearrange(batch["coords"][:, :, BB, :], "b n c d -> b (n c) d")
    coords_mask = batch["mask_dict"]["coords"][..., BB, 0]; mask = coords_mask[..., 1]
    cmf = rearrange(coords_mask, "b n c -> b (n c)")
    x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), cmf)
    batch.update({"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": int(x_1.shape[-2] // 4)})
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(dev)
    z = ae.encoder(batch)["single_repr"].float().cpu().numpy()
    return [z[i, :it["coords"].shape[0]] for i, it in enumerate(batch_items)]

t0 = time.perf_counter()
if os.path.exists(a.out): os.remove(a.out)
out = h5py.File(a.out, "w"); grp = out.create_group("val")
for s in range(0, len(items), 16):
    chunk = items[s:s + 16]
    for it, z in zip(chunk, encode(chunk)):
        g = grp.create_group(it["id"])
        g.create_dataset("z", data=z.astype(np.float32)); g.create_dataset("ca_coords", data=it["coords"][:, 1, :])
        g.create_dataset("backbone", data=it["coords"][:, [0, 1, 2], :])
        g.attrs["sequence"] = it["seq"]; g.attrs["n_residues"] = len(it["seq"]); g.attrs["casp"] = it["casp"]
        g.attrs["dropped_residues"] = it["dropped"]; g.attrs["resolution"] = -1.0 if it["resolution"] is None else it["resolution"]
        g.attrs["source_file"] = os.path.relpath(it["file"], ROOT)
    print(f"  encoded {s+len(chunk)}/{len(items)}  {time.perf_counter()-t0:.0f}s", flush=True)
out.attrs["source"] = "CASP15 TS-domains public 12.20.2022 + CASP16 monomer trimmed2domains; experimental coordinates"
out.close()
names = sorted(it["id"] for it in items)
open(f"{ROOT}/notes/casp_domains_le{a.max_len}.txt", "w").write("\n".join(names) + "\n")
for casp in ("casp15", "casp16"):
    open(f"{ROOT}/notes/{casp}_domains_le{a.max_len}.txt", "w").write("\n".join(n for n in names if n.startswith(casp)) + "\n")
json.dump({"kept": [{k: v for k, v in it.items() if k != "coords"} for it in items], "skipped": skipped}, open(f"{ROOT}/notes/gate13_casp_build.json", "w"), indent=1)
print(f"wrote {a.out} with {len(items)} domains; name lists in notes/")
