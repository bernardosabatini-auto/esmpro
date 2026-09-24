"""Gate 15b (GPU job): build experimental-structure training targets from the
chains chosen by gate15_select_pdb.py.

Per chain: download the PDB-format entry from RCSB, take the author chain,
keep residues with N/CA/C/O (MSE -> M), require >= --min-obs observed
residues and >= --min-frac of the SEQRES length observed (gapped chains are
kept as the observed residues only, like the CASP evaluation units),
PCA-canonicalise, encode with the frozen ProteinAE encoder, and store in the
training layout (train/<pdb>_<chain>/{z, ca_coords, backbone}, attrs
sequence, n_residues, resolution, method). Then run gate11 for embeddings.

  python gate15_build_pdb.py --out data/phase1_dataset/dataset_pdb_train.h5 --workers 14
"""
import os, sys, time, json, argparse, urllib.request
import numpy as np, torch, h5py
from multiprocessing import Pool
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/ProteinAE_v1"); sys.path.insert(0, ROOT + "/code")
from gate8_build_afdb import three_to_one
DATA = f"{ROOT}/data/phase1_dataset"

def _fetch(pid):
    try:
        return urllib.request.urlopen(f"https://files.rcsb.org/download/{pid.upper()}.pdb", timeout=60).read().decode("utf-8", "ignore")
    except Exception:
        return None

def parse_chain(text, chain):
    order, atoms = [], {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"): break
        if not (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] == "MSE")): continue
        if line[21].strip() != chain: continue
        alt = line[16]
        if alt not in (" ", "A", "1"): continue
        key = line[22:27]; nm = line[12:16].strip()
        if key not in atoms: atoms[key] = {"res": line[17:20]}; order.append(key)
        if nm in ("N", "CA", "C", "O") and nm not in atoms[key]:
            atoms[key][nm] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    keep = [k for k in order if all(x in atoms[k] for x in ("N", "CA", "C", "O"))]
    seq = "".join("M" if atoms[k]["res"] == "MSE" else three_to_one.get(atoms[k]["res"], "X") for k in keep)
    coords = np.full((len(keep), 37, 3), 1e-5, dtype=np.float32)
    for i, k in enumerate(keep):
        at = atoms[k]; coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 4] = at["N"], at["CA"], at["C"], at["O"]
    return seq, coords

MIN_OBS, MIN_FRAC, MAX_LEN = 32, 0.8, 256
def _one(row):
    pid, ch, n_seqres, res, meth = row
    from canonicalize import canonicalize_pyg_data
    from torch_geometric.data import Data
    text = _fetch(pid)
    if text is None: return pid, "download"
    seq, c = parse_chain(text, ch)
    n = len(seq)
    if n < MIN_OBS or n < MIN_FRAC * n_seqres: return pid, "too few observed"
    if n > MAX_LEN: return pid, "too long"
    if "X" in seq: return pid, "nonstandard"
    try:
        item, _ = canonicalize_pyg_data(Data(coords=torch.from_numpy(c)))
        cc = item.coords.numpy().astype(np.float32)
        if cc.shape[0] != n or not np.isfinite(cc[:, :3]).all(): return pid, "canonicalise"
    except Exception as e:
        return pid, "canonicalise"
    # crude chain-break count: consecutive Ca farther than 4.5 A
    d = np.linalg.norm(cc[1:, 1] - cc[:-1, 1], axis=-1); breaks = int((d > 4.5).sum())
    return {"id": f"{pid}_{ch}", "seq": seq, "coords": cc, "resolution": res, "method": meth, "n_seqres": n_seqres, "breaks": breaks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=f"{ROOT}/data/pdb/selected_chains.tsv"); ap.add_argument("--out", default=f"{DATA}/dataset_pdb_train.h5")
    ap.add_argument("--workers", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enc-batch", type=int, default=64); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rows = []
    for l in list(open(a.list))[1:]:
        pid, ch, n, r, m, cs = l.rstrip("\n").split("\t"); rows.append((pid, ch, int(n), float(r), m))
    if a.limit: rows = rows[:a.limit]
    out = h5py.File(a.out, "a"); grp = out.require_group("train")
    todo = [r for r in rows if f"{r[0]}_{r[1]}" not in grp]
    print(f"{len(rows)} chains listed, {len(todo)} to do", flush=True)
    os.chdir(ROOT + "/ProteinAE_v1")
    import hydra
    from einops import rearrange
    from proteinfoundation.proteinflow.proteinae import ProteinAE
    from proteinfoundation.autoencode import ProteinProcessor
    from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
    from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
    from proteinfoundation.utils.coors_utils import ang_to_nm
    from torch_geometric.data import Data
    dev = torch.device(a.device)
    with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
        hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False, map_location=dev).eval().to(dev)
    fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0); BB = ProteinProcessor.BACKBONE_ATOM_INDICES

    @torch.no_grad()
    def encode(items):
        datas = [Data(coords=torch.from_numpy(it["coords"]), id=it["id"]) for it in items]
        batch = next(iter(DensePaddingDataLoader(datas, batch_size=len(datas))))
        x_1 = rearrange(batch["coords"][:, :, BB, :], "b n c d -> b (n c) d")
        coords_mask = batch["mask_dict"]["coords"][..., BB, 0]; mask = coords_mask[..., 1]
        cmf = rearrange(coords_mask, "b n c -> b (n c)")
        x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), cmf)
        batch.update({"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": int(x_1.shape[-2] // 4)})
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.to(dev)
        z = ae.encoder(batch)["single_repr"].float().cpu().numpy()
        return [z[i, :it["coords"].shape[0]] for i, it in enumerate(items)]

    t0, n_ok, fails, buf = time.perf_counter(), 0, {}, []
    def flush():
        nonlocal buf, n_ok
        if not buf: return
        for it, z in zip(buf, encode(buf)):
            g = grp.create_group(it["id"])
            g.create_dataset("z", data=z.astype(np.float32), compression="gzip", compression_opts=4)
            g.create_dataset("ca_coords", data=it["coords"][:, 1, :], compression="gzip", compression_opts=4)
            g.create_dataset("backbone", data=it["coords"][:, [0, 1, 2], :], compression="gzip", compression_opts=4)
            g.attrs["sequence"] = it["seq"]; g.attrs["n_residues"] = len(it["seq"]); g.attrs["resolution"] = it["resolution"]
            g.attrs["method"] = it["method"]; g.attrs["n_seqres"] = it["n_seqres"]; g.attrs["chain_breaks"] = it["breaks"]; g.attrs["source"] = "RCSB PDB experimental"
            n_ok += 1
        buf = []; out.flush()
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, todo, chunksize=4)):
            if isinstance(r, dict):
                buf.append(r)
                if len(buf) >= a.enc_batch: flush()
            else:
                fails[r[1]] = fails.get(r[1], 0) + 1
            if (k + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {k+1}/{len(todo)}  ok {n_ok}  fails {fails}  {el:.0f}s  eta {(len(todo)-k-1)/((k+1)/el)/60:.0f} min", flush=True)
    flush()
    out.attrs["source"] = "RCSB PDB chains, X-ray/EM <= 3 A, 32-256 aa, 50 % identity clusters, CASP homologs removed (gate15_select_pdb.py)"
    out.close(); print(f"done: {n_ok} written, fails {fails}, {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
