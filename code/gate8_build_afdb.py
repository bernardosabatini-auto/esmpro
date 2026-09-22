"""Gate 8: stream a large AFDB training set into an HDF5 of latents.

For each entry of the Genie2 AFDB index (AFDB cluster representatives, length
<= 256, pLDDT >= 80) that is NOT already in dataset_100k.h5: download the v6
PDB from EBI, parse with ProteinAE's ProteinProcessor, PCA-canonicalise
(canonicalize.py, identical to the 100k build), encode with the frozen
ProteinAE encoder, and store per protein:

    z          (n, 8)  float32   ProteinAE latent (the flow model's target)
    ca_coords  (n, 3)  float32   canonical-frame alpha carbons
    backbone   (n, 3, 3) float32 N, CA, C (for true-frame losses)
    attrs      sequence, plddt_mean, afdb_version

No ESM embeddings are stored: train with `gate7_latent_flow.py --esm online`.
Everything lands in ONE group, "train"; validation/test remain the 100k
splits so all numbers stay comparable. Download and parsing run in a
process pool; encoding is batched on the GPU. Resumable: entries already in
the output file are skipped.

  python gate8_build_afdb.py --out dataset_afdb_train.h5 --workers 24 --limit 0
"""
import os as _os
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
import os, sys, io, time, json, argparse, urllib.request, tempfile, random
import numpy as np, h5py, torch
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, ROOT + "/ProteinAE_v1"); sys.path.insert(0, ROOT + "/code")
PROJECT = Path(ROOT); DATA = PROJECT / "data" / "phase1_dataset"
MAX_LEN, MIN_LEN = 256, 32

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

_proc = None
FAST = os.environ.get("GATE8_FAST_PARSE", "1") == "1"
def _init():
    global _proc
    os.chdir(ROOT + "/ProteinAE_v1")
    if not FAST:
        from proteinfoundation.autoencode import ProteinProcessor
        _proc = ProteinProcessor()

def fast_atom37(text):
    """(n, 37, 3) float32 in ProteinProcessor's layout for a single-chain AFDB
    model: N, CA, C, O at atom37 indices 0, 1, 2, 4, every other slot 1e-5
    (the processor's fill value). Residues in order of first appearance.
    Verified identical to ProteinProcessor.process_pdb on the backbone atoms."""
    order, atoms = [], {}
    for line in text.splitlines():
        if line.startswith("ATOM"):
            key = (line[21], line[22:27]); nm = line[12:16].strip()
            if key not in atoms: atoms[key] = {}; order.append(key)
            if nm in ("N", "CA", "C", "O") and nm not in atoms[key]:
                atoms[key][nm] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        elif line.startswith("ENDMDL"):
            break
    coords = np.full((len(order), 37, 3), 1e-5, dtype=np.float32)
    for i, key in enumerate(order):
        a = atoms[key]
        if not all(k in a for k in ("N", "CA", "C", "O")): return None
        coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 4] = a["N"], a["CA"], a["C"], a["O"]
    return coords

def _fetch(base_id):
    for v in ("v6", "v4"):
        try:
            return v, urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/files/{base_id}-model_{v}.pdb", timeout=60).read()
        except Exception:
            continue
    return None, None

def _seq_plddt(text):
    seq, pl, seen = [], [], set()
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            key = (line[21], line[22:27].strip())
            if key in seen: continue
            seen.add(key); seq.append(three_to_one.get(line[17:20], "X")); pl.append(float(line[60:66]))
        elif line.startswith("ENDMDL"):
            break
    return "".join(seq), float(np.mean(pl)) if pl else 0.0

def _one(base_id):
    """Download + parse + canonicalise. Returns dict or (base_id, reason)."""
    from canonicalize import canonicalize_pyg_data
    v, data = _fetch(base_id)
    if data is None: return base_id, "download"
    text = data.decode("utf-8", "ignore")
    seq, plddt = _seq_plddt(text)
    n = len(seq)
    if not (MIN_LEN <= n <= MAX_LEN): return base_id, f"length {n}"
    if "X" in seq: return base_id, "nonstandard"
    try:
        if FAST:
            from torch_geometric.data import Data
            c = fast_atom37(text)
            if c is None: return base_id, "parse-missing-atom"
            item = Data(coords=torch.from_numpy(c))
        else:
            tmp = Path(tempfile.gettempdir()) / f"{base_id}-model_{v}.pdb"
            tmp.write_text(text)
            try:
                item = _proc.process_pdb(tmp)
            finally:
                tmp.unlink(missing_ok=True)
        item, _ = canonicalize_pyg_data(item)
        coords = item.coords.numpy().astype(np.float32)           # (n, 37, 3) canonical frame
        if coords.shape[0] != n or not np.isfinite(coords[:, :3]).all(): return base_id, "parse-mismatch"
        return {"id": base_id, "v": v, "seq": seq, "plddt": plddt, "coords": coords}
    except Exception as e:
        return base_id, f"parse: {str(e)[:40]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROJECT / "data" / "genie2_afdb_index.txt"))
    ap.add_argument("--out", default=str(DATA / "dataset_afdb_train.h5"))
    ap.add_argument("--exclude", default=str(DATA / "dataset_100k.h5"), help="skip ids present in this file")
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enc-batch", type=int, default=64); ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="k/N: build only every N-th entry starting at k (separate --out per shard)")
    a = ap.parse_args()
    ids = [l.strip().replace("-model_v4", "") for l in open(a.index) if l.strip()]
    have = set()
    if a.exclude and Path(a.exclude).exists():
        with h5py.File(a.exclude, "r") as h:
            for s in h.keys(): have |= {n.split("-model_")[0] for n in h[s].keys()}
    out = h5py.File(a.out, "a"); grp = out.require_group("train")
    done = {n.split("-model_")[0] for n in grp.keys()}
    todo = [i for i in ids if i not in have and i not in done]
    random.seed(0); random.shuffle(todo)
    k, N = (int(x) for x in a.shard.split("/")); todo = todo[k::N]
    if a.limit: todo = todo[:a.limit]
    print(f"index {len(ids)}, excluded {len(have)}, already built {len(done)}, to do {len(todo)}", flush=True)

    os.chdir(ROOT + "/ProteinAE_v1")
    import lightning as Lt, hydra
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
        """items: list of dicts with canonical (n,37,3) coords. Returns list of (n,8) latents.
        Mirrors gate6_100k.step3_encode exactly, batched."""
        datas = [Data(coords=torch.from_numpy(it["coords"]), id=it["id"]) for it in items]
        loader = DensePaddingDataLoader(datas, batch_size=len(datas))
        batch = next(iter(loader))
        x_1 = rearrange(batch["coords"][:, :, BB, :], "b n c d -> b (n c) d")
        coords_mask = batch["mask_dict"]["coords"][..., BB, 0]
        mask = coords_mask[..., 1]
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
        zs = encode(buf)
        for it, z in zip(buf, zs):
            g = grp.create_group(f"{it['id']}-model_{it['v']}")
            g.create_dataset("z", data=z.astype(np.float32), compression="gzip", compression_opts=4)
            g.create_dataset("ca_coords", data=it["coords"][:, 1, :], compression="gzip", compression_opts=4)
            g.create_dataset("backbone", data=it["coords"][:, [0, 1, 2], :], compression="gzip", compression_opts=4)
            g.attrs["sequence"] = it["seq"]; g.attrs["plddt_mean"] = it["plddt"]; g.attrs["afdb_version"] = it["v"]
            g.attrs["n_residues"] = it["coords"].shape[0]
            n_ok += 1
        buf = []
        out.flush()

    with Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, todo, chunksize=8)):
            if isinstance(r, dict):
                buf.append(r)
                if len(buf) >= a.enc_batch: flush()
            else:
                fails[r[1].split(":")[0].split(" ")[0]] = fails.get(r[1].split(":")[0].split(" ")[0], 0) + 1
            if (k + 1) % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"  {k+1}/{len(todo)}  ok {n_ok}  fails {fails}  {el:.0f}s  {(k+1)/el:.1f}/s  "
                      f"eta {(len(todo)-k-1)/((k+1)/el)/60:.0f} min", flush=True)
    flush()
    out.attrs["source"] = "genie2 afdbreps_l-256_plddt_80, EBI AFDB v6, minus dataset_100k ids"
    out.close()
    print(f"done: {n_ok} written, fails {fails}, {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
