"""
Gate 6 (100k): Scale to 100k structures with 6-layer d256 head.
Uses existing Foldseek clustering, downloads additional structures as needed.
Also implements trimmed loss (downweight worst 10% per batch).
"""
import os as _os
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
import os, sys, time, json, random, subprocess
import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from torch.utils.data import Dataset, DataLoader

os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, ".")
sys.path.insert(0, ROOT)

import lightning as L
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
from canonicalize import canonicalize_pyg_data
from einops import rearrange
from Bio.PDB import PDBParser

PROJECT = Path(ROOT)
DATA_DIR = PROJECT / "data" / "phase1_dataset"
STRUCT_DIR = DATA_DIR / "structures"
INDEX_FILE = PROJECT / "data" / "genie2_afdb_index.txt"
FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
N_TARGET = 100000

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def extract_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", str(pdb_path))
    seq = []
    for m in s:
        for c in m:
            for r in c:
                if r.get_resname() in three_to_one:
                    seq.append(three_to_one[r.get_resname()])
            break
        break
    return "".join(seq)


def fast_residue_count(pdb_path):
    residues = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                residues.add((line[21], line[22:27].strip()))
            if line.startswith("ENDMDL"):
                break
    return len(residues)


def download_one(afid, out_dir):
    base_id = afid.replace("-model_v4", "")
    for version in ["v6", "v4"]:
        fname = f"{base_id}-model_{version}.pdb"
        out_path = out_dir / fname
        if out_path.exists():
            return {"id": base_id, "path": str(out_path), "status": "ok"}
        url = f"https://alphafold.ebi.ac.uk/files/{base_id}-model_{version}.pdb"
        try:
            urllib.request.urlretrieve(url, out_path)
            return {"id": base_id, "path": str(out_path), "status": "ok"}
        except Exception:
            continue
    return {"id": base_id, "status": "failed"}


def step1_download():
    """Download structures up to 120k (buffer for ~15% failure rate)."""
    existing = list(STRUCT_DIR.glob("*.pdb"))
    n_existing = len(existing)
    print(f"Existing: {n_existing}")

    n_needed = int((N_TARGET - n_existing) * 1.2)
    if n_needed <= 0:
        print("Sufficient structures already downloaded.")
        return existing

    with open(INDEX_FILE) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    existing_ids = set(p.stem.split("-model_")[0] for p in existing)
    remaining = [a for a in all_ids if a.replace("-model_v4", "") not in existing_ids]
    random.seed(42)
    to_download = random.sample(remaining, min(n_needed, len(remaining)))
    print(f"Downloading {len(to_download)} additional structures...")

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_one, a, STRUCT_DIR): a for a in to_download}
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 5000 == 0:
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  {i+1}/{len(to_download)}: {ok} ok")
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "failed")
    print(f"Downloaded {ok}, failed {fail} in {time.perf_counter()-t0:.0f}s")
    return list(STRUCT_DIR.glob("*.pdb"))


def step2_cluster(pdb_paths):
    """Foldseek structural clustering."""
    cluster_dir = DATA_DIR / "clustering_foldseek_100k"
    cluster_dir.mkdir(exist_ok=True)
    cluster_tsv = cluster_dir / "clusters.tsv"

    if cluster_tsv.exists():
        print("Cluster file exists, loading...")
        clusters = {}
        with open(cluster_tsv) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) == 2:
                    clusters.setdefault(p[0], []).append(p[1])
        print(f"  {len(clusters)} clusters")
        return clusters

    pdb_dir = cluster_dir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for p in pdb_paths:
        dst = pdb_dir / p.name
        if not dst.exists():
            try:
                os.symlink(p, dst)
            except FileExistsError:
                pass

    db = cluster_dir / "db"
    res = cluster_dir / "result"
    subprocess.run([FOLDSEEK, "createdb", str(pdb_dir), str(db)],
                  check=True, capture_output=True)
    print("Clustering with Foldseek...")
    t0 = time.perf_counter()
    subprocess.run([FOLDSEEK, "cluster", str(db), str(res),
                   str(cluster_dir / "tmp"), "--min-seq-id", "0", "-c", "0.5"],
                  check=True, capture_output=True)
    subprocess.run([FOLDSEEK, "createtsv", str(db), str(db), str(res), str(cluster_tsv)],
                  check=True, capture_output=True)
    print(f"  Done in {time.perf_counter()-t0:.0f}s")

    clusters = {}
    with open(cluster_tsv) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 2:
                clusters.setdefault(p[0], []).append(p[1])
    print(f"  {len(clusters)} clusters")
    return clusters


def step3_encode(pdb_paths, clusters):
    """Encode 100k structures into HDF5."""
    from transformers import AutoTokenizer, AutoModel

    # Split
    cluster_ids = sorted(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_ids)
    n = len(cluster_ids)
    splits = {}
    for cid in cluster_ids[:int(0.8*n)]:
        for m in clusters[cid]:
            splits[m] = ("train", cid)
    for cid in cluster_ids[int(0.8*n):int(0.9*n)]:
        for m in clusters[cid]:
            splits[m] = ("val", cid)
    for cid in cluster_ids[int(0.9*n):]:
        for m in clusters[cid]:
            splits[m] = ("test", cid)

    counts = {"train": 0, "val": 0, "test": 0}
    for _, (s, _) in splits.items():
        counts[s] += 1
    print(f"Split: {counts}")

    # Filter valid
    name_to_path = {}
    for p in pdb_paths:
        name = p.stem
        if name in splits:
            try:
                nr = fast_residue_count(p)
                if 32 <= nr <= 256:
                    name_to_path[name] = p
            except Exception:
                pass
    names = sorted(name_to_path.keys())[:N_TARGET]
    print(f"  {len(names)} valid structures for encoding")

    # Load models
    print("Loading ProteinAE...")
    ae_model = ProteinAE.load_from_checkpoint(
        "checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False
    ).eval().cuda()
    proc = ProteinProcessor()
    fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)

    print("Loading ESM-2 via HuggingFace transformers...")
    esm_tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval().cuda()

    h5_path = DATA_DIR / "dataset_100k.h5"
    h5 = h5py.File(str(h5_path), "w")
    for s in ["train", "val", "test"]:
        h5.create_group(s)

    n_ok, failed = 0, []
    t0 = time.perf_counter()

    for i, name in enumerate(names):
        pdb_path = name_to_path[name]
        split_name, cluster_id = splits[name]

        try:
            item = proc.process_pdb(pdb_path)
            n_res = item.coords.shape[0]
            item_canon, _ = canonicalize_pyg_data(item)
            seq = extract_sequence(pdb_path)

            # ProteinAE encode
            loader = DensePaddingDataLoader([item_canon])
            batch = next(iter(loader))
            x_1 = batch["coords"][:, :, proc.BACKBONE_ATOM_INDICES, :]
            x_1 = rearrange(x_1, "b n c d -> b (n c) d")
            coords_mask = batch["mask_dict"]["coords"][..., proc.BACKBONE_ATOM_INDICES, 0]
            mask = coords_mask[..., 1]
            coords_mask_flat = rearrange(coords_mask, "b n c -> b (n c)")
            x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), coords_mask_flat)
            nn_atoms = x_1.shape[-2]
            batch.update({
                "x_1": x_1, "mask": mask, "coords_mask": coords_mask_flat,
                "nsamples": 1, "nres": int(nn_atoms // 4),
            })
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            with torch.no_grad():
                z = ae_model.encoder(batch).get("single_repr", None)
            z_np = z.squeeze(0).cpu().numpy()

            # ESM-2 via HuggingFace
            inputs = esm_tok([seq], return_tensors="pt", padding=False).to("cuda")
            with torch.no_grad():
                out = esm_model(**inputs)
            emb = out.last_hidden_state[0, 1:len(seq)+1, :].cpu().half().numpy()

            ca = item_canon.coords[:, 1, :].numpy()

            grp = h5[split_name].create_group(name)
            grp.create_dataset("z", data=z_np, compression="gzip", compression_opts=4)
            grp.create_dataset("esm2_emb", data=emb, compression="gzip", compression_opts=4)
            grp.create_dataset("ca_coords", data=ca.astype(np.float32),
                             compression="gzip", compression_opts=4)
            grp.attrs["sequence"] = seq
            grp.attrs["n_residues"] = n_res
            grp.attrs["cluster_id"] = cluster_id
            n_ok += 1

        except Exception as e:
            failed.append({"name": name, "error": str(e)[:80]})

        if (i + 1) % 5000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i+1) / elapsed
            eta = (len(names)-i-1) / rate / 60
            print(f"  {i+1}/{len(names)} ({n_ok} ok, {len(failed)} fail), "
                  f"{elapsed:.0f}s, ~{eta:.0f}min left")

    h5.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_ok}/{len(names)} in {elapsed:.0f}s ({len(failed)} failed)")

    meta = {"n_total": n_ok, "n_failed": len(failed), "clustering": "foldseek_structural"}
    with open(DATA_DIR / "metadata_100k.json", "w") as f:
        json.dump(meta, f, indent=2)
    return n_ok


def step4_train():
    """Train 6-layer d256 head with trimmed loss."""

    class ProteinDataset(Dataset):
        def __init__(self, h5_path, split, max_len=256):
            self.h5 = h5py.File(h5_path, "r")
            self.split = split
            self.names = list(self.h5[split].keys())
            self.max_len = max_len
        def __len__(self):
            return len(self.names)
        def __getitem__(self, idx):
            grp = self.h5[self.split][self.names[idx]]
            esm = torch.from_numpy(grp["esm2_emb"][:].astype(np.float32))
            z = torch.from_numpy(grp["z"][:])
            n = esm.shape[0]
            if n < self.max_len:
                esm = torch.cat([esm, torch.zeros(self.max_len-n, 1280)], 0)
                z = torch.cat([z, torch.zeros(self.max_len-n, 8)], 0)
            else:
                esm, z = esm[:self.max_len], z[:self.max_len]
                n = self.max_len
            mask = torch.zeros(self.max_len, dtype=torch.bool)
            mask[:n] = True
            return esm, z, mask, n

    class Head(nn.Module):
        def __init__(self, d_in=1280, d_model=256, d_out=8, n_layers=6, n_heads=8, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(d_in, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.output_proj = nn.Linear(d_model, d_out)
        def forward(self, x, mask):
            h = self.input_proj(x)
            h = self.encoder(h, src_key_padding_mask=~mask)
            return self.output_proj(h)

    h5_path = DATA_DIR / "dataset_100k.h5"
    device = torch.device("cuda")
    torch.manual_seed(42)

    train_ds = ProteinDataset(h5_path, "train")
    val_ds = ProteinDataset(h5_path, "val")
    test_ds = ProteinDataset(h5_path, "test")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Baseline
    h5 = h5py.File(str(h5_path), "r")
    z_sum, z_n = np.zeros(8, np.float64), 0
    for name in h5["train"]:
        z = h5["train"][name]["z"][:]
        z_sum += z.sum(0); z_n += z.shape[0]
    mean_z = z_sum / z_n
    bl_sum, bl_n = 0.0, 0
    for name in h5["test"]:
        z = h5["test"][name]["z"][:]
        bl_sum += ((z-mean_z)**2).sum(); bl_n += z.shape[0]*8
    baseline_mse = bl_sum/bl_n
    baseline_sigma = float(np.sqrt(baseline_mse))
    h5.close()

    print(f"\nTraining 6L-d256 head on 100k data")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Baseline sigma: {baseline_sigma:.4f}")

    model = Head().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_val, best_epoch = float('inf'), 0
    history = []
    print(f"\n{'Ep':>4s}  {'Train':>10s}  {'Val':>10s}  {'Ratio':>8s}  {'Time':>6s}")

    for epoch in range(40):
        t0 = time.perf_counter()
        model.train()
        tl, nb = 0.0, 0
        for esm, z, mask, _ in train_loader:
            esm, z, mask = esm.to(device), z.to(device), mask.to(device)
            pred = model(esm, mask)
            # Per-residue MSE, then trim worst 10% of residues per batch
            per_res = ((pred-z)**2).sum(-1)  # (B, L)
            per_res = per_res * mask.float()
            # Flatten valid residues
            valid = per_res[mask]  # (n_valid,)
            # Trim worst 10%
            n_valid = valid.numel()
            k = int(n_valid * 0.9)
            if k > 0:
                trimmed, _ = torch.topk(valid, k, largest=False)
                loss = trimmed.mean() / 8
            else:
                loss = valid.mean() / 8
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); nb += 1
        train_mse = tl/nb

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for esm, z, mask, _ in val_loader:
                esm, z, mask = esm.to(device), z.to(device), mask.to(device)
                diff = ((model(esm,mask)-z)**2).sum(-1)*mask.float()
                vs += diff.sum().item(); vn += mask.float().sum().item()*8
        val_mse = vs/vn
        scheduler.step()
        ratio = val_mse/baseline_mse
        history.append({"epoch": epoch+1, "train": train_mse, "val": val_mse, "ratio": ratio})
        print(f"{epoch+1:4d}  {train_mse:10.6f}  {val_mse:10.6f}  {ratio:8.4f}  "
              f"{time.perf_counter()-t0:6.1f}s")

        if val_mse < best_val:
            best_val = val_mse; best_epoch = epoch+1
            torch.save(model.state_dict(), str(DATA_DIR/"best_100k.pt"))
        if epoch+1 - best_epoch >= 10:
            print(f"\nEarly stopping (best: {best_epoch})"); break

    model.load_state_dict(torch.load(str(DATA_DIR/"best_100k.pt"), weights_only=True))
    model.eval()
    ts, tn = 0.0, 0
    with torch.no_grad():
        for esm, z, mask, _ in test_loader:
            esm, z, mask = esm.to(device), z.to(device), mask.to(device)
            diff = ((model(esm,mask)-z)**2).sum(-1)*mask.float()
            ts += diff.sum().item(); tn += mask.float().sum().item()*8
    test_mse = ts/tn
    test_sigma = float(np.sqrt(test_mse))

    print(f"\n{'='*60}")
    print(f"100k result: sigma={test_sigma:.4f}, ratio={test_mse/baseline_mse:.4f}")
    print(f"  vs 20k 2L-d128:  sigma=0.6206")
    print(f"  vs ESMFold:       sigma=0.4866")
    print(f"  D1 target (2A):   sigma<0.200")

    output = {"gate6_100k": {
        "n_train": len(train_ds), "n_test": len(test_ds),
        "d_model": 256, "n_layers": 6, "n_params": n_params,
        "test_sigma": test_sigma, "test_mse": test_mse,
        "baseline_sigma": baseline_sigma, "ratio": test_mse/baseline_mse,
        "best_epoch": best_epoch, "trimmed_loss": True,
        "history": history,
    }}
    with open(PROJECT/"notes"/"gate6_100k_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to notes/gate6_100k_results.json")


def main():
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    random.seed(42)

    print("=" * 60)
    print("Gate 6 (100k): Scale to 100k with 6L-d256 head + trimmed loss")
    print("=" * 60)

    print("\n=== Step 1: Download ===")
    pdb_paths = step1_download()
    print(f"Total: {len(pdb_paths)}")

    print("\n=== Step 2: Cluster ===")
    clusters = step2_cluster(pdb_paths)

    print("\n=== Step 3: Encode ===")
    n_ok = step3_encode(pdb_paths, clusters)

    print("\n=== Step 4: Train ===")
    step4_train()


if __name__ == "__main__":
    main()
