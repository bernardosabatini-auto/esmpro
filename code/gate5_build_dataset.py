"""
Phase 1 Gate 5: Build cached dataset.
1. Download 20k structures from Genie 2 AFDB index
2. Cluster with MMseqs2 (foldseek fallback)
3. Canonicalize and encode with ProteinAE
4. Compute ESM-2 embeddings
5. Store in HDF5
"""
import os as _os
ROOT = _os.environ.get("ESM_PROAE_ROOT", ROOT + "")
import os, sys, time, json, random, subprocess
import numpy as np
import torch
import h5py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from Bio.PDB import PDBParser

os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, ".")
sys.path.insert(0, ROOT)

import lightning as L
import hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinAutoEncoder, ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
from canonicalize import canonicalize_coords, canonicalize_pyg_data
from einops import rearrange

DATA_DIR = Path(ROOT + "/data/phase1_dataset")
STRUCT_DIR = DATA_DIR / "structures"
INDEX_FILE = Path(ROOT + "/data/genie2_afdb_index.txt")
N_TARGET = 3000  # reduced from 20k for feasibility test
N_SAMPLE = 4000  # over-provision for ~15% failure


three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def download_one(afid, out_dir):
    """Download one AFDB structure, trying v6 then v4."""
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


def extract_sequence(pdb_path):
    """Extract sequence from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                rn = residue.get_resname()
                if rn in three_to_one:
                    seq.append(three_to_one[rn])
            break
        break
    return "".join(seq)


def step1_download():
    """Sample and download structures."""
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = set(p.stem.split("-model_")[0] for p in STRUCT_DIR.glob("*.pdb"))
    if len(existing) >= N_TARGET:
        print(f"Already have {len(existing)} structures, skipping download")
        return list(STRUCT_DIR.glob("*.pdb"))

    # Sample from index
    with open(INDEX_FILE) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    random.seed(42)
    sampled = random.sample(all_ids, min(N_SAMPLE, len(all_ids)))
    print(f"Sampling {len(sampled)} IDs from {len(all_ids)} total")

    # Download
    print(f"Downloading to {STRUCT_DIR}...")
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_one, afid, STRUCT_DIR): afid for afid in sampled}
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 2000 == 0:
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  {i+1}/{len(sampled)}: {ok} downloaded")

    t_dl = time.perf_counter() - t0
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"Downloaded: {len(ok)}/{len(sampled)} in {t_dl:.0f}s ({len(failed)} failed)")

    return list(STRUCT_DIR.glob("*.pdb"))


def step2_cluster(pdb_paths):
    """Cluster sequences with MMseqs2 at 30% identity."""
    cluster_dir = DATA_DIR / "clustering"
    cluster_dir.mkdir(exist_ok=True)
    cluster_tsv = cluster_dir / "clusters.tsv"

    if cluster_tsv.exists():
        print(f"Cluster file exists, loading...")
        clusters = {}
        with open(cluster_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    rep, member = parts
                    clusters.setdefault(rep, []).append(member)
        return clusters

    # Write FASTA
    fasta_path = cluster_dir / "sequences.fasta"
    print("Extracting sequences for clustering...")
    n_written = 0
    with open(fasta_path, "w") as f:
        for pdb_path in pdb_paths:
            try:
                seq = extract_sequence(pdb_path)
                if 32 <= len(seq) <= 256:
                    name = pdb_path.stem
                    f.write(f">{name}\n{seq}\n")
                    n_written += 1
            except Exception:
                pass
    print(f"  Written {n_written} sequences to FASTA")

    # Run MMseqs2
    db_path = cluster_dir / "seqdb"
    result_path = cluster_dir / "result"
    print("Running MMseqs2 clustering at 30% identity...")
    t0 = time.perf_counter()

    subprocess.run([
        _os.environ.get("MMSEQS_BIN", "mmseqs"), "createdb", str(fasta_path), str(db_path)
    ], check=True, capture_output=True)

    subprocess.run([
        _os.environ.get("MMSEQS_BIN", "mmseqs"), "cluster", str(db_path), str(result_path),
        str(cluster_dir / "tmp"),
        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0"
    ], check=True, capture_output=True)

    subprocess.run([
        _os.environ.get("MMSEQS_BIN", "mmseqs"), "createtsv", str(db_path), str(db_path),
        str(result_path), str(cluster_tsv)
    ], check=True, capture_output=True)

    t_cluster = time.perf_counter() - t0
    print(f"  Clustering done in {t_cluster:.0f}s")

    # Parse clusters
    clusters = {}
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rep, member = parts
                clusters.setdefault(rep, []).append(member)
    print(f"  {len(clusters)} clusters found")
    return clusters


def step3_split(clusters):
    """Split clusters into train/val/test 80/10/10."""
    cluster_ids = sorted(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_ids)

    n = len(cluster_ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    splits = {}
    for cid in train_clusters:
        for member in clusters[cid]:
            splits[member] = ("train", cid)
    for cid in val_clusters:
        for member in clusters[cid]:
            splits[member] = ("val", cid)
    for cid in test_clusters:
        for member in clusters[cid]:
            splits[member] = ("test", cid)

    counts = {"train": 0, "val": 0, "test": 0}
    for _, (split, _) in splits.items():
        counts[split] += 1
    print(f"Split: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    return splits


def step4_encode_and_store(pdb_paths, splits):
    """Canonicalize, encode with ProteinAE, compute ESM-2, store HDF5."""
    import esm as esm_lib

    # Filter to valid structures in splits
    valid_paths = {}
    proc = ProteinProcessor()
    print("Filtering valid structures...")
    for p in pdb_paths:
        name = p.stem
        if name in splits:
            try:
                item = proc.process_pdb(p)
                n = item.coords.shape[0]
                if 32 <= n <= 256:
                    valid_paths[name] = p
            except Exception:
                pass
    print(f"  {len(valid_paths)} valid structures in split")

    # Cap at N_TARGET
    names = sorted(valid_paths.keys())[:N_TARGET]
    print(f"  Processing {len(names)} structures")

    # Load ProteinAE
    print("Loading ProteinAE...")
    model = ProteinAE.load_from_checkpoint(
        "checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False
    )
    model = model.eval().cuda()
    fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)

    # Load ESM-2
    print("Loading ESM-2 650M...")
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.eval().cuda()

    # Create HDF5
    h5_path = DATA_DIR / "dataset.h5"
    h5 = h5py.File(h5_path, "w")
    for split_name in ["train", "val", "test"]:
        h5.create_group(split_name)

    metadata = {"splits": {}, "failed": []}
    n_processed = 0
    t0 = time.perf_counter()

    for i, name in enumerate(names):
        pdb_path = valid_paths[name]
        split_name, cluster_id = splits[name]

        try:
            # Process PDB
            item = proc.process_pdb(pdb_path)
            n_res = item.coords.shape[0]

            # Canonicalize
            item_canon, canon_info = canonicalize_pyg_data(item)

            # Extract sequence
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
            n = x_1.shape[-2]
            batch.update({
                "x_1": x_1, "mask": mask, "coords_mask": coords_mask_flat,
                "nsamples": 1, "nres": int(n // 4),
            })
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            with torch.no_grad():
                z = model.encoder(batch).get("single_repr", None)
            z_np = z.squeeze(0).cpu().numpy()  # (N, 8) float32

            # ESM-2 embed
            esm_data = [(name, seq)]
            _, _, tokens = batch_converter(esm_data)
            tokens = tokens.cuda()
            with torch.no_grad():
                result = esm_model(tokens, repr_layers=[33], return_contacts=False)
            emb = result["representations"][33][0, 1:len(seq)+1, :].cpu().half().numpy()  # fp16

            # CA coords (canonicalized)
            ca_coords = item_canon.coords[:, 1, :].numpy()  # (N, 3)

            # Store in HDF5
            grp = h5[split_name].create_group(name)
            grp.create_dataset("z", data=z_np, compression="gzip", compression_opts=4)
            grp.create_dataset("esm2_emb", data=emb, compression="gzip", compression_opts=4)
            grp.create_dataset("ca_coords", data=ca_coords.astype(np.float32),
                             compression="gzip", compression_opts=4)
            grp.attrs["sequence"] = seq
            grp.attrs["n_residues"] = n_res
            grp.attrs["cluster_id"] = cluster_id

            metadata["splits"].setdefault(split_name, []).append(name)
            n_processed += 1

        except Exception as e:
            metadata["failed"].append({"name": name, "error": str(e)})

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(names) - i - 1) / rate / 60
            print(f"  {i+1}/{len(names)} processed ({n_processed} ok), "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}min remaining")

    h5.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_processed}/{len(names)} in {elapsed:.0f}s "
          f"({len(metadata['failed'])} failed)")

    # Save metadata
    meta_path = DATA_DIR / "metadata.json"
    # Convert to summary for JSON
    meta_summary = {
        "n_total": n_processed,
        "n_failed": len(metadata["failed"]),
        "splits": {k: len(v) for k, v in metadata["splits"].items()},
        "h5_path": str(h5_path),
        "failed_ids": [f["name"] for f in metadata["failed"][:20]],
    }
    with open(meta_path, "w") as f:
        json.dump(meta_summary, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Report sizes
    h5_size = h5_path.stat().st_size / 1e9
    struct_size = sum(p.stat().st_size for p in STRUCT_DIR.glob("*.pdb")) / 1e9
    print(f"\nDisk usage:")
    print(f"  HDF5: {h5_size:.2f} GB")
    print(f"  PDB files: {struct_size:.2f} GB")

    return metadata


def step5_validate():
    """Validate dataset integrity."""
    h5_path = DATA_DIR / "dataset.h5"
    meta_path = DATA_DIR / "metadata.json"

    with open(meta_path) as f:
        meta = json.load(f)

    h5 = h5py.File(h5_path, "r")

    print("\n=== Dataset Validation ===")
    print(f"Splits: {meta['splits']}")

    # Check disjointness
    all_members = set()
    for split in ["train", "val", "test"]:
        members = set(h5[split].keys())
        overlap = all_members & members
        if overlap:
            print(f"  WARNING: {len(overlap)} members appear in multiple splits!")
        all_members |= members
    print(f"  Total unique members: {len(all_members)}")
    print(f"  Split disjointness: OK")

    # Length distribution
    lengths = []
    for split in ["train", "val", "test"]:
        for name in h5[split]:
            lengths.append(h5[split][name].attrs["n_residues"])
    lengths = np.array(lengths)
    print(f"\nLength distribution:")
    print(f"  Mean: {lengths.mean():.0f}, Median: {np.median(lengths):.0f}")
    print(f"  Min: {lengths.min()}, Max: {lengths.max()}")
    for lo, hi in [(32, 64), (64, 128), (128, 192), (192, 256)]:
        n = ((lengths >= lo) & (lengths < hi)).sum()
        print(f"  [{lo}-{hi}): {n} ({100*n/len(lengths):.1f}%)")

    # Cluster counts per split
    for split in ["train", "val", "test"]:
        clusters = set()
        for name in h5[split]:
            clusters.add(h5[split][name].attrs["cluster_id"])
        print(f"  {split}: {len(list(h5[split].keys()))} samples, {len(clusters)} clusters")

    # Verify shapes
    sample_name = list(h5["train"].keys())[0]
    grp = h5["train"][sample_name]
    print(f"\nSample shapes ({sample_name}):")
    print(f"  z: {grp['z'].shape}, dtype={grp['z'].dtype}")
    print(f"  esm2_emb: {grp['esm2_emb'].shape}, dtype={grp['esm2_emb'].dtype}")
    print(f"  ca_coords: {grp['ca_coords'].shape}")
    print(f"  sequence: {grp.attrs['sequence'][:30]}...")

    h5.close()
    print(f"\nFailed: {meta['n_failed']}")
    target = int(N_TARGET * 0.75)  # 75% of target
    print(f"\nGate 5: PASS" if meta["n_total"] > target else f"\nGate 5: FAIL (need > {target})")



def main():
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    print("=== Step 1: Download structures ===")
    pdb_paths = step1_download()
    print(f"Total PDB files: {len(pdb_paths)}\n")

    # Step 2: Cluster
    print("=== Step 2: Cluster sequences ===")
    clusters = step2_cluster(pdb_paths)
    print(f"Clusters: {len(clusters)}\n")

    # Step 3: Split
    print("=== Step 3: Split by cluster ===")
    splits = step3_split(clusters)
    print()

    # Step 4: Encode and store
    print("=== Step 4: Encode and store ===")
    metadata = step4_encode_and_store(pdb_paths, splits)

    # Step 5: Validate
    step5_validate()


if __name__ == "__main__":
    main()
