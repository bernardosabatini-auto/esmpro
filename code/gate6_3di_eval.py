"""
3Di invariant evaluation: compare ground truth, ProteinAE reconstruction,
and head prediction using Foldseek's rotation-invariant 3Di alphabet.

For each test protein, produces 3 structures:
  A) Ground truth (from AFDB)
  B) ProteinAE reconstruction (encode ground truth → z → decode)
  C) Head prediction (ESM → head → predicted z → decode)

Then computes pairwise 3Di similarity via Foldseek structurealphabet.
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, json, time, tempfile, subprocess
import numpy as np
import torch
import h5py
from pathlib import Path
os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, ".")
sys.path.insert(0, ROOT)

import lightning as L
import hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinAutoEncoder
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from gate6_deep_head import DeepHead

PROJECT = Path(ROOT)
H5_100K = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"
FOLDSEEK = _FOLDSEEK


def decode_z_tensor(autoencoder, z_np, cfg):
    """Decode z (N,8) → atom37 coords (N,37,3)."""
    cfg.ae_mode = "decode"
    autoencoder.model.configure_inference(cfg, nn_ag=None)
    z_t = torch.from_numpy(z_np).float()
    item = {"single_repr": z_t, "mask": torch.ones(z_t.shape[0], dtype=torch.bool)}
    loader = DensePaddingDataLoader([item])
    preds = autoencoder.trainer.predict(autoencoder.model, loader)
    return preds[0]["pred_coords"][0]  # (N, 37, 3)


def encode_structure(autoencoder, ca_coords, cfg):
    """Encode Ca coords (N,3) → z (N,8) via ProteinAE encoder."""
    cfg.ae_mode = "encode"
    autoencoder.model.configure_inference(cfg, nn_ag=None)
    # Build atom37 from Ca only (index 1)
    n = ca_coords.shape[0]
    atom37 = np.zeros((n, 37, 3), dtype=np.float32)
    atom37[:, 1, :] = ca_coords
    atom37_mask = np.zeros((n, 37), dtype=np.float32)
    atom37_mask[:, 1] = 1.0
    item = {
        "coords": torch.from_numpy(atom37).float(),
        "coords_mask": torch.from_numpy(atom37_mask).float(),
        "mask": torch.ones(n, dtype=torch.bool),
    }
    loader = DensePaddingDataLoader([item])
    preds = autoencoder.trainer.predict(autoencoder.model, loader)
    return preds[0]["single_repr"][0].numpy()[:n]  # (N, 8)


def save_backbone_pdb(atom37_coords, path):
    """Save N, Ca, C backbone atoms as PDB. atom37_coords: (N, 37, 3)."""
    atom_names = {0: (" N  ", "N"), 1: (" CA ", "C"), 2: (" C  ", "C")}
    with open(path, "w") as f:
        serial = 1
        for i in range(atom37_coords.shape[0]):
            for aidx, (aname, element) in atom_names.items():
                x, y, z = atom37_coords[i, aidx]
                f.write(f"ATOM  {serial:5d} {aname} ALA A{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}\n")
                serial += 1
        f.write("END\n")


def ca_to_pseudo_backbone(ca_coords):
    """Build pseudo N, Ca, C from Ca-only trace using ideal geometry.
    Returns (N, 3, 3) array with [N, Ca, C] per residue."""
    n = len(ca_coords)
    backbone = np.zeros((n, 3, 3), dtype=np.float32)
    backbone[:, 1, :] = ca_coords  # Ca

    for i in range(n):
        ca = ca_coords[i]
        # Direction to next/prev Ca for placing N and C
        if i > 0 and i < n - 1:
            d_prev = ca_coords[i-1] - ca
            d_next = ca_coords[i+1] - ca
        elif i == 0 and n > 1:
            d_prev = -(ca_coords[1] - ca)
            d_next = ca_coords[1] - ca
        else:
            d_prev = np.array([1.0, 0.0, 0.0])
            d_next = np.array([-1.0, 0.0, 0.0])

        # Normalize and place N/C at ~1.47 A from Ca
        d_prev_n = d_prev / (np.linalg.norm(d_prev) + 1e-8)
        d_next_n = d_next / (np.linalg.norm(d_next) + 1e-8)
        backbone[i, 0, :] = ca + d_prev_n * 1.47  # N
        backbone[i, 2, :] = ca + d_next_n * 1.52  # C

    return backbone


def save_pseudo_backbone_pdb(ca_coords, path):
    """Build pseudo-backbone from Ca trace and save as PDB."""
    bb = ca_to_pseudo_backbone(ca_coords)
    # Convert to atom37-like shape for save_backbone_pdb
    n = len(ca_coords)
    atom37 = np.zeros((n, 37, 3), dtype=np.float32)
    atom37[:, 0, :] = bb[:, 0, :]  # N
    atom37[:, 1, :] = bb[:, 1, :]  # Ca
    atom37[:, 2, :] = bb[:, 2, :]  # C
    save_backbone_pdb(atom37, path)


def get_3di_string(pdb_path, tmpdir):
    """Run Foldseek structurealphabet on a PDB file, return 3Di string."""
    os.makedirs(tmpdir, exist_ok=True)
    db_path = os.path.join(tmpdir, "db")
    # Create Foldseek DB from single PDB
    result = subprocess.run([FOLDSEEK, "createdb", pdb_path, db_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"createdb failed: {result.stderr}")
    # Link header DB to _ss DB (required by convert2fasta)
    subprocess.run([FOLDSEEK, "lndb", db_path + "_h", db_path + "_ss_h"],
                   capture_output=True, check=True)
    # Extract 3Di sequence
    fasta_out = os.path.join(tmpdir, "3di.fasta")
    result = subprocess.run(
        [FOLDSEEK, "convert2fasta", db_path + "_ss", fasta_out],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"convert2fasta failed: {result.stderr}")
    with open(fasta_out) as f:
        lines = f.read().strip().split("\n")
    seq_3di = "".join(l for l in lines if not l.startswith(">"))
    return seq_3di


def compare_3di(s1, s2):
    """Fraction of matching 3Di characters (identity)."""
    n = min(len(s1), len(s2))
    if n == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(s1[:n], s2[:n]))
    return matches / n


def main():
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    device = torch.device("cuda")

    print("=" * 60)
    print("3Di invariant evaluation")
    print("=" * 60)

    # Load ProteinAE
    print("Loading ProteinAE...")
    with hydra.initialize_config_dir(
        config_dir=f"{os.getcwd()}/configs/experiment_config",
        version_base=hydra.__version__,
    ):
        cfg = hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae_model = ProteinAE.load_from_checkpoint(
        "checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False
    )
    trainer = L.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
    autoencoder = ProteinAutoEncoder(ae_model, trainer)

    # Load best head (deep10L)
    print("Loading deep10L head...")
    head = DeepHead(n_layers=10, d_model=256, dropout=0.15, use_conv=False).to(device)
    head.load_state_dict(torch.load(
        str(PROJECT / "data" / "phase1_dataset" / "best_deep10L.pt"), weights_only=True))
    head.eval()

    # Sample test proteins
    h5 = h5py.File(str(H5_100K), "r")
    test_names = list(h5["test"].keys())
    np.random.seed(42)
    sample_idx = np.random.choice(len(test_names), size=30, replace=False)
    sample_names = [test_names[i] for i in sample_idx]

    print(f"\nEvaluating {len(sample_names)} test proteins")
    print(f"{'Name':>25s}  {'N':>4s}  {'GT↔Recon':>8s}  {'GT↔Pred':>8s}  {'Recon↔Pred':>8s}")

    results = []
    for name in sample_names:
        grp = h5["test"][name]
        esm_emb = grp["esm2_emb"][:].astype(np.float32)
        z_true = grp["z"][:]
        ca_native = grp["ca_coords"][:]
        n_res = z_true.shape[0]

        # Predict z with head
        esm_t = torch.from_numpy(esm_emb).unsqueeze(0).to(device)
        mask = torch.ones(1, n_res, dtype=torch.bool).to(device)
        # Pad to 256 for the head
        if n_res < 256:
            esm_padded = torch.zeros(1, 256, 1280, device=device)
            esm_padded[0, :n_res] = esm_t[0]
            mask_padded = torch.zeros(1, 256, dtype=torch.bool, device=device)
            mask_padded[0, :n_res] = True
        else:
            esm_padded = esm_t[:, :256]
            mask_padded = torch.ones(1, 256, dtype=torch.bool, device=device)
            n_res = 256

        with torch.no_grad():
            z_pred = head(esm_padded, mask_padded).squeeze(0).cpu().numpy()[:n_res]

        # Decode both z_true and z_pred
        try:
            coords_recon = decode_z_tensor(autoencoder, z_true, cfg)
            coords_pred = decode_z_tensor(autoencoder, z_pred, cfg)
        except Exception as e:
            print(f"{name:>25s}  {n_res:4d}  DECODE ERROR: {e}")
            continue

        # Get 3Di strings for all three
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_pdb = os.path.join(tmpdir, "gt.pdb")
            recon_pdb = os.path.join(tmpdir, "recon.pdb")
            pred_pdb = os.path.join(tmpdir, "pred.pdb")

            save_pseudo_backbone_pdb(ca_native[:n_res], gt_pdb)
            save_backbone_pdb(coords_recon[:n_res].numpy(), recon_pdb)
            save_backbone_pdb(coords_pred[:n_res].numpy(), pred_pdb)

            try:
                s_gt = get_3di_string(gt_pdb, os.path.join(tmpdir, "gt_db"))
                s_recon = get_3di_string(recon_pdb, os.path.join(tmpdir, "recon_db"))
                s_pred = get_3di_string(pred_pdb, os.path.join(tmpdir, "pred_db"))
            except Exception as e:
                print(f"{name:>25s}  {n_res:4d}  FOLDSEEK ERROR: {e}")
                continue

        gt_recon = compare_3di(s_gt, s_recon)
        gt_pred = compare_3di(s_gt, s_pred)
        recon_pred = compare_3di(s_recon, s_pred)

        results.append({
            "name": name, "n_res": n_res,
            "gt_recon_3di": gt_recon, "gt_pred_3di": gt_pred,
            "recon_pred_3di": recon_pred,
            "s_gt": s_gt, "s_recon": s_recon, "s_pred": s_pred,
        })
        print(f"{name:>25s}  {n_res:4d}  {gt_recon:8.3f}  {gt_pred:8.3f}  {recon_pred:8.3f}")

    h5.close()

    # Summary
    print(f"\n{'='*60}")
    print("Summary (3Di identity)")
    print(f"{'='*60}")
    if results:
        gt_recon = [r["gt_recon_3di"] for r in results]
        gt_pred = [r["gt_pred_3di"] for r in results]
        recon_pred = [r["recon_pred_3di"] for r in results]

        print(f"  {'Comparison':>20s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}")
        print(f"  {'GT ↔ Recon':>20s}  {np.mean(gt_recon):8.3f}  {np.median(gt_recon):8.3f}  {np.std(gt_recon):8.3f}")
        print(f"  {'GT ↔ Pred':>20s}  {np.mean(gt_pred):8.3f}  {np.median(gt_pred):8.3f}  {np.std(gt_pred):8.3f}")
        print(f"  {'Recon ↔ Pred':>20s}  {np.mean(recon_pred):8.3f}  {np.median(recon_pred):8.3f}  {np.std(recon_pred):8.3f}")

        print(f"\n  Interpretation:")
        print(f"    GT↔Recon: how faithful is the autoencoder?")
        print(f"    GT↔Pred:  how good is our full pipeline?")
        print(f"    Recon↔Pred: how much does the head degrade structure?")

    out = {"gate6_3di_eval": {"n_test": len(results), "per_target": results}}
    out_path = str(PROJECT / "notes" / "gate6_3di_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
