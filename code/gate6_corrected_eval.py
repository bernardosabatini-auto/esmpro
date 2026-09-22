"""
Re-score every model with POSE-INVARIANT metrics after the FAPE bug fix.

The previous fape_loss expressed both structures in the TRUE frames, which
cancels the frame index and reduces to un-superposed coordinate error. It
scored a perfect-but-rotated structure at 0.95, so every number produced with
it is invalid.

Reports per model:
  fape_fixed  - pose-invariant FAPE (each structure in its own frames)
  rmsd_kabsch - optimally superposed Ca RMSD, Angstroms (interpretable)
  fape_broken - the old metric, for reference against historical numbers
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, ".")
sys.path.insert(0, ROOT)

import lightning as L
import hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
from gate6_fape_train import (DifferentiableDecoder, ProteinDatasetFAPE,
                              fape_loss, pose_sensitive_coord_error)
from gate6_deep_head import DeepHead

PROJECT = Path(ROOT)
H5_PATH = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"
CKPT = PROJECT / "data" / "phase1_dataset"
ODE_STEPS = 20          # the decoder's design value, for a definitive read
N_PROTEINS = 200


def kabsch_rmsd(P, Q, mask):
    """Optimally superposed RMSD per structure. P,Q: (B,N,3), mask: (B,N)."""
    out = []
    for b in range(P.shape[0]):
        m = mask[b]
        if m.sum() < 3:
            out.append(float("nan")); continue
        p = P[b][m].double(); q = Q[b][m].double()
        p = p - p.mean(0, keepdim=True)
        q = q - q.mean(0, keepdim=True)
        U, S, Vt = torch.linalg.svd(p.T @ q)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64, device=P.device))
        R = Vt.T @ D @ U.T
        out.append(torch.sqrt((((R @ p.T).T - q) ** 2).sum(-1).mean()).item())
    return out


def main():
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    print("=" * 78)
    print(f"Corrected evaluation: pose-invariant metrics, {ODE_STEPS} ODE steps")
    print("=" * 78, flush=True)

    with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config",
                                     version_base=hydra.__version__):
        hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt",
                                        strict=True, weights_only=False)
    ae.eval().to(device)
    dec = DifferentiableDecoder(ae, n_steps=ODE_STEPS).to(device)

    val_full = ProteinDatasetFAPE(H5_PATH, "val")
    torch.manual_seed(42)
    idx = torch.randperm(len(val_full))[:N_PROTEINS].tolist()
    loader = DataLoader(torch.utils.data.Subset(val_full, idx), batch_size=25,
                        shuffle=False, num_workers=2, pin_memory=True)

    # Mean latent over train, then projected to the manifold.
    import h5py
    h5 = h5py.File(str(H5_PATH), "r")
    acc, n = np.zeros(8, np.float64), 0
    for k in list(h5["train"].keys())[:4000]:
        z = h5["train"][k]["z"][:]; acc += z.sum(0); n += z.shape[0]
    h5.close()
    mean_z = torch.tensor(acc / n, dtype=torch.float32, device=device)

    # (checkpoint, normalize_out) — must match how each head was trained.
    heads = {}
    for nm, norm in [("best_deep10L.pt", False), ("best_fape_fixed.pt", True)]:
        if not (CKPT / nm).exists():
            print(f"  [skip] {nm} not found"); continue
        h = DeepHead(n_layers=10, d_model=256, dropout=0.15, use_conv=False,
                     normalize_out=norm).to(device)
        h.load_state_dict(torch.load(str(CKPT / nm), weights_only=True)); h.eval()
        heads[nm] = h

    arms = ["true_z", "mean_z", "random_z"] + list(heads)
    print(f"\n  {'model':>20s} {'FAPE_fixed':>11s} {'RMSD_kabsch':>12s} {'FAPE_broken':>12s}")
    results = []
    for arm in arms:
        ff, fb, rm, nb = 0.0, 0.0, [], 0
        g = torch.Generator(device="cuda").manual_seed(7)
        with torch.no_grad():
            for esm, z_true, ca, mask, _ in loader:
                esm, z_true = esm.to(device), z_true.to(device)
                ca, mask = ca.to(device), mask.to(device)
                B, N, _ = z_true.shape
                if arm == "true_z":
                    z = z_true
                elif arm == "mean_z":
                    z = F.layer_norm(mean_z.view(1, 1, 8).expand(B, N, 8), (8,))
                elif arm == "random_z":
                    z = F.layer_norm(torch.randn(z_true.shape, device=device, generator=g), (8,))
                else:
                    z = heads[arm](esm, mask)
                pred = dec(z, mask)
                ff += fape_loss(pred, ca, mask).item()
                fb += pose_sensitive_coord_error(pred, ca, mask).item()
                rm += kabsch_rmsd(pred, ca, mask)
                nb += 1
        r = {"model": arm, "fape_fixed": ff / nb, "fape_broken": fb / nb,
             "rmsd_kabsch_mean": float(np.nanmean(rm)),
             "rmsd_kabsch_median": float(np.nanmedian(rm))}
        results.append(r)
        print(f"  {arm:>20s} {r['fape_fixed']:11.4f} {r['rmsd_kabsch_mean']:11.2f} A "
              f"{r['fape_broken']:12.4f}", flush=True)

    print(f"\n  RMSD medians: " + ", ".join(f"{r['model']}={r['rmsd_kabsch_median']:.1f}A"
                                            for r in results))
    print("  ESMFold reference: 4.33 A mean Ca RMSD")
    with open(PROJECT / "notes" / "gate6_corrected_eval_results.json", "w") as f:
        json.dump({"ode_steps": ODE_STEPS, "n_proteins": N_PROTEINS,
                   "results": results}, f, indent=2)
    print("\nSaved notes/gate6_corrected_eval_results.json")


if __name__ == "__main__":
    main()
