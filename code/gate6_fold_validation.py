"""
Validate the hand-rolled FAPE against established fold measures.

FAPE-on-Ca-pseudo-frames improved (0.958 -> 0.926) while Kabsch RMSD did not.
That decoupling suggests the metric may reward local backbone geometry without
the global fold. This scores the same models with TM-score (Foldseek TMalign)
and 3Di identity, which are established fold descriptors, to find out.

Both ground truth and predictions get the SAME pseudo-backbone construction
from Ca, so the comparison is symmetric.
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, json, shutil, subprocess, tempfile
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
import h5py
from proteinfoundation.proteinflow.proteinae import ProteinAE
from gate6_fape_train import DifferentiableDecoder, ProteinDatasetFAPE, fape_loss
from gate6_deep_head import DeepHead
from gate6_3di_eval import (ca_to_pseudo_backbone, save_backbone_pdb,
                            get_3di_string, compare_3di)

PROJECT = Path(ROOT)
H5_PATH = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"
CKPT = PROJECT / "data" / "phase1_dataset"
FOLDSEEK = _FOLDSEEK
ODE_STEPS, N_PROT = 20, 100


def write_ca_as_pdb(ca, path):
    n = len(ca)
    a37 = np.zeros((n, 37, 3), np.float32)
    bb = ca_to_pseudo_backbone(ca)
    a37[:, 0], a37[:, 1], a37[:, 2] = bb[:, 0], bb[:, 1], bb[:, 2]
    save_backbone_pdb(a37, path)


def tm_scores(pred_dir, gt_dir, tmp):
    """Foldseek TMalign, matched pairs only. Returns {name: tmscore}."""
    os.makedirs(os.path.join(tmp, "fstmp"), exist_ok=True)
    out = os.path.join(tmp, "aln.tsv")
    r = subprocess.run([FOLDSEEK, "easy-search", pred_dir, gt_dir, out,
                        os.path.join(tmp, "fstmp"), "--alignment-type", "1",
                        "-e", "inf", "--max-seqs", "2000", "--exact-tmscore", "1",
                        # Without this, Foldseek's 3Di k-mer prefilter silently drops
                        # structures with no 3Di signal, so they never reach TMalign
                        # and get imputed as 0. That makes TM a function of 3Di.
                        "--exhaustive-search", "1",
                        "--format-output", "query,target,alntmscore,qtmscore,ttmscore"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"foldseek easy-search failed: {r.stderr[-400:]}")
    best = {}
    with open(out) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0].replace(".pdb", ""), p[1].replace(".pdb", "")
            if q != t:
                continue
            best[q] = max(float(p[3]), float(p[4]))   # qtmscore / ttmscore
    return best


def main():
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    print("=" * 76)
    print(f"Fold validation: TM-score + 3Di vs hand-rolled FAPE ({N_PROT} proteins)")
    print("=" * 76, flush=True)

    with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config",
                                     version_base=hydra.__version__):
        hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt",
                                        strict=True, weights_only=False)
    ae.eval().to(device)
    dec = DifferentiableDecoder(ae, n_steps=ODE_STEPS).to(device)

    val_full = ProteinDatasetFAPE(H5_PATH, "val")
    torch.manual_seed(42)
    idx = torch.randperm(len(val_full))[:N_PROT].tolist()
    names = [val_full.names[i] for i in idx]
    loader = DataLoader(torch.utils.data.Subset(val_full, idx), batch_size=25,
                        shuffle=False, num_workers=2, pin_memory=True)

    h5 = h5py.File(str(H5_PATH), "r")
    acc, n = np.zeros(8, np.float64), 0
    for k in list(h5["train"].keys())[:4000]:
        z = h5["train"][k]["z"][:]; acc += z.sum(0); n += z.shape[0]
    h5.close()
    mean_z = torch.tensor(acc / n, dtype=torch.float32, device=device)

    heads = {}
    for nm, norm in [("best_deep10L.pt", False), ("best_fape_fixed.pt", True)]:
        if not (CKPT / nm).exists():
            continue
        h = DeepHead(n_layers=10, d_model=256, dropout=0.15, use_conv=False,
                     normalize_out=norm).to(device)
        h.load_state_dict(torch.load(str(CKPT / nm), weights_only=True)); h.eval()
        heads[nm] = h

    work = tempfile.mkdtemp(prefix="foldval_")
    gt_dir = os.path.join(work, "gt"); os.makedirs(gt_dir)
    gt_3di, k = {}, 0
    for esm, z_true, ca, mask, lengths in loader:
        for b in range(ca.shape[0]):
            L_ = int(lengths[b]); nm_ = names[k]; k += 1
            write_ca_as_pdb(ca[b, :L_].numpy(), os.path.join(gt_dir, f"{nm_}.pdb"))
    for nm_ in names:
        gt_3di[nm_] = get_3di_string(os.path.join(gt_dir, f"{nm_}.pdb"),
                                     os.path.join(work, "db_gt", nm_))
    print(f"  ground truth: {len(gt_3di)} structures written\n", flush=True)

    arms = ["true_z", "mean_z"] + list(heads)
    print(f"  {'model':>20s} {'TM(meas)':>9s} {'coverage':>8s} {'TM>0.5':>7s} "
          f"{'3Di id':>7s} {'FAPE':>7s}")
    results = []
    for arm in arms:
        d = os.path.join(work, arm); os.makedirs(d, exist_ok=True)
        ff, nb, k = 0.0, 0, 0
        with torch.no_grad():
            for esm, z_true, ca, mask, lengths in loader:
                esm, z_true = esm.to(device), z_true.to(device)
                ca_d, mask = ca.to(device), mask.to(device)
                B, N, _ = z_true.shape
                if arm == "true_z":
                    z = z_true
                elif arm == "mean_z":
                    z = F.layer_norm(mean_z.view(1, 1, 8).expand(B, N, 8), (8,))
                else:
                    z = heads[arm](esm, mask)
                pred = dec(z, mask)
                ff += fape_loss(pred, ca_d, mask).item(); nb += 1
                for b in range(pred.shape[0]):
                    L_ = int(lengths[b]); nm_ = names[k]; k += 1
                    write_ca_as_pdb(pred[b, :L_].cpu().numpy(),
                                    os.path.join(d, f"{nm_}.pdb"))
        tms = tm_scores(d, gt_dir, os.path.join(work, f"tm_{arm}"))
        ids = []
        for nm_ in names:
            try:
                ids.append(compare_3di(get_3di_string(os.path.join(d, f"{nm_}.pdb"),
                                                      os.path.join(work, f"db_{arm}", nm_)),
                                       gt_3di[nm_]))
            except Exception:
                pass
        measured = [tms[nm_] for nm_ in names if nm_ in tms]
        cover = len(measured) / len(names)
        r = {"model": arm,
             "tm_mean_measured": float(np.mean(measured)) if measured else None,
             "tm_median_measured": float(np.median(measured)) if measured else None,
             "tm_coverage": cover,
             "tm_frac_above_0.5": float(np.mean([tms.get(n_, 0.0) > 0.5 for n_ in names])),
             "di3_mean": float(np.mean(ids)) if ids else None,
             "fape_fixed": ff / nb}
        results.append(r)
        print(f"  {arm:>20s} {(r['tm_mean_measured'] or float('nan')):9.3f} "
              f"{cover:8.2f} {r['tm_frac_above_0.5']:7.2f} "
              f"{(r['di3_mean'] or 0):7.3f} {r['fape_fixed']:7.3f}", flush=True)

    print("\n  TM-score: >0.5 = same fold, <0.3 = unrelated. 3Di random ~0.05.")
    print("  Reference: ProteinAE reconstruction 3Di id was 0.83 (2026-09-13).")
    with open(PROJECT / "notes" / "gate6_fold_validation_results.json", "w") as f:
        json.dump({"ode_steps": ODE_STEPS, "n_proteins": N_PROT,
                   "results": results}, f, indent=2)
    print("\nSaved notes/gate6_fold_validation_results.json")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
