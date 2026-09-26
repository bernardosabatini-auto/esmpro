"""
Gate 6 Option C: FAPE training loop.

Train the head with a rotation-invariant structural loss instead of MSE on z.
Pipeline: ESM embedding → head → predicted z → decoder (frozen) → predicted coords
Loss: FAPE(predicted_coords, true_coords)

The decoder is a flow-matching model. We write a custom differentiable ODE loop
(Euler integration, few steps) that allows gradients to flow back through the
decoder to the head.
"""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, ".")
sys.path.insert(0, ROOT)

import lightning as L
import hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
from gate6_deep_head import DeepHead

PROJECT = Path(ROOT)
H5_PATH = PROJECT / "data" / "phase1_dataset" / "dataset_100k.h5"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ProteinDatasetFAPE(Dataset):
    """Returns ESM embeddings AND Ca coordinates for FAPE loss."""
    def __init__(self, h5_path, split, max_len=256, backbone_path=None):
        self.h5 = h5py.File(h5_path, "r")
        self.split = split
        self.names = list(self.h5[split].keys())
        self.max_len = max_len
        # Optional N/CA/C side file (backbone_100k.h5) for true-frame FAPE.
        # When given, __getitem__ returns two extra items: bb (max_len, 3, 3)
        # in Angstroms, atom order [N, CA, C], and bb_valid (bool), False for
        # the ~70 structures the extractor could not process (bb is then zero).
        self.bb = h5py.File(backbone_path, "r") if backbone_path else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        grp = self.h5[self.split][name]
        esm = torch.from_numpy(grp["esm2_emb"][:].astype(np.float32))
        z = torch.from_numpy(grp["z"][:])
        ca = torch.from_numpy(grp["ca_coords"][:])
        n = esm.shape[0]
        if n < self.max_len:
            esm = torch.cat([esm, torch.zeros(self.max_len - n, esm.shape[1])], 0)
            z = torch.cat([z, torch.zeros(self.max_len - n, 8)], 0)
            ca = torch.cat([ca, torch.zeros(self.max_len - n, 3)], 0)
        else:
            esm, z, ca = esm[:self.max_len], z[:self.max_len], ca[:self.max_len]
            n = self.max_len
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:n] = True
        if self.bb is None:
            return esm, z, ca, mask, n
        bb = torch.zeros(self.max_len, 3, 3)
        bb_valid = False
        if name in self.bb[self.split]:
            arr = torch.from_numpy(self.bb[self.split][name][:])[:n]
            if arr.shape[0] == n and torch.isfinite(arr).all():
                bb[:n] = arr
                bb_valid = True
        return esm, z, ca, mask, n, bb, bb_valid


# ---------------------------------------------------------------------------
# Lightweight FAPE loss (no OpenFold Rigid dependency)
# ---------------------------------------------------------------------------
def build_frames_from_ca(ca_coords, mask, fix_ends=False):
    """Build pseudo-frames from Ca trace for FAPE.

    fix_ends: the first residue has no Ca_{i-1} and the LAST REAL residue's
    Ca_{i+1} is a padded zero, which does not move with the protein. That
    made one frame per chain pose-dependent (measured 8.7 A error on a
    perfect structure under a rigid pose change, ~1% of the loss). With
    fix_ends the two end residues reuse the rotation of their inner
    neighbour, which is built from real atoms only; the origin stays Ca_i.

    For each residue i, define a local frame using:
      origin = Ca_i
      x-axis = Ca_{i+1} - Ca_{i-1} (normalized)
      z-axis = cross(Ca_{i+1} - Ca_i, Ca_{i-1} - Ca_i) (normalized)
      y-axis = cross(z, x)

    Args:
        ca_coords: (B, N, 3)
        mask: (B, N) bool
    Returns:
        rotations: (B, N, 3, 3) rotation matrices
        translations: (B, N, 3) origins (= Ca positions)
    """
    B, N, _ = ca_coords.shape

    # Shifted Ca positions
    ca_prev = torch.cat([ca_coords[:, :1], ca_coords[:, :-1]], dim=1)
    ca_next = torch.cat([ca_coords[:, 1:], ca_coords[:, -1:]], dim=1)

    # Local frame vectors
    d_forward = ca_next - ca_coords  # (B, N, 3)
    d_backward = ca_prev - ca_coords  # (B, N, 3)

    # x-axis: forward direction (normalized)
    x_axis = ca_next - ca_prev
    x_norm = torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-6)
    x_axis = x_axis / x_norm

    # z-axis: cross product of forward and backward (normal to plane)
    z_axis = torch.cross(d_forward, d_backward, dim=-1)
    z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True).clamp(min=1e-6)
    z_axis = z_axis / z_norm

    # y-axis: completes right-handed frame
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    # Rotation matrix: columns are local frame axes
    rotations = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, N, 3, 3)
    translations = ca_coords  # (B, N, 3)

    if fix_ends and N > 2:
        last = mask.long().sum(1).clamp(min=2) - 1                  # (B,)
        bidx = torch.arange(B, device=ca_coords.device)
        rot_first = rotations[:, 1]                                   # (B, 3, 3)
        rot_last = rotations[bidx, last - 1]                          # (B, 3, 3)
        idx = torch.arange(N, device=ca_coords.device).unsqueeze(0)  # (1, N)
        is_first = (idx == 0).unsqueeze(-1).unsqueeze(-1)
        is_last = (idx == last.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
        rotations = torch.where(is_first, rot_first.unsqueeze(1), rotations)
        rotations = torch.where(is_last, rot_last.unsqueeze(1), rotations)

    return rotations, translations


def _to_local(coords, rot, trans, N):
    """Express every point in every residue frame: out[b,i,j] = R_i^T (x_j - t_i)."""
    local = coords.unsqueeze(1) - trans.unsqueeze(2)        # (B, N_frames, N_pts, 3)
    rot_inv = rot.transpose(-1, -2).unsqueeze(2)            # (B, N_frames, 1, 3, 3)
    return torch.einsum('bfpij,bfpj->bfpi',
                        rot_inv.expand(-1, -1, N, -1, -1), local)


def fape_loss(pred_ca, true_ca, mask, clamp_distance=10.0, frac_unclamped=0.0,
              fix_ends=False):
    """Frame Aligned Point Error, invariant to global rotation and translation.

    Each structure is expressed in ITS OWN residue frames, then the two local
    coordinate sets are compared. That is what makes the metric pose-invariant:
    a global rotation Q maps R_i -> Q R_i and x_j -> Q x_j, and
    (Q R_i)^T (Q x_j - Q t_i) = R_i^T (x_j - t_i), so the local coordinates are
    unchanged.

    NOTE: an earlier version of this function used the TRUE frames for BOTH
    structures. That makes the frame index cancel algebraically, since
    R_i^T(p_j - t_i) - R_i^T(x_j - t_i) = R_i^T(p_j - x_j) and R_i is
    orthogonal, reducing the whole thing to un-superposed per-point distance
    ||p_j - x_j||. It scored a perfect-but-rotated structure at 0.95 and was
    therefore dominated by pose rather than fold. All numbers produced with it
    are invalid. See `pose_sensitive_coord_error` below to reproduce them.

    Args:
        pred_ca: (B, N, 3) predicted Ca coordinates
        true_ca: (B, N, 3) ground truth Ca coordinates
        mask: (B, N) bool mask
        clamp_distance: cutoff in Angstroms above which errors are ignored
        frac_unclamped: fraction of samples left unclamped, as in AlphaFold2,
            so that badly-placed residues keep supplying gradient
    Returns:
        scalar loss
    """
    B, N, _ = pred_ca.shape

    rot_pred, trans_pred = build_frames_from_ca(pred_ca, mask, fix_ends)
    rot_true, trans_true = build_frames_from_ca(true_ca, mask, fix_ends)

    pred_local = _to_local(pred_ca, rot_pred, trans_pred, N)
    true_local = _to_local(true_ca, rot_true, trans_true, N)

    error = torch.sqrt(((pred_local - true_local) ** 2).sum(-1) + 1e-8)

    if clamp_distance is not None:
        clamped = torch.clamp(error, max=clamp_distance)
        if frac_unclamped > 0.0:
            # Per-sample choice, matching AF2's 10% unclamped batches.
            keep = (torch.rand(B, 1, 1, device=error.device) < frac_unclamped).float()
            error = keep * error + (1.0 - keep) * clamped
        else:
            error = clamped

    frame_mask = mask.unsqueeze(2).float()
    point_mask = mask.unsqueeze(1).float()
    pair_mask = frame_mask * point_mask

    error = error / 10.0
    return (error * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)


def build_frames_from_backbone(bb, mask):
    """True AlphaFold-style frames from N, CA, C of a SINGLE residue.

    bb: (B, N, 3, 3) with atom order [N, CA, C].

    Gram-Schmidt exactly as AlphaFold2 supplement alg. 21:
      origin = CA
      x   = normalize(C - CA)
      y   = normalize((N - CA) - ((N-CA).x) x)
      z   = x cross y

    Unlike the Ca-pseudo-frames these use rigid intra-residue bonds (~1.5 A),
    so the frame does not swing with backbone dihedrals. That makes the loss far
    less sensitive to small coordinate error.
    """
    n_at, ca, c_at = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    v1 = c_at - ca
    v2 = n_at - ca
    x = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=1e-6)
    proj = (v2 * x).sum(-1, keepdim=True) * x
    y = v2 - proj
    y = y / torch.linalg.norm(y, dim=-1, keepdim=True).clamp(min=1e-6)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1), ca


def fape_loss_backbone(pred_bb, true_bb, mask, clamp_distance=10.0,
                       frac_unclamped=0.0, all_atom_points=False):
    """Pose-invariant FAPE using true N-CA-C frames.

    pred_bb / true_bb: (B, N, 3, 3), atom order [N, CA, C].
    all_atom_points: if True use all three backbone atoms as points (AF2 style),
        otherwise alpha carbons only. The point set only changes the size of the
        (frames x points) tensor, which is small next to the decoder.
    """
    B, N = pred_bb.shape[0], pred_bb.shape[1]
    rot_p, tr_p = build_frames_from_backbone(pred_bb, mask)
    rot_t, tr_t = build_frames_from_backbone(true_bb, mask)

    if all_atom_points:
        pts_p, pts_t = pred_bb.reshape(B, N * 3, 3), true_bb.reshape(B, N * 3, 3)
        pmask = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, N * 3)
    else:
        pts_p, pts_t = pred_bb[..., 1, :], true_bb[..., 1, :]
        pmask = mask

    P = pts_p.shape[1]
    loc_p = _to_local(pts_p, rot_p, tr_p, P)
    loc_t = _to_local(pts_t, rot_t, tr_t, P)
    error = torch.sqrt(((loc_p - loc_t) ** 2).sum(-1) + 1e-8)

    if clamp_distance is not None:
        clamped = torch.clamp(error, max=clamp_distance)
        if frac_unclamped > 0.0:
            keep = (torch.rand(B, 1, 1, device=error.device) < frac_unclamped).float()
            error = keep * error + (1.0 - keep) * clamped
        else:
            error = clamped

    pair_mask = mask.unsqueeze(2).float() * pmask.unsqueeze(1).float()
    error = error / 10.0
    return (error * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)


# Ideal backbone bond lengths (Engh & Huber), Angstroms.
BOND_N_CA, BOND_CA_C, BOND_C_N = 1.458, 1.525, 1.329


def backbone_bond_lengths(bb, mask):
    """bb: (B, N, >=3, 3) Angstroms, atom order [N, CA, C, ...].
    Returns (d_nca, d_cac, d_cn, m_intra, m_pept): per-residue bond lengths and
    the masks of residues/peptide bonds that exist."""
    n_at, ca, c_at = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    d_nca = torch.linalg.norm(ca - n_at, dim=-1)
    d_cac = torch.linalg.norm(c_at - ca, dim=-1)
    d_cn = torch.linalg.norm(n_at[:, 1:] - c_at[:, :-1], dim=-1)   # C_i - N_{i+1}
    m_intra = mask.float()
    m_pept = (mask[:, 1:] & mask[:, :-1]).float()
    return d_nca, d_cac, d_cn, m_intra, m_pept


def bond_length_penalty(bb, mask):
    """Mean absolute deviation (Angstroms) of N-CA, CA-C and C-N(+1) bond
    lengths from ideal values. Rotation- and translation-invariant by
    construction; pushes the head toward latents that decode to a chemically
    sane backbone (CLAUDE.md section 8: fix local geometry first)."""
    d_nca, d_cac, d_cn, m_intra, m_pept = backbone_bond_lengths(bb, mask)
    dev = ((d_nca - BOND_N_CA).abs() * m_intra).sum() \
        + ((d_cac - BOND_CA_C).abs() * m_intra).sum() \
        + ((d_cn - BOND_C_N).abs() * m_pept).sum()
    cnt = 2.0 * m_intra.sum() + m_pept.sum()
    return dev / cnt.clamp(min=1.0)


def pose_sensitive_coord_error(pred_ca, true_ca, mask, clamp_distance=10.0):
    """The ORIGINAL, defective 'fape_loss'. Kept only to reproduce old numbers.

    Not pose-invariant: reduces to clamped mean ||pred_j - true_j|| with no
    superposition. Do not use for training or reporting.
    """
    d = torch.sqrt(((pred_ca - true_ca) ** 2).sum(-1) + 1e-8)
    d = torch.clamp(d, max=clamp_distance) / 10.0
    m = mask.float()
    return (d * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Differentiable decoder wrapper
# ---------------------------------------------------------------------------
class DifferentiableDecoder(nn.Module):
    """Wraps ProteinAE decoder for differentiable z → coords mapping.

    Uses Euler ODE integration with gradients flowing through each step.
    """
    def __init__(self, ae_model, n_steps=10):
        super().__init__()
        self.decoder = ae_model.decoder
        self.fm = ae_model.fm
        self.ae_model = ae_model
        self.n_steps = n_steps

        # Freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Check parameterization
        self.target_pred = ae_model.cfg_exp.model.target_pred

    @torch.no_grad()
    def _sample_initial_noise(self, batch_size, n_atoms, device):
        """Sample initial Gaussian noise for backbone atoms."""
        x = torch.randn(batch_size, n_atoms, 3, device=device) * self.fm.scale_ref
        return x

    def _nn_out_to_x_clean(self, nn_out, x_t, t):
        """Convert decoder output to clean coords prediction."""
        nn_pred = nn_out["coors_pred"]
        if self.target_pred == "x_1":
            return nn_pred
        elif self.target_pred == "v":
            t_ext = t[:, None, None]
            return x_t + (1.0 - t_ext) * nn_pred
        else:
            raise ValueError(f"Unknown target_pred: {self.target_pred}")

    def forward(self, z, mask, return_backbone=False):
        """Decode z to Ca coordinates with differentiable ODE.

        return_backbone: also return the full (B, N, 4, 3) backbone in
        Angstroms, atom order [N, CA, C, O], as (ca, backbone).

        The decoder operates in backbone mode (4 atoms per residue: N, Ca, C, O).
        x_t shape: (B, N*4, 3), coords_mask shape: (B, N*4).

        Args:
            z: (B, N, 8) predicted latent vectors
            mask: (B, N) bool mask
        Returns:
            ca_coords: (B, N, 3) predicted Ca coordinates in Angstroms
        """
        B, N, _ = z.shape
        device = z.device

        # Backbone mode: 4 atoms per residue
        n_atoms = N * 4
        coords_mask = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, n_atoms)

        # Time schedule: uniform from 0 to 1
        ts = torch.linspace(0, 1, self.n_steps + 1, device=device)

        # Start from noise
        x = self._sample_initial_noise(B, n_atoms, device)

        for step in range(self.n_steps):
            t_val = ts[step]
            dt = ts[step + 1] - ts[step]
            t = t_val * torch.ones(B, device=device)

            batch_nn = {
                "x_t": x,
                "t": t,
                "mask": mask,
                "coords_mask": coords_mask,
                "single_repr": z,
            }

            # Forward through frozen decoder (gradients flow through z only)
            nn_out = self.decoder(batch_nn)
            x_clean = self._nn_out_to_x_clean(nn_out, x, t)

            # Euler step: v = (x_clean - x_t) / (1 - t), x_{t+dt} = x_t + v * dt
            v = (x_clean - x) / (1.0 - t_val + 1e-6)
            x = x + v * dt.item()

            # Zero out masked atoms
            x = x * coords_mask.unsqueeze(-1).float()

        # Extract Ca atoms (index 1 of each 4-atom group) and convert nm → Angstroms
        # Reshape to (B, N, 4, 3), take atom index 1
        x_backbone = x.reshape(B, N, 4, 3)
        ca_coords = x_backbone[:, :, 1, :] * 10.0

        if return_backbone:
            return ca_coords, x_backbone * 10.0
        return ca_coords


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TM-score evaluation (the metric we actually care about)
# ---------------------------------------------------------------------------
# FAPE is the training signal because it must be differentiable. It is a good
# surrogate for DIRECTION but not for MAGNITUDE: between epochs 16 and 22 of the
# 40k run the FAPE gain shrank 59% while the correct-fold fraction GREW 60%.
# Selecting checkpoints or stopping on FAPE therefore risks ending a run, or
# keeping the wrong file, while fold quality is still improving. These helpers
# are deliberately self-contained so this module has no import cycle with the
# evaluation scripts.
FOLDSEEK_BIN = _FOLDSEEK


def _write_pseudo_backbone_pdb(ca, path):
    """Write a Ca trace as N/CA/C backbone. Foldseek rejects Ca-only PDBs."""
    n = len(ca)
    prev = np.concatenate([ca[:1], ca[:-1]], 0)
    nxt = np.concatenate([ca[1:], ca[-1:]], 0)
    dp = prev - ca
    dn = nxt - ca
    dp = dp / (np.linalg.norm(dp, axis=-1, keepdims=True) + 1e-8) * 1.47
    dn = dn / (np.linalg.norm(dn, axis=-1, keepdims=True) + 1e-8) * 1.52
    with open(path, "w") as f:
        sr = 1
        for i in range(n):
            for nm, el, xyz in ((" N  ", "N", ca[i] + dp[i]),
                                (" CA ", "C", ca[i]),
                                (" C  ", "C", ca[i] + dn[i])):
                f.write(f"ATOM  {sr:5d} {nm} ALA A{i+1:4d}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        f"  1.00  0.00           {el}\n")
                sr += 1
        f.write("END\n")


FOLDSEEK_CHUNK = int(os.environ.get("FOLDSEEK_CHUNK", "64"))

def _foldseek_tm(pred_dir, gt_dir, tmp):
    """Matched-pair TM via TMalign. --exhaustive-search is REQUIRED: without it
    the 3Di k-mer prefilter silently drops dissimilar structures, which
    previously produced a TM column that was 85-90% imputed zeros.
    Only the diagonal (pred_i vs gt_i) is used, but exhaustive search aligns
    every query against every target: N^2 TM-aligns. For N > FOLDSEEK_CHUNK the
    structures are split into chunks of that size (same names in both halves), so the
    cost is N * chunk instead of N^2 (1000 proteins: 64k alignments instead of 1M).
    Every matched pair is still aligned exhaustively; results are identical."""
    import subprocess, shutil as _sh
    names = sorted(f[:-4] for f in os.listdir(pred_dir) if f.endswith(".pdb") and os.path.exists(os.path.join(gt_dir, f)))
    if len(names) <= FOLDSEEK_CHUNK:
        return _foldseek_tm_dir(pred_dir, gt_dir, tmp)
    best = {}
    for c, s in enumerate(range(0, len(names), FOLDSEEK_CHUNK)):
        cp, cg = os.path.join(tmp, f"c{c}", "pr"), os.path.join(tmp, f"c{c}", "gt")
        os.makedirs(cp, exist_ok=True); os.makedirs(cg, exist_ok=True)
        for nm in names[s:s + FOLDSEEK_CHUNK]:
            os.symlink(os.path.abspath(os.path.join(pred_dir, nm + ".pdb")), os.path.join(cp, nm + ".pdb"))
            os.symlink(os.path.abspath(os.path.join(gt_dir, nm + ".pdb")), os.path.join(cg, nm + ".pdb"))
        best.update(_foldseek_tm_dir(cp, cg, os.path.join(tmp, f"c{c}", "tmp")))
        _sh.rmtree(os.path.join(tmp, f"c{c}"), ignore_errors=True)
    return best


def _foldseek_tm_dir(pred_dir, gt_dir, tmp):
    import subprocess
    os.makedirs(os.path.join(tmp, "fs"), exist_ok=True)
    out = os.path.join(tmp, "aln.tsv")
    r = subprocess.run([FOLDSEEK_BIN, "easy-search", pred_dir, gt_dir, out,
                        os.path.join(tmp, "fs"), "--alignment-type", "1",
                        "-e", "inf", "--max-seqs", "2000", "--exact-tmscore", "1",
                        "--exhaustive-search", "1",
                        "--format-output", "query,target,alntmscore,qtmscore,ttmscore"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    best = {}
    for line in open(out):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5 and f[0] == f[1]:
            best[f[0].replace(".pdb", "")] = max(float(f[3]), float(f[4]))
    return best


@torch.no_grad()
def evaluate_tm(head, diff_decoder, val_loader, device, n_proteins=100,
                use_amp=True):
    """Return (tm_mean, frac_tm_above_0.5, coverage). Never raises: a Foldseek
    failure must not kill a multi-day training run."""
    import tempfile, shutil
    head.eval()
    work = tempfile.mkdtemp(prefix="tmeval_")
    gt_d, pr_d = os.path.join(work, "gt"), os.path.join(work, "pred")
    os.makedirs(gt_d); os.makedirs(pr_d)
    try:
        k = 0
        for batch in val_loader:
            esm, _z, ca, mask, lengths = batch[:5]
            if k >= n_proteins:
                break
            esm, mask = esm.to(device), mask.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = diff_decoder(head(esm, mask), mask)
            pred = pred.float().cpu().numpy()
            for b in range(pred.shape[0]):
                if k >= n_proteins:
                    break
                L_ = int(lengths[b])
                _write_pseudo_backbone_pdb(ca[b, :L_].numpy(),
                                           os.path.join(gt_d, f"p{k:05d}.pdb"))
                _write_pseudo_backbone_pdb(pred[b, :L_],
                                           os.path.join(pr_d, f"p{k:05d}.pdb"))
                k += 1
        tms = _foldseek_tm(pr_d, gt_d, os.path.join(work, "tm"))
        names = [f"p{i:05d}" for i in range(k)]
        meas = [tms[n] for n in names if n in tms]
        if not meas:
            return float("nan"), float("nan"), 0.0
        return (float(np.mean(meas)),
                float(np.mean([tms.get(n, 0.0) > 0.5 for n in names])),
                len(meas) / max(k, 1))
    except Exception as e:
        print(f"    [tm eval failed: {e}]", flush=True)
        return float("nan"), float("nan"), 0.0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        head.train()


def train_fape(head, diff_decoder, train_loader, val_loader, device,
               lr=5e-5, n_epochs=30, patience=8, label="fape", use_amp=False,
               clamp=10.0, frac_unclamped=0.1, aux_mse_weight=0.0,
               tm_every=0, tm_n=100, select_on="fape", arch=None, resume=None,
               loss_mode="ca", bb_points="all", bond_weight=0.0, fix_ends=False):
    # Fail fast on an inconsistent config, before any model or data setup.
    if select_on == "tm" and tm_every <= 0:
        raise ValueError("select_on='tm' requires --tm-every > 0")
    need_bb = loss_mode != "ca" or bond_weight > 0.0
    loss_cfg = {"loss": loss_mode, "bb_points": bb_points,
                "bond_weight": bond_weight, "fix_ends": fix_ends}

    def structure_loss(ca_pred, bb_pred, ca_true, bb_true, bb_valid, mask,
                       clamp_d, frac_unc):
        """Returns (total, parts) where parts holds the detached components.
        ca  : Ca-point FAPE in Ca pseudo-frames (the inherited loss)
        bb  : FAPE in true N-CA-C frames; points = CA or all of N/CA/C
        bond: mean |bond length - ideal| over N-CA, CA-C, C-N(+1), Angstroms
        Structures without a backbone record (bb_valid False) are masked out
        of the bb term only."""
        parts = {}
        total = torch.zeros((), device=ca_pred.device)
        if loss_mode in ("ca", "ca+bb"):
            l_ca = fape_loss(ca_pred, ca_true, mask, clamp_distance=clamp_d,
                             frac_unclamped=frac_unc, fix_ends=fix_ends)
            parts["ca"] = l_ca.detach(); total = total + l_ca
        if loss_mode in ("bb", "ca+bb"):
            m_bb = mask & bb_valid.unsqueeze(1)
            l_bb = fape_loss_backbone(bb_pred[:, :, :3], bb_true, m_bb,
                                      clamp_distance=clamp_d, frac_unclamped=frac_unc,
                                      all_atom_points=(bb_points == "all"))
            parts["bb"] = l_bb.detach(); total = total + l_bb
        if bond_weight > 0.0:
            l_bond = bond_length_penalty(bb_pred, mask)
            parts["bond"] = l_bond.detach(); total = total + bond_weight * l_bond
        return total, parts
    n_params = sum(p.numel() for p in head.parameters())
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_epoch = float('inf'), 0
    best_tm, best_tm_epoch = -1.0, 0
    history = []
    start_epoch = 0
    arch = dict(arch or {})

    # Full-state resume. Weights alone are NOT enough: restarting without the
    # AdamW moments and the cosine schedule position silently degrades a long
    # run with no error, which matters on a preemptible partition.
    if resume:
        rp = Path(resume)
        if not rp.is_absolute():
            rp = PROJECT / "data" / "phase1_dataset" / rp
        st = torch.load(str(rp), weights_only=False, map_location=device)
        head.load_state_dict(st["head"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if st.get("scaler") is not None:
            scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"]
        best_val, best_epoch = st["best_val"], st["best_epoch"]
        best_tm, best_tm_epoch = st["best_tm"], st["best_tm_epoch"]
        history = st.get("history", [])
        if st.get("torch_rng") is not None:
            torch.set_rng_state(st["torch_rng"].cpu())
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        print(f"  RESUMED from epoch {start_epoch} "
              f"(best_val {best_val:.4f}, best_tm {best_tm:.3f}), "
              f"lr now {scheduler.get_last_lr()[0]:.2e}", flush=True)
        if st.get("arch") and arch and st["arch"] != arch:
            raise ValueError(f"architecture mismatch: checkpoint {st['arch']} "
                             f"vs requested {arch}")
        arch = st.get("arch", arch)

    print(f"\n  {label}: {n_params:,} params, lr={lr}, amp={use_amp}")
    print(f"  loss: {loss_cfg}")
    print(f"  {'Ep':>4s}  {'Train':>10s}  {'Val':>10s}  {'Time':>6s}")

    for epoch in range(start_epoch, n_epochs):
        t0 = time.perf_counter()
        head.train()
        tl, nb = 0.0, 0

        for batch in train_loader:
            esm, z_true, ca_true, mask, lengths = batch[:5]
            esm = esm.to(device)
            ca_true = ca_true.to(device)
            z_true_d = z_true.to(device)
            mask = mask.to(device)
            if need_bb:
                bb_true = batch[5].to(device); bb_valid = batch[6].to(device)
            else:
                bb_true = bb_valid = None

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z_pred = head(esm, mask)
                    if need_bb:
                        ca_pred, bb_pred = diff_decoder(z_pred, mask, return_backbone=True)
                    else:
                        ca_pred, bb_pred = diff_decoder(z_pred, mask), None
                    loss_fape, parts = structure_loss(ca_pred, bb_pred, ca_true, bb_true,
                                                      bb_valid, mask, clamp, frac_unclamped)
                    # Auxiliary latent MSE. Free: z_true is already loaded and was
                    # previously discarded, and this adds no decoder passes.
                    if aux_mse_weight > 0.0:
                        m = mask.unsqueeze(-1).float()
                        loss_aux = (((z_pred - z_true_d) ** 2) * m).sum() / \
                                   (m.sum() * z_pred.shape[-1]).clamp(min=1.0)
                        loss = loss_fape + aux_mse_weight * loss_aux
                    else:
                        loss_aux = torch.zeros((), device=device)
                        loss = loss_fape

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                tl += loss.item()
                nb += 1
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at batch {nb+1}, skipping", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            if nb == 1 or nb % 100 == 0:
                with torch.no_grad():
                    zl2 = z_pred.detach()[mask].float().norm(dim=-1)
                pmsg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                zmsg = (f"fape={loss_fape.item():.4f} [{pmsg}] aux={loss_aux.item():.4f} "
                        f"z_L2={zl2.mean().item():.3f}")
                if nb == 1:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"    batch 1: loss={loss.item():.4f}, {zmsg}, "
                          f"peak GPU={mem:.1f} GB", flush=True)
                else:
                    print(f"    batch {nb}: loss={loss.item():.4f}, {zmsg}", flush=True)

        train_loss = tl / max(nb, 1)

        # Validation
        head.eval()
        vl, vn = 0.0, 0
        vparts = {}
        with torch.no_grad():
            for batch in val_loader:
                esm, z_true, ca_true, mask, lengths = batch[:5]
                esm = esm.to(device)
                ca_true = ca_true.to(device)
                mask = mask.to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z_pred = head(esm, mask)
                    if need_bb:
                        bb_true = batch[5].to(device); bb_valid = batch[6].to(device)
                        ca_pred, bb_pred = diff_decoder(z_pred, mask, return_backbone=True)
                        _, vp = structure_loss(ca_pred, bb_pred, ca_true, bb_true,
                                               bb_valid, mask, 10.0, 0.0)
                        # always report N-CA scatter, the section-8 quantity
                        d_nca, _, _, m_i, _ = backbone_bond_lengths(bb_pred.float(), mask)
                        vp["nca_dev"] = ((d_nca - BOND_N_CA).abs() * m_i).sum() / m_i.sum()
                        for k, v in vp.items():
                            vparts[k] = vparts.get(k, 0.0) + v.item()
                    else:
                        ca_pred = diff_decoder(z_pred, mask)
                    # the inherited metric, identical across arms
                    loss = fape_loss(ca_pred, ca_true, mask, clamp_distance=10.0,
                                     fix_ends=fix_ends)
                vl += loss.item()
                vn += 1

        val_loss = vl / max(vn, 1)
        vparts = {k: v / max(vn, 1) for k, v in vparts.items()}
        if vparts:
            print("        val parts: " + " ".join(f"{k}={v:.4f}" for k, v in vparts.items()),
                  flush=True)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        # TM-score: the metric we actually care about. FAPE tracks it in
        # direction but not magnitude, so selection and stopping use TM when
        # available and fall back to FAPE otherwise.
        tm_mean = tm_frac = float("nan")
        if tm_every > 0 and (epoch + 1) % tm_every == 0:
            t1 = time.perf_counter()
            tm_mean, tm_frac, cov = evaluate_tm(head, diff_decoder, val_loader,
                                                device, n_proteins=tm_n,
                                                use_amp=use_amp)
            print(f"        TM {tm_mean:.3f}  TM>0.5 {tm_frac:.2f}  "
                  f"coverage {cov:.2f}  [{time.perf_counter()-t1:.0f}s]", flush=True)

        rec = {"epoch": epoch + 1, "train": train_loss, "val": val_loss}
        rec.update({f"val_{k}": v for k, v in vparts.items()})
        if not np.isnan(tm_mean):
            rec.update({"tm_mean": tm_mean, "tm_frac_above_0.5": tm_frac})
        history.append(rec)
        print(f"  {epoch+1:4d}  {train_loss:10.4f}  {val_loss:10.4f}  {elapsed:6.1f}s")

        improved_fape = val_loss < best_val
        if improved_fape:
            best_val, best_epoch = val_loss, epoch + 1
        improved_tm = (not np.isnan(tm_mean)) and tm_mean > best_tm
        if improved_tm:
            best_tm, best_tm_epoch = tm_mean, epoch + 1

        save = improved_tm if select_on == "tm" else improved_fape
        if save:
            ckpt_path = PROJECT / "data" / "phase1_dataset" / f"best_{label}.pt"
            torch.save(head.state_dict(), str(ckpt_path))
            meta = {"epoch": epoch + 1, "val_fape": val_loss,
                    "selected_on": select_on,
                    "tm_mean": None if np.isnan(tm_mean) else tm_mean,
                    "tm_frac_above_0.5": None if np.isnan(tm_frac) else tm_frac}
            meta.update(arch)          # the ACTUAL head, not hardcoded constants
            meta["loss_cfg"] = loss_cfg
            with open(str(ckpt_path) + ".meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # Resumable full state, rewritten every epoch regardless of improvement.
        last_path = PROJECT / "data" / "phase1_dataset" / f"last_{label}.ckpt"
        torch.save({"head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "epoch": epoch + 1, "best_val": best_val,
                    "best_epoch": best_epoch, "best_tm": best_tm,
                    "best_tm_epoch": best_tm_epoch, "history": history,
                    "arch": arch, "label": label, "loss_cfg": loss_cfg,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None},
                   str(last_path) + ".tmp")
        os.replace(str(last_path) + ".tmp", str(last_path))  # atomic: a preempt
                                                             # mid-write cannot
                                                             # corrupt the file

        # Patience counts epochs since the last improvement in the SELECTION
        # metric. On TM that is measured only every tm_every epochs, so scale
        # patience accordingly rather than counting un-evaluated epochs.
        if select_on == "tm":
            since = (epoch + 1) - best_tm_epoch
            if best_tm_epoch > 0 and since >= patience * tm_every:
                print(f"  Early stopping on TM (best: epoch {best_tm_epoch}, "
                      f"TM {best_tm:.3f})")
                break
        elif epoch + 1 - best_epoch >= patience:
            print(f"  Early stopping on FAPE (best: {best_epoch})")
            break

    return {"label": label, "best_epoch": best_epoch, "best_val": best_val,
            "best_tm": None if best_tm < 0 else best_tm,
            "best_tm_epoch": best_tm_epoch, "select_on": select_on,
            "history": history}


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=0, help="0 = full 80k")
    p.add_argument("--n-val", type=int, default=0, help="0 = full 10k")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warm-start", type=str, default="best_deep10L.pt",
                   help="head weights to load (weights ONLY); 'none' = random init")
    p.add_argument("--normalize-out", action="store_true",
                   help="project head output onto the encoder's LayerNorm latent manifold")
    p.add_argument("--clamp", type=float, default=10.0,
                   help="FAPE clamp in Angstroms for the TRAINING loss")
    p.add_argument("--frac-unclamped", type=float, default=0.1,
                   help="fraction of samples left unclamped, as in AlphaFold2")
    p.add_argument("--aux-mse-weight", type=float, default=0.0,
                   help="weight on auxiliary latent-MSE term (0 = pure FAPE)")
    p.add_argument("--tm-every", type=int, default=0,
                   help="evaluate TM-score every N epochs (0 = never). ~5 min vs 1.2h/epoch")
    p.add_argument("--tm-n", type=int, default=100, help="proteins per TM evaluation")
    p.add_argument("--select-on", choices=["fape", "tm"], default="fape",
                   help="metric for checkpoint selection AND early stopping")
    p.add_argument("--n-layers", type=int, default=10, help="head depth")
    p.add_argument("--d-model", type=int, default=256, help="head width")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--resume", type=str, default=None,
                   help="last_<label>.ckpt — restores optimizer, scheduler, epoch, RNG")
    p.add_argument("--label", type=str, default="fape_full")
    p.add_argument("--loss", choices=["ca", "bb", "ca+bb"], default="ca",
                   help="ca: Ca points in Ca pseudo-frames (inherited); "
                        "bb: true N-CA-C frames from backbone_100k.h5; ca+bb: sum")
    p.add_argument("--bb-points", choices=["ca", "all"], default="all",
                   help="points for the bb term: CA only, or all of N/CA/C")
    p.add_argument("--bond-weight", type=float, default=0.0,
                   help="weight on the backbone bond-length penalty (Angstrom L1)")
    p.add_argument("--fix-end-frames", action="store_true",
                   help="pose-invariant chain-end frames (see build_frames_from_ca)")
    return p.parse_args()


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    device = torch.device("cuda")

    print("=" * 60)
    print("Gate 6 Option C: FAPE training")
    print("=" * 60)

    # Load ProteinAE
    print("Loading ProteinAE decoder...")
    with hydra.initialize_config_dir(
        config_dir=f"{os.getcwd()}/configs/experiment_config",
        version_base=hydra.__version__,
    ):
        cfg = hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae_model = ProteinAE.load_from_checkpoint(
        "checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False
    )
    ae_model.eval()
    ae_model.to(device)

    # Wrap decoder for differentiable decoding
    N_STEPS = args.ode_steps
    diff_decoder = DifferentiableDecoder(ae_model, n_steps=N_STEPS).to(device)

    # Quick sanity with bf16
    print(f"Sanity check: {N_STEPS} ODE steps, bf16...")
    test_z = torch.randn(1, 50, 8, device=device, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        test_ca = diff_decoder(test_z, torch.ones(1, 50, dtype=torch.bool, device=device))
    test_ca.sum().backward()
    print(f"  Shape: {test_ca.shape}, grad norm: {test_z.grad.norm().item():.1f}")
    del test_z, test_ca
    torch.cuda.empty_cache()

    # Dataset (optionally subsampled, seeded to match the PoC control)
    need_bb = args.loss != "ca" or args.bond_weight > 0.0
    BB_PATH = PROJECT / "data" / "phase1_dataset" / "backbone_100k.h5"
    bb_path = str(BB_PATH) if need_bb else None
    if need_bb:
        print(f"Backbone side file: {BB_PATH}")
    train_ds = ProteinDatasetFAPE(H5_PATH, "train", backbone_path=bb_path)
    val_ds = ProteinDatasetFAPE(H5_PATH, "val", backbone_path=bb_path)
    torch.manual_seed(42)
    if args.n_train:
        train_ds = torch.utils.data.Subset(
            train_ds, torch.randperm(len(train_ds))[:args.n_train].tolist())
    if args.n_val:
        val_ds = torch.utils.data.Subset(
            val_ds, torch.randperm(len(val_ds))[:args.n_val].tolist())

    BATCH_SIZE = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batch: {BATCH_SIZE}")
    print(f"ODE steps: {N_STEPS}, precision: bf16, normalize_out: {args.normalize_out}")

    # Warm-start from MSE-trained checkpoint
    print("Initializing head (MSE warm-start)...")
    ARCH = {"n_layers": args.n_layers, "d_model": args.d_model,
            "dropout": args.dropout, "use_conv": False,
            "normalize_out": args.normalize_out}
    head = DeepHead(**ARCH)
    if args.resume:
        print("  --resume given; skipping warm-start (full state restored below)")
    elif args.warm_start.lower() in ("", "none"):
        # A wider or deeper head cannot take the 10x256 weights; it needs a
        # fresh run (CLAUDE.md section 5). Random init, no checkpoint.
        print("  --warm-start none: random init")
    else:
        warm = PROJECT / "data" / "phase1_dataset" / args.warm_start
        head.load_state_dict(torch.load(str(warm), weights_only=True))
        print(f"  Loaded: {warm}")

    # Train with bf16
    result = train_fape(head, diff_decoder, train_loader, val_loader, device,
                        lr=args.lr, n_epochs=args.epochs, patience=args.patience,
                        label=args.label, use_amp=True,
                        clamp=args.clamp, frac_unclamped=args.frac_unclamped,
                        aux_mse_weight=args.aux_mse_weight,
                        tm_every=args.tm_every, tm_n=args.tm_n,
                        select_on=args.select_on, arch=ARCH, resume=args.resume,
                        loss_mode=args.loss, bb_points=args.bb_points,
                        bond_weight=args.bond_weight, fix_ends=args.fix_end_frames)
    result["config"] = vars(args)

    # Save results
    print(f"\n{'='*60}")
    print(f"Best val FAPE: {result['best_val']:.4f} at epoch {result['best_epoch']}")

    out_path = PROJECT / "notes" / f"gate6_{args.label}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main(parse_args())
