"""
PCA-based canonical orientation for protein structures.

Resolves ProteinAE's rotational non-invariance by placing each structure
in a deterministic reference frame before encoding.

Sign convention: skewness-based (force projection skewness >= 0 on each axis).
"""
import numpy as np
import torch
from pathlib import Path
from scipy.stats import skew


def canonicalize_coords(ca_coords: np.ndarray):
    """
    PCA canonicalization with deterministic sign convention.

    Args:
        ca_coords: (N, 3) CA coordinates

    Returns:
        canonical_coords: (N, 3) canonicalized coordinates
        rotation_matrix: (3, 3) the rotation applied (R such that X_canon = X_centered @ R)
        info: dict with eigenvalues, eigenvalue ratios, degeneracy flags
    """
    assert ca_coords.ndim == 2 and ca_coords.shape[1] == 3
    N = ca_coords.shape[0]

    # 1. Center
    centroid = ca_coords.mean(axis=0)
    X = ca_coords - centroid

    # 2. Covariance matrix and eigendecomposition
    C = X.T @ X / N
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # ascending order

    # 3. Sort by eigenvalue descending (largest variance = first axis)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 4. Sign convention: force skewness >= 0 along each axis
    projections = X @ eigenvectors  # (N, 3)
    for i in range(3):
        s = skew(projections[:, i])
        if s < 0:
            eigenvectors[:, i] *= -1
            projections[:, i] *= -1

    # 5. Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # 6. Apply rotation
    canonical = X @ eigenvectors

    # 7. Degeneracy info
    ev_ratios = {}
    if eigenvalues[0] > 1e-10:
        ev_ratios["lambda2_over_lambda1"] = float(eigenvalues[1] / eigenvalues[0])
    if eigenvalues[1] > 1e-10:
        ev_ratios["lambda3_over_lambda2"] = float(eigenvalues[2] / eigenvalues[1])
    degenerate = any(v > 0.9 for v in ev_ratios.values())

    info = {
        "eigenvalues": eigenvalues.tolist(),
        "ev_ratios": ev_ratios,
        "degenerate": degenerate,
        "centroid": centroid.tolist(),
    }

    return canonical, eigenvectors, info


def canonicalize_all_atom_coords(all_coords: np.ndarray, ca_coords: np.ndarray):
    """
    Apply PCA rotation (computed from CA) to all-atom coordinates.

    Args:
        all_coords: (N, A, 3) all-atom coordinates (A atoms per residue)
        ca_coords: (N, 3) CA coordinates used to compute the rotation

    Returns:
        canonical_all: (N, A, 3) canonicalized all-atom coordinates
        R: (3, 3) rotation matrix
        info: dict with eigenvalues etc.
    """
    _, R, info = canonicalize_coords(ca_coords)
    centroid = np.array(info["centroid"])

    # Center and rotate all atoms
    centered = all_coords - centroid[None, None, :]
    canonical_all = centered @ R

    return canonical_all, R, info


def canonicalize_pyg_data(data):
    """
    Apply PCA canonicalization to a PyG protein Data object.
    Rotates coords in-place using PCA of CA atoms.

    The CA atom is at index 1 in OpenFold atom ordering
    (after PDB_TO_OPENFOLD_INDEX_TENSOR reordering).

    Args:
        data: torch_geometric.data.Data with coords field (N, 37, 3)

    Returns:
        data: modified Data object
        info: canonicalization info dict
    """
    coords_np = data.coords.numpy()  # (N, 37, 3)
    ca_coords = coords_np[:, 1, :]  # CA = index 1 in OpenFold ordering

    _, R, info = canonicalize_coords(ca_coords)
    centroid = np.array(info["centroid"])

    # Apply to all atoms
    centered = coords_np - centroid[None, None, :]
    canonical = centered @ R

    data.coords = torch.from_numpy(canonical).float()
    return data, info
