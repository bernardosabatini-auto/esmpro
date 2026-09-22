"""Does the ProteinAE encoder's latent depend on the input's orientation?
Encode each structure as-is and under random rotations / reflections and
compare per-residue latents. If latents are rotation-invariant, a latent-space
loss (no decoder in the loop) is viable; if not, this shows how they move.
CPU is fine for a handful of structures."""
import os, sys, math, glob, numpy as np, torch, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, "."); sys.path.insert(0, ROOT + "/code")
import lightning as Lt, hydra
from einops import rearrange
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.autoencode import ProteinProcessor
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.coors_utils import ang_to_nm
import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--pdb-dir", default=ROOT + "/logs/pdb_sample/structures")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args()
dev = torch.device(a.device)
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False, map_location=dev).eval().to(dev)
proc = ProteinProcessor(); fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)

def encode(item):
    loader = DensePaddingDataLoader([item]); batch = next(iter(loader))
    x_1 = batch["coords"][:, :, proc.BACKBONE_ATOM_INDICES, :]; x_1 = rearrange(x_1, "b n c d -> b (n c) d")
    coords_mask = batch["mask_dict"]["coords"][..., proc.BACKBONE_ATOM_INDICES, 0]; mask = coords_mask[..., 1]
    cmf = rearrange(coords_mask, "b n c -> b (n c)")
    x_1 = fm._mask_and_zero_com(ang_to_nm(x_1), cmf)
    batch.update({"x_1": x_1, "mask": mask, "coords_mask": cmf, "nsamples": 1, "nres": int(x_1.shape[-2] // 4)})
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(dev)
    with torch.no_grad():
        return ae.encoder(batch)["single_repr"].squeeze(0).float().cpu()

def rand_rot(seed):
    g = torch.Generator().manual_seed(seed); q = torch.randn(4, generator=g); q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def transformed(item, R, t=None):
    it = item.clone(); c = it.coords.clone(); ok = torch.isfinite(c).all(-1)
    c2 = c @ R.T + (t if t is not None else 0.0); c2[~ok] = c[~ok]; it.coords = c2; return it

pdbs = sorted(glob.glob(a.pdb_dir + "/*.pdb"))
print(f"{len(pdbs)} structures, device {dev}")
print(f"{'structure':<32s} {'n':>4s} | {'|z|':>5s} | rot L2 diff (3 seeds)      cos | trans 20A L2 | mirror L2  cos | 90deg x-rot L2")
for p in pdbs:
    from pathlib import Path
    item = proc.process_pdb(Path(p)); z0 = encode(item); n = z0.shape[0]
    diffs, coss = [], []
    for s in range(3):
        zr = encode(transformed(item, rand_rot(s))); diffs.append((zr - z0).norm(dim=-1).mean().item()); coss.append(F.cosine_similarity(zr, z0, dim=-1).mean().item())
    zt = encode(transformed(item, torch.eye(3), torch.tensor([20., -5., 7.])))
    zm = encode(transformed(item, torch.diag(torch.tensor([1., 1., -1.]))))
    zx = encode(transformed(item, torch.tensor([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])))
    print(f"{os.path.basename(p)[:32]:<32s} {n:4d} | {z0.norm(dim=-1).mean():5.2f} | "
          f"{' '.join(f'{d:.3f}' for d in diffs)}  {np.mean(coss):.3f} | {(zt-z0).norm(dim=-1).mean():.4f} | "
          f"{(zm-z0).norm(dim=-1).mean():.3f}  {F.cosine_similarity(zm, z0, dim=-1).mean():.3f} | {(zx-z0).norm(dim=-1).mean():.3f}")
print("\nReference: true latents have per-residue L2 = 2.83; two unrelated residues differ by ~4.")
