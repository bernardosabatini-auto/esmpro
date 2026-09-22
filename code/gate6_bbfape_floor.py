"""How much bb-FAPE floor does a PERFECT latent carry, and where does it come
from: bf16 autocast, ODE step count, or the decoder itself? Also reports the
Ca-FAPE floor for comparison. Run on a GPU."""
import os, sys, numpy as np, torch, h5py, time
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
from gate6_fape_train import (fape_loss, fape_loss_backbone, backbone_bond_lengths,
                              DifferentiableDecoder, H5_PATH, PROJECT, BOND_N_CA)
from gate6_corrected_eval import kabsch_rmsd
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import lightning as Lt, hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
dev = torch.device("cuda")
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False).eval().to(dev)
h5 = h5py.File(H5_PATH, "r"); bbf = h5py.File(PROJECT / "data/phase1_dataset/backbone_100k.h5", "r")
names = [n for n in list(h5["val"].keys())[:120] if n in bbf["val"]][:40]
L = 256; B = len(names)
ca = torch.zeros(B, L, 3); bb = torch.zeros(B, L, 3, 3); z = torch.zeros(B, L, 8); mask = torch.zeros(B, L, dtype=torch.bool)
for i, n in enumerate(names):
    c = torch.from_numpy(h5["val"][n]["ca_coords"][:])[:L]; b = torch.from_numpy(bbf["val"][n][:])[:L]; zz = torch.from_numpy(h5["val"][n]["z"][:])[:L]
    ca[i, :len(c)] = c; bb[i, :len(b)] = b; z[i, :len(zz)] = zz; mask[i, :len(c)] = True
print(f"{B} val proteins, lengths {mask.sum(1).min().item()}-{mask.sum(1).max().item()}")
print(f"{'config':<16s} {'Ca-FAPE':>8s} {'bb-FAPE/ca':>10s} {'bb-FAPE/all':>11s} {'N-CA std':>9s} {'Ca RMSD':>8s} {'s/decode':>8s}")
for steps in (3, 10):
    for amp in (True, False):
        dec = DifferentiableDecoder(ae, n_steps=steps).to(dev)
        torch.manual_seed(0); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            cap, bbp = dec(z.to(dev), mask.to(dev), return_backbone=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        cap, bbp = cap.float().cpu(), bbp.float().cpu()
        d = backbone_bond_lengths(bbp, mask)
        print(f"{'bf16' if amp else 'fp32':<5s}{steps:>2d} steps      "
              f"{fape_loss(cap, ca, mask, fix_ends=True).item():8.4f} "
              f"{fape_loss_backbone(bbp[:, :, :3], bb, mask).item():10.4f} "
              f"{fape_loss_backbone(bbp[:, :, :3], bb, mask, all_atom_points=True).item():11.4f} "
              f"{d[0][mask].std().item():9.3f} {np.nanmean(kabsch_rmsd(cap, ca, mask)):8.3f} {dt:8.2f}")
# same latent, two noise seeds: how much of the floor is the ODE's random start?
dec = DifferentiableDecoder(ae, n_steps=3).to(dev)
outs = []
for seed in (0, 1):
    torch.manual_seed(seed)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outs.append(dec(z.to(dev), mask.to(dev), return_backbone=True)[1].float().cpu())
print(f"\nseed-to-seed (bf16, 3 steps): bb-FAPE/all between two decodes of the SAME latent = "
      f"{fape_loss_backbone(outs[0][:, :, :3], outs[1][:, :, :3], mask, all_atom_points=True).item():.4f}, "
      f"Ca-FAPE = {fape_loss(outs[0][:, :, 1], outs[1][:, :, 1], mask, fix_ends=True).item():.4f}")
