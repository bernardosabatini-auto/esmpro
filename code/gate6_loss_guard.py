"""Guards for every structural loss term (CLAUDE.md section 6, bug 1), with
the pose change applied to REAL residues only so padded zeros stay put, as
they do in training. Also (with --gpu) decodes true latents to confirm the
decoder's backbone atom order and units via bond lengths.
Exit 1 on any failure."""
import os, sys, math, argparse
import numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]
sys.path.insert(0, ROOT + "/code")
from gate6_fape_train import (fape_loss, fape_loss_backbone, bond_length_penalty,
                              backbone_bond_lengths, H5_PATH, PROJECT,
                              BOND_N_CA, BOND_CA_C, BOND_C_N)
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", action="store_true")
ap.add_argument("--ckpt", default="smallscale_final_40k.pt"); a = ap.parse_args()

torch.manual_seed(0)
h5 = h5py.File(H5_PATH, "r"); bbf = h5py.File(PROJECT / "data/phase1_dataset/backbone_100k.h5", "r")
names = [n for n in list(h5["val"].keys())[:40] if n in bbf["val"]][:12]
L = 256; B = len(names)
ca = torch.zeros(B, L, 3); bb = torch.zeros(B, L, 3, 3); mask = torch.zeros(B, L, dtype=torch.bool)
for i, n in enumerate(names):
    c = torch.from_numpy(h5["val"][n]["ca_coords"][:])[:L]; b = torch.from_numpy(bbf["val"][n][:])[:L]
    ca[i, :len(c)] = c; bb[i, :len(b)] = b; mask[i, :len(c)] = True
assert torch.equal(bb[:, :, 1][mask], ca[mask]), "backbone CA != dataset CA"
print(f"backbone CA matches dataset CA exactly for {B} proteins (lengths {mask.sum(1).min().item()}-{mask.sum(1).max().item()})")

def rot(axis, deg):
    ax = torch.tensor(axis, dtype=torch.float32); ax = ax / ax.norm(); t = math.radians(deg)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return torch.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
def pose(x, R, t):   # move REAL residues only
    m = mask.view(B, L, *([1] * (x.dim() - 2))).float()
    return (x @ R.T + t) * m

ok = True
def check(label, val, cond):
    global ok; good = cond(val); ok &= good
    print(f"  {label:<46s} {val:.6f}   {'OK' if good else 'FAIL'}")

poses = [("rot 30", rot([1, 2, 3], 30), torch.zeros(3)), ("rot 90", rot([0, 1, 0], 90), torch.zeros(3)),
         ("rot 180", rot([1, 0, 0], 180), torch.zeros(3)), ("trans 1 A", torch.eye(3), torch.tensor([1., 0, 0])),
         ("trans 20 A", torch.eye(3), torch.tensor([20., -20, 5])), ("rot 90 + trans 20", rot([1, 1, 0], 90), torch.tensor([20., 3, -9]))]
print("\n[Ca-FAPE, legacy end frames] expect a SMALL nonzero leak (documented):")
for lab, R, t in poses:
    check(lab, fape_loss(pose(ca, R, t), ca, mask).item(), lambda v: v < 0.05)
print("[Ca-FAPE, fix_ends] expect ~0:")
for lab, R, t in poses:
    check(lab, fape_loss(pose(ca, R, t), ca, mask, fix_ends=True).item(), lambda v: v < 1e-3)
check("1 A noise (must be nonzero)", fape_loss(ca + torch.randn_like(ca), ca, mask, fix_ends=True).item(), lambda v: v > 0.05)
for pts in ("ca", "all"):
    print(f"[backbone-FAPE, true frames, points={pts}] expect ~0:")
    for lab, R, t in poses:
        check(lab, fape_loss_backbone(pose(bb, R, t), bb, mask, all_atom_points=(pts == "all")).item(), lambda v: v < 1e-3)
    check("1 A noise (must be nonzero)", fape_loss_backbone(bb + torch.randn_like(bb), bb, mask, all_atom_points=(pts == "all")).item(), lambda v: v > 0.05)
    check("mirror image (must be nonzero)", fape_loss_backbone(bb * torch.tensor([1., 1., -1.]), bb, mask, all_atom_points=(pts == "all")).item(), lambda v: v > 0.05)
print("[bond penalty]")
b0 = bond_length_penalty(bb, mask).item()
check("true backbone (near-ideal bonds)", b0, lambda v: v < 0.05)
check("true backbone, rot 90 + trans 20 (same)", abs(bond_length_penalty(pose(bb, rot([1, 1, 0], 90), torch.tensor([20., 3, -9])), mask).item() - b0), lambda v: v < 1e-4)
check("0.3 A noise (must rise)", bond_length_penalty(bb + 0.3 * torch.randn_like(bb), mask).item(), lambda v: v > 0.1)
d = backbone_bond_lengths(bb, mask)
print(f"  true bond lengths: N-CA {d[0][mask].mean():.3f}+-{d[0][mask].std():.3f}  CA-C {d[1][mask].mean():.3f}+-{d[1][mask].std():.3f}  C-N {d[2][d[4].bool()].mean():.3f}+-{d[2][d[4].bool()].std():.3f}")

if a.gpu:
    print("\n[decoder atom order / units via bond lengths]")
    os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
    import lightning as Lt, hydra, json
    from proteinfoundation.proteinflow.proteinae import ProteinAE
    from gate6_fape_train import DifferentiableDecoder
    from gate6_deep_head import DeepHead
    dev = torch.device("cuda")
    with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
        hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
    ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False).eval().to(dev)
    dec = DifferentiableDecoder(ae, n_steps=3).to(dev)
    z = torch.zeros(B, L, 8); esm = torch.zeros(B, L, 1280)
    for i, n in enumerate(names):
        zz = torch.from_numpy(h5["val"][n]["z"][:])[:L]; z[i, :len(zz)] = zz
        ee = torch.from_numpy(h5["val"][n]["esm2_emb"][:].astype(np.float32))[:L]; esm[i, :len(ee)] = ee
    def stats(bbp, tag):
        d = backbone_bond_lengths(bbp.float().cpu(), mask)
        s = (f"N-CA {d[0][mask].mean():.3f}+-{d[0][mask].std():.3f}  CA-C {d[1][mask].mean():.3f}+-{d[1][mask].std():.3f}  "
             f"C-N {d[2][d[4].bool()].mean():.3f}+-{d[2][d[4].bool()].std():.3f}")
        print(f"  {tag:<22s} {s}"); return d
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ca_p, bb_p = dec(z.to(dev), mask.to(dev), return_backbone=True)
    d = stats(bb_p, "true latent -> decoder")
    check("true-latent N-CA mean within 0.05 of 1.458", abs(d[0][mask].mean().item() - BOND_N_CA), lambda v: v < 0.05)
    check("true-latent CA-C mean within 0.05 of 1.525", abs(d[1][mask].mean().item() - BOND_CA_C), lambda v: v < 0.05)
    check("true-latent C-N mean within 0.05 of 1.329", abs(d[2][d[4].bool()].mean().item() - BOND_C_N), lambda v: v < 0.05)
    # Measured floor (gate6_bbfape_floor.py, 40 proteins): 0.131 for a perfect
    # latent, from the decoder's stochastic sampling; Ca-FAPE floor is 0.053.
    # An atom-order or unit error would read ~1, so 0.25 still discriminates.
    check("true-latent bb-FAPE vs truth (floor ~0.13)", fape_loss_backbone(bb_p[:, :, :3].float().cpu(), bb, mask).item(), lambda v: v < 0.25)
    check("true-latent Ca-FAPE vs truth (floor ~0.05)", fape_loss(ca_p.float().cpu(), ca, mask, fix_ends=True).item(), lambda v: v < 0.12)
    ck = PROJECT / "data/phase1_dataset" / a.ckpt; meta = json.loads(open(str(ck) + ".meta.json").read())
    h = DeepHead(n_layers=meta["n_layers"], d_model=meta["d_model"], normalize_out=meta["normalize_out"]).to(dev)
    h.load_state_dict(torch.load(str(ck), weights_only=True)); h.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ca_h, bb_h = dec(h(esm.to(dev), mask.to(dev)), mask.to(dev), return_backbone=True)
    stats(bb_h, f"head ({a.ckpt}) -> decoder")
    print(f"  head bond penalty = {bond_length_penalty(bb_h.float().cpu(), mask).item():.4f} A;  "
          f"head bb-FAPE(all) = {fape_loss_backbone(bb_h[:, :, :3].float().cpu(), bb, mask, all_atom_points=True).item():.4f};  "
          f"head Ca-FAPE = {fape_loss(ca_h.float().cpu(), ca, mask, fix_ends=True).item():.4f}")
print("\nPASS" if ok else "\nFAIL"); sys.exit(0 if ok else 1)
