"""How brittle is the decoder to latent error? Add noise to TRUE latents
(on-manifold: re-layer-normed) and measure TM / RMSD vs noise level, and
compare with the head's actual latent error. Answers whether the head's
target is a smooth basin or a needle. GPU."""
import os, sys, json, tempfile, shutil, numpy as np, torch, h5py, torch.nn.functional as F
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
from gate6_fape_train import DifferentiableDecoder, fape_loss, H5_PATH, PROJECT
from gate6_fold_validation import write_ca_as_pdb, tm_scores
from gate6_corrected_eval import kabsch_rmsd
from gate6_deep_head import DeepHead
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import lightning as Lt, hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--ckpt", default="smallscale_final_40k.pt"); a = ap.parse_args()
dev = torch.device("cuda"); Lt.seed_everything(42)
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config", version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True, weights_only=False).eval().to(dev)
dec = DifferentiableDecoder(ae, n_steps=3).to(dev)
h5 = h5py.File(H5_PATH, "r"); names = list(h5["val"].keys()); torch.manual_seed(42)
idx = torch.randperm(len(names))[:a.n].tolist(); names = [names[i] for i in idx]
L = 256; B = len(names)
ca = torch.zeros(B, L, 3); z = torch.zeros(B, L, 8); esm = torch.zeros(B, L, 1280); mask = torch.zeros(B, L, dtype=torch.bool)
for i, n in enumerate(names):
    g = h5["val"][n]; c = torch.from_numpy(g["ca_coords"][:])[:L]; zz = torch.from_numpy(g["z"][:])[:L]
    e = torch.from_numpy(g["esm2_emb"][:].astype(np.float32))[:L]
    ca[i, :len(c)] = c; z[i, :len(zz)] = zz; esm[i, :len(e)] = e; mask[i, :len(c)] = True
ck = PROJECT / "data/phase1_dataset" / a.ckpt; meta = json.loads(open(str(ck) + ".meta.json").read())
head = DeepHead(n_layers=meta["n_layers"], d_model=meta["d_model"], normalize_out=meta["normalize_out"]).to(dev)
head.load_state_dict(torch.load(str(ck), weights_only=True)); head.eval()

def evaluate(zin, tag):
    w = tempfile.mkdtemp(prefix="tol_"); gt = os.path.join(w, "gt"); pd_ = os.path.join(w, "pred"); os.makedirs(gt); os.makedirs(pd_)
    fa, rm = 0.0, []
    for s in range(0, B, 10):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = dec(zin[s:s+10].to(dev), mask[s:s+10].to(dev)).float().cpu()
        fa += fape_loss(pred, ca[s:s+10], mask[s:s+10], fix_ends=True).item() * pred.shape[0]
        rm += kabsch_rmsd(pred, ca[s:s+10], mask[s:s+10])
        for b in range(pred.shape[0]):
            n = mask[s+b].sum().item(); nm = names[s+b]
            write_ca_as_pdb(ca[s+b, :n].numpy(), os.path.join(gt, f"{nm}.pdb")); write_ca_as_pdb(pred[b, :n].numpy(), os.path.join(pd_, f"{nm}.pdb"))
    tms = tm_scores(pd_, gt, os.path.join(w, "tm")); meas = [tms[n] for n in names if n in tms]
    shutil.rmtree(w, ignore_errors=True)
    print(f"  {tag:<34s} TM {np.mean(meas):.3f}  TM>0.5 {np.mean([tms.get(n,0)>0.5 for n in names]):.2f}  "
          f"RMSD {np.nanmean(rm):5.2f} A  Ca-FAPE {fa/B:.3f}  coverage {len(meas)}/{B}", flush=True)

def lat_err(zp):   # per-residue L2 between latents, masked mean (true latents have L2 = 2.83)
    return ((zp - z).norm(dim=-1)[mask]).mean().item()

print(f"{B} val proteins. True latent per-residue L2 = {z[mask].norm(dim=-1).mean():.3f}")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    zh = torch.cat([head(esm[s:s+20].to(dev), mask[s:s+20].to(dev)).float().cpu() for s in range(0, B, 20)])
print(f"head latent error (L2 to true, per residue) = {lat_err(zh):.3f}; cosine = {F.cosine_similarity(zh[mask], z[mask], dim=-1).mean():.3f}")
evaluate(z, "true latent")
evaluate(zh, f"head ({a.ckpt})")
torch.manual_seed(0)
for sig in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
    zn = F.layer_norm(z + sig * torch.randn_like(z), (8,)) * mask.unsqueeze(-1)   # back onto the manifold
    evaluate(zn, f"true + N(0,{sig}) renormed (err {lat_err(zn):.2f})")
# structured error: interpolate between true and head latents
for t in (0.25, 0.5, 0.75):
    zi = F.layer_norm((1 - t) * z + t * zh, (8,)) * mask.unsqueeze(-1)
    evaluate(zi, f"lerp true->head t={t} (err {lat_err(zi):.2f})")
