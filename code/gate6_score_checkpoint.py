"""TM-score for a given head checkpoint. Small batch + 3 ODE steps so it can run
alongside training (head output is step-invariant: verified 0.9217 at 3/10/20)."""
import os as _os
# Portable root. Set ESM_PROAE_ROOT to relocate the project; set
# FOLDSEEK_BIN if foldseek is not on PATH.
ROOT = _os.environ.get("ESM_PROAE_ROOT", "/home/guest/projects/esm_proae")
_FOLDSEEK = _os.environ.get("FOLDSEEK_BIN", "foldseek")
_os.makedirs(_os.path.join(ROOT, "notes"), exist_ok=True)  # results dir may not exist in a fresh checkout
import os, sys, json, tempfile, shutil, argparse, pathlib
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
os.chdir(ROOT + "/ProteinAE_v1")
sys.path.insert(0, "."); sys.path.insert(0, ROOT)
import lightning as L, hydra
from proteinfoundation.proteinflow.proteinae import ProteinAE
from gate6_fape_train import DifferentiableDecoder, ProteinDatasetFAPE, fape_loss
from gate6_deep_head import DeepHead
from gate6_fold_validation import write_ca_as_pdb, tm_scores
from gate6_corrected_eval import kabsch_rmsd

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--bs", type=int, default=10)
ap.add_argument("--n-layers", type=int, default=None, help="override sidecar arch")
ap.add_argument("--d-model", type=int, default=None, help="override sidecar arch")
ap.add_argument("--offset", type=int, default=0,
                help="skip this many proteins of the seed-42 permutation (0 = gate set; the training "
                     "runs select on the first 200, so use >=1000 for a selection-free score)")
a = ap.parse_args()
P = Path(ROOT)
L.seed_everything(42); torch.set_float32_matmul_precision("high"); dev = torch.device("cuda")
with hydra.initialize_config_dir(config_dir=f"{os.getcwd()}/configs/experiment_config",
                                 version_base=hydra.__version__):
    hydra.compose(config_name="inference_proteinae", return_hydra_config=True)
ae = ProteinAE.load_from_checkpoint("checkpoints/ae_r1_d8_v1.ckpt", strict=True,
                                     weights_only=False).eval().to(dev)
dec = DifferentiableDecoder(ae, n_steps=3).to(dev)
vf = ProteinDatasetFAPE(P/"data"/"phase1_dataset"/"dataset_100k.h5", "val")
torch.manual_seed(42); idx = torch.randperm(len(vf))[a.offset:a.offset + a.n].tolist()
names = [vf.names[i] for i in idx]
loader = DataLoader(torch.utils.data.Subset(vf, idx), batch_size=a.bs, num_workers=2)
# Read the architecture from the checkpoint's sidecar rather than assuming it,
# so a head-size sweep can be scored. Falls back to the 10x256 head that every
# checkpoint predating the sweep used.
_ck = P/"data"/"phase1_dataset"/a.ckpt
_meta_path = pathlib.Path(str(_ck) + ".meta.json")
_DEFAULT = {"n_layers": 10, "d_model": 256, "dropout": 0.15,
            "use_conv": False, "normalize_out": True}
if _meta_path.exists():
    _m = json.loads(_meta_path.read_text())
    arch = {k: _m.get(k, v) for k, v in _DEFAULT.items()}
    src = "sidecar"
else:
    arch = dict(_DEFAULT); src = "DEFAULT (no sidecar found)"
if a.n_layers: arch["n_layers"] = a.n_layers      # explicit override
if a.d_model:  arch["d_model"] = a.d_model
print(f"  arch       : {arch['n_layers']}L d{arch['d_model']} "
      f"normalize_out={arch['normalize_out']}  [{src}]")
h = DeepHead(**arch).to(dev)
try:
    h.load_state_dict(torch.load(str(_ck), weights_only=True))
except RuntimeError as e:
    raise SystemExit(f"\nArchitecture does not match the weights.\n  tried: {arch}\n"
                     f"  pass --n-layers/--d-model explicitly.\n  {str(e)[:300]}")
h.eval()
w = tempfile.mkdtemp(prefix="tmck_"); gt = os.path.join(w,"gt"); pd_ = os.path.join(w,"pred")
os.makedirs(gt); os.makedirs(pd_)
ff, nb, rm, k = 0.0, 0, [], 0
with torch.no_grad():
    for esm, z_true, ca, mask, lengths in loader:
        esm, ca_d, mask = esm.to(dev), ca.to(dev), mask.to(dev)
        pred = dec(h(esm, mask), mask)
        ff += fape_loss(pred, ca_d, mask).item(); nb += 1
        rm += kabsch_rmsd(pred, ca_d, mask)
        for b in range(pred.shape[0]):
            Lb = int(lengths[b]); nm = names[k]; k += 1
            write_ca_as_pdb(ca[b,:Lb].numpy(), os.path.join(gt, f"{nm}.pdb"))
            write_ca_as_pdb(pred[b,:Lb].cpu().numpy(), os.path.join(pd_, f"{nm}.pdb"))
tms = tm_scores(pd_, gt, os.path.join(w,"tm"))
meas = [tms[n] for n in names if n in tms]
print(f"\n  checkpoint : {a.ckpt}   proteins: offset {a.offset}, n {a.n}")
print(f"  FAPE       : {ff/nb:.4f}")
print(f"  TM mean    : {np.mean(meas):.3f}   median {np.median(meas):.3f}   coverage {len(meas)}/{len(names)}")
print(f"  TM > 0.5   : {np.mean([tms.get(n,0)>0.5 for n in names]):.2f}")
print(f"  TM > 0.3   : {np.mean([tms.get(n,0)>0.3 for n in names]):.2f}")
print(f"  RMSD mean  : {np.nanmean(rm):.2f} A   median {np.nanmedian(rm):.2f} A")
json.dump({"ckpt":a.ckpt,"arch":arch,"fape":ff/nb,"tm_mean":float(np.mean(meas)),
           "tm_median":float(np.median(meas)),
           "tm_gt_0.5":float(np.mean([tms.get(n,0)>0.5 for n in names])),
           "tm_gt_0.3":float(np.mean([tms.get(n,0)>0.3 for n in names])),
           "rmsd_mean":float(np.nanmean(rm))},
          open(P/"notes"/f"tm_{a.ckpt.replace('.pt','')}{'' if a.offset == 0 else f'_off{a.offset}'}.json","w"), indent=2)
shutil.rmtree(w, ignore_errors=True)
