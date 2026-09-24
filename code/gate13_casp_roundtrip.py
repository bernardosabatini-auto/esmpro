"""Control: does the frozen ProteinAE round trip (true latent -> decoder) hold on
EXPERIMENTAL structures with gaps? TM of decode(z_true) vs truth on the CASP set
(and optionally any other val-layout file). On AFDB training data it is 0.997."""
import os, sys, tempfile, shutil, argparse, numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code"); os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
import gate7_latent_flow as G
from gate6_fape_train import _write_pseudo_backbone_pdb, _foldseek_tm
from gate6_corrected_eval import kabsch_rmsd
ap = argparse.ArgumentParser(); ap.add_argument("--h5", default=f"{ROOT}/data/phase1_dataset/dataset_casp.h5"); ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args(); dev = torch.device("cuda"); dec = G.load_decoder(dev)
h = h5py.File(a.h5, "r")[a.split]; names = list(h.keys())[: a.n or None]
work = tempfile.mkdtemp(prefix="rt_"); os.makedirs(f"{work}/gt"); os.makedirs(f"{work}/pr"); rms = []
with torch.no_grad():
    for i in range(0, len(names), 16):
        b = names[i:i+16]; L = max(h[n]["z"].shape[0] for n in b)
        z = torch.zeros(len(b), L, 8); m = torch.zeros(len(b), L, dtype=torch.bool); ca = torch.zeros(len(b), L, 3)
        for j, n in enumerate(b):
            zz = h[n]["z"][:]; z[j, :len(zz)] = torch.from_numpy(zz); m[j, :len(zz)] = True; ca[j, :len(zz)] = torch.from_numpy(h[n]["ca_coords"][:])
        pred = dec(z.to(dev), m.to(dev)).float().cpu()
        rms += kabsch_rmsd(pred, ca, m)
        for j, n in enumerate(b):
            k = int(m[j].sum()); _write_pseudo_backbone_pdb(ca[j, :k].numpy(), f"{work}/gt/{n}.pdb"); _write_pseudo_backbone_pdb(pred[j, :k].numpy(), f"{work}/pr/{n}.pdb")
tm = _foldseek_tm(f"{work}/pr", f"{work}/gt", f"{work}/tm"); shutil.rmtree(work, ignore_errors=True)
v = [tm.get(n, 0.0) for n in names]
print(f"round trip on {os.path.basename(a.h5)} ({len(names)}): TM {np.mean(v):.3f}  TM>0.5 {np.mean(np.array(v) > 0.5):.2f}  RMSD {np.nanmean(rms):.2f} A  coverage {len(tm)/len(names):.2f}")
worst = sorted(range(len(names)), key=lambda i: v[i])[:5]; print("  worst:", [(names[i], round(v[i], 2), int(h[names[i]].attrs.get('dropped_residues', -1))) for i in worst])
