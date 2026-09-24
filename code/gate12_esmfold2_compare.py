"""Gate 12: external comparison — ESMFold2-Fast (single sequence) on our
evaluation proteins, scored with the SAME pipeline (Ca pseudo-backbone PDBs,
Foldseek TM-align exhaustive search, Kabsch RMSD) as our models.

Sets: the seed-42 held-out slice (positions 1000-1099) and the no-neighbour
subsets (notes/val_noneighbour_lt0.6.txt, lt0.5.txt). Uses transformers'
native EsmFold2Model (no `esm` package). Records ESMFold2's own pTM/pLDDT too.

  python gate12_esmfold2_compare.py --model data/esmfold2_fast --loops 3 --steps 50
"""
import os, sys, json, time, argparse, tempfile, shutil
import numpy as np, torch, h5py
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
from gate6_fape_train import _write_pseudo_backbone_pdb, _foldseek_tm, H5_PATH, PROJECT
from gate6_corrected_eval import kabsch_rmsd
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(PROJECT / "data/esmfold2_fast"))
ap.add_argument("--loops", type=int, default=3); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--sets", default="heldout,lt0.6,lt0.5"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features, output_to_pdb
dev = torch.device("cuda")
t0 = time.perf_counter()
model = EsmFold2Model.from_pretrained(a.model, dtype=torch.bfloat16).eval().to(dev)
print(f"loaded ESMFold2-Fast in {time.perf_counter()-t0:.0f}s, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
      f"{torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

h = h5py.File(H5_PATH, "r"); vnames = list(h["val"].keys())
torch.manual_seed(42); perm = torch.randperm(len(vnames)).tolist()
sets = {}
if "heldout" in a.sets: sets["heldout"] = [vnames[i] for i in perm[1000:1100]]
for tag in ("lt0.6", "lt0.5"):
    if tag in a.sets: sets[tag] = [l.strip() for l in open(PROJECT / "notes" / f"val_noneighbour_{tag}.txt") if l.strip()]
todo = sorted({n for v in sets.values() for n in v})
if a.limit: todo = todo[:a.limit]
print(f"{len(todo)} unique proteins across sets {list(sets)}", flush=True)

work = tempfile.mkdtemp(prefix="esmfold2_"); gt_d, pr_d = f"{work}/gt", f"{work}/pred"; os.makedirs(gt_d); os.makedirs(pr_d)
per = {}
for k, nm in enumerate(todo):
    g = h["val"][nm]; seq = str(g.attrs["sequence"]); ca_true = torch.from_numpy(g["ca_coords"][:])
    t1 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feats = prepare_protein_features(seq, device=dev)
        out = model.fold(**feats, num_loops=a.loops, num_sampling_steps=a.steps, num_diffusion_samples=1)
    pdb = output_to_pdb(out, feats)
    ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.splitlines()
                   if l.startswith("ATOM") and l[12:16].strip() == "CA"], dtype=np.float32)
    if ca.shape[0] != len(seq):
        print(f"  {nm}: CA count {ca.shape[0]} != {len(seq)}", flush=True); continue
    L = len(seq); m = torch.ones(1, L, dtype=torch.bool)
    rmsd = float(kabsch_rmsd(torch.from_numpy(ca)[None], ca_true[None], m)[0])
    _write_pseudo_backbone_pdb(ca_true.numpy(), f"{gt_d}/{nm}.pdb"); _write_pseudo_backbone_pdb(ca, f"{pr_d}/{nm}.pdb")
    per[nm] = {"len": L, "rmsd": rmsd, "ptm": float(out["ptm"].mean()), "plddt": float(out["plddt"].mean()), "sec": time.perf_counter() - t1}
    if (k + 1) % 25 == 0: print(f"  {k+1}/{len(todo)}  {time.perf_counter()-t0:.0f}s", flush=True)
tms = _foldseek_tm(pr_d, gt_d, f"{work}/tm")
for nm in per: per[nm]["tm"] = tms.get(nm)
shutil.rmtree(work, ignore_errors=True)
summary = {}
print(f"\n{'set':8s} {'n':>4s} {'TM':>6s} {'TM>0.5':>7s} {'RMSD':>6s} {'pTM':>5s} {'cov':>6s} {'s/prot':>6s}")
for tag, names in sets.items():
    rows = [per[n] for n in names if n in per]
    tmv = [r["tm"] for r in rows if r["tm"] is not None]
    s = {"n": len(rows), "tm": float(np.mean(tmv)) if tmv else None, "tm_frac": float(np.mean([(r["tm"] or 0) > 0.5 for r in rows])),
         "rmsd": float(np.mean([r["rmsd"] for r in rows])), "ptm": float(np.mean([r["ptm"] for r in rows])),
         "coverage": len(tmv) / max(len(rows), 1), "sec_per_protein": float(np.mean([r["sec"] for r in rows]))}
    summary[tag] = s
    print(f"{tag:8s} {s['n']:4d} {s['tm']:6.3f} {s['tm_frac']:7.2f} {s['rmsd']:6.2f} {s['ptm']:5.2f} {s['coverage']:6.2f} {s['sec_per_protein']:6.2f}", flush=True)
json.dump({"model": a.model, "loops": a.loops, "steps": a.steps, "summary": summary, "per_protein": per},
          open(PROJECT / "notes" / "gate12_esmfold2_fast.json", "w"), indent=1)
print("done")
