"""Gate 31: designability with a real inverse-folding model.

gate29 could not answer whether our generated structures are designable, because the control
exposed the pipeline's own ceiling: genuine experimental structures scored only scTM 0.376
through our 174M co-design head. This repeats the loop with ProteinMPNN's CA-only model, which
takes exactly what our decoder produces (an alpha-carbon trace) and needs no fabricated backbone.

  structure -> ProteinMPNN (CA-only, --num-seq per backbone) -> ESMFold2-Fast -> TM back

Standard protocol (RFdiffusion / Genie): best scTM over the designed sequences, designable if
above 0.5. Every set is measured the same way, including real structures as the control.

  python gate31_designability_mpnn.py --dirs notes/gate28/real,notes/gate28/uncond --n 32
"""
import os, sys, json, glob, shutil, argparse, subprocess, tempfile, time
import numpy as np, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
from gate6_fape_train import PROJECT, _write_pseudo_backbone_pdb, _foldseek_tm

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", default="notes/gate28/real,notes/gate28/uncond,notes/gate30/prior,notes/gate30/contact_s0.2,notes/gate30/inpaint,notes/gate30/noiseinterp0.5")
ap.add_argument("--n", type=int, default=32, help="backbones per set")
ap.add_argument("--num-seq", type=int, default=8, help="ProteinMPNN sequences per backbone")
ap.add_argument("--temp", type=float, default=0.1)
ap.add_argument("--tag-out", default="gate31_designability_mpnn")
ap.add_argument("--mpnn", default=str(PROJECT / "tools/ProteinMPNN"))
ap.add_argument("--esmfold", default=str(PROJECT / "data/esmfold2_fast"))
ap.add_argument("--loops", type=int, default=3); ap.add_argument("--fold-steps", type=int, default=50)
a = ap.parse_args(); dev = torch.device("cuda")
W = tempfile.mkdtemp(prefix="g31_")

def mpnn_designs(src_dir, tag):
    """Run ProteinMPNN CA-only on every PDB in src_dir; returns {name: [seq, ...]}."""
    ind = f"{W}/{tag}_in"; os.makedirs(ind, exist_ok=True)
    files = sorted(glob.glob(f"{src_dir}/*.pdb"))[: a.n]
    for f in files: shutil.copy(f, ind)
    pj = f"{W}/{tag}.jsonl"; od = f"{W}/{tag}_mpnn"
    subprocess.run([sys.executable, f"{a.mpnn}/helper_scripts/parse_multiple_chains.py",
                    f"--input_path={ind}", f"--output_path={pj}"], check=True, capture_output=True)
    subprocess.run([sys.executable, f"{a.mpnn}/protein_mpnn_run.py", "--jsonl_path", pj, "--out_folder", od,
                    "--ca_only", "--path_to_model_weights", f"{a.mpnn}/ca_model_weights",
                    "--num_seq_per_target", str(a.num_seq), "--sampling_temp", str(a.temp),
                    "--seed", "1", "--batch_size", "1"], check=True, capture_output=True)
    out = {}
    for fa in glob.glob(f"{od}/seqs/*.fa"):
        nm = os.path.basename(fa)[:-3]; seqs = []; hdr = None
        for line in open(fa):
            line = line.strip()
            if line.startswith(">"): hdr = line; continue
            if hdr and "sample=" in hdr: seqs.append(line)     # skip the poly-A input line
        if seqs: out[nm] = seqs
    return out, {os.path.basename(f)[:-4]: f for f in files}

print(f"loading ESMFold2-Fast", flush=True); t0 = time.perf_counter()
from transformers import EsmFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features, output_to_pdb
fold = EsmFold2Model.from_pretrained(a.esmfold, dtype=torch.float32).eval().to(dev)
print(f"  {time.perf_counter()-t0:.0f}s", flush=True)

summary = {}
for src in [d.strip() for d in a.dirs.split(",")]:
    tag = src.replace("/", "_").replace("notes_", "")
    full = src if os.path.isabs(src) else str(PROJECT / src)
    if not os.path.isdir(full): print(f"  {src}: missing", flush=True); continue
    designs, srcfiles = mpnn_designs(full, tag)
    print(f"\n== {src}: {len(designs)} backbones, {a.num_seq} sequences each", flush=True)
    ref = f"{W}/{tag}_ref"; os.makedirs(ref, exist_ok=True)
    # _foldseek_tm only scores basenames present in BOTH directories, so each design index gets
    # its own directory and keeps the backbone's name.
    preds = [f"{W}/{tag}_pred{j}" for j in range(a.num_seq)]
    for d in preds: os.makedirs(d, exist_ok=True)
    for k, (nm, seqs) in enumerate(sorted(designs.items())):
        shutil.copy(srcfiles[nm], f"{ref}/{nm}.pdb")
        for j, sq in enumerate(seqs[: a.num_seq]):
            with torch.no_grad():
                feats = prepare_protein_features(sq, device=dev)
                o = fold.fold(**feats, num_loops=a.loops, num_sampling_steps=a.fold_steps, num_diffusion_samples=1)
            pdb = output_to_pdb(o, feats)
            ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.splitlines()
                           if l.startswith("ATOM") and l[12:16].strip() == "CA"], dtype=np.float32)
            if ca.shape[0] != len(sq): continue
            _write_pseudo_backbone_pdb(ca, f"{preds[j]}/{nm}.pdb")
        if (k + 1) % 10 == 0: print(f"  {k+1}/{len(designs)}", flush=True)
    best = {}
    for j, pd in enumerate(preds):
        tm = _foldseek_tm(pd, ref, f"{W}/w_{tag}_{j}")
        for nm, val in tm.items(): best[nm] = max(best.get(nm, 0.0), val)
    cov = len(best) / max(len(designs), 1)
    if cov < 1.0: print(f"  WARNING: coverage {cov:.2f} -- {len(designs)-len(best)} backbones scored no matched pair", flush=True)
    v = np.array([best[n] for n in best]) if best else np.array([0.0])
    summary[src] = {"n": len(best), "scTM_best_of_k": float(v.mean()), "designable_frac": float((v > 0.5).mean()),
                    "median": float(np.median(v)), "coverage": float(cov)}
    s = summary[src]
    print(f"  scTM (best of {a.num_seq}) {s['scTM_best_of_k']:.3f}, median {s['median']:.3f}, designable {s['designable_frac']:.0%}, coverage {s['coverage']:.2f}", flush=True)

print(f"\n{'set':>34s} {'n':>4s} {'scTM':>7s} {'median':>7s} {'designable':>11s}")
for k, s in summary.items():
    print(f"{k:>34s} {s['n']:4d} {s['scTM_best_of_k']:7.3f} {s['median']:7.3f} {s['designable_frac']:11.2f}")
print("\nread every row against notes/gate28/real: that is what genuine structures score here")
json.dump(summary, open(PROJECT / "notes" / (a.tag_out + ".json"), "w"), indent=1)
shutil.rmtree(W, ignore_errors=True)
print("ALLDONE")
