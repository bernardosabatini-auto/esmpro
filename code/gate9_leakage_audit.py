"""Gate 9: structural leakage audit of the training set against validation.

For each validation protein (the seed-42 2000-subset, which contains the gate
set = first 100 and the held-out slice = 1000-1099), find the most similar
TRAINING structure by Foldseek TM-align, separately for the original 80k
train split and for the 393k AFDB additions. If the additions contain
near-duplicates of held-out proteins, the held-out gain of the 473k model is
inflated. Uses Foldseek's prefilter (not exhaustive): we are looking for
CLOSE hits, which the prefilter finds; a miss means no close homolog, which
is the benign case. Writes Ca pseudo-backbone PDBs to node-local scratch.

  python gate9_leakage_audit.py --threads 14 --work /tmp/audit
"""
import os, sys, json, time, argparse, subprocess, collections
import numpy as np, h5py, torch
ROOT = os.environ["ESM_PROAE_ROOT"]; sys.path.insert(0, ROOT + "/code")
os.chdir(ROOT + "/ProteinAE_v1"); sys.path.insert(0, ".")
from gate6_fape_train import _write_pseudo_backbone_pdb, H5_PATH, PROJECT, _FOLDSEEK
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=14)
ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp") + "/audit"); ap.add_argument("--n-val", type=int, default=2000)
a = ap.parse_args()
W = a.work; os.makedirs(f"{W}/val", exist_ok=True); os.makedirs(f"{W}/train", exist_ok=True)
t0 = time.perf_counter()

h = h5py.File(H5_PATH, "r")
vnames = list(h["val"].keys()); torch.manual_seed(42); vidx = torch.randperm(len(vnames))[:a.n_val].tolist()
vsel = [vnames[i] for i in vidx]
subset = {}
for pos, nm in enumerate(vsel):
    subset[nm] = "gate" if pos < 100 else ("heldout" if 1000 <= pos < 1100 else "other")
for nm in vsel:
    _write_pseudo_backbone_pdb(h["val"][nm]["ca_coords"][:], f"{W}/val/{nm}.pdb")
print(f"val: {len(vsel)} written ({time.perf_counter()-t0:.0f}s)", flush=True)

n = 0
for nm in h["train"].keys():
    _write_pseudo_backbone_pdb(h["train"][nm]["ca_coords"][:], f"{W}/train/k_{nm}.pdb"); n += 1
print(f"train 100k: {n} written ({time.perf_counter()-t0:.0f}s)", flush=True)
for k in (0, 1):
    hh = h5py.File(str(PROJECT / "data/phase1_dataset" / f"dataset_afdb_train_{k}.h5"), "r")["train"]
    m = 0
    for nm in hh.keys():
        _write_pseudo_backbone_pdb(hh[nm]["ca_coords"][:], f"{W}/train/a_{nm}.pdb"); m += 1
        if m % 100000 == 0: print(f"  shard {k}: {m} ({time.perf_counter()-t0:.0f}s)", flush=True)
    n += m
print(f"train total: {n} written ({time.perf_counter()-t0:.0f}s)", flush=True)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(cmd[:3] + [r.stderr[-500:]])
run([_FOLDSEEK, "createdb", f"{W}/train", f"{W}/trainDB", "--threads", str(a.threads)])
run([_FOLDSEEK, "createdb", f"{W}/val", f"{W}/valDB", "--threads", str(a.threads)])
print(f"dbs built ({time.perf_counter()-t0:.0f}s)", flush=True)
os.makedirs(f"{W}/tmp", exist_ok=True)
run([_FOLDSEEK, "search", f"{W}/valDB", f"{W}/trainDB", f"{W}/aln", f"{W}/tmp", "--alignment-type", "1",
     "-s", "9.5", "--max-seqs", "2000", "-e", "10", "--exact-tmscore", "1", "-a", "--threads", str(a.threads)])
run([_FOLDSEEK, "convertalis", f"{W}/valDB", f"{W}/trainDB", f"{W}/aln", f"{W}/aln.tsv",
     "--format-output", "query,target,qtmscore,ttmscore,alntmscore", "--threads", str(a.threads)])
print(f"search done ({time.perf_counter()-t0:.0f}s)", flush=True)

best = collections.defaultdict(lambda: {"k": 0.0, "a": 0.0, "k_hit": None, "a_hit": None})
for line in open(f"{W}/aln.tsv"):
    q, t, qtm, ttm, atm = line.rstrip("\n").split("\t")[:5]
    q = q.replace(".pdb", ""); tm = max(float(qtm), float(ttm)); src = "k" if t.startswith("k_") else "a"
    if tm > best[q][src]: best[q][src] = tm; best[q][src + "_hit"] = t.replace(".pdb", "")
out = {"per_protein": {}, "summary": {}}
for nm in vsel:
    b = best[nm]; out["per_protein"][nm] = {"subset": subset[nm], "max_tm_to_100k_train": b["k"], "max_tm_to_afdb_train": b["a"],
                                          "hit_100k": b["k_hit"], "hit_afdb": b["a_hit"]}
for sub in ("gate", "heldout", "other"):
    rows = [v for v in out["per_protein"].values() if v["subset"] == sub]
    k = np.array([r["max_tm_to_100k_train"] for r in rows]); aa = np.array([r["max_tm_to_afdb_train"] for r in rows]); both = np.maximum(k, aa)
    s = {"n": len(rows)}
    for thr in (0.5, 0.7, 0.9):
        s[f"frac_gt{thr}_100k"] = float((k > thr).mean()); s[f"frac_gt{thr}_afdb"] = float((aa > thr).mean()); s[f"frac_gt{thr}_either"] = float((both > thr).mean())
    s["mean_max_tm_100k"] = float(k.mean()); s["mean_max_tm_afdb"] = float(aa.mean())
    s["frac_afdb_closer_than_100k"] = float((aa > k).mean())
    out["summary"][sub] = s
    print(f"{sub:8s} n={len(rows):4d}  nearest-train TM>0.5: 100k {s['frac_gt0.5_100k']:.2f} afdb {s['frac_gt0.5_afdb']:.2f} either {s['frac_gt0.5_either']:.2f} | "
          f">0.7: {s['frac_gt0.7_100k']:.2f} {s['frac_gt0.7_afdb']:.2f} {s['frac_gt0.7_either']:.2f} | >0.9: {s['frac_gt0.9_100k']:.2f} {s['frac_gt0.9_afdb']:.2f} {s['frac_gt0.9_either']:.2f} | "
          f"mean max TM {s['mean_max_tm_100k']:.3f} / {s['mean_max_tm_afdb']:.3f}", flush=True)
json.dump(out, open(PROJECT / "notes" / "gate9_leakage_audit.json", "w"), indent=1)
print(f"done ({time.perf_counter()-t0:.0f}s) -> notes/gate9_leakage_audit.json")
