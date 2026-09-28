"""Gate 27: choose an EXPERIMENTAL held-out evaluation set.

Every selection and held-out number in this project except CASP and CAMEO has been measured
against AFDB, i.e. against AlphaFold predictions. This builds a validation/selection set of
real structures instead, separated from everything we train on and from the CASP/CAMEO test
sets, so model selection and reporting can stop using predicted references.

Candidates: the PDB cluster representatives already chosen by gate15_select_pdb.py (50 %
identity clusters, resolution <= 3 A) that are NOT in the training files. Each candidate is
then dropped if it is >= --min-id identical (with >= 50 % coverage either way) to ANY training
sequence, or to any CASP or CAMEO benchmark sequence.

  python gate27_select_exp_val.py --n 1200 --threads 16
"""
import os, sys, subprocess, tempfile, argparse, collections, random
import h5py, numpy as np
ROOT = os.environ["ESM_PROAE_ROOT"]; D = f"{ROOT}/data/phase1_dataset"; P = f"{ROOT}/data/pdb"
ap = argparse.ArgumentParser()
ap.add_argument("--min-id", type=float, default=0.3, help="drop candidates at or above this identity to any training or test sequence")
ap.add_argument("--n", type=int, default=1200); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--min-len", type=int, default=32); ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--out", default=f"{P}/exp_val_chains.tsv")
a = ap.parse_args()
MM = os.environ.get("MMSEQS_BIN", "mmseqs")
W = tempfile.mkdtemp(prefix="expval_")

TRAIN_H5 = ["dataset_100k.h5", "dataset_pdb_train.h5", "dataset_pdb_long.h5",
            "dataset_afdb_long_0.h5", "dataset_afdb_long_1.h5", "dataset_afdb_long_2.h5", "dataset_afdb_long_3.h5",
            "dataset_afdb_short2_0.h5", "dataset_afdb_short2_1.h5", "dataset_afdb_short2_2.h5", "dataset_afdb_short2_3.h5",
            "dataset_afdb_train_0.h5", "dataset_afdb_train_1.h5"]
TEST_H5 = ["dataset_casp512.h5", "dataset_casp.h5", "dataset_cameo22.h5", "dataset_apo.h5", "dataset_codnas.h5"]

def dump(files, path, splits=("train",)):
    n = 0; seen_names = set()
    with open(path, "w") as f:
        for fn in files:
            p = f"{D}/{fn}"
            if not os.path.exists(p): print(f"  (missing {fn})", flush=True); continue
            h = h5py.File(p, "r")
            for sp in splits:
                if sp not in h: continue
                g = h[sp]
                for nm in g:
                    s = g[nm].attrs.get("sequence")
                    if s is None: continue
                    f.write(f">{fn}|{sp}|{nm}\n{str(s)}\n"); n += 1; seen_names.add(nm)
            h.close()
            print(f"  {fn}: running total {n}", flush=True)
    return n, seen_names

print("collecting training sequences...", flush=True)
n_train, train_names = dump(TRAIN_H5, f"{W}/train.fasta", ("train",))
print(f"{n_train} training sequences", flush=True)
print("collecting benchmark (test) sequences...", flush=True)
n_test, _ = dump(TEST_H5, f"{W}/test.fasta", ("val", "train"))
print(f"{n_test} benchmark sequences", flush=True)
subprocess.run(f"cat {W}/train.fasta {W}/test.fasta > {W}/exclude.fasta", shell=True, check=True)

# candidates: selected clusters not already built into a training file
import gzip
seqs = {}
with gzip.open(f"{P}/pdb_seqres.txt.gz", "rt") as fh:
    hdr = None
    for l in fh:
        if l.startswith(">"): hdr = l[1:].split()[0]; continue
        seqs[hdr] = l.strip()
cand = {}
for tsv in (f"{P}/selected_chains.tsv", f"{P}/selected_chains_long.tsv"):
    if not os.path.exists(tsv): continue
    for i, l in enumerate(open(tsv)):
        if i == 0: continue
        pid, ch, ln, res, meth, cs = l.rstrip("\n").split("\t")
        nm = f"{pid}_{ch}"
        if nm in train_names: continue
        s = seqs.get(nm)
        if s is None or not (a.min_len <= len(s) <= a.max_len): continue
        cand[nm] = (pid, ch, int(ln), float(res), meth, s)
print(f"{len(cand)} candidate chains not already in a training file", flush=True)
with open(f"{W}/cand.fasta", "w") as f:
    for nm, v in cand.items(): f.write(f">{nm}\n{v[5]}\n")

print("searching candidates against training + benchmark sequences...", flush=True)
subprocess.run([MM, "easy-search", f"{W}/cand.fasta", f"{W}/exclude.fasta", f"{W}/hits.m8", f"{W}/tmp",
                "-s", "6.0", "-e", "1e-3", "--max-seqs", "2000", "--threads", str(a.threads),
                "--format-output", "query,target,fident,qcov,tcov", "-v", "1"], check=True)
bad = set()
for l in open(f"{W}/hits.m8"):
    q, t, fid, qc, tc = l.split()
    if float(fid) >= a.min_id and (float(qc) >= 0.5 or float(tc) >= 0.5): bad.add(q)
keep = [nm for nm in cand if nm not in bad]
print(f"{len(bad)} candidates within {a.min_id:.0%} identity of training or benchmark data; {len(keep)} clean", flush=True)

# stratify: half short (<=256), half long (257-512), best resolution first inside each stratum
short = sorted([n for n in keep if cand[n][2] <= 256], key=lambda n: cand[n][3])
long_ = sorted([n for n in keep if cand[n][2] > 256], key=lambda n: cand[n][3])
half = a.n // 2
sel = short[:half] + long_[:a.n - half]
random.Random(0).shuffle(sel)
with open(a.out, "w") as f:
    f.write("pdb_id\tchain\tlength\tresolution\tmethod\tcluster_size\n")
    for nm in sel:
        pid, ch, ln, res, meth, s = cand[nm]
        f.write(f"{pid}\t{ch}\t{ln}\t{res}\t{meth}\t1\n")
L = np.array([cand[n][2] for n in sel]); R = np.array([cand[n][3] for n in sel])
print(f"wrote {len(sel)} chains to {a.out}: {int((L<=256).sum())} <=256, {int((L>256).sum())} 257-512, "
      f"length median {np.median(L):.0f}, resolution median {np.median(R):.2f}", flush=True)
