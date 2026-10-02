"""pd04: cluster the proteins by sequence identity so cross-validation folds cannot leak.

This is one proteome, dense with paralogs and isoform families whose ESMC embeddings are
nearly identical. Splitting proteins at random would put a protein's close relative in the
training set and report a score that is partly memorisation. Folds are therefore grouped by
MMseqs2 cluster.

  python pd04_cluster.py --min-id 0.3
"""
import os, sys, csv, subprocess, tempfile, shutil, argparse, collections
import numpy as np

ROOT = os.environ["ESM_PROAE_ROOT"]
MM = os.environ.get("MMSEQS_BIN", "mmseqs")
ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default=f"{ROOT}/data/pd_data/sequences.fasta")
ap.add_argument("--out", default=f"{ROOT}/data/pd_data/clusters.tsv")
ap.add_argument("--min-id", type=float, default=0.3)
ap.add_argument("--cov", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()

W = tempfile.mkdtemp(prefix="pd04_")
# strip the description from the headers so cluster ids are bare accessions
clean = f"{W}/in.fasta"
with open(clean, "w") as fh:
    for line in open(a.fasta):
        fh.write(">" + line[1:].split("|")[0].strip() + "\n" if line.startswith(">") else line)

print(f"clustering at {a.min_id:.0%} identity, {a.cov:.0%} coverage", flush=True)
subprocess.run([MM, "easy-cluster", clean, f"{W}/clu", f"{W}/tmp",
                "--min-seq-id", str(a.min_id), "-c", str(a.cov), "--cov-mode", "0",
                "--threads", str(a.threads), "-v", "1"], check=True, capture_output=True)

member_of = {}
for line in open(f"{W}/clu_cluster.tsv"):
    rep, mem = line.split()
    member_of[mem] = rep
shutil.rmtree(W, ignore_errors=True)

sizes = collections.Counter(member_of.values())
n = len(member_of)
print(f"  {n} sequences -> {len(sizes)} clusters", flush=True)
print(f"  singletons {sum(1 for v in sizes.values() if v == 1)}, "
      f"largest {sizes.most_common(1)[0][1]}, median size {int(np.median(list(sizes.values())))}", flush=True)
big = sizes.most_common(6)
print(f"  biggest clusters: {big}", flush=True)
print(f"  fraction of proteins in a cluster of >1: "
      f"{sum(v for v in sizes.values() if v > 1)/n:.3f}  "
      f"<- this is the share that random splits would leak", flush=True)

with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["accession", "cluster"])
    for mem, rep in sorted(member_of.items()):
        w.writerow([mem, rep])
print(f"wrote {a.out}", flush=True)
print("PD04_DONE")
