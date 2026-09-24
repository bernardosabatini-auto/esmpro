"""Gate 15a (CPU): choose experimental PDB chains to add to the training targets.

Filters pdb_seqres protein chains to SEQRES length [32, 256], X-ray or EM
entries with resolution <= --max-res, drops chains with non-standard letters,
excludes CASP15/16 target entries and anything > 30 % identical to a CASP
benchmark domain (keeps casp_benchmark.md clean), clusters the rest with
MMseqs2 at --min-id identity and keeps the best-resolution chain per cluster.
Writes data/pdb/selected_chains.tsv (pdb_id, chain, length, resolution, method, cluster size).
"""
import os, sys, gzip, subprocess, collections, argparse, tempfile
ROOT = os.environ["ESM_PROAE_ROOT"]; D = f"{ROOT}/data/pdb"
ap = argparse.ArgumentParser(); ap.add_argument("--max-res", type=float, default=3.0); ap.add_argument("--min-id", type=float, default=0.5)
ap.add_argument("--min-len", type=int, default=32); ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
method = {}
for l in open(f"{D}/pdb_entry_type.txt"):
    f = l.split()
    if len(f) == 3: method[f[0].lower()] = (f[1], f[2])
res = {}
for l in open(f"{D}/resolu.idx"):
    f = [x.strip() for x in l.split(";")]
    if len(f) == 2 and len(f[0]) == 4:
        try: res[f[0].lower()] = float(f[1])
        except ValueError: pass
# CASP target PDB ids are not recorded in the domain files; the sequence filter below removes their homologs.
AA = set("ACDEFGHIKLMNPQRSTVWY")
cands = {}; stats = collections.Counter()
with gzip.open(f"{D}/pdb_seqres.txt.gz", "rt") as fh:
    hdr = None
    for l in fh:
        if l.startswith(">"): hdr = l; continue
        seq = l.strip(); f = hdr[1:].split(); pid, ch = f[0].split("_", 1); pid = pid.lower()
        if f[1] != "mol:protein": stats["not protein"] += 1; continue
        n = len(seq)
        if not (a.min_len <= n <= a.max_len): stats["length"] += 1; continue
        if set(seq) - AA: stats["nonstandard"] += 1; continue
        m = method.get(pid)
        if m is None or m[0] not in ("prot", "prot-nuc") or m[1] not in ("diffraction", "EM"): stats["method"] += 1; continue
        r = res.get(pid, -1.0)
        if not (0 < r <= a.max_res): stats["resolution"] += 1; continue
        if seq in cands and cands[seq][2] <= r: stats["dup seq"] += 1; continue
        cands[seq] = (pid, ch, r, m[1])
print(f"{len(cands)} unique candidate sequences; dropped {dict(stats)}", flush=True)
W = tempfile.mkdtemp(prefix="pdbsel_")
with open(f"{W}/cand.fasta", "w") as f:
    for seq, (pid, ch, r, m) in cands.items(): f.write(f">{pid}_{ch}\n{seq}\n")
# remove CASP homologs
subprocess.run(["mmseqs", "easy-search", f"{ROOT}/data/phase1_dataset/casp_seqs.fasta", f"{W}/cand.fasta", f"{W}/casp.m8", f"{W}/tmp",
                "-s", "7.5", "-e", "1e-3", "--max-seqs", "5000", "--threads", str(a.threads), "--format-output", "query,target,fident,qcov,tcov", "-v", "0"], check=True)
bad = set()
for l in open(f"{W}/casp.m8"):
    q, t, fid, qc, tc = l.split(); 
    if float(fid) >= 0.3 and (float(qc) >= 0.5 or float(tc) >= 0.5): bad.add(t)
print(f"{len(bad)} chains within 30 % identity of a CASP benchmark domain removed", flush=True)
with open(f"{W}/clean.fasta", "w") as f:
    for seq, (pid, ch, r, m) in cands.items():
        if f"{pid}_{ch}" not in bad: f.write(f">{pid}_{ch}\n{seq}\n")
subprocess.run(["mmseqs", "easy-cluster", f"{W}/clean.fasta", f"{W}/clu", f"{W}/tmp", "--min-seq-id", str(a.min_id), "-c", "0.8", "--cov-mode", "0",
                "--threads", str(a.threads), "-v", "0"], check=True)
byname = {f"{pid}_{ch}": (pid, ch, r, m, len(seq)) for seq, (pid, ch, r, m) in cands.items()}
clusters = collections.defaultdict(list)
for l in open(f"{W}/clu_cluster.tsv"):
    rep, mem = l.split(); clusters[rep].append(mem)
rows = []
for rep, mems in clusters.items():
    best = min(mems, key=lambda n: byname[n][2]); pid, ch, r, m, n = byname[best]
    rows.append((pid, ch, n, r, m, len(mems)))
rows.sort()
with open(f"{D}/selected_chains.tsv", "w") as f:
    f.write("pdb_id\tchain\tlength\tresolution\tmethod\tcluster_size\n")
    for r_ in rows: f.write("\t".join(map(str, r_)) + "\n")
import numpy as np
L = np.array([r_[2] for r_ in rows]); R = np.array([r_[3] for r_ in rows])
print(f"{len(rows)} clusters at {a.min_id:.0%} identity -> selected_chains.tsv; length median {np.median(L):.0f}, resolution median {np.median(R):.2f}, "
      f"{sum(r_[4]=='EM' for r_ in rows)} EM; {len(set(r_[0] for r_ in rows))} distinct entries")
