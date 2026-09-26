#!/bin/bash
# Sequence guards for the 627k extra short AFDB set, then queue the 840M main line behind the running cycle2 job.
set -e
R=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae; D=$R/data/phase1_dataset; source $R/slurm/env.sh; cd $R
python - <<'PY'
import h5py, os
D=os.environ["ESM_PROAE_ROOT"]+"/data/phase1_dataset"
with open(f"{D}/short2_seqs.fasta","w") as f:
    for k in range(4):
        h=h5py.File(f"{D}/dataset_afdb_short2_{k}.h5","r")["train"]
        for n in h: f.write(f">{n}\n{h[n].attrs['sequence']}\n")
        print("fasta shard",k,flush=True)
PY
W=/tmp/claude-59393/g2; rm -rf $W; mkdir -p $W
cat $D/all_val_seqs.fasta $D/casp512_seqs.fasta > $W/targets.fasta
mmseqs easy-search $D/short2_seqs.fasta $W/targets.fasta $W/res.m8 $W/tmp -s 7.5 -e 1e-3 --max-seqs 300 --threads 8 --format-output "query,target,fident,qcov,tcov" -v 0
python - <<PY
import collections, h5py, os
D=os.environ["ESM_PROAE_ROOT"]+"/data/phase1_dataset"
bad=set()
for l in open("$W/res.m8"):
    q,t,fid,qc,tc=l.split(); fid=float(fid); cov=max(float(qc),float(tc))
    if cov>=0.5 and ((t.startswith("casp") and fid>=0.9) or (not t.startswith("casp") and fid>0.5)): bad.add(q)
print("short2 proteins to remove (val >50% id or CASP >=90% id):", len(bad))
rm=0
for k in range(4):
    for fn in (f"dataset_afdb_short2_{k}.h5", f"dataset_afdb_short2_{k}_esmc.h5"):
        h=h5py.File(f"{D}/{fn}","a"); g=h["train"]
        for n in list(g.keys()):
            if n in bad: del g[n]; rm+=1
        h.attrs["guards"]="val homologs (>50% id) and CASP<=512 near-identical (>=90%) removed 2026-09-26"; h.close()
print("removed groups (both files):", rm)
tot=sum(len(h5py.File(f"{D}/dataset_afdb_short2_{k}.h5","r")["train"]) for k in range(4)); print("short2 total:", tot)
PY
rm -rf $W
echo GUARD_DONE
# queue the 840M main line behind cycle2 (job 48736293)
S2="$D/dataset_afdb_short2_0_esmc.h5,$D/dataset_afdb_short2_1_esmc.h5,$D/dataset_afdb_short2_2_esmc.h5,$D/dataset_afdb_short2_3_esmc.h5"
J=$(LABEL=pf_840M_p128x8_long512_1p3M DM=1280 NL=28 NH=20 R=8 BS=12 EPOCHS=30 LR=3e-4 WARMUP=1500 PATIENCE=8 TLATE=1 EXTRA_H5="$S2" SBATCH_EXTRA="--parsable --dependency=afterany:48736293" bash slurm/launch_long_finetune.sh)
echo "840M main line queued: $J"
CKPT=last_pf_840M_p128x8_long512_1p3M.ckpt N=1000 K=4 sbatch --parsable --dependency=afterok:$J --export=ALL slurm/score_long.sbatch
CKPT=last_pf_840M_p128x8_long512_1p3M.ckpt sbatch --parsable --dependency=afterok:$J --export=ALL slurm/score_full.sbatch
echo QUEUED
