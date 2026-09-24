#!/bin/bash
#SBATCH --job-name=casp_leakage
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# CPU-only: nearest TRAINING structure (Foldseek TM-align) for every CASP benchmark domain.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
W=${TMPDIR:-/tmp}/casp_audit_$SLURM_JOB_ID
python -u gate9_leakage_audit.py --threads 14 --work $W --query-h5 $ESM_PROAE_ROOT/data/phase1_dataset/dataset_casp.h5 --tag gate13_casp_leakage 2>&1 | grep -v -i -E "warn|INFO"
rm -rf $W
echo AUDIT_DONE
