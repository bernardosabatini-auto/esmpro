#!/bin/bash
#SBATCH --job-name=leakage_audit
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u gate9_leakage_audit.py --threads 14 --work /tmp/audit_$SLURM_JOB_ID 2>&1 | grep -v -i -E "warn|INFO"
rm -rf /tmp/audit_$SLURM_JOB_ID
echo AUDIT_DONE
