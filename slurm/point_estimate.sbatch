#!/bin/bash
#SBATCH --job-name=point_est
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset
export ESM_PROAE_MAX_LEN=${MAXLEN:-256}
python -u gate26_point_estimate.py --ckpt ${CKPT:?} ${ARGS:-} 2>&1 | grep -v -i -E "warn|INFO|return data|Loading weights"
echo ALLDONE
