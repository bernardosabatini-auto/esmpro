#!/bin/bash
#SBATCH -J ga09
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00
#SBATCH -o logs/ga09_%j.log
#SBATCH -e logs/ga09_%j.log
# GA_33 ensemble-MLP readout + per-sample likelihoods. proteinae env: its torch has sm_120 kernels.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/ga09_mlp.py "$@"
RC=$?
echo "GA09_EXIT=$RC"
exit $RC
