#!/bin/bash
#SBATCH -J ga10
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00
#SBATCH -o logs/ga10_%j.log
#SBATCH -e logs/ga10_%j.log
# GA_33 shared-trunk and frozen-trunk transfer experiment (ga10). proteinae env: sm_120 kernels.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/ga10_shared.py "$@"
RC=$?
echo "GA10_EXIT=$RC"
exit $RC
