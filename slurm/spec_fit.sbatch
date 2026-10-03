#!/bin/bash
#SBATCH -J ga13spec
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -o logs/ga13_%j.log
#SBATCH -e logs/ga13_%j.log
# Co-chaperone specificity fits (ga13): kernel ridge on the GPU.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/ga13_spec_fit.py "$@"
RC=$?
echo "GA13_EXIT=$RC"
exit $RC
