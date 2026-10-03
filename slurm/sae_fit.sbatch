#!/bin/bash
#SBATCH -J pd13fit
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -o logs/pd13_%j.log
#SBATCH -e logs/pd13_%j.log
# SAE-feature ridge fits: kernel products and eigendecompositions on one RTX card.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/pd13_sae_fit.py "$@"
RC=$?
echo "PD13_EXIT=$RC"
exit $RC
