#!/bin/bash
#SBATCH -J pd09
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -o logs/pd09_%j.log
#SBATCH -e logs/pd09_%j.log
# 48 readout configurations x 5 outer folds x 2 doses, each a 16-member ensemble trained in one
# vectorised pass; then three per-sample likelihoods on the top three. One RTX card is plenty.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/pd09_mlp.py "$@"
RC=$?
echo "PD09_EXIT=$RC"
exit $RC
