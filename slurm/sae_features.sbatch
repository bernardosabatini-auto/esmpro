#!/bin/bash
#SBATCH -J pd12sae
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-03:00
#SBATCH -o logs/pd12_%j.log
#SBATCH -e logs/pd12_%j.log
# ESMC-6B layer-60 SAE features for every protein in both experiments. One RTX card.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/pd12_sae_features.py "$@"
RC=$?
echo "PD12_EXIT=$RC"
exit $RC
