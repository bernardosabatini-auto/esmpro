#!/bin/bash
#SBATCH -J cm01emb
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-01:00
#SBATCH -o logs/cm01_%j.log
#SBATCH -e logs/cm01_%j.log
# Campaign: ESMC-6B SAE + 9 pooled layers for proteins lacking them. One RTX card (25 GB model).
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm01_embed.py "$@"
RC=$?
echo "CM01_EXIT=$RC"
exit $RC
