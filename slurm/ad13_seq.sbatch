#!/bin/bash
#SBATCH -J ad13seq
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -o logs/ad13_%j.log
#SBATCH -e logs/ad13_%j.log
# Sequence vs input-lysate quantities (ad13): ridge per ESMC layer + stacking, GPU.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/ad13_enrichment_seq.py "$@"
RC=$?
echo "AD13_EXIT=$RC"
exit $RC
