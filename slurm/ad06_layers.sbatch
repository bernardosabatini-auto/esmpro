#!/bin/bash
#SBATCH -J ad06layers
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#SBATCH -o logs/ad06_%j.log
#SBATCH -e logs/ad06_%j.log
# Combined-layer models (ad06): ridge per layer, stacking, concatenation on the GPU.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/ad06_layers.py "$@"
RC=$?
echo "AD06_EXIT=$RC"
exit $RC
