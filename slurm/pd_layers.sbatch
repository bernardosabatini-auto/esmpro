#!/bin/bash
#SBATCH -J pd03b
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-02:00
#SBATCH -o logs/pd03b_%j.log
#SBATCH -e logs/pd03b_%j.log
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/pd03b_layers.py "$@"
RC=$?
echo "PD03B_EXIT=$RC"
exit $RC
