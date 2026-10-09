#!/bin/bash
#SBATCH -J cm01b
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-03:00
#SBATCH -o logs/cm01b_%j.log
#SBATCH -e logs/cm01b_%j.log
# Campaign: per-residue ESMC-6B layer 80 + layer-60 SAE codes for attention pooling. One RTX card.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm01b_residues.py "$@"
RC=$?
echo "CM01B_EXIT=$RC"
exit $RC
