#!/bin/bash
#SBATCH -J cm13
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-02:00
#SBATCH -o logs/cm13_%j.log
#SBATCH -e logs/cm13_%j.log
# AlphaFold structural features (code/cm13_structure.py) and the residual test (code/cm14_struct_test.py).
# The work is single-core CPU -- freesasa and cKDTree over 20k models, ~11 min -- but the CPU account has
# MaxSubmit 0, so it takes the smallest GPU partition and leaves the card idle. Nothing here uses CUDA.
set -eu
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
source ~/.bashrc; conda activate dendritic_env
python -u code/cm13_structure.py
python -u code/cm14_struct_test.py
