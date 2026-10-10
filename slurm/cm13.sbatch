#!/bin/bash
#SBATCH -J cm13
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-03:00
#SBATCH -o logs/cm13_%j.log
#SBATCH -e logs/cm13_%j.log
# AlphaFold structural features and the tests built on them:
#   cm13  stream the AlphaFold human proteome tar, per-protein structural features + per-residue SASA/pLDDT
#   cm15  pool the ESMC residue states over the structural masks
#   cm14  does any of it explain the residual of the sequence model?
# All three are single-core CPU work (freesasa, cKDTree, mmap reads over res_L80.npy). The CPU account has
# MaxSubmit 0, so this takes the smallest GPU partition and leaves the card idle. Nothing here uses CUDA.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
$PY -u code/cm13_structure.py && $PY -u code/cm15_maskpool.py && $PY -u code/cm14_struct_test.py
echo "CM13_EXIT=$?"
