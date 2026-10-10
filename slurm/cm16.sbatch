#!/bin/bash
#SBATCH -J cm16
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-03:00
#SBATCH -o logs/cm16_%j.log
#SBATCH -e logs/cm16_%j.log
# cm16: does the complex-level residual sharing of cm12 extend to the wider STRING physical network?
# Single-core CPU work (sparse matvecs over a 1 M-edge graph). The CPU account has
# MaxSubmit 0, so this takes the smallest GPU partition and leaves the card idle. Nothing here uses CUDA.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
$PY -u code/cm16_network.py
echo "CM16_EXIT=$?"
