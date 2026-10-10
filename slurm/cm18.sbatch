#!/bin/bash
#SBATCH -J cm18
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0-03:00
#SBATCH -o logs/cm18_%j.log
#SBATCH -e logs/cm18_%j.log
# Tier 0 of the complex plan: native-partner occlusion from experimental structures.
#   cm18  SASA of every measured chain, alone and in its biological assembly (freesasa, multiprocessing)
#   cm19  does the occlusion explain the sequence model's residual?
# Pure CPU work -- there is no GPU kernel here. The card is held only because the CPU account has
# MaxSubmit 0; 16 cores is one eighth of a 128-core node, matching the 8 GPUs it has.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
$PY -u code/cm18_interface.py && $PY -u code/cm19_iface_test.py
echo "CM18_EXIT=$?"
