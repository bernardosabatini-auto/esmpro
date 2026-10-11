#!/bin/bash
#SBATCH -J cm26
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=120G
#SBATCH -t 0-04:00
#SBATCH -o logs/cm26_%j.log
#SBATCH -e logs/cm26_%j.log
# cm26: is the complex effect of 3c a capture nonlinearity, co-abundance, or co-capture?
# Dense numpy over an 11.5k x 11.5k co-membership matrix plus cluster bootstraps. Single-threaded CPU
# work with no CUDA in it; the CPU account bsabatini_lab has MaxSubmit 0 (verified), so this takes the
# smallest GPU partition and leaves the card idle. kempner_rtx caps cores at 15 per GPU.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
$PY -u code/cm26_nonlinearity.py
echo "CM26_EXIT=$?"
