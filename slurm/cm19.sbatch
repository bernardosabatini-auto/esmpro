#!/bin/bash
#SBATCH -J cm19
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-02:00
#SBATCH -o logs/cm19_%j.log
#SBATCH -e logs/cm19_%j.log
# Re-aggregate the per-chain interface records (cheap: reads iface_entry.tsv, no structures touched) and run
# the test. CM18_AGG_ONLY exists so that fixing the aggregation never costs the SASA pass again.
# Single-core CPU work; the card is held only because the CPU account has MaxSubmit 0.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
CM18_AGG_ONLY=1 $PY -u code/cm18_interface.py && $PY -u code/cm19_iface_test.py
echo "CM19_EXIT=$?"
