#!/bin/bash
#SBATCH --job-name=embed_layer
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Re-embed a dataset reading an EARLIER conditioner layer (not layer mixing: one layer).
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset
python -u gate11_precompute_esmc.py --layer-frac ${FRAC:?} --out $D/${OUT:?} --sort --bs ${BS:-48} 2>&1 | grep -v -i -E "warn|INFO|return data"
echo ALLDONE
