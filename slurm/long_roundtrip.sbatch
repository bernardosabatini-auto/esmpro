#!/bin/bash
#SBATCH --job-name=long_rt
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u gate17_long_roundtrip.py --n 250 --min-len 257 --max-len 640 2>&1 | grep -v -i -E "warn|INFO|return data"
echo LONG_DONE
