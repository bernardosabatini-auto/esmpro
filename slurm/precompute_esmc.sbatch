#!/bin/bash
#SBATCH --job-name=precompute_esmc
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
python -u $ESM_PROAE_ROOT/code/gate11_precompute_esmc.py --splits val,train --bs 64 2>&1 | grep -v -i -E "warn|Loading weights|UNEXPECTED|^Key|^---|^Notes|can be ignored"
echo PRECOMPUTE_DONE
