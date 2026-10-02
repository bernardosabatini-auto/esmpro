#!/bin/bash
#SBATCH --job-name=pd_embed
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# One RTX card: ESMC-6B is 24.8 GB in bf16, so an H200 would be wasted fairshare.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u pd03_embed.py --budget ${BUDGET:-20000} ${EXTRA:-} 2>&1 \
  | grep -v -i -E "^\[|warn|INFO|Loading weights|return data"
echo PD_EMBED_DONE
