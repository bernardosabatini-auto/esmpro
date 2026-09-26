#!/bin/bash
#SBATCH --job-name=infer_cost
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; export ESM_PROAE_MAX_LEN=512
python -u gate21_inference_cost.py --ckpt ${CKPT:-last_pf_459M_p128x8_long512_scratch.ckpt} 2>&1 | grep -v -i -E "warn|INFO|Loading weights|return data"
echo ALLDONE
