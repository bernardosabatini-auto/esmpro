#!/bin/bash
#SBATCH --job-name=cd_eval
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u gate24_codesign_eval.py --ckpt ${CKPT:-best_cd_174M_p64x6_esmc.pt} --n ${N:-100} 2>&1 | grep -v -i -E "warn|INFO|return data|important use|Loading weights"
echo ALLDONE
