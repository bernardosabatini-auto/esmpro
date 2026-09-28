#!/bin/bash
#SBATCH --job-name=latent_explore
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
export ESM_PROAE_MAX_LEN=${MAXLEN:-512}
python -u gate28_latent_explore.py --ckpt ${CKPT:?} ${ARGS:-} 2>&1 | grep -v -i -E "warn|INFO|return data|Loading weights|softmax"
echo ALLDONE
