#!/bin/bash
#SBATCH --job-name=steering
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
export ESM_PROAE_MAX_LEN=${MAXLEN:-256}
python -u gate30_steering.py --ckpt ${CKPT:?} ${ARGS:-} 2>&1 | grep -v -i -E "warn|INFO|return data|Loading weights|softmax|Consider using"
echo ALLDONE
