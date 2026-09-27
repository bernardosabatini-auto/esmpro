#!/bin/bash
#SBATCH --job-name=prof_nopair
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Cost of the pair track, same GPU, same data, same repeated batching.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO|Loading weights'
D=$ESM_PROAE_ROOT/data/phase1_dataset
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "per step|tokens|peak|params|OutOfMemory|Error"; }
export ESM_PROAE_MAX_LEN=512 REPEAT_COPIES=${R:-4} TOKEN_CAP=${TOKEN_CAP:-24000}
run "pair 128x8 (fused), R=$REPEAT_COPIES" env STEP_BUDGET=${BUDGET:-16} python -u step_profile.py --fused --h5 $D/dataset_afdb_long_0_esmc.h5 --n-steps 8
run "pair-free latent DiT,  R=$REPEAT_COPIES" env STEP_BUDGET=${BUDGET:-16} python -u step_profile.py --no-pair --h5 $D/dataset_afdb_long_0_esmc.h5 --n-steps 8
echo PROFILE_DONE
