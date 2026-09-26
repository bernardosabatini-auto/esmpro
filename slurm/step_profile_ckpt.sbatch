#!/bin/bash
#SBATCH --job-name=prof_ckpt
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO|AUTOTUNE'
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "per step|tokens|peak|OutOfMemory|Error"; }
export REPEAT_COPIES=4
run "R=4 budget 12, pair checkpointing ON (recipe)" env STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
run "R=4 budget 12, pair checkpointing OFF" env PAIR_CKPT=0 STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
run "R=4 budget 18, pair checkpointing OFF" env PAIR_CKPT=0 STEP_BUDGET=18 python -u step_profile.py --fused --n-steps 12
run "R=4 budget 24, pair checkpointing OFF" env PAIR_CKPT=0 STEP_BUDGET=24 python -u step_profile.py --fused --n-steps 8
run "R=8 budget 6, pair checkpointing ON" env REPEAT_COPIES=8 STEP_BUDGET=6 python -u step_profile.py --fused --n-steps 16
run "R=8 budget 6, pair checkpointing OFF" env REPEAT_COPIES=8 PAIR_CKPT=0 STEP_BUDGET=6 python -u step_profile.py --fused --n-steps 16
echo PROFILE_DONE
