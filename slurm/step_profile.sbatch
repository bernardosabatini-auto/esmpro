#!/bin/bash
#SBATCH --job-name=step_profile
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO'
B=${BUDGET:-48}
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "per step|tokens|peak|OutOfMemory|Error"; }
run "unfused (current H200 code path)" env STEP_BUDGET=$B python -u step_profile.py
run "FUSED triangle projections" env STEP_BUDGET=$B python -u step_profile.py --fused
run "FUSED, no pair checkpointing (memory for recompute)" env STEP_BUDGET=$B PAIR_CKPT=0 python -u step_profile.py --fused --n-steps 6
run "FUSED + recycling" env STEP_BUDGET=$B python -u step_profile.py --fused --recycle
run "FUSED, budget 64" env STEP_BUDGET=64 python -u step_profile.py --fused --n-steps 6
echo PROFILE_DONE
