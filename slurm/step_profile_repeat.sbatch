#!/bin/bash
#SBATCH --job-name=prof_repeat
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO'
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "per step|tokens|peak|OutOfMemory|Error"; }
run "R=1 budget 48 (reference)" env STEP_BUDGET=48 python -u step_profile.py --fused
run "R=2 budget 24 (same samples/step)" env REPEAT_COPIES=2 STEP_BUDGET=24 python -u step_profile.py --fused
run "R=4 budget 12 (same samples/step)" env REPEAT_COPIES=4 STEP_BUDGET=12 python -u step_profile.py --fused
run "R=2 budget 48 (same proteins/step, 2x samples)" env REPEAT_COPIES=2 STEP_BUDGET=48 python -u step_profile.py --fused --n-steps 6
echo PROFILE_DONE
