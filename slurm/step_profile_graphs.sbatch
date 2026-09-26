#!/bin/bash
#SBATCH --job-name=prof_graphs
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO'
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "per step|tokens|peak|OutOfMemory|Error|recompil"; }
export REPEAT_COPIES=4
run "R=4 budget 12, compile default, pad 8 (current recipe)" env STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
run "R=4 budget 12, pad 32, compile default" env PAD_MULT=32 STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
run "R=4 budget 12, pad 32, reduce-overhead (CUDA graphs)" env PAD_MULT=32 DIT_COMPILE_MODE=reduce-overhead STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 24
run "R=4 budget 12, pad 32, max-autotune-no-cudagraphs" env PAD_MULT=32 DIT_COMPILE_MODE=max-autotune-no-cudagraphs STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
run "R=4 budget 12, pad 8, DIT_COMPILE=0 (eager trunk, for reference)" env DIT_COMPILE=0 STEP_BUDGET=12 python -u step_profile.py --fused --n-steps 16
echo PROFILE_DONE
