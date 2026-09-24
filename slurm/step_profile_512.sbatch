#!/bin/bash
#SBATCH --job-name=prof_512
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Memory and step time of the 459M pair flow on REAL 257-512 residue batches, to size the H200 budget for the long stage.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO'; H=$ESM_PROAE_ROOT/data/phase1_dataset/dataset_afdb_long_0_esmc.h5
export ESM_PROAE_MAX_LEN=512
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "window|per step|tokens|peak|OutOfMemory|Error"; }
run "L<=512, budget 24 (=6 proteins at 512), R=1" env STEP_BUDGET=24 python -u step_profile.py --fused --h5 $H --n-steps 8
run "L<=512, budget 36, R=1" env STEP_BUDGET=36 python -u step_profile.py --fused --h5 $H --n-steps 8
run "L<=512, budget 48, R=1" env STEP_BUDGET=48 python -u step_profile.py --fused --h5 $H --n-steps 6
run "L<=512, budget 12, R=4" env REPEAT_COPIES=4 STEP_BUDGET=12 python -u step_profile.py --fused --h5 $H --n-steps 8
run "L<=512, budget 24, R=2 + recycling" env REPEAT_COPIES=2 STEP_BUDGET=24 python -u step_profile.py --fused --recycle --h5 $H --n-steps 6
echo PROFILE_DONE
