#!/bin/bash
#SBATCH --job-name=prof_big
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Size a larger trunk / deeper pair track at the 512 window on real long batches (on an H200 directly).
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; F='warn|Inductor|dynamo|return data|INFO|AUTOTUNE'; H=$ESM_PROAE_ROOT/data/phase1_dataset/dataset_afdb_long_0_esmc.h5
export ESM_PROAE_MAX_LEN=512 REPEAT_COPIES=8
run() { echo "== $1"; shift; "$@" 2>&1 | grep -v -i -E "$F" | grep -E "params|per step|tokens|peak|OutOfMemory|Error"; }
run "459M (d1024x24, pair 128x8), R=8 budget 10, 512 window" env STEP_BUDGET=16 python -u step_profile.py --fused --h5 $H --n-steps 8
run "~750M trunk (d1280x28, 20 heads), pair 128x8, R=8 budget 8" env STEP_BUDGET=12 python -u step_profile.py --fused --h5 $H --n-steps 8 --d-model 1280 --n-layers 28 --n-heads 20
run "459M trunk, deeper pair 128x12, R=8 budget 8" env STEP_BUDGET=12 python -u step_profile.py --fused --h5 $H --n-steps 8 --n-pair-blocks 12
run "459M trunk, wider pair 192x8, R=8 budget 8" env STEP_BUDGET=12 python -u step_profile.py --fused --h5 $H --n-steps 8 --d-pair 192
echo PROFILE_DONE
