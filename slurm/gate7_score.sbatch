#!/bin/bash
#SBATCH --job-name=gate7_score
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Usage: sbatch slurm/gate7_score.sbatch <ckpt> [n] [k] [cfg_w list]
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python gate7_score.py --ckpt ${1:-best_lf_base.pt} --n ${2:-100} --k ${3:-8} --cfg-w ${4:-1,1.5,2,3,4} --offset ${5:-0} --steps ${6:-50} ${7:+--names-file $7} 2>&1 | grep -v -i -E "warn|INFO|return data"
echo SCORE_DONE
