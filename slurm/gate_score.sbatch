#!/bin/bash
#SBATCH --job-name=gate_score
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Usage: sbatch slurm/gate_score.sbatch <ckpt_name> [n] [bs]
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
CKPT=${1:-smallscale_final_40k.pt}; N=${2:-100}; BS=${3:-10}
echo "scoring $CKPT n=$N bs=$BS on $(hostname)"
time python gate6_score_checkpoint.py --ckpt $CKPT --n $N --bs $BS --offset ${4:-0}
echo GATE_DONE
