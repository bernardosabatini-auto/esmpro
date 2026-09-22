#!/bin/bash
#SBATCH --job-name=loss_guard
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python gate6_loss_guard.py --gpu 2>&1 | grep -v -i warn
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then echo "GUARD_FAILED rc=$rc"; exit 1; fi
echo GUARD_PASSED
cd $ESM_PROAE_ROOT
S="--fix-end-frames"
LABEL=h200_ca_bond   EXTRA="$S --loss ca --bond-weight 1.0"                    sbatch --export=ALL slurm/train_h200_full.sbatch
LABEL=h200_bb_bond   EXTRA="$S --loss bb --bb-points all --bond-weight 1.0"    sbatch --export=ALL slurm/train_h200_full.sbatch
LABEL=h200_bb        EXTRA="$S --loss bb --bb-points all"                      sbatch --export=ALL slurm/train_h200_full.sbatch
LABEL=h200_cabb_bond EXTRA="$S --loss ca+bb --bb-points all --bond-weight 1.0" sbatch --export=ALL slurm/train_h200_full.sbatch
echo ARMS_SUBMITTED
