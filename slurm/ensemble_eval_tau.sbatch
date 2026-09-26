#!/bin/bash
#SBATCH --job-name=ens_tau
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; export ESM_PROAE_MAX_LEN=512; F='warn|INFO|return data'
for tau in 0.3 0.6 1.0; do for s in apo codnas; do echo "== $s tau=$tau"; python -u gate23_ensemble_eval.py --ckpt ${CKPT:-last_pf_459M_p128x8_long512_scratch.ckpt} --set $s --k 5 --cfg-w 1 --tau $tau 2>&1 | grep -v -i -E "$F" | grep -E "^w="; done; done
echo ALLDONE
