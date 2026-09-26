#!/bin/bash
#SBATCH --job-name=ens_eval
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# knobs: CKPT, K, W
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; export ESM_PROAE_MAX_LEN=512; F='warn|INFO|return data'
for s in apo codnas; do echo "== $s"; python -u gate23_ensemble_eval.py --ckpt ${CKPT:-last_pf_459M_p128x8_long512_scratch.ckpt} --set $s --k ${K:-5} --cfg-w ${W:-1,2} 2>&1 | grep -v -i -E "$F"; done
echo "== cameo22 (folding benchmark)"; python gate7_score.py --ckpt ${CKPT:-last_pf_459M_p128x8_long512_scratch.ckpt} --h5 $ESM_PROAE_ROOT/data/phase1_dataset/dataset_cameo22_esmc.h5 --names-file $ESM_PROAE_ROOT/notes/cameo22_names.txt --n 300 --k 4 --cfg-w 2 --steps 50 --bs 8 2>&1 | grep -v -i -E "$F"
echo ALLDONE
