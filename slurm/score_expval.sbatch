#!/bin/bash
#SBATCH --job-name=score_expval
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Score a checkpoint on the EXPERIMENTAL held-out set (626 real structures, 30 % identity
# separated from all training files and from every benchmark). Real references only.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; N_=$ESM_PROAE_ROOT/notes; F='warn|INFO|return data|Loading weights'
export ESM_PROAE_MAX_LEN=512; C=${CKPT:?}
echo "== $C exp_val (all 626 real structures)"
python -u gate7_score.py --ckpt $C --h5 $D/dataset_exp_val_esmc.h5 --names-file $N_/exp_val_names.txt --n 700 --k ${K:-4} --cfg-w ${W:-2} --steps 50 --bs 10 2>&1 | grep -v -i -E "$F"
echo "== $C exp_val long only (167 of 257-512)"
python -u gate7_score.py --ckpt $C --h5 $D/dataset_exp_val_esmc.h5 --names-file $N_/exp_val_long_names.txt --n 200 --k ${K:-4} --cfg-w ${W:-2} --steps 50 --bs 10 2>&1 | grep -v -i -E "$F"
echo ALLDONE
