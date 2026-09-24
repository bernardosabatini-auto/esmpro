#!/bin/bash
#SBATCH --job-name=sample_sel
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# knobs: CKPT, K, STEPS
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; N=$ESM_PROAE_ROOT/notes
C=${CKPT:-best_pf_459M_esmc_afdb.pt}; F='warn|INFO|return data'
echo "== held-out offset 1000"; python -u gate14_sample_selector.py --ckpt $C --offset 1000 --n 100 --k ${K:-8} --steps ${STEPS:-50} 2>&1 | grep -v -i -E "$F"
echo "== casp"; python -u gate14_sample_selector.py --ckpt $C --h5 $D/dataset_casp_esmc.h5 --names-file $N/casp_domains_le256.txt --k ${K:-8} --steps ${STEPS:-50} 2>&1 | grep -v -i -E "$F"
echo "== no-neighbour lt0.6"; python -u gate14_sample_selector.py --ckpt $C --names-file $N/val_noneighbour_lt0.6.txt --k ${K:-8} --steps ${STEPS:-50} 2>&1 | grep -v -i -E "$F"
echo ALLDONE
