#!/bin/bash
#SBATCH --job-name=exp_val
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# An EXPERIMENTAL held-out evaluation set: real structures, 30 % identity separated from every
# training file and from the CASP/CAMEO/apo/CoDNaS benchmarks. Replaces the AFDB selection set.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
echo "== select"
python -u gate27_select_exp_val.py --n ${N:-1200} --threads 16 --min-id ${MINID:-0.3} 2>&1 | grep -v -i -E "$F"
echo "== build (val group)"
python -u gate15_build_pdb.py --list $ESM_PROAE_ROOT/data/pdb/exp_val_chains.tsv --out $D/dataset_exp_val.h5 \
   --group val --workers 8 --enc-batch 48 2>&1 | grep -v -i -E "$F"
echo "== ESMC embeddings"
python -u gate11_precompute_esmc.py --src $D/dataset_exp_val.h5 --splits val --out $D/dataset_exp_val_esmc.h5 --kind esmc --bs 48 2>&1 | grep -v -i -E "$F"
echo "== autoencoder round-trip control on the new set"
python -u gate13_casp_roundtrip.py --h5 $D/dataset_exp_val.h5 --split val --n 200 2>&1 | grep -v -i -E "$F"
echo ALLDONE
