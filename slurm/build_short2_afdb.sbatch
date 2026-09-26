#!/bin/bash
#SBATCH --job-name=build_short2
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# 627k additional AFDB v6 cluster representatives <= 256 residues, pLDDT >= 80, not in the 473k set; shard $1 of $2; then ESMC embeddings.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
K=${1:-0}; N=${2:-4}
python -u gate8_build_afdb.py --index $ESM_PROAE_ROOT/data/afdb_short_extra_index.txt --exclude "" --shard $K/$N --out $D/dataset_afdb_short2_$K.h5 --workers 14 --enc-batch 64 2>&1 | grep -v -i -E "$F"
python -u gate11_precompute_esmc.py --src $D/dataset_afdb_short2_$K.h5 --splits train --out $D/dataset_afdb_short2_${K}_esmc.h5 --kind esmc --bs 64 --sort 2>&1 | grep -v -i -E "$F"
echo ALLDONE
