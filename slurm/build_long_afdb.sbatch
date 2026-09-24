#!/bin/bash
#SBATCH --job-name=build_long_afdb
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# 500-residue data, part 2: AFDB v6 cluster representatives 257-512 residues, pLDDT >= 70 (161k), shard $1 of $2, then ESMC embeddings.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
K=${1:-0}; N=${2:-4}
export BUILD_MAX_LEN=512 BUILD_MIN_LEN=257
python -u gate8_build_afdb.py --index $ESM_PROAE_ROOT/data/afdb_long_index.txt --exclude "" --shard $K/$N --out $D/dataset_afdb_long_$K.h5 --workers 14 --enc-batch 32 2>&1 | grep -v -i -E "$F"
python -u gate11_precompute_esmc.py --src $D/dataset_afdb_long_$K.h5 --splits train --out $D/dataset_afdb_long_${K}_esmc.h5 --kind esmc --bs 16 --sort 2>&1 | grep -v -i -E "$F"
echo ALLDONE
