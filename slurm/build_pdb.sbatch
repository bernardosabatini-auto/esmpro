#!/bin/bash
#SBATCH --job-name=build_pdb
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
echo "== CASP round trip control"; python -u gate13_casp_roundtrip.py 2>&1 | grep -v -i -E "$F"
echo "== build"; python -u gate15_build_pdb.py --out $D/dataset_pdb_train.h5 --workers 14 --enc-batch 64 ${LIMIT:+--limit $LIMIT} 2>&1 | grep -v -i -E "$F"
echo "== ESMC embeddings"; python -u gate11_precompute_esmc.py --src $D/dataset_pdb_train.h5 --splits train --out $D/dataset_pdb_train_esmc.h5 --kind esmc --bs 64 2>&1 | grep -v -i -E "$F"
echo "== PDB round trip control"; python -u gate13_casp_roundtrip.py --h5 $D/dataset_pdb_train.h5 --split train --n 200 2>&1 | grep -v -i -E "$F"
echo ALLDONE
