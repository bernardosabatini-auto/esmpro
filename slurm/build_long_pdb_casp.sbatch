#!/bin/bash
#SBATCH --job-name=build_long_pdb
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# 500-residue data, part 1: CASP15/16 domains <= 512, experimental PDB chains 257-512, ESMC embeddings, round-trip controls.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
export BUILD_MAX_LEN=512
echo "== CASP <=512"; python -u gate13_build_casp.py --device cuda --max-len 512 --out $D/dataset_casp512.h5 2>&1 | grep -v -i -E "$F" | grep -E "unique|wrote"
echo "== PDB 257-512 build"; python -u gate15_build_pdb.py --list $ESM_PROAE_ROOT/data/pdb/selected_chains_long.tsv --out $D/dataset_pdb_long.h5 --workers 14 --enc-batch 32 2>&1 | grep -v -i -E "$F"
echo "== ESMC embeddings (CASP512 val, PDB long train)"
python -u gate11_precompute_esmc.py --src $D/dataset_casp512.h5 --splits val --out $D/dataset_casp512_esmc.h5 --kind esmc --bs 16 --sort 2>&1 | grep -v -i -E "$F"
python -u gate11_precompute_esmc.py --src $D/dataset_pdb_long.h5 --splits train --out $D/dataset_pdb_long_esmc.h5 --kind esmc --bs 16 --sort 2>&1 | grep -v -i -E "$F"
echo "== round trips"; python -u gate13_casp_roundtrip.py --h5 $D/dataset_casp512.h5 2>&1 | grep -v -i -E "$F"; python -u gate13_casp_roundtrip.py --h5 $D/dataset_pdb_long.h5 --split train --n 200 2>&1 | grep -v -i -E "$F"
echo ALLDONE
