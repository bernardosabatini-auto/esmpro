#!/bin/bash
#SBATCH --job-name=build_ens
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; F='warn|INFO|return data|positions\['
export BUILD_MAX_LEN=512
python -u gate22_build_ensembles.py 2>&1 | grep -v -i -E "$F"
for s in apo codnas cameo22; do python -u gate11_precompute_esmc.py --src $D/dataset_$s.h5 --splits val --out $D/dataset_${s}_esmc.h5 --kind esmc --bs 16 --sort 2>&1 | grep -v -i -E "$F" | tail -1; done
echo ALLDONE
