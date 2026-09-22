#!/bin/bash
#SBATCH --job-name=build_afdb
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Usage: sbatch slurm/build_afdb.sbatch <shard k/N> <out file>   (resumable: rerun to continue)
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u gate8_build_afdb.py --shard ${1:-0/1} --out ${2:-$ESM_PROAE_ROOT/data/phase1_dataset/dataset_afdb_train.h5} --workers 14 --enc-batch 64 2>&1 | grep -v -i -E "warn|INFO|return data|positions\["
echo BUILD_DONE
