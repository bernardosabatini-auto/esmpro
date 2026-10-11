#!/bin/bash
#SBATCH -J study
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=120G
#SBATCH -t 0-06:00
#SBATCH -o logs/study_%j.log
#SBATCH -e logs/study_%j.log
# Run every analysis level on one study:   sbatch slurm/study.sbatch <study> [extra args]
#
# Mostly CPU (pandas, numpy, MMseqs2) with one GPU-bound level (embed) and one that benefits from a
# GPU (predict). The CPU account bsabatini_lab has MaxSubmit 0, so everything runs on the smallest GPU
# partition regardless; kempner_rtx caps cores at 15 per GPU. Ask for more GPUs only to get more cores
# (15 each, up to 120 on a whole node) and only when a level is genuinely core-bound.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
export MMSEQS_BIN=${MMSEQS_BIN:-/n/holylabs/bsabatini_lab/Users/bsabatini/mmseqs/bin/mmseqs}
cd $ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
STUDY=${1:?usage: sbatch slurm/study.sbatch <study> [args...]}; shift || true
$PY -u -m pdpipe.study --study "$STUDY" "$@"
echo "STUDY_EXIT=$?"
