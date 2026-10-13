#!/bin/bash
#SBATCH -J study
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=48G
#SBATCH -t 0-06:00
#SBATCH -o logs/study_%j.log
#SBATCH -e logs/study_%j.log
# Run every analysis level on one study:   sbatch slurm/study.sbatch <study> [extra args]
#
# Mostly CPU (pandas, numpy, MMseqs2) with one GPU-bound level (embed) and one that benefits from a
# GPU (predict). The CPU account bsabatini_lab has MaxSubmit 0, so everything runs on the smallest GPU
# partition regardless; kempner_rtx caps cores at 15 per GPU. Ask for more GPUs only to get more cores
# (15 each, up to 120 on a whole node) and only when a level is genuinely core-bound.
#
# 15 cores is sized for a FULL run, where the CPU ridge in `predict` eigendecomposes a 5,120 x 5,120 Gram
# per target per fold and reaches 87 % CPU efficiency (measured, job 52153861). Every other level is
# serial, so a subset run wastes most of them -- 10.6 % on folds+decompose. Override at submit time,
# where the command line beats these directives:
#   sbatch -c 2 --mem 16G slurm/study.sbatch mystudy --levels complex,decompose --resume
# Memory is sized at ~30x the 1.64 GB that GA_22 actually peaked at, to leave room for a larger study.
set -u
# Pin the maths libraries to the allocation. BLAS/OMP otherwise size their pools from the cores they can
# SEE (128 on this node), not the cores the cgroup allows, and oversubscribe every solve.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
export MMSEQS_BIN=${MMSEQS_BIN:-/n/holylabs/bsabatini_lab/Users/bsabatini/mmseqs/bin/mmseqs}
cd $ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
STUDY=${1:?usage: sbatch slurm/study.sbatch <study> [args...]}; shift || true
$PY -u -m pdpipe.study --study "$STUDY" "$@"
echo "STUDY_EXIT=$?"
