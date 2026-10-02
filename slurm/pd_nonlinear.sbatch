#!/bin/bash
#SBATCH -J pd06
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -o logs/pd06_%j.log
#SBATCH -e logs/pd06_%j.log
# Gradient boosting runs on the 16 CPU cores; the MLP runs on the GPU. The card is requested
# because kempner_requeue refuses CPU-only jobs and the lab account that owns the CPU
# partitions has a submit limit of 0, so this is the only reachable partition.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python -u code/pd06_nonlinear.py "$@"
RC=$?
echo "PD06_EXIT=$RC"
exit $RC
