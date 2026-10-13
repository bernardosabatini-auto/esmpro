#!/bin/bash
#SBATCH -J cm01emb
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH -o logs/cm01_%j.log
#SBATCH -e logs/cm01_%j.log
# Campaign: ESMC-6B SAE + 9 pooled layers for proteins lacking them. One RTX card (25 GB model).
set -u
# Pin the maths libraries to the allocation. BLAS/OMP otherwise size their pools from the cores they can
# SEE (128 on this node), not the cores the cgroup allows, and oversubscribe every solve.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm01_embed.py "$@"
RC=$?
echo "CM01_EXIT=$RC"
exit $RC
