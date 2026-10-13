#!/bin/bash
#SBATCH -J cm26
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-04:00
#SBATCH -o logs/cm26_%j.log
#SBATCH -e logs/cm26_%j.log
# cm26: is the complex effect of 3c a capture nonlinearity, co-abundance, or co-capture?
# Dense numpy over an 11.5k x 11.5k co-membership matrix plus cluster bootstraps. Single-threaded CPU
# work with no CUDA in it; the CPU account bsabatini_lab has MaxSubmit 0 (verified), so this takes the
# smallest GPU partition and leaves the card idle. 2 cores, because the work is serial: the partition
# caps at 15 per GPU, but asking for cores that sit idle is what the Job Defense Shield reports.
set -u
# Pin the maths libraries to the allocation. BLAS/OMP otherwise size their pools from the cores they can
# SEE (128 on this node), not the cores the cgroup allows, and oversubscribe every solve.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
export PYTHONPATH=$ESM_PROAE_ROOT/code
PY=/n/home08/bsabatini/.conda/envs/dendritic_env/bin/python
$PY -u code/cm26_nonlinearity.py
echo "CM26_EXIT=$?"
