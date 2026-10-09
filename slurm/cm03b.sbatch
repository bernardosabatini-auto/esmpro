#!/bin/bash
#SBATCH -J cm03bb
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -o logs/cm03b_%j.log
#SBATCH -e logs/cm03b_%j.log
# Campaign ridge baselines (code/cm03b_ridge.py), GPU fp64 solves; one RTX card.
# small MLPs; H100/H200 memory would sit idle. GPU utilisation is sampled every 30 s into the log.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 30 > logs/cm03_${SLURM_JOB_ID}.smi &
SMI=$!
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm03b_ridge.py "$@"
RC=$?
kill $SMI
echo "GPU util (mean of 30 s samples): $(awk -F', ' '{gsub(/ %/,"",$2); s+=$2; n++} END {if(n) printf "%.0f%% over %d samples", s/n, n}' logs/cm03_${SLURM_JOB_ID}.smi)"
echo "CM03_EXIT=$RC"
exit $RC
