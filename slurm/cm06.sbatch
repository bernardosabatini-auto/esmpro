#!/bin/bash
#SBATCH -J cm06
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -o logs/cm06_%j.log
#SBATCH -e logs/cm06_%j.log
# Campaign attention-pooling MLP (code/cm06_attn.py): 37 GB of per-residue states on one 96 GB RTX card.
# small MLPs; H100/H200 memory would sit idle. GPU utilisation is sampled every 30 s into the log.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 30 > logs/cm06_${SLURM_JOB_ID}.smi &
SMI=$!
G=${SLURM_JOB_GPUS:-0}; G=${G%%,*}
( dcgmi dmon -e 1002,1003,1004,1005 -i $G -d 5000 -c 100000 > logs/cm06_${SLURM_JOB_ID}.dcgm 2>/dev/null ) &
DCGM=$!
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm06_attn.py "$@"
RC=$?
kill $SMI $DCGM 2>/dev/null
echo "DCGM mean (sm_active occupancy tensor_active dram_active): $(awk '$1=="GPU" && $3+0==$3 {for(i=3;i<=6;i++) s[i]+=$i; n++} END {if(n) printf "%.2f %.2f %.2f %.2f over %d samples", s[3]/n, s[4]/n, s[5]/n, s[6]/n, n; else print "n/a"}' logs/cm06_${SLURM_JOB_ID}.dcgm)"
echo "GPU util (mean of 30 s samples): $(awk -F', ' '{gsub(/ %/,"",$2); s+=$2; n++} END {if(n) printf "%.0f%% over %d samples", s/n, n}' logs/cm06_${SLURM_JOB_ID}.smi)"
echo "CM03_EXIT=$RC"
exit $RC
