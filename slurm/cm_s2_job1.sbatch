#!/bin/bash
#SBATCH -J cm03_s2_1
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00
#SBATCH -o logs/cm03_%j.log
#SBATCH -e logs/cm03_%j.log
# Campaign MLP population training (code/cm03_mlp.py). One RTX card: data ~1 GB + a vectorised population of
# small MLPs; H100/H200 memory would sit idle. GPU utilisation is sampled every 30 s into the log.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 30 > logs/cm03_${SLURM_JOB_ID}.smi &
SMI=$!
G=${SLURM_JOB_GPUS:-0}; G=${G%%,*}
( dcgmi dmon -e 1002,1003,1004,1005 -i $G -d 5000 -c 100000 > logs/cm03_${SLURM_JOB_ID}.dcgm 2>/dev/null ) &
DCGM=$!
/n/home08/bsabatini/.conda/envs/proteinae/bin/python -u code/cm03_mlp.py --spec code/specs/s2/job1_0_refine_Lall.json --out data/campaign/runs/s2
RC=$?
kill $SMI $DCGM 2>/dev/null
echo "DCGM mean (sm_active occupancy tensor_active dram_active): $(awk '$1=="GPU" && $3+0==$3 {for(i=3;i<=6;i++) s[i]+=$i; n++} END {if(n) printf "%.2f %.2f %.2f %.2f over %d samples", s[3]/n, s[4]/n, s[5]/n, s[6]/n, n; else print "n/a"}' logs/cm03_${SLURM_JOB_ID}.dcgm)"
echo "GPU util (mean of 30 s samples): $(awk -F', ' '{gsub(/ %/,"",$2); s+=$2; n++} END {if(n) printf "%.0f%% over %d samples", s/n, n}' logs/cm03_${SLURM_JOB_ID}.smi)"
echo "CM03_EXIT=$RC"
exit $RC
