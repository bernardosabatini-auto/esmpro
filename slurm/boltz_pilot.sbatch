#!/bin/bash
#SBATCH -J bzpilot
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=120G
#SBATCH -t 0-08:00
#SBATCH -o logs/boltz_pilot_%j.log
#SBATCH -e logs/boltz_pilot_%j.log
# Design A pilot: 4,787 HSPB1-ACD-dimer + client-window co-folds.
#
# EFFICIENCY. kempner_rtx allows at most 15 cores per GPU, so this is one whole GPU's share. Each job is ~192 tokens, which is tiny: a single Boltz process leaves most of a 96 GB
# Blackwell card idle, and the measured rate was ~4 s/job, i.e. ~5.3 GPU-hours if run one at a time.
# So four processes share the one card, each on its own shard. That is the right way to use this
# hardware for many small inferences -- more cards would each be just as underused. An H200 would be
# strictly worse here: same underutilisation, more fairshare. GPU use is logged every 30 s so the
# claim is checked, not assumed.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
B=$HOME/.conda/envs/boltz/bin/boltz
IN=data/campaign/boltz/in
OUT=data/campaign/boltz/out
NPROC=4

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"START %s"
( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
      | sed 's/,//' | awk -v t=$(date +%s) '{print "GPUMON",t,$1,$2}'
    sleep 30
  done ) > logs/boltz_gpumon_$SLURM_JOB_ID.log 2>&1 &
MON=$!

PIDS=()
for i in $(seq 0 $((NPROC-1))); do
  s=$(printf "%02d" $i)
  $B predict $IN/shard_$s \
     --out_dir $OUT/shard_$s \
     --cache $HOME/.boltz \
     --accelerator gpu --devices 1 \
     --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
     --output_format mmcif --num_workers 2 --override \
     > logs/boltz_shard_$s.log 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait $p || RC=1; done
kill $MON 2>/dev/null

date +"END %s"
echo "predictions written: $(ls -d $OUT/shard_*/boltz_results_shard_*/predictions/* 2>/dev/null | wc -l)"
grep -h "Number of failed examples" logs/boltz_shard_*.log
awk '/GPUMON/{u+=$3;m+=$4;if($4>mx)mx=$4;n++} END{if(n)printf "GPU over %d samples: mean util %.1f%%, mean mem %.1f GB, peak mem %.1f GB\n",n,u/n,m/n/1024,mx/1024}' logs/boltz_gpumon_$SLURM_JOB_ID.log
echo "BZPILOT_EXIT=$RC"
