#!/bin/bash
#SBATCH --job-name=lf_multinode
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
#SBATCH --signal=B:USR1@900
# Gate 7 latent flow across several H200 nodes: one srun task per GPU,
# torch.distributed env:// init from SLURM ranks, IPv4 master address
# (hostname-based c10d rendezvous fails on these nodes: IPv6, errno 97).
# --batch-size in EXTRA is PER GPU. Env knobs: LABEL EXTRA. Resumes from
# last_<label>.ckpt and resubmits itself before the time limit.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
LABEL=${LABEL:-lf_multinode}
LAST=$ESM_PROAE_ROOT/data/phase1_dataset/last_${LABEL}.ckpt
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(getent hosts $MASTER | awk '{print $1}' | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=4
resubmit() { echo "[$(date)] resubmitting for resume"; cd $ESM_PROAE_ROOT && LABEL=$LABEL EXTRA="$EXTRA" sbatch --nodes=$SLURM_JOB_NUM_NODES --export=ALL slurm/train_latent_flow_multinode.sbatch; }
trap resubmit USR1
RES=""; [ -f "$LAST" ] && RES="--resume last_${LABEL}.ckpt"
echo "nodes: $SLURM_JOB_NODELIST  master: $MASTER -> $MASTER_ADDR:$MASTER_PORT  tasks: $SLURM_NTASKS"
srun --kill-on-bad-exit=1 bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python -u gate7_latent_flow.py --label '"$LABEL"' '"${EXTRA:-}"' '"$RES" &
PID=$!
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "[$(date)] srun exited with $RC"
