#!/bin/bash
#SBATCH --job-name=lf_multinode
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
#SBATCH --signal=B:USR1@900
# Gate 7 latent flow across several H200 nodes: one torchrun per node, 4 GPUs
# each, c10d rendezvous on the first node. --batch-size in EXTRA is PER GPU.
# Env knobs: LABEL EXTRA. Resumes from last_<label>.ckpt; resubmits itself.
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
LABEL=${LABEL:-lf_multinode}
LAST=$ESM_PROAE_ROOT/data/phase1_dataset/last_${LABEL}.ckpt
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
resubmit() { echo "[$(date)] resubmitting for resume"; cd $ESM_PROAE_ROOT && LABEL=$LABEL EXTRA="$EXTRA" sbatch --nodes=$SLURM_JOB_NUM_NODES --export=ALL slurm/train_latent_flow_multinode.sbatch; }
trap resubmit USR1
RES=""; [ -f "$LAST" ] && RES="--resume last_${LABEL}.ckpt"
echo "nodes: $SLURM_JOB_NODELIST  master: $MASTER:$PORT"
srun --kill-on-bad-exit=1 torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT --rdzv-id=$SLURM_JOB_ID \
  gate7_latent_flow.py --label $LABEL ${EXTRA:-} $RES &
PID=$!
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "[$(date)] torchrun exited with $RC"
