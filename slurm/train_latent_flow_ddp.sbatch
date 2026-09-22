#!/bin/bash
#SBATCH --job-name=latent_flow_ddp
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
#SBATCH --signal=B:USR1@900
# Gate 7 latent flow, data-parallel over all GPUs of one node via torchrun.
# --batch-size in EXTRA is PER GPU. Env knobs: LABEL EXTRA NGPU (default 8)
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
LABEL=${LABEL:-lf_ddp}; NGPU=${NGPU:-8}
LAST=$ESM_PROAE_ROOT/data/phase1_dataset/last_${LABEL}.ckpt
resubmit() { echo "[$(date)] resubmitting for resume"; cd $ESM_PROAE_ROOT && LABEL=$LABEL EXTRA="$EXTRA" NGPU=$NGPU sbatch -p $SLURM_JOB_PARTITION --gres=gpu:$NGPU --export=ALL slurm/train_latent_flow_ddp.sbatch; }
trap resubmit USR1
RES=""; [ -f "$LAST" ] && RES="--resume last_${LABEL}.ckpt"
torchrun --standalone --nproc-per-node=$NGPU gate7_latent_flow.py --label $LABEL ${EXTRA:-} $RES &
PID=$!
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "[$(date)] torchrun exited with $RC"
