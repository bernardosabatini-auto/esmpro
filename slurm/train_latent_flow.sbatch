#!/bin/bash
#SBATCH --job-name=latent_flow
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
#SBATCH --signal=B:USR1@900
# Gate 7 latent flow model. Whole train split is cached in host RAM (52 GB
# of fp16 ESM embeddings), hence --mem=200G. Resumes from last_<label>.ckpt
# and resubmits itself before the time limit. Env knobs: LABEL EXTRA
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
LABEL=${LABEL:-lf_base}
LAST=$ESM_PROAE_ROOT/data/phase1_dataset/last_${LABEL}.ckpt
resubmit() { echo "[$(date)] resubmitting for resume"; cd $ESM_PROAE_ROOT && LABEL=$LABEL EXTRA="$EXTRA" sbatch -p $SLURM_JOB_PARTITION --export=ALL slurm/train_latent_flow.sbatch; }
trap resubmit USR1
if [ -f "$LAST" ]; then
  python -u gate7_latent_flow.py --label $LABEL ${EXTRA:-} --resume last_${LABEL}.ckpt &
else
  python -u gate7_latent_flow.py --label $LABEL ${EXTRA:-} &
fi
PID=$!
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "[$(date)] python exited with $RC"
