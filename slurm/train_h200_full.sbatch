#!/bin/bash
#SBATCH --job-name=esm_proae_train
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
#SBATCH --signal=B:USR1@900

# CLAUDE.md section 5 job. First submission warm-starts from the inherited
# checkpoint (or random init with WARM=none for a different head size); every
# later submission RESUMES from last_<label>.ckpt, which carries the optimizer,
# schedule position, epoch counter and RNG. Never restart with --warm-start
# after a preemption. When the 2-day limit approaches (USR1 15 min before),
# the script resubmits itself; the new job resumes.
#   env knobs: LABEL BS NL DM WARM EPOCHS EXTRA (extra trainer flags, e.g. loss)
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
LABEL=${LABEL:-h200_full}
LAST=$ESM_PROAE_ROOT/data/phase1_dataset/last_${LABEL}.ckpt
DONE=$ESM_PROAE_ROOT/notes/gate6_${LABEL}_results.json
COMMON="--n-train 79653 --n-val 10355 --batch-size ${BS:-96} --ode-steps 3 \
  --lr 1e-4 --clamp 20.0 --frac-unclamped 0.1 --normalize-out \
  --tm-every 1 --tm-n 200 --select-on tm \
  --n-layers ${NL:-10} --d-model ${DM:-256} \
  --epochs ${EPOCHS:-60} --patience 6 --label $LABEL ${EXTRA:-}"

resubmit() {
  echo "[$(date)] time limit approaching; resubmitting for resume"
  cd $ESM_PROAE_ROOT && LABEL=$LABEL BS=$BS NL=$NL DM=$DM WARM=$WARM EPOCHS=$EPOCHS EXTRA="$EXTRA" \
    sbatch --export=ALL slurm/train_h200_full.sbatch
}
trap resubmit USR1

if [ -f "$DONE" ]; then echo "results file exists ($DONE); nothing to do"; exit 0; fi
if [ -f "$LAST" ]; then
  echo "RESUMING from $LAST"
  python -u gate6_fape_train.py $COMMON --resume last_${LABEL}.ckpt &
else
  echo "FRESH START (warm-start ${WARM:-smallscale_final_40k.pt})"
  python -u gate6_fape_train.py $COMMON --warm-start ${WARM:-smallscale_final_40k.pt} &
fi
PID=$!
# a trapped signal interrupts wait; keep waiting so the run uses its full allocation
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "[$(date)] python exited with $RC"
