#!/bin/bash
# ESMC-6B-conditioned latent flow on 473k proteins, 2 H200 nodes (8 GPUs),
# STORED embeddings (gate11 files, ragged RAM cache ~42 GB per rank).
#   SCRIPT=gate10_pair_flow.py PAIR="--d-pair 64 --n-pair-blocks 6" ./slurm/launch_esmc_afdb_big.sh   # with pair track
R=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
D=$R/data/phase1_dataset
LABEL=${LABEL:-lf_459M_esmc_afdb} BS=${BS:-192}
# knobs: EXTRA_H5 (more comma-separated train files, e.g. the PDB set repeated), WARM (best_*.pt weights-only init),
#        LR, WARMUP, PATIENCE, EPOCHS, BUDGET, BS, PAIR, DM/NL/NH, SCRIPT, MEM, SBATCH_EXTRA
EXTRA="--h5-path $D/dataset_100k_esmc.h5 --extra-train-h5 $D/dataset_afdb_train_0_esmc.h5,$D/dataset_afdb_train_1_esmc.h5${EXTRA_H5:+,$EXTRA_H5} \
  ${PAIR:-} --d-model ${DM:-1024} --n-layers ${NL:-24} --n-heads ${NH:-16} --batch-size $BS --lr ${LR:-4e-4} --warmup ${WARMUP:-1000} \
  --epochs ${EPOCHS:-60} --eval-every 1 --eval-n 100 --patience ${PATIENCE:-8} --cfg-w 2 --workers 4 --sample-steps 25 --budget-cap ${BUDGET:-3} \
  ${WARM:+--warm-start $WARM}"
LABEL=$LABEL EXTRA="$EXTRA" SCRIPT="${SCRIPT:-}" sbatch --nodes=2 --mem=${MEM:-1400G} ${SBATCH_EXTRA:-} --export=ALL $R/slurm/train_latent_flow_multinode.sbatch
