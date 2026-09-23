#!/bin/bash
# Launch the AFDB-scale latent flow model on 2 H200 nodes (8 GPUs).
# 459M DiT, online ESM-2 with learnable all-layer mix, train = 100k train
# split + both Gate 8 AFDB shards (~470k proteins), val = the 100k val split.
# --batch-size is per GPU: 192 x 8 = 1536 effective. Adjust BS after the
# batch-1 peak-GPU print (rule: fill the 144 GB).
R=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
D=$R/data/phase1_dataset
LABEL=${LABEL:-lf_459M_afdb} BS=${BS:-192}
EXTRA="--esm online --extra-train-h5 $D/dataset_afdb_train_0.h5,$D/dataset_afdb_train_1.h5 \
  --d-model 1024 --n-layers 24 --n-heads 16 --batch-size $BS --lr 4e-4 --warmup 1000 \
  --epochs ${EPOCHS:-60} --eval-every 1 --eval-n 100 --patience 8 --cfg-w 2 --workers 4 --sample-steps 25"
LABEL=$LABEL EXTRA="$EXTRA" sbatch --nodes=2 ${SBATCH_EXTRA:-} --export=ALL $R/slurm/train_latent_flow_multinode.sbatch
