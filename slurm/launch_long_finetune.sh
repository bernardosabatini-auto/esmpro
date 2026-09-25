#!/bin/bash
# 512-residue stage: fine-tune the best 256-window checkpoint on 473k short + 160k long AFDB + 13.5k long PDB,
# selection on the 1,000 held-out long proteins. 2 H200 nodes (8 GPUs). Repeated batching R copies per protein.
#   fine-tune: LABEL=pf_459M_p128x8_long512 WARM=best_pf_459M_p128x8_esmc_afdb.pt R=2 BS=54 EPOCHS=12 ./slurm/launch_long_finetune.sh
#   from scratch: LABEL=pf_459M_p128x8_long512_scratch R=4 BS=36 EPOCHS=40 LR=4e-4 WARMUP=1000 TLATE=1 ./slurm/launch_long_finetune.sh
R_=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae; D=$R_/data/phase1_dataset
LABEL=${LABEL:-pf_459M_p128x8_long512}
EXTRAS="$D/dataset_afdb_train_0_esmc.h5,$D/dataset_afdb_train_1_esmc.h5,$D/dataset_afdb_long_0_esmc.h5,$D/dataset_afdb_long_1_esmc.h5,$D/dataset_afdb_long_2_esmc.h5,$D/dataset_afdb_long_3_esmc.h5,$D/dataset_pdb_long_esmc.h5${EXTRA_H5:+,$EXTRA_H5}"
EXTRA="--h5-path $D/dataset_100k_esmc.h5 --extra-train-h5 $EXTRAS --val-h5 $D/dataset_long_val_esmc.h5 --n-val 1000 \
  ${PAIR:---d-pair 128 --n-pair-blocks 8} --d-model ${DM:-1024} --n-layers ${NL:-24} --n-heads ${NH:-16} --batch-size ${BS:-60} \
  --lr ${LR:-1e-4} --warmup ${WARMUP:-300} --epochs ${EPOCHS:-12} --eval-every 1 --eval-n 100 --patience ${PATIENCE:-6} --cfg-w 2 --workers 4 \
  --sample-steps 25 --budget-cap ${BUDGET:-3} ${WARM:+--warm-start $WARM}"
# TLATE=1 -> SimpleFold late-t resampling (logit-normal m 0.8 s 1.7, 2 % uniform)
if [ -n "$TLATE" ]; then export T_LOGIT_M=0.8 T_LOGIT_S=1.7 T_UNIF=0.02; fi
ESM_PROAE_MAX_LEN=512 REPEAT_COPIES=${R:-2} LABEL=$LABEL EXTRA="$EXTRA" SCRIPT="${SCRIPT:-gate10_pair_flow.py}" \
  sbatch --nodes=2 --mem=${MEM:-1400G} ${SBATCH_EXTRA:-} --export=ALL $R_/slurm/train_latent_flow_multinode.sbatch
