#!/bin/bash
#SBATCH --job-name=score_full
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Full scoring of one checkpoint: held-out w sweep (K=8), no-neighbour <0.6 / <0.5, CASP (ESMC or ESM-2 file chosen from the meta).
# knobs: CKPT (required; best_*.pt or last_*.ckpt = final EMA weights), K, STEPS, W, CASP_H5 (dataset_casp_esmc.h5 | dataset_casp_esm2.h5)
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; N=$ESM_PROAE_ROOT/notes; F='warn|INFO|return data'
C=${CKPT:?}; K=${K:-8}; S=${STEPS:-50}
echo "== $C held-out offset 1000"; python gate7_score.py --ckpt $C --n 100 --k $K --cfg-w ${W:-1.5,2} --offset 1000 --steps $S 2>&1 | grep -v -i -E "$F"
for nf in val_noneighbour_lt0.6 val_noneighbour_lt0.5; do echo "== $C $nf"; python gate7_score.py --ckpt $C --n 400 --k $K --cfg-w 2 --steps $S --names-file $N/$nf.txt 2>&1 | grep -v -i -E "$F"; done
if [ -n "$CASP_H5" ]; then h5=$D/$CASP_H5; elif [[ "$C" == *.ckpt ]]; then h5=$D/dataset_casp_esmc.h5; else
  h5=$D/dataset_casp_esmc.h5; python -c "import json,sys;m=json.load(open('$D/$C.meta.json'));sys.exit(0 if 'esmc' in str(m.get('h5_path','')) else 1)" || h5=$D/dataset_casp_esm2.h5; fi
echo "== $C casp ($(basename $h5))"; python gate7_score.py --ckpt $C --h5 $h5 --names-file $N/casp_domains_le256.txt --n 200 --k $K --cfg-w 2 --steps $S 2>&1 | grep -v -i -E "$F"
echo ALLDONE
