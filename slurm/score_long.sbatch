#!/bin/bash
#SBATCH --job-name=score_long
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# Long-protein scoring at the 512 window: long val (first N), CASP <= 512 (all 109) and the long CASP subset. knobs: CKPT (required), N, K, W
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code; D=$ESM_PROAE_ROOT/data/phase1_dataset; N_=$ESM_PROAE_ROOT/notes; F='warn|INFO|return data'
export ESM_PROAE_MAX_LEN=512; C=${CKPT:?}
echo "== $C long val (first ${N:-200} of 1000)"; python gate7_score.py --ckpt $C --h5 $D/dataset_long_val_esmc.h5 --n ${N:-200} --offset 0 --k ${K:-4} --cfg-w ${W:-2} --steps 50 --bs 10 2>&1 | grep -v -i -E "$F"
grep -E "casp1[56]_" $N_/casp_domains_le512.txt > $N_/casp_domains_le512_all.txt
python - <<PY
import h5py; h=h5py.File("$D/dataset_casp512.h5","r")["val"]
open("$N_/casp_domains_257_512.txt","w").write("\n".join(n for n in h if h[n].attrs["n_residues"]>256)+"\n")
PY
echo "== $C CASP <=512 (all 109)"; python gate7_score.py --ckpt $C --h5 $D/dataset_casp512_esmc.h5 --names-file $N_/casp_domains_le512.txt --n 200 --k ${K:-4} --cfg-w ${W:-2} --steps 50 --bs 10 2>&1 | grep -v -i -E "$F"
echo "== $C CASP 257-512 only"; python gate7_score.py --ckpt $C --h5 $D/dataset_casp512_esmc.h5 --names-file $N_/casp_domains_257_512.txt --n 200 --k ${K:-4} --cfg-w ${W:-2} --steps 50 --bs 10 2>&1 | grep -v -i -E "$F"
echo ALLDONE
