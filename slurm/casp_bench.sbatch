#!/bin/bash
#SBATCH --job-name=casp_bench
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
# CASP15/16 experimental benchmark: embeddings (ESMC-6B and ESM-2), our checkpoints, ESMFold2-Fast.
# knobs: CKPTS (space-separated best_*.pt), W (cfg weights), K, STEPS, SKIP_ESMFOLD=1
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
D=$ESM_PROAE_ROOT/data/phase1_dataset; N=$ESM_PROAE_ROOT/notes
F='warn|INFO|return data|Loading weights|UNEXPECTED|^Key|^---|^Notes|can be ignored'
[ -e $D/dataset_casp_esmc.h5 ] || python -u gate11_precompute_esmc.py --src $D/dataset_casp.h5 --splits val --out $D/dataset_casp_esmc.h5 --kind esmc --bs 16 2>&1 | grep -v -i -E "$F"
[ -e $D/dataset_casp_esm2.h5 ] || python -u gate11_precompute_esmc.py --src $D/dataset_casp.h5 --splits val --out $D/dataset_casp_esm2.h5 --kind esm2 --bs 16 2>&1 | grep -v -i -E "$F"
for c in ${CKPTS:-best_pf_459M_esmc_afdb.pt best_lf_174M_esmc.pt best_pf_174M_p64x6_esmc.pt best_lf_459M_afdb.pt best_lf_174M.pt best_pf_174M_p128x8.pt}; do
  h5=$D/dataset_casp_esmc.h5; python -c "import json;m=json.load(open('$D/${c%.pt}.meta.json'));import sys;sys.exit(0 if m.get('esm_kind','esm2')=='esmc' or 'esmc' in str(m.get('h5_path','')) else 1)" || h5=$D/dataset_casp_esm2.h5
  echo "== $c on $(basename $h5)"
  for nf in casp_domains_le256; do
    python gate7_score.py --ckpt $c --h5 $h5 --names-file $N/$nf.txt --n 200 --k ${K:-8} --cfg-w ${W:-1.5,2} --steps ${STEPS:-50} 2>&1 | grep -v -i -E "$F"
  done
done
[ -n "$SKIP_ESMFOLD" ] || \
  python -u gate12_esmfold2_compare.py --loops 3 --steps 50 --h5 $D/dataset_casp.h5 --sets names:$N/casp_domains_le256.txt,names:$N/casp15_domains_le256.txt,names:$N/casp16_domains_le256.txt --tag esmfold2_fast_casp 2>&1 | grep -v -i -E "$F"
echo ALLDONE
