#!/bin/bash
#SBATCH -J bzval
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-01:30
#SBATCH -o logs/boltz_val_%j.log
#SBATCH -e logs/boltz_val_%j.log
# Design A validation gate: can Boltz-2 single-sequence reproduce the HSPB1 ACD dimer and dock the
# held-out 9ef3 IxI peptide into the beta4/beta8 groove? Also measures seconds per co-fold, which
# sizes the full pilot. One RTX PRO 6000 (96 GB) is far more than this needs; H200 is not justified.
set -u
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
cd $ESM_PROAE_ROOT
B=$HOME/.conda/envs/boltz/bin/boltz
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"START %s"
$B predict data/campaign/boltz/val/in \
   --out_dir data/campaign/boltz/val/out \
   --cache $HOME/.boltz \
   --accelerator gpu --devices 1 \
   --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
   --output_format mmcif --num_workers 4 --override
date +"END %s"
echo "BZVAL_EXIT=$?"
