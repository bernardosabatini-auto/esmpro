#!/bin/bash
#SBATCH --job-name=design_mpnn
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
python -u gate31_designability_mpnn.py ${ARGS:-} 2>&1 | grep -v -i -E "warn|INFO|return data|Loading weights|softmax|UNEXPECTED|MISSING"
echo ALLDONE
