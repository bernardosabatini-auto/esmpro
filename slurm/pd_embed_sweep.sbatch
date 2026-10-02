#!/bin/bash
#SBATCH --job-name=pd_sweep
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
cd $ESM_PROAE_ROOT/code
for B in 8000 30000 80000 160000; do
  echo "=== budget $B"
  python -u pd03_embed.py --budget $B --limit 900 --dcgm --out /tmp/sweep_$B.npz 2>&1 \
    | grep -E "batches,|residues/s, peak|GPU: SM|OutOfMemory" | tail -3
done
echo PD_SWEEP_DONE
