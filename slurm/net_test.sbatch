#!/bin/bash
#SBATCH --job-name=net_test
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(getent hosts $MASTER | awk '{print $1}' | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "nodes $SLURM_JOB_NODELIST master $MASTER -> $MASTER_ADDR:$MASTER_PORT"
srun bash -c 'echo "$(hostname -s): ifaces $(hostname -I)"; export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0; python $ESM_PROAE_ROOT/code/net_test.py'
echo NET_TEST_DONE
