#!/bin/bash
#SBATCH --job-name=pyg_build_gate
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/logs/%x_%j.out
set -x
source /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/slurm/env.sh
module load gcc/13.2.0-fasrc01 cuda/13.3.1-fasrc01
export CUDA_HOME=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16
export FORCE_CUDA=1
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install --no-cache-dir setuptools wheel ninja
echo "=== BUILD torch-scatter ==="
pip install --no-cache-dir --no-build-isolation -v torch-scatter==2.1.2 > $ESM_PROAE_ROOT/logs/build_scatter_$SLURM_JOB_ID.log 2>&1 || tail -30 $ESM_PROAE_ROOT/logs/build_scatter_$SLURM_JOB_ID.log
echo "=== BUILD torch-cluster ==="
pip install --no-cache-dir --no-build-isolation -v torch-cluster==1.6.3 > $ESM_PROAE_ROOT/logs/build_cluster_$SLURM_JOB_ID.log 2>&1 || tail -30 $ESM_PROAE_ROOT/logs/build_cluster_$SLURM_JOB_ID.log
python -c "
import torch, torch_scatter, torch_cluster
x = torch.randn(10, 4, device='cuda'); i = torch.tensor([0,0,1,1,2,2,3,3,4,4], device='cuda')
print('scatter_mean cuda ok', torch_scatter.scatter_mean(x, i, dim=0).shape)
print('BUILD_OK')" || { echo BUILD_FAILED; exit 1; }
echo "=== DECODER IMPORT ==="
cd $ESM_PROAE_ROOT/ProteinAE_v1 && python -c "
import sys; sys.path.insert(0,'.')
from proteinfoundation.proteinflow.proteinae import ProteinAE
print('IMPORT_OK')" || { echo IMPORT_FAILED; exit 1; }
echo "=== FAPE INVARIANCE GUARD ==="
cd $ESM_PROAE_ROOT/code && python gate6_fape_invariance_check.py || { echo GUARD_FAILED; exit 1; }
echo "=== GATE: score inherited checkpoint ==="
cd $ESM_PROAE_ROOT/code && time python gate6_score_checkpoint.py --ckpt smallscale_final_40k.pt --n 100 --bs 10
echo GATE_DONE
