# Source this before any script in this project.
export ESM_PROAE_ROOT=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae
source /n/sw/Miniforge3-26.7.2-0/etc/profile.d/conda.sh
conda activate proteinae
export FOLDSEEK_BIN=$CONDA_PREFIX/bin/foldseek
export MMSEQS_BIN=$CONDA_PREFIX/bin/mmseqs
export HDF5_USE_FILE_LOCKING=FALSE
