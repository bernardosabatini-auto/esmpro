# H200 cluster setup — Harvard FASRC, 2026-09-22

Everything below was verified by the gate in CLAUDE.md section 4, which on this
cluster reproduced the Spark numbers exactly: TM 0.427, TM>0.5 0.32,
RMSD 11.56 A, coverage 100/100 (job 47757668, 2.5 min wall).

## SLURM

- Account `kempner_bsabatini_lab`, partition `kempner_h200` (2-day limit).
  The plain `bsabatini_lab` account has MaxSubmit 0; `gpu_h200` refuses jobs.
- H200 nodes: 143 GB HBM, driver 610.57, 1.5 TB RAM, 844 GB local `/tmp`.
- Job scripts in `slurm/`:
  - `env.sh` — sets `ESM_PROAE_ROOT`, activates the env, sets `FOLDSEEK_BIN`.
  - `gate_score.sbatch <ckpt> [n] [bs]` — scores any checkpoint.
  - `train_h200_full.sbatch` — the section 5 job. Auto-resumes from
    `last_<label>.ckpt` if present and resubmits itself 15 min before the
    time limit. Env knobs: `LABEL BS NL DM WARM EPOCHS`.
    Example: `LABEL=h200_full sbatch --export=ALL slurm/train_h200_full.sbatch`
  - `build_pyg_and_gate.sbatch` — rebuilds the PyG extensions (see below).

## Environment (`conda env proteinae`, python 3.11)

torch 2.11.0+cu130, torch-geometric 2.8.0, lightning 2.6.6, transformers 5.17,
numpy 2.4, foldseek 10 and mmseqs2 from bioconda. Two things bit:

1. **PyG extension wheels need glibc 2.32; the cluster has 2.28.** So
   `torch_scatter` (the decoder imports it) and `torch_cluster` were built
   from source on a compute node with `gcc/13.2.0-fasrc01` and
   `cuda/13.3.1-fasrc01`, `TORCH_CUDA_ARCH_LIST=9.0`.
2. **nvcc 13.3 cannot compile torch 2.11's `ATen/core/List_inl.h:202`.**
   The env's copy is patched to `static_cast<std::ptrdiff_t>(pos)`, original
   kept as `List_inl.h.orig`. Reinstalling torch undoes the patch, and the
   extensions must then be rebuilt.

`gate6_fape_train.py` gained `--warm-start none` (random init) for head-size
sweep members, which cannot take the 10x256 weights.

## Guards run before training

`code/gate6_fape_invariance_check.py` (CLAUDE.md bug 1): true structure under
rotations 30/90/180 deg and translations 1/5/20 A all scored FAPE 1e-5; 1 A
noise scored 0.73; mirror image 0.84. PASS.

## Measured on the H200 (2026-09-22)

- 10L d256, batch 96: 65.9 GB peak, 1.7 s/batch, 24.5 min/epoch on 79,653
  proteins including validation; TM eval on 200 proteins adds 30 s.
- 16L d512: 72 GB peak. 24L d768: 83 GB peak. 10L d256 batch 160: 110 GB peak.
- Epoch 1 of the warm-started section 5 run reproduced the inherited numbers:
  val FAPE 0.858, TM 0.433, TM>0.5 0.37, coverage 200/200.

## Loss arms (2026-09-22)

`gate6_fape_train.py` gained `--loss {ca,bb,ca+bb}`, `--bb-points {ca,all}`,
`--bond-weight W` and `--fix-end-frames`. The decoder wrapper returns the full
N/CA/C/O backbone on request (`return_backbone=True`); the dataset returns
N/CA/C from `backbone_100k.h5` when a backbone term is on. Guards:
`gate6_loss_guard.py [--gpu]` (pose invariance of every term with real
residues only moved; decoder atom order via bond lengths) and
`gate6_bbfape_floor.py` (loss floor for a perfect latent).

Findings before launch:
- Legacy Ca pseudo-frames leak pose at the last real residue (its Ca_{i+1}
  is a padded zero): 8.7 A error on that one frame, ~1% of the loss.
  `--fix-end-frames` reuses the inner neighbour's rotation; leak gone.
- A perfect latent decodes with intrinsic sampling scatter (N-CA std 0.058 A):
  floors are Ca-FAPE 0.053, bb-FAPE 0.131, seed-to-seed 0.060 / 0.146.
  fp32 and 10 ODE steps only lower bb-FAPE to 0.075, so the floor is the
  decoder, not precision. Both floors sit far below the head (0.87 / 0.89).
- Decoder atom order confirmed [N, CA, C, O], Angstroms after x10.

Arms (all warm-started 10L d256, batch 96, fix_ends on; control = h200_full):
h200_ca_bond, h200_bb_bond, h200_bb, h200_cabb_bond. Bond weight 1.0 (L1, A).

## Two findings that bear on strategy (2026-09-22)

1. **The latent is not rotation-invariant.** `code/gate6_encoder_equivariance.py`:
   rotating a structure changes its per-residue latent by L2 2.1-2.9 (norm
   2.83, cosine ~0.55); translating it changes nothing. The code encodes
   absolute orientation, so a sequence-only head cannot regress it directly;
   FAPE through the decoder works because the head is free to pick a frame.
2. **`canonicalize.py` is missing from the bundle.** `gate6_100k.py` imports
   `canonicalize_pyg_data` from it, and `gate6_extract_backbone.py` imports
   through `gate6_100k`. Section 9b's claim that the dataset is rebuildable
   from the bundle is false until that file is copied from the Spark.

Ready to run when a GPU frees: `code/gate6_decoder_tolerance.py` (TM vs
latent noise, and the head's actual latent error).
