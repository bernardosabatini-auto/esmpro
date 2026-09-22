# ESM-ProteinAE — instructions for the Claude instance on the H200 cluster

You are picking up a working protein structure predictor. Read this fully before
running anything. The project has lost three separate multi-day stretches to
**silent measurement bugs** — no crash, no warning, plausible-looking curves. The
guards below exist because each one already failed once.

---

## 1. What this is

Predict protein structure from sequence alone, through a frozen pipeline:

```
sequence -> ESM-2 650M (FROZEN) -> trainable head -> 8-dim per-residue latent
         -> ProteinAE flow-matching decoder (FROZEN) -> Ca coordinates
```

Only the head trains. Loss is FAPE, which is differentiable. The metric we
actually care about is **TM-score**, which is not.

## 2. State you are inheriting

Small-scale run: 40k structures, 8.2M-param head, converged.

| | TM | correct folds (TM>0.5) | Ca RMSD |
|---|---|---|---|
| True latent through decoder | 0.997 | 100% | 0.17 A |
| **Inherited checkpoint** | **0.427** | **32%** | **11.56 A** |
| MSE-trained head (older approach) | 0.161 | 0% | 14.81 A |
| Mean-latent baseline | 0.061 | 0% | 20.17 A |
| ESMFold reference | — | — | 4.33 A |

Loss AND fold quality plateaued together, so 40k has genuinely saturated an
8.2M head. **Your job: scale data and head size.**

Because the decoder reaches 0.17 A from the true latent, neither the decoder nor
the 8-dimensional channel is the bottleneck. The head is.

---

## 3. Setup

```bash
export ESM_PROAE_ROOT=/path/to/this/bundle      # REQUIRED, no trailing slash
export FOLDSEEK_BIN=$(which foldseek)           # or an absolute path
```

Every script reads `ESM_PROAE_ROOT`. Nothing else needs editing.

**Build the environment; do not copy it.** The Spark is aarch64, the H200 is
x86_64. `docs/pip_versions_spark.txt` lists what worked — match the torch and
lightning majors, treat the rest as flexible. On x86 the PyG extensions
(torch-scatter/sparse/cluster) have wheels and need no source build.

```bash
conda create -n proteinae python=3.11
conda activate proteinae
pip install torch lightning hydra-core omegaconf h5py einops numpy scipy pandas tqdm loguru
conda install -c conda-forge -c bioconda foldseek     # ~700 MB, required for TM
```

**Do not re-clone ProteinAE_v1.** The shipped tree carries two source fixes that
upstream lacks:
- `proteinfoundation/proteinflow/model_trainer_base.py:26` — `from typing import Dict` (upstream says `from torch import Dict`, which does not exist)
- `proteinfoundation/autoencode.py:107` — `weights_only=False` on checkpoint load

## 4. Verify before you train

**Step 0 — structural check. Works before the environment exists:**

```bash
cd $ESM_PROAE_ROOT/code
python check_bundle.py            # imports + required files
python check_bundle.py --deps     # add this once conda is built
```

It resolves every local import against what is actually present, which is the
failure that hit twice: `gate6_100k.py` and then `canonicalize.py` were each
imported by shipped code but never copied, and both surfaced only at runtime.
It also confirms `dataset_100k.h5` and the decoder checkpoint. Exit 0 means
structurally complete. If you add a script that imports a new local module,
re-run it before relying on the script.

**Step 1 — the real contract.**

```bash
cd $ESM_PROAE_ROOT/code
python gate6_score_checkpoint.py --ckpt smallscale_final_40k.pt --n 100 --bs 10
```

Expect **TM ≈ 0.42-0.44, TM>0.5 ≈ 0.32, RMSD ≈ 11.5 A, coverage 100/100**.

If TM is far off, or coverage is below 100/100, **stop and diagnose**. Do not
train. Coverage below 1.0 means section 6 bug 2 has returned.

Checkpoints are keyed to the data file. `--ckpt` resolves against
`$ESM_PROAE_ROOT/data/phase1_dataset/`.

---

## 5. The job to run

Start from the inherited checkpoint; it is 32% of the way, not zero.

```bash
cd $ESM_PROAE_ROOT/code
python -u gate6_fape_train.py \
  --n-train 79653 --n-val 10355 \
  --batch-size 96 --ode-steps 3 \
  --lr 1e-4 --clamp 20.0 --frac-unclamped 0.1 --normalize-out \
  --warm-start smallscale_final_40k.pt \
  --tm-every 1 --tm-n 200 --select-on tm \
  --n-layers 10 --d-model 256 \
  --epochs 60 --patience 6 --label h200_full
```

**On a preemptible partition, resume — do not restart.** Every epoch writes
`last_<label>.ckpt` atomically, carrying the AdamW moments, the cosine schedule
position, the epoch counter, the best-so-far trackers and the RNG state:

```bash
python -u gate6_fape_train.py ... --resume last_h200_full.ckpt
```

`--resume` supersedes `--warm-start`, which loads weights ONLY. Restarting with
`--warm-start` after a preemption resets the optimizer and restarts the learning
rate schedule from the top. Measured on a 12-epoch example: a restart would run
at lr 1.0e-4 where the resume correctly continues at 6.5e-5, with no error
message — exactly the silent-degradation signature this project keeps hitting.
`--resume` also refuses to load a checkpoint whose architecture differs from the
flags you passed.

**Measure memory before scaling the batch.** On the Spark this config peaked at
66 GB, but that was 128 GB of *unified* memory shared with the host. An H200 has
141 GB of dedicated HBM, so you likely have room for a larger batch — but
measure, do not assume. Batch 1 prints `peak GPU=... GB`. Known Spark ceilings:
batch 96 with 3 steps fit; batch 96 with 10 steps and batch 160 with any steps
both OOMed, because the pair representation costs O(batch x residues^2).

**Head size is the untested axis.** 8.2M parameters is far below any published
folding trunk reaching non-trivial TM. Sweep it with `--n-layers` and
`--d-model`. Each checkpoint records its own architecture in its `.meta.json`
sidecar, and `gate6_score_checkpoint.py` reads that sidecar, so sweep members
score correctly without extra flags; `--n-layers`/`--d-model` override it, and a
mismatch aborts with a clear message rather than loading silently. A wider or
deeper head needs a fresh run, not a warm-start from the 10x256 checkpoint.

---

## 6. Three bugs that will silently recur

### Bug 1 — FAPE must use each structure's OWN frames
The original put predicted and true points in the **true** frames. The frame
index then cancels algebraically and the loss collapses to un-superposed point
distance. A perfect structure rotated 90 degrees scored 0.95; real 1 A error
scored 0.16. Every model sat in the band reserved for orientation, so the loss
measured pose — precisely what it existed to ignore. Days of results were void.

**Guard, before trusting any structural loss:** score a true structure against
rotations of 30, 90, 180 degrees and translations of 1, 5, 20 A. All must read
~0. Real 1 A noise must not. Ten lines, seconds to run.

### Bug 2 — Foldseek needs `--exhaustive-search 1`
`-e inf --max-seqs 2000` relaxes the E-value but leaves the 3Di k-mer prefilter
ON. Dissimilar structures never reach TMalign, return no row, and get imputed as
zero. Coverage was 10-20 of 100 for every arm except the positive control, which
made TM a function of 3Di rather than independent of it. It manufactured a fake
agreement between "three independent metrics" and nearly killed a line of work
that was succeeding.

**Guard:** report coverage beside every Foldseek-derived statistic. Never impute
a floor for a missing hit without saying so. Already fixed in the shipped code —
do not remove the flag.

### Bug 3 — stop on TM, not on FAPE
FAPE tracks fold quality in direction but not magnitude. Between epochs 16 and
22 the loss gain shrank 59% while the correct-fold fraction grew 60%. Stopping
or selecting on the loss risks ending a run, or keeping the wrong file, while
structure is still improving.

Already wired: `--tm-every 1 --select-on tm`. Costs 27 s against a 4384 s epoch,
0.6%. Patience scales by `tm_every` so un-evaluated epochs are not counted.

### Also
**3Di identity is useless below about TM 0.4.** Unrelated real proteins score
~0.142, so the 0.146-0.167 range observed across all models was entirely floor.
Use TM.

---

## 7. Settled — do not relitigate

- **3 ODE steps is correct, not a shortcut.** Head output is identical at 3, 5,
  10 and 20 steps (FAPE 0.9217 at all four). Do not pay for 20. This also makes
  concurrent scoring cheap and safe.
- **Keep ESM-2 frozen.** ESMFold freezes its language model and trains only the
  trunk on a learned softmax-weighted mix over ALL layers. We read layer 33 only,
  so all-layer mixing is an untried lever costing 33 parameters against 118M for
  fine-tuning. Try mixing first.
- **Do not add an auxiliary latent-MSE term.** Its target is committed to the
  arbitrary PCA orientation of the encoded structure — the same broken signal
  that capped the older MSE approach at TM 0.161. It pulls back toward the
  failed solution.
- **The latent lives on a manifold.** True latents satisfy
  `layer_norm(z, (8,))` exactly: zero mean, unit variance per residue, L2 =
  sqrt(8) = 2.8284. `--normalize-out` projects onto it. Keep it on. Its measured
  effect on FAPE is below noise, but without it the head drifts (one run reached
  L2 6.9).

## 8. Ready, verified, deliberately OFF

`data/phase1_dataset/backbone_100k.h5` (transferred separately, 797 MB) holds N/CA/C for 99,832
structures, verified to match the stored alpha carbons exactly.
`fape_loss_backbone` implements AlphaFold's Gram-Schmidt frames, verified
invariant to 1e-5.

**Do not enable true frames yet.** Backbone decoded from the head's latents has
N-CA bonds of 1.38 +/- 0.52 A, where truth is 1.46 +/- 0.01 and the *true*
latent decodes to 1.46 +/- 0.03. True frames rest on 1.5 A baselines and would
be dominated by that scatter; the Ca pseudo-frames use 3.8 A and tolerate it.

**Fix local geometry first.** A bond-length penalty on the decoded output is
cheap and rotation-invariant. That is the prerequisite, and it is probably the
single highest-value experiment after the scale-up.

---

## 9. How to report back

Score with `gate6_score_checkpoint.py` and report TM mean, TM>0.5 fraction,
RMSD, **and coverage**. Compare against the inherited 0.427 / 32% / 11.56 A.

Success is TM > 0.5 mean with a majority of proteins folding correctly. ESMFold
at 4.33 A is the external bar; we are at 11.56 A.

If a result looks surprising in either direction, the base rate in this project
is that the measurement is wrong, not the model. Check coverage, check
invariance, check that the metric moves when you feed it a known-good and a
known-bad input. Two of the three bugs above were caught only because someone
asked whether a plateau was real.

## 9b. Rebuilding the dataset

`netscratch` is purged on a retention window, so know what is and is not
reproducible from this bundle.

Shipped: `gate5_build_dataset.py` (splits and MMseqs2 clustering) and
`gate6_100k.py` (ESM-2 embeddings, ProteinAE encoding, PCA canonicalisation —
it also supplies `canonicalize_pyg_data`, which `gate6_extract_backbone.py`
imports at runtime).

Also shipped: **`data/phase1_dataset/structures.tar.gz`**, 2.36 GB, holding all
99,903 source AFDB PDB files. The dataset IS therefore reproducible from this
bundle. Extract only when you actually need a rebuild:

```bash
cd $ESM_PROAE_ROOT/data/phase1_dataset
tar -xzf structures.tar.gz          # -> structures/, 8.9 GB, ~30 min
```

Expect roughly half an hour, and it is not the network. netscratch is a parallel
filesystem where per-file metadata operations dominate: creating these 100k
files runs near 2,500 files/min, while the 2.36 GB archive itself crosses in
about 90 seconds. Deleting the extracted tree is slow for the same reason. That
is also why the archive is the better form to copy somewhere permanent before a
purge — one file, trivially checksummed.

`backbone_100k.h5` is reproducible from `dataset_100k.h5` plus the extracted
structures, via `gate6_extract_backbone.py`.

External tools are read from `FOLDSEEK_BIN` and `MMSEQS_BIN`, defaulting to
whatever is on PATH.

## 10. Layout

```
$ESM_PROAE_ROOT/
  code/                         all scripts, path-portable via ESM_PROAE_ROOT
    gate6_fape_train.py         training; TM selection + stopping wired
    gate6_deep_head.py          the head (n_layers, d_model, normalize_out)
    gate6_score_checkpoint.py   score any checkpoint; safe alongside training
    gate6_fold_validation.py    TM + 3Di against all baselines
    gate6_corrected_eval.py     pose-invariant FAPE + Kabsch RMSD
    gate6_extract_backbone.py   rebuild backbone_100k.h5 (needs the PDB structures)
    gate6_100k.py               dataset builder: ESM-2 + ProteinAE encode
    gate5_build_dataset.py      dataset builder: splits + MMseqs2 clustering
  ProteinAE_v1/                 decoder package + ae_r1_d8_v1.ckpt. DO NOT re-clone.
  data/phase1_dataset/
    dataset_100k.h5             32.7 GB, transferred separately
    backbone_100k.h5            797 MB, transferred separately, unused for now
    structures.tar.gz           2.36 GB, 99,903 source PDBs — extract to rebuild
    smallscale_final_40k.pt     the inherited checkpoint
  docs/
    SMALLSCALE_HANDOFF.md       fuller write-up of the small-scale result
    gate6_smallscale_final.json 28 epochs + all four TM evaluations
    reviews/                    adversarial reviews that caught the bugs
```
