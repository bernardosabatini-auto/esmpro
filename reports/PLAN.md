# Plan: from ~0.75 TM to a defensible breakthrough

Written 2026-09-23 evening, assuming the combined run (`pf_459M_esmc_afdb`:
459M DiT + 64-dim pair track + ESMC-6B + 473k proteins) lands near TM 0.75 /
~90% correct folds / under 7 A mean RMSD on the selection-free held-out slice.

## What we have learned (evidence, not opinion)
| lever | effect (held-out TM, equal everything else) | cost |
|---|---|---|
| ESM-2 -> ESMC-6B conditioner | +0.12 (0.61 -> 0.73), also +0.10 on no-neighbour folds | one 30-min precompute |
| 80k -> 473k proteins | +0.05 (ESM-2); no overfitting at 473k with ESMC | one 75-min build |
| pair track (64-dim) | +0.01-0.02 at plateau, ~2x fewer epochs to a given TM | 2.2x step cost |
| pair track 128-dim x 8 vs 64 x 6 | +0.03 at equal epochs (partly batch) | +50% pair cost |
| best-of-8 sampling | +0.03-0.06 | 8x inference |
| model size 174M -> 459M | ~0 on 80k; untested with ESMC on 473k | 2.6x |
| all-layer ESM mixing, loss variants, sampler steps | 0 | - |

The generative model is now data- and conditioner-limited, not architecture-
limited, and it generalises: the ESMC gain is larger on folds with no training
relative than on the plain held-out slice.

## Phase A (this week, RTX only, in parallel with nothing else on the H200s)
1. **Close the loop on the combined run.** Held-out, no-neighbour and best-of-K
   scores; guidance sweep; report. Cost: one RTX-hour.
2. **External comparison on the same proteins.** Run ESMFold2 (biohub, built
   on the same ESMC-6B) in single-sequence mode and ESMFold on the held-out
   slice and the no-neighbour subsets, same Foldseek TM/RMSD pipeline. Without
   this no claim is credible. Cost: a few RTX-hours.
3. **A clean benchmark.** *(b done 2026-09-24, see casp_benchmark.md; a pending)* (a) Structure-clustered split: Foldseek-cluster the
   573k structures and hold out whole clusters (TM < 0.5 to any training
   protein) so future numbers are generalisation numbers by construction.
   (b) An experimental set: CAMEO/CASP single-chain targets <= 256 residues
   (then <= 512), scored against experimental coordinates.
4. **Sample selector.** *(done 2026-09-24, see sample_selector.md: consensus +0.01, ~20 % of the gap; confidence head deprioritised)* Best-of-8 is worth +0.03-0.06 but needs the answer to
   pick. Train a small confidence head on the flow model's final features to
   predict per-sample TM (labels from our own pipeline on training samples),
   and measure how much of the oracle gap it recovers. If it recovers half,
   inference gets ~+0.02 for 8x compute, no retraining of the generator.

## Phase B (1-2 weeks, 8 H200s for one run at a time)
*2026-09-24: experimental PDB targets built (pdb_build.md) and tested by fine-tuning (pdb_finetune.md): null. Recycling retired (recycling.md).*
*2026-09-26: 512-window data built (long_data_build.md); recipe = repeated batching + late-t + fused pair + distributed eval (efficiency.md, repeated_batching.md, late_t_resampling.md); from-scratch 512 run is the best model on every set (pf_459M_p128x8_long512_scratch.md: held-out 0.785, long CASP 0.763). Next: 840M trunk on ~1.3M proteins.*
5. **Data at the million scale, cleanly.** Extend to the Foldseek AFDB
   cluster representatives (~2.3M at <= 512 residues, ~4x the current set),
   built with `gate8` and clustered against the new held-out split. ESMC
   embeddings for 2M proteins are ~1.5 TB fp16: too much for host RAM, so
   either stream precomputed shards from netscratch or run ESMC online on
   the H200s (1 s per 64 sequences; acceptable at 8 GPUs). Decision after a
   loader benchmark.
6. **Longer proteins.** MAX_LEN 256 -> 512 (the ProteinAE decoder config is
   512-capable; verify reconstruction on long chains first). The pair track's
   L^2 cost is handled by residue-budget batching.
7. **Architecture, in this order.** Pair track 128 x 8 (measured +0.03);
   structure recycling through the frozen decoder (decode a sample, feed its
   distogram to the pair init, resample; 1-2 cycles at inference only);
   then, only if the data step shows headroom, a larger DiT.

## Phase C (after B lands)
8. **Decoder robustness.** Fine-tune the ProteinAE decoder on flow-sampled
   latents so slightly-off latents decode to the nearest good structure.
9. **What only a generative model can do.** Ensemble evaluation on
   fold-switching and apo/holo pairs; guidance-based motif scaffolding and
   conditional design through the same latent; these are the results that
   distinguish this from a cheaper ESMFold.

## What "breakthrough" means numerically
On the structure-clustered held-out set: mean TM >= 0.80, >= 90% correct
folds, mean RMSD ~5 A, at a per-protein inference cost an order of magnitude
below ESMFold2, with calibrated ensembles. On CAMEO singles: within a few
points of ESMFold2 single-sequence mode. Anything short of the external
comparison is an internal number.

## Immediate order of work
A1 -> A2 -> A3(a) -> A4 -> A3(b), with B5's data build starting as soon as
A3(a) defines the split (the build is I/O-bound and runs on one RTX).
