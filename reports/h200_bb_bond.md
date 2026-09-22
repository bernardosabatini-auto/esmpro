# h200_bb_bond — FAPE head, true N-CA-C frames + bond-length penalty

**Job** 47785121, kempner_h200, 2026-09-22 13:15-18:15. Stopped deliberately at epoch 11 (still rising ~0.006 TM/epoch, tracking the control within noise) to free GPUs for the latent-flow line. Log `logs/esm_proae_train_47785121.out`.

## Configuration
`code/gate6_fape_train.py`; head 10L d256 (8.2M), warm-started from `smallscale_final_40k.pt`; batch 96; 3 ODE steps; lr 1e-4 cosine over 60 epochs; clamp 20 A, 10% unclamped; `--normalize-out`; `--fix-end-frames` (pose-invariant chain-end frames); 79,653 train / 10,355 val; TM on 200 val proteins every epoch, selection on TM, patience 6. kempner_h200, one GPU, peak 65.9 GB, ~1490 s/epoch.

**Loss:** FAPE in true N-CA-C frames built from the decoded backbone, all three backbone atoms as points (`--loss bb --bb-points all`), plus a bond-length L1 penalty on N-CA, CA-C and C-N(+1) with weight 1.0 (`--bond-weight 1.0`). Truth frames from `backbone_100k.h5`. The N-CA dev column is the validation mean |N-CA bond - 1.458 A| of the decoded backbone.

## Curve (TM on 200 val proteins each epoch)
| epoch | train | val FAPE | TM | TM>0.5 | coverage | N-CA dev |
|---|---|---|---|---|---|---|
| 1 | 1.7933 | 0.8724 | 0.394 | 24% | 100% | 0.207 |
| 2 | 1.7411 | 0.8658 | 0.409 | 29% | 100% | 0.179 |
| 3 | 1.7026 | 0.8607 | 0.422 | 30% | 100% | 0.158 |
| 4 | 1.6716 | 0.8561 | 0.432 | 32% | 100% | 0.143 |
| 5 | 1.6471 | 0.8505 | 0.442 | 36% | 100% | 0.132 |
| 6 | 1.6216 | 0.8455 | 0.449 | 38% | 100% | 0.123 |
| 7 | 1.6068 | 0.8434 | 0.453 | 40% | 100% | 0.116 |
| 8 | 1.5849 | 0.8387 | 0.462 | 40% | 100% | 0.112 |
| 9 | 1.5671 | 0.8338 | 0.467 | 42% | 100% | 0.106 |
| 10 | 1.5594 | 0.8300 | 0.475 | 43% | 100% | 0.102 |
| 11 | 1.5434 | 0.8268 | 0.480 | 45% | 100% | 0.099 |

Best TM 0.4802005 at epoch 11; best val FAPE 0.8268068648046918 at epoch 11.
Stopped by hand at epoch 11 (see above).

## Outcome
TM 0.480 / 45% at epoch 11 on the 200-protein selection set, against the control's 0.477 / 45% at the same epoch. The bond penalty did exactly what CLAUDE.md section 8 asked for: decoded N-CA scatter fell from 0.39 A (unpenalised control) to 0.10 A by epoch 11, i.e. within a factor of two of what the true latent decodes to (0.06 A). It cost about 0.04 TM in the first epoch while the head re-organised its latents, which was recovered by epoch 8. True-frame FAPE on top of the penalty produced no measurable gain over the Ca pseudo-frame loss: the arms are indistinguishable at the 3.5-point standard error of 200 proteins. Coverage 100% throughout. Held-out (offset 1000) score of `best_h200_bb_bond.pt`: pending, will be appended.

## What it changed
Two things. First, local geometry is fixable cheaply and the fix is now on the shelf (`--bond-weight`), so the section-8 prerequisite is met. Second, the loss function is not where the FAPE line's headroom is: with the geometry repaired, true frames buy nothing, which says the head's errors are global (wrong fold) rather than local (right fold, wrong frame). That, together with the head-size sweep converging to the same 0.47-0.48, is why effort moved to the latent flow model (`reports/lf_base.md`), which passed 0.53 on the held-out slice in 22 minutes.
