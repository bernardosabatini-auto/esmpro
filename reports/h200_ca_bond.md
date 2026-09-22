# h200_ca_bond — FAPE head, Ca-FAPE + bond-length penalty (the section-8 prerequisite)

**Job** 47785117, kempner_h200, 2026-09-22. Stopped by hand after 15 epochs (6 h); tracking the control within noise. Log `logs/esm_proae_train_47785117.out`.

## Configuration
Head 10L d256 (8.2M), warm-started; Ca-FAPE with `--fix-end-frames`, plus bond-length L1 penalty (N-CA, CA-C, C-N+1) weight 1.0. Peak GPU 65.9 GB, ~1440 s/epoch. `code/gate6_fape_train.py`; batch 96; 3 ODE steps; lr 1e-4 cosine over 60 epochs; clamp 20 A, 10% unclamped; `--normalize-out`; 79,653 train / 10,355 val; TM on 200 val proteins every epoch, selection on TM, patience 6; kempner_h200, one GPU.

## Curve (TM on 200 val proteins each epoch)
| epoch | train | val FAPE | TM | TM>0.5 | coverage | N-CA dev |
|---|---|---|---|---|---|---|
| 1 | 1.7786 | 0.8684 | 0.397 | 24% | 100% | 0.214 |
| 2 | 1.7332 | 0.8610 | 0.416 | 31% | 100% | 0.190 |
| 3 | 1.6992 | 0.8553 | 0.426 | 34% | 100% | 0.170 |
| 4 | 1.6725 | 0.8496 | 0.438 | 36% | 100% | 0.159 |
| 5 | 1.6520 | 0.8440 | 0.445 | 36% | 100% | 0.149 |
| 6 | 1.6295 | 0.8378 | 0.460 | 40% | 100% | 0.140 |
| 7 | 1.6171 | 0.8366 | 0.455 | 41% | 100% | 0.132 |
| 8 | 1.5969 | 0.8319 | 0.465 | 42% | 100% | 0.126 |
| 9 | 1.5800 | 0.8278 | 0.469 | 42% | 100% | 0.122 |
| 10 | 1.5734 | 0.8241 | 0.478 | 45% | 100% | 0.117 |
| 11 | 1.5596 | 0.8212 | 0.482 | 46% | 100% | 0.115 |
| 12 | 1.5473 | 0.8187 | 0.486 | 46% | 100% | 0.111 |
| 13 | 1.5356 | 0.8163 | 0.488 | 47% | 100% | 0.108 |
| 14 | 1.5254 | 0.8143 | 0.489 | 47% | 100% | 0.105 |
| 15 | 1.5170 | 0.8119 | 0.490 | 48% | 100% | 0.103 |

Best TM 0.4898205 at epoch 15; best val FAPE 0.8119147624130603 at epoch 15.
Stopped by hand (see above).

## Scores
Best checkpoint `best_h200_ca_bond.pt` (epoch 15, selected on the 200-protein set): selection-set TM 0.490 / 48%.
**Held-out slice (offset 1000, selection-free):** TM 0.479, TM>0.5 45%, TM>0.3 87%, RMSD 12.63 A, coverage 100/100. Same slice: inherited checkpoint 0.420 / 29% / 12.67 A; latent flow 59M (epoch 12) 0.530 / 57% / 11.14 A.

## Outcome
Best selection-set TM 0.490 / 48% at epoch 15, held-out 0.479 / 45% / 12.63 A, versus the control's 0.487 / 48% / 11.36 A: no gain, possibly a small loss, within one standard error. The penalty did its job on geometry: decoded N-CA scatter fell from 0.39 A to 0.12 A by epoch 9 (val `nca_dev` column), at a cost of ~0.04 TM in the first epoch that was recovered by epoch 8. Coverage 100% throughout.

## What it changed
Repairing local geometry, the experiment CLAUDE.md section 8 called the highest-value one after the scale-up, does not improve fold accuracy for this head. The head's errors are global. The bond penalty stays available (`--bond-weight`) for any consumer that needs chemically sane decoded backbones.
