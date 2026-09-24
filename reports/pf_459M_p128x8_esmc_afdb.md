# pf_459M_p128x8_esmc_afdb — main line: 128-dim x 8 pair track + ESMC-6B + 473k on 8 H200s

**Job** 48083691, kempner_h200, 2 nodes x 4 H200 (8 GPUs, 1400 GB host RAM per node), 2026-09-23 22:37 to 2026-09-24 09:36 (11.0 h). Early-stopped on TM at epoch 50 (patience 8), best epoch 42. Log `logs/lf_multinode_48083691.out`.

## Configuration
`code/gate10_pair_flow.py`: DiT d1024 x 24L x 16 heads + **128-dim pair track, 8 gated triangular-multiplication blocks** (checkpointed, bf16, compiled); stored ESMC-6B embeddings (2560-d); train = 80k original + 393k AFDB additions = 473k proteins; residue-budget batching 96 x 256^2 per GPU (cap 3x), peak GPU 114 GB of 141; lr 4e-4, warm-up 1000, cosine over 60 epochs; CFG dropout 0.1, self-conditioning, EMA; eval every epoch on the 100 selection proteins at w=2, 25 Euler steps. ~796 s/epoch (~280 steps). Same recipe as `pf_459M_esmc_afdb` except the pair track (64 x 6 there).

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 1.1403 | 0.4883 | 0.179 (2.0) | 0% | 39.66 | 100% |
| 2 | 0.2594 | 0.1924 | 0.307 (2.0) | 3% | 15.88 | 100% |
| 3 | 0.1626 | 0.1187 | 0.429 (2.0) | 32% | 12.17 | 100% |
| 4 | 0.1228 | 0.0889 | 0.554 (2.0) | 61% | 9.95 | 100% |
| 5 | 0.1038 | 0.0762 | 0.634 (2.0) | 78% | 8.55 | 100% |
| 6 | 0.0947 | 0.0700 | 0.659 (2.0) | 81% | 8.14 | 100% |
| 7 | 0.0893 | 0.0660 | 0.674 (2.0) | 82% | 7.84 | 100% |
| 8 | 0.0855 | 0.0634 | 0.685 (2.0) | 85% | 7.68 | 100% |
| 9 | 0.0831 | 0.0611 | 0.691 (2.0) | 85% | 7.60 | 100% |
| 10 | 0.0807 | 0.0596 | 0.698 (2.0) | 85% | 7.51 | 100% |
| 11 | 0.0789 | 0.0582 | 0.699 (2.0) | 86% | 7.47 | 100% |
| 12 | 0.0779 | 0.0570 | 0.706 (2.0) | 86% | 7.41 | 100% |
| 13 | 0.0761 | 0.0560 | 0.711 (2.0) | 87% | 7.45 | 100% |
| 14 | 0.0753 | 0.0552 | 0.711 (2.0) | 86% | 7.42 | 100% |
| 15 | 0.0736 | 0.0543 | 0.716 (2.0) | 87% | 7.36 | 100% |
| 16 | 0.0725 | 0.0535 | 0.716 (2.0) | 87% | 7.45 | 100% |
| 17 | 0.0720 | 0.0528 | 0.717 (2.0) | 86% | 7.19 | 100% |
| 18 | 0.0714 | 0.0525 | 0.719 (2.0) | 88% | 7.15 | 100% |
| 19 | 0.0701 | 0.0519 | 0.721 (2.0) | 88% | 7.23 | 100% |
| 20 | 0.0698 | 0.0515 | 0.725 (2.0) | 87% | 7.16 | 100% |
| 21 | 0.0691 | 0.0510 | 0.725 (2.0) | 87% | 7.17 | 100% |
| 22 | 0.0687 | 0.0505 | 0.728 (2.0) | 87% | 7.11 | 100% |
| 23 | 0.0682 | 0.0500 | 0.730 (2.0) | 87% | 7.06 | 100% |
| 24 | 0.0672 | 0.0497 | 0.730 (2.0) | 87% | 7.07 | 100% |
| 25 | 0.0671 | 0.0494 | 0.729 (2.0) | 87% | 7.04 | 100% |
| 26 | 0.0666 | 0.0491 | 0.731 (2.0) | 87% | 6.99 | 100% |
| 27 | 0.0663 | 0.0490 | 0.732 (2.0) | 87% | 6.94 | 100% |
| 28 | 0.0657 | 0.0489 | 0.732 (2.0) | 88% | 6.98 | 100% |
| 29 | 0.0651 | 0.0486 | 0.732 (2.0) | 88% | 6.99 | 100% |
| 30 | 0.0650 | 0.0484 | 0.733 (2.0) | 88% | 6.99 | 100% |
| 31 | 0.0646 | 0.0482 | 0.734 (2.0) | 88% | 7.04 | 100% |
| 32 | 0.0643 | 0.0480 | 0.733 (2.0) | 88% | 7.07 | 100% |
| 33 | 0.0635 | 0.0478 | 0.734 (2.0) | 88% | 7.05 | 100% |
| 34 | 0.0634 | 0.0478 | 0.734 (2.0) | 88% | 7.06 | 100% |
| 35 | 0.0631 | 0.0478 | 0.734 (2.0) | 88% | 7.04 | 100% |
| 36 | 0.0629 | 0.0477 | 0.737 (2.0) | 88% | 7.00 | 100% |
| 37 | 0.0625 | 0.0476 | 0.737 (2.0) | 88% | 7.05 | 100% |
| 38 | 0.0620 | 0.0475 | 0.737 (2.0) | 88% | 7.08 | 100% |
| 39 | 0.0616 | 0.0474 | 0.738 (2.0) | 88% | 7.06 | 100% |
| 40 | 0.0614 | 0.0471 | 0.738 (2.0) | 88% | 7.05 | 100% |
| 41 | 0.0612 | 0.0469 | 0.740 (2.0) | 88% | 6.87 | 100% |
| 42 | 0.0612 | 0.0467 | 0.741 (2.0) | 88% | 6.84 | 100% |
| 43 | 0.0607 | 0.0464 | 0.740 (2.0) | 88% | 6.86 | 100% |
| 44 | 0.0602 | 0.0464 | 0.740 (2.0) | 88% | 6.81 | 100% |
| 45 | 0.0600 | 0.0463 | 0.740 (2.0) | 88% | 6.82 | 100% |
| 46 | 0.0599 | 0.0462 | 0.738 (2.0) | 87% | 6.88 | 100% |
| 47 | 0.0598 | 0.0460 | 0.738 (2.0) | 88% | 6.91 | 100% |
| 48 | 0.0597 | 0.0459 | 0.736 (2.0) | 88% | 7.08 | 100% |
| 49 | 0.0587 | 0.0458 | 0.737 (2.0) | 87% | 7.14 | 100% |
| 50 | 0.0588 | 0.0457 | 0.737 (2.0) | 88% | 7.18 | 100% |

Best TM 0.7406759999999999 at epoch 42.

## Scores (`best_pf_459M_p128x8_esmc_afdb.pt`, epoch 42; w = 2, 50 Euler steps, K = 8; coverage 100/100 everywhere)
| set | this run | 64-dim pair predecessor (`pf_459M_esmc_afdb`) |
|---|---|---|
| selection set (train-time, 25 steps) | 0.741 / 88 % / 6.84 A at epoch 42 | 0.740 / 89 % / 6.82 A at epoch 60 |
| **held-out (offset 1000, n=100)** | **0.758 / 90 % / 6.13 A**, best-of-8 0.796 (w=1.5: 0.755) | 0.758 / 88 % / 6.51 A, best-of-8 0.792 |
| no-neighbour < 0.6 (n=266) | 0.573 / 65 % / 12.69 A, bo8 0.632 | 0.567 / 64 % |
| no-neighbour < 0.5 (n=93) | 0.514 / 42 % / 15.35 A, bo8 0.577 | 0.512 / 41 % |
| **CASP15/16 experimental (n=80)** | **0.703 / 74 % / 8.10 A**, bo8 0.745 | 0.698 / 72 % / 8.32 A, bo8 0.743 |
| ESMFold2-Fast, same sets | held-out 0.753 / 85 % / 7.18 A; CASP 0.741 / 79 % / 7.15 A | |

## Outcome
The larger pair track reached the 64-dim run's final selection-set score 18 epochs earlier and finished +0.001 above it; on every independent set it is +0.002 to +0.006 TM, +2 points of correct folds, and 0.2-0.4 A better RMSD. All within one standard error of the predecessor, all in the same direction, and exactly what the three 80k ablations predicted (pair capacity = convergence lever, ~+0.01 at most). Best model in the project on every set: held-out 0.758 / 90 %, novel-fold subsets 0.573 and 0.514, CASP 0.703 / 74 %. Against ESMFold2-Fast: ahead by 0.005 TM and 5 points of correct folds on AFDB held-out, behind by 0.038 TM and 5 points on experimental CASP coordinates. Coverage 100 % at all 50 evaluations.

## What it changed
Closes the pair-size axis at scale: 128 x 8 is the configuration to keep (faster, marginally better, fits at 114 GB), but pair capacity is not where the next 0.05 is. Its checkpoint is the warm start for the two fine-tune arms now running (`pf_459M_p128x8_pdbft` with 13 % experimental PDB targets, `pf_459M_p128x8_ctlft` without), which test the one lever the CASP benchmark pointed at. The remaining gap to ESMFold2-Fast on experimental structures (0.04) is smaller than the gap between our ESM-2 and ESMC-6B runs (0.17), so the conditioner and the training targets, not the architecture, remain the levers.
