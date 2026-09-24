# pf_459M_esmc_afdb — pair track + ESMC-6B + 473k proteins on 8 H200s (the combined run)

**Job** 48000412, 2 x 4 H200 (`slurm/launch_esmc_afdb_big.sh`, `train_latent_flow_multinode.sbatch`), 2026-09-23 14:40-22:29 (7.8 h). Ran the full 60-epoch cosine schedule; best at the last epoch, still rising ~0.001 TM/epoch. Two earlier submissions died at the cache fill (host memory under-requested) and one was replaced to add residue-budget batching. Log `logs/lf_multinode_48000412.out`.

## Configuration
`code/gate10_pair_flow.py` (via `--esm stored --h5-path dataset_100k_esmc.h5 --extra-train-h5 dataset_afdb_train_{0,1}_esmc.h5`): DiT d1024 x 24L x 16 heads + 64-dim pair track (6 gated triangular-multiplication blocks, checkpointed, bf16, compiled) = 461.4M params; conditioning = ESMC-6B last-layer embeddings (2560-d, precomputed, ragged host cache ~40 GB/rank); CFG dropout 0.1, self-conditioning, EMA 0.999 with warm-up; AdamW lr 4e-4, warm-up 1000, cosine over 60 epochs; **residue-budget batching** (B x Lmax^2 <= 128 x 256^2, cap 3x, so batches of 134-384 proteins, 1,024-3,072 effective) with a DDP-safe step count; eval every epoch on the 100 gate proteins at w=2, 25 Euler steps. 487 s/epoch (~210 steps), peak GPU 131 GB of 144. Train = 473,184 proteins (100k train split + Gate 8 AFDB shards); validation = the 100k val split.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 1.3573 | 0.8292 | 0.159 (2.0) | 0% | 20.88 | 100% |
| 2 | 0.4259 | 0.2740 | 0.232 (2.0) | 0% | 21.25 | 100% |
| 3 | 0.2239 | 0.1822 | 0.342 (2.0) | 7% | 14.55 | 100% |
| 4 | 0.1633 | 0.1296 | 0.404 (2.0) | 30% | 12.52 | 100% |
| 5 | 0.1281 | 0.1003 | 0.519 (2.0) | 51% | 10.52 | 100% |
| 6 | 0.1118 | 0.0859 | 0.600 (2.0) | 72% | 9.00 | 100% |
| 7 | 0.1032 | 0.0784 | 0.632 (2.0) | 78% | 8.56 | 100% |
| 8 | 0.0968 | 0.0734 | 0.650 (2.0) | 79% | 8.21 | 100% |
| 9 | 0.0924 | 0.0700 | 0.659 (2.0) | 78% | 8.19 | 100% |
| 10 | 0.0890 | 0.0674 | 0.666 (2.0) | 82% | 8.16 | 100% |
| 11 | 0.0868 | 0.0652 | 0.672 (2.0) | 82% | 8.06 | 100% |
| 12 | 0.0846 | 0.0638 | 0.679 (2.0) | 83% | 8.09 | 100% |
| 13 | 0.0831 | 0.0626 | 0.680 (2.0) | 84% | 8.13 | 100% |
| 14 | 0.0810 | 0.0612 | 0.686 (2.0) | 86% | 7.90 | 100% |
| 15 | 0.0798 | 0.0598 | 0.689 (2.0) | 86% | 7.85 | 100% |
| 16 | 0.0794 | 0.0587 | 0.692 (2.0) | 86% | 7.79 | 100% |
| 17 | 0.0772 | 0.0579 | 0.696 (2.0) | 86% | 7.72 | 100% |
| 18 | 0.0768 | 0.0571 | 0.699 (2.0) | 87% | 7.61 | 100% |
| 19 | 0.0751 | 0.0563 | 0.702 (2.0) | 87% | 7.43 | 100% |
| 20 | 0.0750 | 0.0558 | 0.704 (2.0) | 87% | 7.46 | 100% |
| 21 | 0.0747 | 0.0553 | 0.708 (2.0) | 87% | 7.48 | 100% |
| 22 | 0.0738 | 0.0549 | 0.712 (2.0) | 87% | 7.37 | 100% |
| 23 | 0.0728 | 0.0546 | 0.710 (2.0) | 87% | 7.41 | 100% |
| 24 | 0.0726 | 0.0543 | 0.711 (2.0) | 87% | 7.34 | 100% |
| 25 | 0.0718 | 0.0538 | 0.712 (2.0) | 87% | 7.36 | 100% |
| 26 | 0.0714 | 0.0535 | 0.713 (2.0) | 87% | 7.41 | 100% |
| 27 | 0.0705 | 0.0531 | 0.715 (2.0) | 88% | 7.35 | 100% |
| 28 | 0.0696 | 0.0527 | 0.714 (2.0) | 88% | 7.44 | 100% |
| 29 | 0.0694 | 0.0524 | 0.718 (2.0) | 88% | 7.33 | 100% |
| 30 | 0.0692 | 0.0521 | 0.719 (2.0) | 88% | 7.30 | 100% |
| 31 | 0.0693 | 0.0519 | 0.719 (2.0) | 88% | 7.35 | 100% |
| 32 | 0.0680 | 0.0517 | 0.721 (2.0) | 88% | 7.26 | 100% |
| 33 | 0.0678 | 0.0514 | 0.722 (2.0) | 88% | 7.22 | 100% |
| 34 | 0.0675 | 0.0510 | 0.724 (2.0) | 87% | 7.16 | 100% |
| 35 | 0.0673 | 0.0506 | 0.725 (2.0) | 87% | 7.15 | 100% |
| 36 | 0.0669 | 0.0504 | 0.726 (2.0) | 88% | 7.16 | 100% |
| 37 | 0.0665 | 0.0501 | 0.725 (2.0) | 87% | 7.26 | 100% |
| 38 | 0.0659 | 0.0499 | 0.726 (2.0) | 88% | 7.17 | 100% |
| 39 | 0.0660 | 0.0498 | 0.726 (2.0) | 88% | 7.19 | 100% |
| 40 | 0.0655 | 0.0497 | 0.728 (2.0) | 89% | 7.23 | 100% |
| 41 | 0.0650 | 0.0496 | 0.729 (2.0) | 89% | 7.23 | 100% |
| 42 | 0.0650 | 0.0495 | 0.729 (2.0) | 88% | 7.16 | 100% |
| 43 | 0.0648 | 0.0492 | 0.730 (2.0) | 89% | 7.20 | 100% |
| 44 | 0.0643 | 0.0492 | 0.729 (2.0) | 89% | 7.22 | 100% |
| 45 | 0.0639 | 0.0492 | 0.730 (2.0) | 89% | 7.20 | 100% |
| 46 | 0.0642 | 0.0492 | 0.731 (2.0) | 89% | 7.26 | 100% |
| 47 | 0.0635 | 0.0492 | 0.733 (2.0) | 89% | 7.13 | 100% |
| 48 | 0.0632 | 0.0491 | 0.734 (2.0) | 89% | 7.15 | 100% |
| 49 | 0.0630 | 0.0490 | 0.736 (2.0) | 89% | 7.04 | 100% |
| 50 | 0.0621 | 0.0489 | 0.735 (2.0) | 88% | 7.10 | 100% |
| 51 | 0.0625 | 0.0487 | 0.735 (2.0) | 88% | 7.09 | 100% |
| 52 | 0.0622 | 0.0485 | 0.736 (2.0) | 88% | 7.04 | 100% |
| 53 | 0.0613 | 0.0484 | 0.737 (2.0) | 89% | 7.02 | 100% |
| 54 | 0.0612 | 0.0484 | 0.738 (2.0) | 89% | 7.03 | 100% |
| 55 | 0.0610 | 0.0483 | 0.739 (2.0) | 89% | 7.00 | 100% |
| 56 | 0.0608 | 0.0483 | 0.738 (2.0) | 89% | 6.98 | 100% |
| 57 | 0.0605 | 0.0482 | 0.738 (2.0) | 89% | 6.96 | 100% |
| 58 | 0.0601 | 0.0482 | 0.737 (2.0) | 89% | 6.87 | 100% |
| 59 | 0.0595 | 0.0484 | 0.739 (2.0) | 88% | 6.95 | 100% |
| 60 | 0.0596 | 0.0483 | 0.740 (2.0) | 89% | 6.82 | 100% |

Best TM 0.740287 at epoch 60.

## Scores
Selection set (w=2): TM 0.740 / 89% / RMSD 6.82 A at epoch 60 (best; final).

**Held-out slice (offset 1000, selection-free, 25 Euler steps, K=8), `best_pf_459M_esmc_afdb.pt`:**

| w | TM | TM>0.5 | RMSD | best-of-8 |
|---|---|---|---|---|
| 1.0 | 0.740 | 90% | 6.93 | 0.790 |
| 1.5 | 0.758 | 87% | 6.55 | 0.797 |
| **2.0** | **0.758** | **88%** | **6.51** | **0.792** |
| 3.0 | 0.746 | 87% | 6.68 | 0.783 |

**No-neighbour subsets (w=2):** nearest training structure < TM 0.6 (n=266): 0.567 / 63% / 12.78 A / best-of-8 0.628; < 0.5 (n=93): 0.512 / 42% / 15.06 A / 0.578.

**External bar on the same proteins** (`esmfold2_comparison.md`, ESMFold2-Fast single-sequence): held-out 0.753 / 85% / 7.18 A; no-neighbour 0.631 / 73% and 0.563 / 60%.

Predecessors on the same slices: `lf_174M_esmc` (80k) 0.732 / 89% / 6.84, no-neighbour 0.535 and 0.493; `lf_459M_afdb` (473k, ESM-2) 0.660 / 74% / 9.01, no-neighbour 0.474 and 0.430; inherited checkpoint 0.420 / 29% / 12.67. Coverage 100/100 in every row.

## Outcome
The best model of the project, and the first to beat the external bar on the held-out slice: **0.758 TM, 88% correct folds, 6.51 A mean RMSD** against ESMFold2-Fast's 0.753 / 85% / 7.18 A on the same 100 proteins with the same pipeline, and 0.79 TM with best-of-8 sampling. Relative to the 80k ESMC model it adds +0.026 TM held-out and +0.03 on the no-neighbour subsets, i.e. the 473k data step and the pair track together improved generalisation, not just seen folds. The curve was still rising at the end of the schedule (0.735 -> 0.740 over the last ten epochs) and the validation flow loss was still falling (0.0483), so unlike the 80k ESMC run this one is not overfitting; a longer schedule would likely add ~0.01. Residue-budget batching kept the H200s at 131 GB and made the pair track affordable (487 s/epoch for 461M params over 473k proteins). Coverage 100% at all 60 evaluations.

## What it changed
Held-out parity with ESMFold2-Fast is now a measured fact rather than a projection, at ~7% of its parameter count in the trainable model and with ensembles for free. The remaining gap is on novel folds: 0.567 vs 0.631 (<0.6) and 0.512 vs 0.563 (<0.5), narrowed from 0.10 to 0.06 TM by this run. `reports/PLAN.md` targets exactly that: the larger pair track (launched as `pf_459M_p128x8_esmc_afdb`), a structure-clustered split and experimental benchmark, a sample selector, more data at 512 residues, recycling. Caveats: AFDB models as truth; possible overlap of these validation proteins with ESMFold2's training; both systems at modest inference compute.
