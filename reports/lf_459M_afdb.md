# lf_459M_afdb — latent flow 459M on 473k AFDB proteins, online ESM-2 (the data scale-up)

**Jobs** 47864287 (epochs 1-20, 6 x RTX Pro 6000, 2026-09-22 22:00-01:47, label `lf_459M_afdb_rtx`) then 47902962 (epochs 20-63, 2 H200 nodes = 8 GPUs, 2026-09-23 01:52-06:50, resumed from the RTX checkpoint). Early-stopped at epoch 63 (patience 8 after the best at epoch 55). Logs `logs/latent_flow_ddp_47864287.out`, `logs/lf_multinode_47902962.out`.

## Configuration
`code/gate7_latent_flow.py --esm online`: DiT d1024 x 24L x 16 heads (459.2M), conditioned on frozen ESM-2 650M run per batch (fp32 weights under bf16 autocast) through a learned all-layer mix (which settled at ~0.8 on layer 33), CFG dropout 0.1, self-conditioning, EMA 0.999 with warm-up, AdamW lr 4e-4, warm-up 1000, cosine; eval every epoch on the 100 gate proteins at w=2 with 25 Euler steps. **Training data: 473,184 proteins** = the 79,653-protein train split of `dataset_100k.h5` plus 393,531 new AFDB entries from the Gate 8 build (`reports/afdb_build.md`), sequences + latents in RAM. Validation unchanged. RTX phase: batch 128/GPU x 6 = 768, 958 s/epoch, 69.6 GB/GPU. H200 phase: batch 192/GPU x 8 = 1536, 354 s/epoch, 107.1 GB/GPU; `--epochs 100` so the cosine stayed continuous in step space across the hand-over.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 0.6858 | 0.2184 | 0.304 (2.0) | 1% | 15.60 | 100% |
| 2 | 0.1757 | 0.1282 | 0.481 (2.0) | 43% | 11.44 | 100% |
| 3 | 0.1344 | 0.1084 | 0.539 (2.0) | 54% | 10.40 | 100% |
| 4 | 0.1213 | 0.1004 | 0.554 (2.0) | 58% | 10.00 | 100% |
| 5 | 0.1150 | 0.0951 | 0.571 (2.0) | 58% | 9.81 | 100% |
| 6 | 0.1099 | 0.0921 | 0.580 (2.0) | 60% | 9.65 | 100% |
| 7 | 0.1064 | 0.0895 | 0.589 (2.0) | 63% | 9.68 | 100% |
| 8 | 0.1037 | 0.0875 | 0.590 (2.0) | 63% | 9.77 | 100% |
| 9 | 0.1021 | 0.0854 | 0.597 (2.0) | 66% | 9.64 | 100% |
| 10 | 0.0994 | 0.0839 | 0.604 (2.0) | 67% | 9.44 | 100% |
| 11 | 0.0984 | 0.0829 | 0.607 (2.0) | 67% | 9.42 | 100% |
| 12 | 0.0969 | 0.0819 | 0.609 (2.0) | 68% | 9.33 | 100% |
| 13 | 0.0954 | 0.0807 | 0.612 (2.0) | 71% | 9.45 | 100% |
| 14 | 0.0945 | 0.0796 | 0.613 (2.0) | 70% | 9.48 | 100% |
| 15 | 0.0931 | 0.0787 | 0.613 (2.0) | 70% | 9.46 | 100% |
| 16 | 0.0924 | 0.0782 | 0.615 (2.0) | 71% | 9.37 | 100% |
| 17 | 0.0912 | 0.0777 | 0.620 (2.0) | 75% | 9.28 | 100% |
| 18 | 0.0906 | 0.0772 | 0.621 (2.0) | 71% | 9.27 | 100% |
| 19 | 0.0896 | 0.0766 | 0.622 (2.0) | 71% | 9.14 | 100% |
| 20 | 0.0884 | 0.0760 | 0.626 (2.0) | 75% | 9.18 | 100% |
| 21 | 0.0850 | 0.0752 | 0.627 (2.0) | 75% | 9.19 | 100% |
| 22 | 0.0848 | 0.0746 | 0.631 (2.0) | 79% | 9.10 | 100% |
| 23 | 0.0837 | 0.0741 | 0.627 (2.0) | 77% | 9.31 | 100% |
| 24 | 0.0836 | 0.0737 | 0.624 (2.0) | 76% | 9.31 | 100% |
| 25 | 0.0828 | 0.0732 | 0.630 (2.0) | 75% | 9.21 | 100% |
| 26 | 0.0824 | 0.0729 | 0.634 (2.0) | 77% | 9.10 | 100% |
| 27 | 0.0819 | 0.0726 | 0.628 (2.0) | 75% | 9.00 | 100% |
| 28 | 0.0817 | 0.0723 | 0.631 (2.0) | 76% | 9.05 | 100% |
| 29 | 0.0806 | 0.0721 | 0.632 (2.0) | 75% | 9.17 | 100% |
| 30 | 0.0805 | 0.0720 | 0.635 (2.0) | 76% | 9.16 | 100% |
| 31 | 0.0803 | 0.0718 | 0.636 (2.0) | 77% | 9.14 | 100% |
| 32 | 0.0797 | 0.0717 | 0.638 (2.0) | 78% | 9.16 | 100% |
| 33 | 0.0793 | 0.0713 | 0.639 (2.0) | 79% | 9.06 | 100% |
| 34 | 0.0790 | 0.0712 | 0.642 (2.0) | 80% | 8.97 | 100% |
| 35 | 0.0787 | 0.0710 | 0.643 (2.0) | 79% | 8.94 | 100% |
| 36 | 0.0784 | 0.0708 | 0.644 (2.0) | 79% | 8.92 | 100% |
| 37 | 0.0777 | 0.0706 | 0.646 (2.0) | 79% | 8.89 | 100% |
| 38 | 0.0773 | 0.0704 | 0.646 (2.0) | 79% | 8.92 | 100% |
| 39 | 0.0770 | 0.0704 | 0.646 (2.0) | 80% | 8.86 | 100% |
| 40 | 0.0766 | 0.0703 | 0.644 (2.0) | 80% | 8.86 | 100% |
| 41 | 0.0764 | 0.0701 | 0.644 (2.0) | 80% | 8.81 | 100% |
| 42 | 0.0761 | 0.0702 | 0.647 (2.0) | 79% | 8.75 | 100% |
| 43 | 0.0759 | 0.0702 | 0.640 (2.0) | 77% | 8.82 | 100% |
| 44 | 0.0758 | 0.0701 | 0.645 (2.0) | 79% | 8.80 | 100% |
| 45 | 0.0755 | 0.0700 | 0.643 (2.0) | 78% | 8.85 | 100% |
| 46 | 0.0746 | 0.0699 | 0.642 (2.0) | 79% | 8.87 | 100% |
| 47 | 0.0746 | 0.0699 | 0.644 (2.0) | 79% | 8.80 | 100% |
| 48 | 0.0744 | 0.0698 | 0.647 (2.0) | 79% | 8.85 | 100% |
| 49 | 0.0739 | 0.0698 | 0.643 (2.0) | 78% | 8.94 | 100% |
| 50 | 0.0738 | 0.0697 | 0.645 (2.0) | 81% | 8.77 | 100% |
| 51 | 0.0729 | 0.0697 | 0.642 (2.0) | 80% | 8.92 | 100% |
| 52 | 0.0732 | 0.0697 | 0.644 (2.0) | 81% | 8.80 | 100% |
| 53 | 0.0727 | 0.0698 | 0.650 (2.0) | 81% | 8.70 | 100% |
| 54 | 0.0724 | 0.0698 | 0.653 (2.0) | 82% | 8.74 | 100% |
| 55 | 0.0719 | 0.0698 | 0.656 (2.0) | 82% | 8.71 | 100% |
| 56 | 0.0717 | 0.0698 | 0.654 (2.0) | 81% | 8.68 | 100% |
| 57 | 0.0715 | 0.0699 | 0.648 (2.0) | 80% | 8.81 | 100% |
| 58 | 0.0711 | 0.0698 | 0.651 (2.0) | 82% | 8.69 | 100% |
| 59 | 0.0708 | 0.0699 | 0.648 (2.0) | 81% | 8.73 | 100% |
| 60 | 0.0707 | 0.0700 | 0.650 (2.0) | 80% | 8.63 | 100% |
| 61 | 0.0704 | 0.0701 | 0.652 (2.0) | 79% | 8.70 | 100% |
| 62 | 0.0703 | 0.0702 | 0.651 (2.0) | 80% | 8.76 | 100% |
| 63 | 0.0698 | 0.0703 | 0.651 (2.0) | 80% | 8.78 | 100% |

Best TM 0.655711 at epoch 55.

## Scores
Selection set (gate proteins, w=2): best TM 0.656 / 82% / RMSD 8.71 A at epoch 55.

**Held-out slice (offset 1000, selection-free, `gate7_score.py`, 25 Euler steps, K=8):**

| w | TM | TM>0.5 | RMSD | best-of-8 |
|---|---|---|---|---|
| 1.0 | 0.627 | 69% | 9.78 | 0.698 |
| 1.5 | 0.656 | 72% | 9.33 | 0.717 |
| **2.0** | **0.660** | **74%** | **9.01** | **0.716** |
| 3.0 | 0.641 | 72% | 9.45 | 0.702 |

Coverage 100/100 in every row. The epoch-42 checkpoint scored 0.657 / 74% / 8.87 A on the same slice, so the model was at its plateau from epoch ~40.

Same slice, for reference: inherited checkpoint 0.420 / 29% / 12.67 A; best FAPE head 0.510 / 53% / 11.62 A; 80k flow reference (`lf_459M_ddp6`) 0.609 / 71% / 10.06 A, best-of-8 0.668.

## Outcome
Six times more training data moved the same 459M model from the 80k ceiling of 0.609 / 71% / 10.06 A to **0.660 / 74% / 9.01 A** held-out, with best-of-8 sampling reaching 0.716. The curve was indistinguishable from the 80k run per optimizer step until ~0.61 (epoch 14), then kept rising where the 80k runs had stopped: 0.626 at epoch 20, 0.644 at 40, 0.656 at 55. Guidance w=2 remains the optimum (w=1.5 is within noise). From epoch 55 the validation flow loss began to rise (0.0697 -> 0.0703) while the training loss kept falling, the first sign of overfitting seen in this project: at 473k proteins a 459M model is starting to be data-limited again, not capacity-limited. The 8-GPU H200 phase ran 2.7x faster per epoch than 6 RTX GPUs. Coverage 100% at all 63 evaluations.

## What it changed
This is the best model of the project: held-out mean TM 0.66, three quarters of proteins folded correctly, mean RMSD 9.0 A against ESMFold's 4.3 A and the inherited 12.7 A on this slice. It confirms the data-limit reading from the 80k plateau and sets the next step: more data (the remaining AFDB clusters beyond the Genie2 index are ~4x more) rather than a larger head. Two things must be checked before any claim is made outside the lab: (1) **leakage** — the new training proteins were not structurally clustered against the validation/test splits, so near-duplicates of held-out proteins may have entered training (audit in progress, `code/gate9_leakage_audit.py`); (2) evaluation on experimental structures (CAMEO/CASP) rather than AFDB models.
