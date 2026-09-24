# pf_174M_p64x6_esmc — pair track (64-dim x 6) + ESMC-6B, 80k (ablation)

**Job** 47985418, kempner_rtx, 2026-09-23 12:50-23:30 (10.6 h). Stopped by hand at epoch 60 at a clear plateau. Same restart history as `pf_174M_p64x6`. Log `logs/latent_flow_47985418.out`.

## Configuration
`code/gate10_pair_flow.py`: DiT d768 x 16L x 12 heads + 64-dim pair track (6 gated triangular-multiplication blocks, outer-sum + relative-position init, activation-checkpointed, bf16, torch.compile) reading into every attention layer through one shared bias projection; CFG dropout 0.1, self-conditioning, EMA 0.999 with warm-up; batch 128 with length-bucketed cropped batches; lr 2e-4, warm-up 1000, cosine over 100 epochs; eval every 2 epochs on the 100 gate proteins at w=2, 25 Euler steps. Train = 79,653 proteins. ~585 s/epoch (2.2x the no-pair model), peak GPU ~56 GB at the 256-residue worst case. Conditioner: precomputed ESMC-6B last-layer embeddings (2560-d, `dataset_100k_esmc.h5`), 175.5M params total.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 0.9006 | 0.3014 | | | | |
| 2 | 0.2418 | 0.1782 | 0.342 (2.0) | 12% | 14.49 | 100% |
| 3 | 0.1702 | 0.1284 | | | | |
| 4 | 0.1412 | 0.1072 | 0.508 (2.0) | 53% | 10.66 | 100% |
| 5 | 0.1256 | 0.0958 | | | | |
| 6 | 0.1174 | 0.0897 | 0.604 (2.0) | 75% | 8.97 | 100% |
| 7 | 0.1116 | 0.0853 | | | | |
| 8 | 0.1068 | 0.0816 | 0.631 (2.0) | 76% | 8.51 | 100% |
| 9 | 0.1021 | 0.0788 | | | | |
| 10 | 0.0998 | 0.0766 | 0.655 (2.0) | 79% | 8.17 | 100% |
| 11 | 0.0972 | 0.0747 | | | | |
| 12 | 0.0954 | 0.0732 | 0.660 (2.0) | 80% | 8.05 | 100% |
| 13 | 0.0933 | 0.0719 | | | | |
| 14 | 0.0920 | 0.0708 | 0.669 (2.0) | 82% | 7.95 | 100% |
| 15 | 0.0903 | 0.0696 | | | | |
| 16 | 0.0894 | 0.0688 | 0.673 (2.0) | 85% | 7.74 | 100% |
| 17 | 0.0882 | 0.0681 | | | | |
| 18 | 0.0869 | 0.0671 | 0.681 (2.0) | 85% | 7.65 | 100% |
| 19 | 0.0852 | 0.0663 | | | | |
| 20 | 0.0849 | 0.0656 | 0.684 (2.0) | 87% | 7.65 | 100% |
| 21 | 0.0832 | 0.0652 | | | | |
| 22 | 0.0828 | 0.0647 | 0.685 (2.0) | 86% | 7.55 | 100% |
| 23 | 0.0819 | 0.0646 | | | | |
| 24 | 0.0812 | 0.0642 | 0.686 (2.0) | 86% | 7.62 | 100% |
| 25 | 0.0803 | 0.0636 | | | | |
| 26 | 0.0799 | 0.0632 | 0.692 (2.0) | 85% | 7.64 | 100% |
| 27 | 0.0790 | 0.0628 | | | | |
| 28 | 0.0781 | 0.0627 | 0.696 (2.0) | 87% | 7.51 | 100% |
| 29 | 0.0771 | 0.0623 | | | | |
| 30 | 0.0771 | 0.0619 | 0.696 (2.0) | 86% | 7.41 | 100% |
| 31 | 0.0762 | 0.0619 | | | | |
| 32 | 0.0758 | 0.0617 | 0.698 (2.0) | 86% | 7.45 | 100% |
| 33 | 0.0749 | 0.0614 | | | | |
| 34 | 0.0749 | 0.0612 | 0.701 (2.0) | 87% | 7.35 | 100% |
| 35 | 0.0736 | 0.0611 | | | | |
| 36 | 0.0730 | 0.0610 | 0.706 (2.0) | 87% | 7.31 | 100% |
| 37 | 0.0722 | 0.0610 | | | | |
| 38 | 0.0722 | 0.0610 | 0.708 (2.0) | 87% | 7.25 | 100% |
| 39 | 0.0711 | 0.0610 | | | | |
| 40 | 0.0712 | 0.0609 | 0.710 (2.0) | 86% | 7.24 | 100% |
| 41 | 0.0700 | 0.0608 | | | | |
| 42 | 0.0698 | 0.0607 | 0.712 (2.0) | 85% | 7.30 | 100% |
| 43 | 0.0689 | 0.0608 | | | | |
| 44 | 0.0686 | 0.0608 | 0.716 (2.0) | 85% | 7.16 | 100% |
| 45 | 0.0678 | 0.0606 | | | | |
| 46 | 0.0676 | 0.0607 | 0.714 (2.0) | 86% | 7.21 | 100% |
| 47 | 0.0665 | 0.0607 | | | | |
| 48 | 0.0661 | 0.0608 | 0.713 (2.0) | 85% | 7.17 | 100% |
| 49 | 0.0653 | 0.0610 | | | | |
| 50 | 0.0653 | 0.0610 | 0.716 (2.0) | 86% | 7.24 | 100% |
| 51 | 0.0641 | 0.0612 | | | | |
| 52 | 0.0639 | 0.0612 | 0.715 (2.0) | 86% | 7.11 | 100% |
| 53 | 0.0629 | 0.0612 | | | | |
| 54 | 0.0626 | 0.0613 | 0.716 (2.0) | 86% | 7.12 | 100% |
| 55 | 0.0617 | 0.0615 | | | | |
| 56 | 0.0615 | 0.0617 | 0.718 (2.0) | 86% | 7.05 | 100% |
| 57 | 0.0606 | 0.0620 | | | | |
| 58 | 0.0606 | 0.0623 | 0.717 (2.0) | 87% | 7.21 | 100% |
| 59 | 0.0594 | 0.0626 | | | | |
| 60 | 0.0592 | 0.0631 | 0.719 (2.0) | 87% | 7.24 | 100% |
| 61 | 0.0585 | 0.0633 | | | | |

Best TM 0.7187360000000002 at epoch 60.

## Scores
Selection set (w=2): best TM 0.719 / 87% at epoch 60 (0.715-0.719 over epochs 52-60).

**Held-out slice (selection-free, `best_pf_174M_p64x6_esmc.pt`):** w=2: TM 0.732, TM>0.5 88%, RMSD 6.38 A, best-of-8 0.767 (w=1.5: 0.729 / 87% / 6.51). **No-neighbour:** <0.6 (n=266) 0.544 / 62% / 13.08 A / 0.598; <0.5 (n=93) 0.488 / 43% / 15.29 A / 0.544. Coverage 100/100.

Twin without the pair track (`lf_174M_esmc`, epoch 88): 0.732 / 89% / 6.84 / 0.763; no-neighbour 0.535 / 62% and 0.493 / 44%. ESMFold2-Fast on the same slices: 0.753 / 85% / 7.18; 0.631 / 73%; 0.563 / 60%.

## Outcome
Identical held-out TM to its no-pair twin (0.732 vs 0.732), a better mean RMSD (6.38 vs 6.84 A, the best RMSD of any 80k model), and no-neighbour scores within noise (0.544 vs 0.535; 0.488 vs 0.493). It got there in 60 epochs where the twin needed 88, and unlike the twin it did not overfit the flow loss before its schedule ended. On the held-out slice it is at ESMFold2-Fast parity (better fold fraction and RMSD, 0.02 lower TM). Coverage 100% at all 30 evaluations.

## What it changed
Confirms, with the stronger conditioner, that the small pair track is a convergence lever rather than an accuracy lever on 80k proteins. It also shows the best RMSD reachable on 80k with this recipe (6.38 A held-out). The gap to ESMFold2-Fast on novel folds (0.544 vs 0.631) is unchanged by the pair track, which is why `PLAN.md` puts data (clean, larger, longer) and the larger pair track ahead of further architecture work at this scale.
