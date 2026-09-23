# lf_174M_esmc — latent flow 174M conditioned on ESMC-6B (80k train, ran to completion)

**Job** 47955320, kempner_rtx (one RTX Pro 6000), 2026-09-23 10:25-17:55 (7.5 h). Ran all 100 epochs. The only change from `lf_174M` is the conditioner: ESMC-6B (biohub/ESMC-6B) last-layer embeddings, precomputed by `code/gate11_precompute_esmc.py` into `dataset_100k_esmc.h5` and read from a ragged host cache. Log `logs/latent_flow_47955320.out`.

## Configuration
`code/gate7_latent_flow.py --h5-path dataset_100k_esmc.h5`: DiT d768 x 16L x 12 heads (174.6M; d_cond 2560), CFG dropout 0.1, self-conditioning, EMA 0.999 with warm-up; one GPU, batch 128, lr 2e-4, warm-up 1000, cosine over 100 epochs; eval every 2 epochs on the 100 gate proteins at w=2 with 25 Euler steps. 268 s/epoch, peak GPU 31.3 GB. Train = 79,653 proteins; conditioning = ESMC-6B (6B params, 80 layers, 2560-d) single last layer, no layer mixing.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 0.9226 | 0.3101 | | | | |
| 2 | 0.2446 | 0.1799 | 0.354 (2.0) | 18% | 14.04 | 100% |
| 3 | 0.1741 | 0.1317 | | | | |
| 4 | 0.1454 | 0.1125 | 0.478 (2.0) | 46% | 11.25 | 100% |
| 5 | 0.1305 | 0.1010 | | | | |
| 6 | 0.1222 | 0.0941 | 0.586 (2.0) | 66% | 9.31 | 100% |
| 7 | 0.1152 | 0.0899 | | | | |
| 8 | 0.1115 | 0.0865 | 0.612 (2.0) | 72% | 8.63 | 100% |
| 9 | 0.1072 | 0.0839 | | | | |
| 10 | 0.1049 | 0.0816 | 0.625 (2.0) | 74% | 8.69 | 100% |
| 11 | 0.1024 | 0.0797 | | | | |
| 12 | 0.1005 | 0.0782 | 0.639 (2.0) | 77% | 8.49 | 100% |
| 13 | 0.0985 | 0.0773 | | | | |
| 14 | 0.0970 | 0.0762 | 0.646 (2.0) | 78% | 8.23 | 100% |
| 15 | 0.0953 | 0.0751 | | | | |
| 16 | 0.0947 | 0.0741 | 0.652 (2.0) | 82% | 8.33 | 100% |
| 17 | 0.0931 | 0.0733 | | | | |
| 18 | 0.0927 | 0.0726 | 0.657 (2.0) | 79% | 8.17 | 100% |
| 19 | 0.0909 | 0.0719 | | | | |
| 20 | 0.0897 | 0.0714 | 0.660 (2.0) | 80% | 8.11 | 100% |
| 21 | 0.0886 | 0.0706 | | | | |
| 22 | 0.0887 | 0.0702 | 0.665 (2.0) | 80% | 7.97 | 100% |
| 23 | 0.0861 | 0.0699 | | | | |
| 24 | 0.0864 | 0.0693 | 0.669 (2.0) | 80% | 7.84 | 100% |
| 25 | 0.0850 | 0.0691 | | | | |
| 26 | 0.0850 | 0.0684 | 0.670 (2.0) | 80% | 7.79 | 100% |
| 27 | 0.0834 | 0.0679 | | | | |
| 28 | 0.0834 | 0.0676 | 0.671 (2.0) | 83% | 7.73 | 100% |
| 29 | 0.0823 | 0.0673 | | | | |
| 30 | 0.0817 | 0.0671 | 0.674 (2.0) | 84% | 7.56 | 100% |
| 31 | 0.0805 | 0.0669 | | | | |
| 32 | 0.0803 | 0.0664 | 0.674 (2.0) | 83% | 7.59 | 100% |
| 33 | 0.0795 | 0.0662 | | | | |
| 34 | 0.0789 | 0.0666 | 0.676 (2.0) | 83% | 7.46 | 100% |
| 35 | 0.0780 | 0.0663 | | | | |
| 36 | 0.0780 | 0.0659 | 0.684 (2.0) | 83% | 7.17 | 100% |
| 37 | 0.0767 | 0.0657 | | | | |
| 38 | 0.0767 | 0.0653 | 0.687 (2.0) | 83% | 7.35 | 100% |
| 39 | 0.0753 | 0.0653 | | | | |
| 40 | 0.0751 | 0.0652 | 0.691 (2.0) | 85% | 7.46 | 100% |
| 41 | 0.0739 | 0.0652 | | | | |
| 42 | 0.0736 | 0.0651 | 0.692 (2.0) | 84% | 7.54 | 100% |
| 43 | 0.0731 | 0.0651 | | | | |
| 44 | 0.0724 | 0.0653 | 0.691 (2.0) | 83% | 7.54 | 100% |
| 45 | 0.0715 | 0.0656 | | | | |
| 46 | 0.0714 | 0.0653 | 0.693 (2.0) | 84% | 7.54 | 100% |
| 47 | 0.0703 | 0.0653 | | | | |
| 48 | 0.0701 | 0.0654 | 0.695 (2.0) | 84% | 7.55 | 100% |
| 49 | 0.0689 | 0.0654 | | | | |
| 50 | 0.0690 | 0.0655 | 0.699 (2.0) | 84% | 7.45 | 100% |
| 51 | 0.0677 | 0.0657 | | | | |
| 52 | 0.0673 | 0.0658 | 0.697 (2.0) | 86% | 7.38 | 100% |
| 53 | 0.0659 | 0.0662 | | | | |
| 54 | 0.0657 | 0.0666 | 0.697 (2.0) | 86% | 7.66 | 100% |
| 55 | 0.0652 | 0.0672 | | | | |
| 56 | 0.0647 | 0.0674 | 0.697 (2.0) | 87% | 7.52 | 100% |
| 57 | 0.0635 | 0.0679 | | | | |
| 58 | 0.0637 | 0.0684 | 0.696 (2.0) | 88% | 7.43 | 100% |
| 59 | 0.0625 | 0.0688 | | | | |
| 60 | 0.0623 | 0.0691 | 0.698 (2.0) | 86% | 7.51 | 100% |
| 61 | 0.0614 | 0.0695 | | | | |
| 62 | 0.0609 | 0.0700 | 0.704 (2.0) | 86% | 7.24 | 100% |
| 63 | 0.0597 | 0.0705 | | | | |
| 64 | 0.0596 | 0.0713 | 0.707 (2.0) | 87% | 7.15 | 100% |
| 65 | 0.0589 | 0.0717 | | | | |
| 66 | 0.0587 | 0.0724 | 0.707 (2.0) | 88% | 7.30 | 100% |
| 67 | 0.0573 | 0.0728 | | | | |
| 68 | 0.0572 | 0.0732 | 0.704 (2.0) | 89% | 7.40 | 100% |
| 69 | 0.0564 | 0.0736 | | | | |
| 70 | 0.0564 | 0.0742 | 0.704 (2.0) | 87% | 7.19 | 100% |
| 71 | 0.0554 | 0.0749 | | | | |
| 72 | 0.0551 | 0.0759 | 0.703 (2.0) | 87% | 7.25 | 100% |
| 73 | 0.0539 | 0.0763 | | | | |
| 74 | 0.0543 | 0.0770 | 0.707 (2.0) | 88% | 7.34 | 100% |
| 75 | 0.0534 | 0.0777 | | | | |
| 76 | 0.0530 | 0.0786 | 0.705 (2.0) | 88% | 7.10 | 100% |
| 77 | 0.0522 | 0.0792 | | | | |
| 78 | 0.0524 | 0.0798 | 0.705 (2.0) | 86% | 7.16 | 100% |
| 79 | 0.0514 | 0.0803 | | | | |
| 80 | 0.0512 | 0.0810 | 0.706 (2.0) | 84% | 7.15 | 100% |
| 81 | 0.0504 | 0.0816 | | | | |
| 82 | 0.0506 | 0.0823 | 0.706 (2.0) | 85% | 7.24 | 100% |
| 83 | 0.0494 | 0.0829 | | | | |
| 84 | 0.0496 | 0.0834 | 0.707 (2.0) | 85% | 7.16 | 100% |
| 85 | 0.0489 | 0.0840 | | | | |
| 86 | 0.0492 | 0.0845 | 0.705 (2.0) | 86% | 7.24 | 100% |
| 87 | 0.0485 | 0.0851 | | | | |
| 88 | 0.0488 | 0.0858 | 0.707 (2.0) | 84% | 7.21 | 100% |
| 89 | 0.0485 | 0.0863 | | | | |
| 90 | 0.0484 | 0.0868 | 0.706 (2.0) | 84% | 7.21 | 100% |
| 91 | 0.0478 | 0.0872 | | | | |
| 92 | 0.0479 | 0.0876 | 0.704 (2.0) | 84% | 7.30 | 100% |
| 93 | 0.0473 | 0.0879 | | | | |
| 94 | 0.0481 | 0.0883 | 0.701 (2.0) | 85% | 7.39 | 100% |
| 95 | 0.0476 | 0.0885 | | | | |
| 96 | 0.0475 | 0.0887 | 0.701 (2.0) | 83% | 7.37 | 100% |
| 97 | 0.0475 | 0.0888 | | | | |
| 98 | 0.0474 | 0.0889 | 0.698 (2.0) | 83% | 7.38 | 100% |
| 99 | 0.0472 | 0.0890 | | | | |
| 100 | 0.0475 | 0.0890 | 0.699 (2.0) | 83% | 7.38 | 100% |

Best TM 0.7072529999999999 at epoch 88.

## Scores
Selection set (w=2): best TM 0.707 / 84% / RMSD 7.21 A at epoch 88; final (epoch 100) 0.699 / 83% / 7.38 A.

**Held-out slice (offset 1000, selection-free, 25 Euler steps, K=8), `best_lf_174M_esmc.pt` (epoch 88):**

| w | TM | TM>0.5 | RMSD | best-of-8 |
|---|---|---|---|---|
| 1.0 | 0.723 | 89% | 7.11 | 0.763 |
| **1.5** | **0.732** | **89%** | **6.84** | **0.763** |
| 2.0 | 0.730 | 89% | 7.10 | 0.761 |
| 3.0 | 0.714 | 87% | 7.75 | 0.749 |

**No-neighbour subsets (w=2):** nearest training structure < TM 0.6 (n=266): 0.535 / 62% / 12.89 A / best-of-8 0.584; < 0.5 (n=93): 0.493 / 44% / 14.84 A / 0.544.

Same slices, ESM-2 twin (`lf_174M`, epoch 78): held-out 0.609 / 69% / 9.92 A; no-neighbour 0.439 / 29% and 0.396 / 14%. ESM-2 473k 459M model: 0.660 / 74% / 9.01; 0.474 / 41% and 0.430 / 24%. Coverage 100/100 in every row.

## Outcome
Swapping the frozen conditioner from ESM-2 650M to ESMC-6B, with nothing else changed, moved the 80k model from 0.609 to **0.732 TM held-out, 89% correct folds and 6.84 A mean RMSD** (best-of-8 0.763). The gain is at least as large on proteins with no close training relative (+0.10 TM, +33 points of correct folds on the <0.6 subset), so it is generalisation, not memorisation. The curve rose fast (0.625 at epoch 10, 0.691 at 40) and kept improving on TM to epoch 88 even though the validation flow loss bottomed at epoch ~40 (0.065) and then rose to 0.089 while training loss fell to 0.048: with ESMC the 174M model overfits the flow objective on 80k proteins, and TM-based selection (CLAUDE.md bug 3) is what kept the right checkpoint. The guidance optimum moved to w=1.5-2. Coverage 100% at all 50 evaluations.

## What it changed
The conditioner is the largest single lever found in this project, worth more than 6x data or 3x model size, and it is cheap: embeddings are precomputed once (30 min, 61 GB for the 100k set). It is also generalisation-positive, unlike the data step. This result set the design of the H200 run `pf_459M_esmc_afdb` (ESMC-6B + pair track + 473k). Overfitting of the flow loss at 80k says the ESMC-conditioned model should be trained on the larger set, which that run does. ESMFold2 is built on the same ESMC-6B and is the external comparison to make next.
