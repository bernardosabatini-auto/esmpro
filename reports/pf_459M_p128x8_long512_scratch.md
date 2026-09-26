# pf_459M_p128x8_long512_scratch — new main line: 512 window from scratch with the full recipe; best model on every set

**Jobs** 48398186 (epochs 1-4, died of CUDA OOM in epoch 5 at budget 36) and 48437245 (auto-resumed from the epoch-4 checkpoint at budget 32; epochs 5-30), 2 nodes x 4 H200, 2026-09-25 05:35 to 2026-09-26 08:27 (26.5 h incl. the restart). Early-stopped at epoch 30 on the long-val TM (patience 8); best epoch 22. Scoring 48437283 (long, 62 min with chunked Foldseek) and 48437284 (short, 25 min). Logs `logs/lf_multinode_48398186.out`, `logs/lf_multinode_48437245.out`.

## Configuration
`code/gate10_pair_flow.py` at `ESM_PROAE_MAX_LEN=512`: DiT 1024 x 24 x 16 heads + fused pair track 128 x 8 (464M), ESMC-6B stored embeddings; **from scratch**; **repeated batching R = 4**, **late-t resampling** (logit-normal 0.8/1.7 + 2 % uniform), residue budget 36 then 32 x 256^2 per GPU (peak 130 GB), lr 4e-4, warm-up 1000, cosine over 40 epochs, CFG dropout 0.1, self-conditioning, EMA; train = 473k short + 159.8k AFDB long + 13.1k PDB long = 646k proteins (80,762 per rank); selection on 100 of the 1,000 held-out long proteins every epoch (w = 2, 25 steps). 2,860-3,000 s/epoch.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 0.3599 | 0.1824 | 0.518 (2.0) | 49% | 19.71 | 100% |
| 2 | 0.1274 | 0.1523 | 0.550 (2.0) | 53% | 18.93 | 100% |
| 3 | 0.1121 | 0.1341 | 0.566 (2.0) | 61% | 17.43 | 100% |
| 4 | 0.1050 | 0.1292 | 0.569 (2.0) | 60% | 17.08 | 100% |
| 5 | 0.1010 | 0.0879 | 0.573 (2.0) | 60% | 16.73 | 100% |
| 6 | 0.0979 | 0.0854 | 0.577 (2.0) | 62% | 16.66 | 100% |
| 7 | 0.0957 | 0.0836 | 0.582 (2.0) | 62% | 16.68 | 100% |
| 8 | 0.0935 | 0.0820 | 0.581 (2.0) | 63% | 16.37 | 100% |
| 9 | 0.0920 | 0.0808 | 0.582 (2.0) | 65% | 16.66 | 100% |
| 10 | 0.0904 | 0.0798 | 0.584 (2.0) | 64% | 16.65 | 100% |
| 11 | 0.0894 | 0.0791 | 0.586 (2.0) | 66% | 16.51 | 100% |
| 12 | 0.0878 | 0.0783 | 0.589 (2.0) | 65% | 16.40 | 100% |
| 13 | 0.0871 | 0.0777 | 0.585 (2.0) | 63% | 16.41 | 100% |
| 14 | 0.0863 | 0.0771 | 0.590 (2.0) | 67% | 16.17 | 100% |
| 15 | 0.0855 | 0.0768 | 0.589 (2.0) | 63% | 16.23 | 100% |
| 16 | 0.0846 | 0.0761 | 0.589 (2.0) | 64% | 16.26 | 100% |
| 17 | 0.0836 | 0.0755 | 0.595 (2.0) | 66% | 15.85 | 100% |
| 18 | 0.0831 | 0.0752 | 0.601 (2.0) | 65% | 15.61 | 100% |
| 19 | 0.0827 | 0.0748 | 0.598 (2.0) | 66% | 15.50 | 100% |
| 20 | 0.0823 | 0.0745 | 0.597 (2.0) | 64% | 15.62 | 100% |
| 21 | 0.0815 | 0.0743 | 0.598 (2.0) | 64% | 15.39 | 100% |
| 22 | 0.0809 | 0.0742 | 0.605 (2.0) | 67% | 15.62 | 100% |
| 23 | 0.0803 | 0.0738 | 0.600 (2.0) | 64% | 15.78 | 100% |
| 24 | 0.0796 | 0.0737 | 0.600 (2.0) | 64% | 15.45 | 100% |
| 25 | 0.0788 | 0.0734 | 0.602 (2.0) | 66% | 15.42 | 100% |
| 26 | 0.0787 | 0.0733 | 0.602 (2.0) | 64% | 15.71 | 100% |
| 27 | 0.0783 | 0.0732 | 0.595 (2.0) | 64% | 15.68 | 100% |
| 28 | 0.0775 | 0.0730 | 0.598 (2.0) | 64% | 15.70 | 100% |
| 29 | 0.0772 | 0.0728 | 0.601 (2.0) | 64% | 15.16 | 100% |
| 30 | 0.0768 | 0.0727 | 0.602 (2.0) | 66% | 15.39 | 100% |

Best TM 0.6045729999999999 at epoch 22.

## Scores (final EMA weights `last_pf_459M_p128x8_long512_scratch.ckpt`, w = 2 unless noted, 50 steps, coverage 100 % everywhere)
| set | this run | previous best (512 fine-tune) | ESMFold2-Fast, same inputs |
|---|---|---|---|
| long val, all 1,000 (257-512 aa; K=4) | **0.591 / 67 % / 17.0 A**, bo4 0.624 | 0.572 / 63 % / 18.7 A | 0.644 / 70 % / 16.7 A (first 200) |
| CASP <= 512, 109 domains | **0.728 / 78 % / 7.54 A**, bo4 0.757 | 0.713 / 79 % / 8.66 A | 0.751 / 82 % / 8.22 A |
| CASP 257-512, 29 domains | **0.763 / 90 % / 8.90 A**, bo4 0.784 | 0.736 / 86 % / 10.6 A | 0.793 / 93 % / 9.31 A |
| held-out short (n=100; K=8) | **0.785 / 91 % / 5.72 A** (w=1.5), bo8 0.821 | 0.770 / 89 % / 6.04 A | 0.753 / 85 % / 7.18 A |
| no-neighbour < 0.6 (n=266) | **0.591 / 66 % / 10.84 A** | 0.582 / 65 % | 0.631 / 73 % |
| no-neighbour < 0.5 (n=93) | **0.541 / 49 % / 12.08 A** | 0.524 / 51 % | 0.563 / 60 % |
| CASP <= 256, 80 domains | **0.715 / 74 % / 6.97 A**, bo8 0.750 | 0.710 / 72 % / 7.37 A | 0.741 / 79 % / 7.15 A |

## Outcome
The full recipe trained from scratch at the 512 window beats every previous checkpoint on every set: +0.019 on the 1,000 long validation proteins, +0.027 on the long CASP domains, +0.015 on short held-out (0.785, the first time above 0.78; ESMFold2-Fast 0.753 there), +0.009/+0.017 on the novel-fold subsets, and the best short-CASP number (0.715) with the best RMSD (6.97 A). Against ESMFold2-Fast the remaining gaps are 0.026 on short CASP, 0.023 on all CASP <= 512, 0.030 on long CASP and ~0.05 on AFDB long validation, down from 0.038 / 0.038 / 0.057 / 0.08 a day earlier. Coverage 100 % at all 30 evaluations and all scoring. What did it: no single change. Relative to `pf_459M_p128x8_esmc_afdb` (0.758 held-out) this run adds the long data (+173k proteins, 512 window), repeated batching (same quality, 2.9x cheaper), late-t resampling (+0.005-0.01), and a from-scratch schedule at the long window. The long-val curve still flattened at ~0.60 from epoch 20, so the long-protein ceiling of this capacity is real; the fine-tune (0.572) versus from-scratch (0.591) difference says the window matters for how the pair track is learned, not only the data.

## What it changed
`last_pf_459M_p128x8_long512_scratch.ckpt` is the project's best model for both length ranges. Recipe confirmed for the next main line: 512 window from scratch, R copies, late-t, fused pair, distributed evaluation. Next lever, per the capacity profile in efficiency.md: the 840M trunk (1.5x cost per residue) on the enlarged data (627k extra short AFDB v6 representatives building now), which is where SimpleFold's scaling curves and our long-protein ceiling both point. A second-cycle fine-tune of these weights at R = 8 is running meanwhile.
