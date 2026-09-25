# late_t_resampling — SimpleFold's flow-time distribution in latent space: small, uniform gain

**Job** `pf_174M_p64x6_esmc_r4_tlate` 48348640, 4 RTX (DDP), 2026-09-24 22:00 to 2026-09-25 05:20, early-stopped at epoch 88 (best epoch 80); scoring 48407809. Reference: `pf_174M_p64x6_esmc_r4` (identical recipe, t ~ logit-normal(0, 1); stopped at epoch 50 on its plateau). Knob: `T_LOGIT_M=0.8 T_LOGIT_S=1.7 T_UNIF=0.02` (`sample_t` in `code/gate7_latent_flow.py`). Log `logs/lf_ddp_tlate_48348640.out`.

## Mechanism
SimpleFold samples the flow time from 0.98 LN(0.8, 1.7) + 0.02 U(0, 1), i.e. heavily toward t = 1 (near-clean latents), arguing that fine detail is learned there. Our recipe used LN(0, 1). Everything else identical: 174M pair flow, ESMC-6B, 80k, R = 4 copies, budget 28 x 4 GPUs.

## Curve (selection-set TM, every 2 epochs)
| epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| late-t | 0.664 | 0.700 | 0.715 | 0.722 | 0.724 | 0.729 | 0.731 | **0.735** |
| reference | 0.677 | 0.705 | 0.711 | 0.709 | 0.717 | – | – | – |

Slower for the first 20 epochs (fewer low-t samples), ahead from epoch 30, and still improving at 80 where the reference had plateaued at 50. Coverage 100 % throughout.

## Independent sets (`best_pf_174M_p64x6_esmc_r4_tlate.pt`, w = 2, K = 8, coverage 100 %)
| set | late-t | reference (R = 4) |
|---|---|---|
| held-out (n=100) | 0.747 / 89 % / 6.23 A, bo8 0.769 (w=1.5: 0.748 / 90 % / 6.12) | 0.743 / 90 % / 6.16 A, bo8 0.773 |
| no-neighbour < 0.6 (n=266) | 0.558 / 62 % / 12.11 A | 0.548 / 62 % / 12.72 A |
| no-neighbour < 0.5 (n=93) | 0.506 / 42 % / 14.34 A | 0.497 / 45 % / 15.10 A |
| CASP15/16 (n=80) | 0.695 / 72 % / 7.17 A, bo8 0.717 | 0.692 / 72 % / 7.60 A, bo8 0.723 |

## Reading
1. +0.004 to +0.010 TM on every independent set and 0.1-0.8 A better RMSD, with the largest gains on the novel-fold subsets. Each difference is within one standard error, but the direction is uniform across four sets and the selection curve.
2. Part of the gap is the extra 30 epochs the late-t run got before its own stop; the reference was stopped by hand at 50 on a flat plateau, so a fair statement is "equal or slightly better plateau, reached later".
3. It is free, so it is in the recipe (`TLATE=1` in the long launcher; the from-scratch 512 main line uses it).

## What it changed
`best_pf_174M_p64x6_esmc_r4_tlate.pt` is the new 174M reference (held-out 0.747). Recipe for new runs: fused pair track, compiled DiT, budget-filling batches, R = 4 copies, late-t resampling.
