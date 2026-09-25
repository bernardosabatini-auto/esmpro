# struct_loss — LDDT auxiliary loss through the differentiable decoder: slower and no better

**Job** `pf_174M_p64x6_esmc_r4_lddt` 48349592, 4 RTX (DDP), 2026-09-24 22:15 to 2026-09-25 03:00, stopped by hand at epoch 40. Code `code/gate19_struct_loss.py`. Reference: `pf_174M_p64x6_esmc_r4` (same recipe without the term; repeated_batching.md). Log `logs/lf_ddp_lddt_48349592.out`.

## Mechanism
SimpleFold reports that an LDDT loss on the one-step estimate is required for local accuracy in coordinate space. Here the one-step latent estimate x1_hat = x_t + (1 - t) v is decoded through the frozen ProteinAE decoder *with gradient* (3 Euler steps), the true latent is decoded without gradient as the target (round trip 0.999), and an AlphaFold3-style LDDT term (pairs within 15 A, thresholds 0.5/1/2/4 A) on 16 samples per step is added to the flow loss with weight 1. This is geometric and pose-free, unlike the latent-MSE term CLAUDE.md warns against.

## Result (174M, pair 64 x 6, ESMC-6B, 80k, R = 4), selection-set TM at equal epochs
| epoch | 4 | 10 | 20 | 30 | 36 | 40 |
|---|---|---|---|---|---|---|
| + LDDT loss (401 s/epoch) | 0.445 | 0.567 | 0.681 | 0.692 | **0.702** | 0.696 |
| reference (261 s/epoch) | 0.558 | 0.677 | 0.705 | 0.711 | 0.709 | 0.709 (0.717 at 50) |

Coverage 100 % at every evaluation.

## Reading
1. The term dominates early training (total loss 0.46 vs a flow loss of ~0.1) and slows convergence badly (-0.11 at epoch 10); by epoch 36 it is still 0.01-0.02 below the reference and flat, at 1.54x the wall-clock per epoch.
2. Why it does not help here, plausibly: the latent flow already has a near-exact geometric decoder, so the flow loss on the latent *is* a structural loss up to a well-conditioned map; the LDDT gradient through three decoder ODE steps is noisy at low t and mostly adds variance. SimpleFold's finding is about Cartesian flows where the flow loss alone under-weights local detail.
3. A smaller weight (0.1) or a late-t-only schedule (alpha(t) = ReLU(t - 0.5), SimpleFold's fine-tuning form) was not tested; given the size of the deficit and the cost, this is not a priority.

## What it changed
Structural-loss-through-the-decoder is retired for now. `gate19` stays as the reference implementation (differentiable decoder in the training loop, LDDT loss) for any future geometric objective.
