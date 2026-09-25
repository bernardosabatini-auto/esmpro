# recycling — structural self-conditioning (decoded-estimate distogram into the pair track): no gain at 1.5x the cost

**Jobs** `pf_459M_p128x8_rec_esmc_afdb`: 48226219 (first attempt, unoptimised code, 2 epochs), 48243251 (fused code, budget 108, stopped by hand at epoch 13), 2 nodes x 4 H200; `pf_174M_p64x6_rec_esmc` 48253913 (4 RTX, stopped at epoch 28 to free GPUs for the 500-aa data build). Code `code/gate16_recycle_flow.py`. Logs `logs/lf_multinode_48243251.out`, `logs/lf_ddp_rec174_48253913.out`.

## Mechanism
At a denoising step the clean-latent estimate x_hat = x_t + (1 - t) v (already computed by the self-conditioning pass) is decoded through the frozen ProteinAE decoder (3 Euler steps, no grad), its Ca distogram (16 bins, 2-20 A) plus the timestep is projected into the pair representation, and the pair track is recomputed with it. Training: on every self-conditioning step (p_rec = 1, i.e. half of all steps). Sampling: pair recomputed with the decoded estimate every 5 of 25 Euler steps. The generative analogue of AlphaFold/ESMFold recycling.

## Result, main line (459M, pair 128 x 8, ESMC-6B, 473k), selection-set TM per epoch
| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recycling (1,235-1,250 s/epoch) | 0.180 0.280 0.378 0.471 0.611 0.648 0.666 0.681 0.691 0.698 0.706 0.700 0.701 |
| reference, no recycling (796 s/epoch) | 0.179 | 0.307 | 0.429 | 0.554 | 0.634 | 0.659 | 0.674 | 0.685 | 0.691 | 0.698 | 0.704 | 0.710 | 0.712 |

Behind through epoch 8, tied at epochs 9-10 (0.698 vs 0.698), behind again by 0.01 at epochs 12-13, at 1.5-1.6x the wall-clock per epoch (about 10 % of that from fewer optimizer steps at the larger budget). Coverage 100 % at every evaluation. Peak GPU 132-135 GB.

**Small-scale twin (174M, pair 64 x 6, ESMC, 80k, 4 RTX, budget batching):** 0.661 at epoch 28 vs 0.696 for the non-recycling twin at the same epoch; the twin had 8x more optimizer steps per epoch (fixed batch 128 on one GPU), so this comparison is confounded and was stopped early.

## Reading
1. Feeding the model its own decoded geometry does not improve the fold estimate at equal epochs and costs 50 % more compute per epoch; at equal wall-clock it is clearly worse. Stopped.
2. Plausible reasons: (i) the latent already carries the geometry — the decoder is a near-identity map (round trip 0.999), so a distogram of the current estimate adds little that the self-conditioned latent x_sc does not; (ii) the pair track's value is small in this regime anyway (pair ablations: +0.01); (iii) at low t the estimate is noise and the pair track has to learn to ignore it, which dilutes the signal.
3. Efficiency lesson (efficiency.md): the mechanism's cost is dominated by the extra decoder pass and the second pair pass; SimpleFold-style repeated batching would hide the pair pass but not the decoder.

## What it changed
- Recycling is retired for the main line. The H200s move to the 512-residue fine-tune (`pf_459M_p128x8_long512`, repeated batching R = 2).
- The distogram hook, decoder-in-the-loop code and DDP fixes remain available (`gate16`) for a structural auxiliary *loss* through the differentiable decoder, which is a different use of the same plumbing and still untested.
