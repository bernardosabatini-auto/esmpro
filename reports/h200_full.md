# h200_full — FAPE head, the CLAUDE.md section-5 job (10L d256, warm-started)

**Job** 47758870, kempner_h200, 2026-09-22. Stopped by hand after 18 epochs (7.3 h) to free H200s for the latent-flow line; rising ~0.003 TM/epoch at the time. Log `logs/esm_proae_train_47758870.out`.

## Configuration
Head 10L d256 (8.2M), warm-started from `smallscale_final_40k.pt`; Ca-FAPE with legacy chain-end frames. Peak GPU 65.9 GB, 1433 s/epoch. `code/gate6_fape_train.py`; batch 96; 3 ODE steps; lr 1e-4 cosine over 60 epochs; clamp 20 A, 10% unclamped; `--normalize-out`; 79,653 train / 10,355 val; TM on 200 val proteins every epoch, selection on TM, patience 6; kempner_h200, one GPU.

## Curve (TM on 200 val proteins each epoch)
| epoch | train | val FAPE | TM | TM>0.5 | coverage |
|---|---|---|---|---|---|
| 1 | 1.5075 | 0.8584 | 0.433 | 37% | 100% |
| 2 | 1.5024 | 0.8567 | 0.434 | 36% | 100% |
| 3 | 1.4918 | 0.8525 | 0.443 | 38% | 100% |
| 4 | 1.4825 | 0.8492 | 0.452 | 38% | 100% |
| 5 | 1.4762 | 0.8463 | 0.457 | 38% | 100% |
| 6 | 1.4657 | 0.8429 | 0.463 | 42% | 100% |
| 7 | 1.4621 | 0.8415 | 0.461 | 42% | 100% |
| 8 | 1.4508 | 0.8390 | 0.467 | 42% | 100% |
| 9 | 1.4417 | 0.8356 | 0.468 | 42% | 100% |
| 10 | 1.4405 | 0.8324 | 0.478 | 44% | 100% |
| 11 | 1.4315 | 0.8309 | 0.477 | 44% | 100% |
| 12 | 1.4235 | 0.8281 | 0.479 | 44% | 100% |
| 13 | 1.4148 | 0.8251 | 0.488 | 48% | 100% |
| 14 | 1.4079 | 0.8237 | 0.485 | 46% | 100% |
| 15 | 1.4019 | 0.8217 | 0.488 | 48% | 100% |
| 16 | 1.3972 | 0.8192 | 0.498 | 51% | 100% |
| 17 | 1.3888 | 0.8179 | 0.495 | 48% | 100% |
| 18 | 1.3847 | 0.8165 | 0.501 | 50% | 100% |

Best TM 0.5009995 at epoch 18; best val FAPE 0.8164656046364043 at epoch 18.
Stopped by hand (see above).

## Scores
Best checkpoint `best_h200_full.pt` (epoch 18, selected on the 200-protein set): selection-set TM 0.501 / 51%.
**Held-out slice (offset 1000, selection-free):** TM 0.487, TM>0.5 48%, TM>0.3 85%, RMSD 11.36 A, coverage 100/100. Same slice: inherited checkpoint 0.420 / 29% / 12.67 A; latent flow 59M (epoch 12) 0.530 / 57% / 11.14 A.

## Outcome
Doubling the data took the inherited head from 0.427 / 32% (gate set) to a best selection-set TM of 0.501 / 51% at epoch 18, and 0.487 / 48% / 11.36 A on the held-out slice (inherited: 0.420 / 29% / 12.67 A there). The gain arrived in two phases: a fast 0.43 -> 0.46 over the first six epochs, then ~0.003 per epoch. Coverage 100% at every evaluation. This is the 'more data' half of the CLAUDE.md prescription, and it worked, but slowly.

## What it changed
It is the reference every other arm is measured against, and it fixes the exchange rate: 2x data bought ~0.06 TM for this head. The latent flow model bought 0.10 TM more than that on the same data in a fraction of the time (`reports/lf_base.md`), which is why the FAPE line was stopped here.
