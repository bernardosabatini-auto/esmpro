# h200_full_bs160 — FAPE head, batch 160

**Job** 47766504, kempner_h200, 2026-09-22 12:00-15:38. Stopped deliberately at
epoch 8 to free the GPU for the first latent flow run.

## Configuration
Identical to the CLAUDE.md section-5 job except batch size: 10L d256 head
(8.2M), warm-started from `smallscale_final_40k.pt`, Ca-FAPE (legacy end
frames), clamp 20 A, 10% unclamped, `--normalize-out`, 3 ODE steps, lr 1e-4
cosine over 60 epochs, 79,653 train / 10,355 val, TM on 200 val proteins
every epoch. Batch 160 instead of 96. Peak GPU 109.7 GB (batch 96: 65.9 GB).

## Curve (val FAPE clamp 10; TM on 200 val proteins, coverage 100% throughout)
| epoch | val FAPE | TM | TM>0.5 | control (batch 96) TM |
|---|---|---|---|---|
| 1 | 0.859 | 0.430 | 36% | 0.433 |
| 2 | 0.857 | 0.434 | 34% | 0.434 |
| 3 | 0.853 | 0.445 | 38% | 0.443 |
| 4 | 0.851 | 0.449 | 38% | 0.452 |
| 5 | 0.850 | 0.446 | 36% | 0.457 |
| 6 | 0.846 | 0.454 | 40% | 0.462 |
| 7 | 0.845 | 0.458 | 41% | 0.465 |
| 8 | 0.842 | 0.462 | 41% | 0.470 |

Epoch time 1424 s (control: 1433 s).

## Outcome
The larger batch neither helped nor hurt: per-epoch TM tracked the batch-96
control within noise (SE about 3.5 points on 200 proteins), and the epoch time
was the same because the GPU was already compute-bound at batch 96. Memory
scales linearly with batch for this decoder (66 GB at 96, 110 GB at 160), so
batch 200 is the practical ceiling on an H200.

## What it changed
Nothing about the model. It established that the H200's spare memory buys no
throughput for FAPE training, which informed the decision to put that memory
into larger latent flow models instead (CLAUDE.md rule: fill the memory).
