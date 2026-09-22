# lf_base — sequence-conditioned latent flow, first prototype (59M)

**Job** 47809117, kempner_h200, 2026-09-22 15:38-16:05. Stopped deliberately at
epoch 12 to give the H200 to a 459M model at batch 256 (memory rule). Its EMA
weights at epoch 12 survive in `last_lf_base.ckpt`; no `best_lf_base.pt` was
written because of a trainer bug fixed the same evening (best-TM comparison
started from NaN).

## What the model is
`code/gate7_latent_flow.py`. Flow matching from Gaussian noise to the stored
ProteinAE 8-dim per-residue latents, conditioned on frozen ESM-2 650M layer-33
embeddings. DiT-style transformer: d512, 12 layers, 8 heads, 58.6M params;
per-token condition added at the input, adaLN-Zero on (time + pooled
condition), learned relative-position attention bias (+-32), self-conditioning
(p 0.5), condition dropout 0.1 for classifier-free guidance, EMA 0.999.
Batch 96, AdamW lr 3e-4, 1000-step warmup then cosine over 100 epochs, bf16.
Train split (79,653) cached in host RAM (52 GB fp16, 67 s to fill).
Sampling: 50 Euler steps, guidance weight w, final per-residue LayerNorm
projection onto the latent manifold. The frozen decoder (3 ODE steps) is used
only at evaluation. Epoch time 112 s. Peak GPU 12.6 GB.

## Curve (val = 2000 held-out proteins for the flow loss; TM on the first 100 of them)
| epoch | train loss | val loss | TM w=1 | TM w=2 | TM>0.5 w=2 | RMSD w=2 |
|---|---|---|---|---|---|---|
| 2 | 0.249 | 0.659 | 0.214 | 0.226 | 0% | 15.8 |
| 4 | 0.173 | 0.197 | 0.344 | 0.381 | 17% | 13.4 |
| 6 | 0.151 | 0.142 | 0.412 | 0.455 | 35% | 12.1 |
| 8 | 0.140 | 0.125 | 0.454 | 0.504 | 44% | 11.1 |
| 10 | 0.133 | 0.118 | 0.476 | 0.522 | 48% | 10.6 |
| 12 | 0.129 | 0.114 | 0.484 | 0.534 | 50% | 10.5 |

Coverage 100/100 at every evaluation. Still improving when stopped.

## Gate-set score (epoch-12 EMA, `gate7_score.py`, the section-4 proteins, K=8)
| w | TM | TM>0.5 | RMSD | best-of-8 TM |
|---|---|---|---|---|
| 1.0 | 0.492 | 46% | 11.24 | 0.557 |
| 1.5 | 0.529 | 50% | 10.55 | 0.590 |
| 2.0 | **0.540** | **56%** | **10.30** | **0.598** |
| 3.0 | 0.531 | 51% | 10.19 | 0.586 |
| 4.0 | 0.510 | 47% | 10.60 | 0.574 |

Inherited checkpoint on the same proteins: 0.427 / 32% / 11.56 A. FAPE
control (`best_h200_full.pt`, epoch 10, TM-selected) on the same proteins:
0.470 / 41% / 10.91 A.

## Held-out score (selection-free)
Both trainers select checkpoints on TM over a prefix of the seed-42 validation
permutation that contains the gate set, so gate-set numbers of a TM-selected
checkpoint are optimistically biased. Proteins 1000-1099 of the same
permutation were never used for selection by any run (`--offset 1000`):

| model | TM | TM>0.5 | RMSD | best-of-8 |
|---|---|---|---|---|
| inherited checkpoint | 0.420 | 29% | 12.67 A | |
| FAPE control, best epoch 10 | 0.466 | 42% | 11.74 A | |
| **latent flow 59M, epoch 12, w=2** | **0.530** | **57%** | **11.14 A** | **0.593** |
| latent flow 59M, epoch 12, w=1 | 0.474 | 43% | 12.45 A | 0.557 |
| latent flow 59M, epoch 12, w=3 | 0.526 | 55% | 11.02 A | 0.586 |

Selection bias was small: 0.010 TM for the flow model, under 0.005 for the
FAPE control.

## Sampler sweep (held-out slice, w=2, epoch-12 EMA)
| ODE steps | K | TM (first sample) | TM>0.5 | RMSD | best-of-K |
|---|---|---|---|---|---|
| 25 | 16 | 0.537 | 61% | 11.25 | 0.614 |
| 50 | 8 | 0.530 | 57% | 11.14 | 0.593 |
| 100 | 16 | 0.544 | 60% | 11.00 | 0.609 |
| 200 | 16 | 0.545 | 59% | 10.98 | 0.609 |

The Euler step count does not matter between 25 and 200 (differences are
within the ~0.01 seed-to-seed noise), so the sampler is not the bottleneck and
evaluation can run at 25 steps. Best-of-16 improves on best-of-8 by only
0.02, so most of the ensemble gain is captured by a handful of samples. The lead of 0.064 TM and 15 points of correct folds over the
FAPE control holds on the clean set (about two standard errors on 100 paired
proteins). Coverage 100/100 in every row.

## Outcome
Meets the CLAUDE.md section-9 success bar (mean TM above 0.5 with a majority
of proteins folding correctly) after 22 minutes of training, with no decoder in
the training loop, on both the gate set and a selection-free held-out set. Guidance is worth about 0.05 TM with an optimum at w=2;
best-of-8 sampling adds another 0.06, the first evidence of usable ensemble
capacity in the latent.

## What it changed
The point-regression-plus-FAPE line is superseded as the main direction. The
latent encodes orientation and the decoder basin is narrow, so p(z | sequence)
is multimodal; modelling it generatively absorbs that instead of fighting it,
and trains 10x faster per epoch. Follow-ups launched the same evening: 174M and
459M single-GPU runs on RTX, 459M at batch 256 on the H200, and a 6-GPU
data-parallel 459M run.
