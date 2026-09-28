# metric_resolution — what a 0.02 TM difference is worth, and whether sampling costs us

**Question.** Every comparison in this project is single-sample TM against one reference, and we trail ESMFold2-Fast by about 0.02 there. Two things were never checked: whether the metric can resolve 0.02 at these sample sizes, and whether a flow-matching model is penalised by a metric built for a point estimator.

**Jobs** `code/gate25_metric_resolution.py` (no GPU), `code/gate26_point_estimate.py` (jobs 48981605 / 48981607, one RTX each). Model `last_pf_840M_p128x8_long512_1p3M.ckpt`. ESMFold2-Fast numbers are the stored per-protein scores from `gate12`, same inputs and same pipeline.

## 1. The gap is not resolvable on experimental structures
Paired per-target comparison, bootstrap over targets (20,000 resamples), exact sign test.

| set | n | ours | ESMFold2-Fast | gap | 95 % CI of the gap | we win / lose | sign test |
|---|---|---|---|---|---|---|---|
| CASP <= 256 | 80 | 0.7199 | 0.7406 | +0.021 | [-0.002, +0.045] | 30 / 49 | 0.042 |
| CASP <= 512 | 109 | 0.7326 | 0.7509 | +0.018 | [-0.001, +0.039] | 38 / 71 | 0.002 |
| CASP 257-512 | 29 | 0.7625 | 0.7931 | +0.031 | [-0.008, +0.063] | 8 / 21 | 0.024 |
| long val (AFDB) | 200 | 0.5918 | 0.6441 | +0.052 | **[+0.038, +0.067]** | 45 / 155 | < 1e-4 |

Per-target differences have sd 0.10 on every set, five times the mean gap, and 18-23 targets per set differ by more than 0.10 in one direction or the other. **On experimental structures the confidence interval includes zero on all three CASP sets.** The sign test is significant, so ESMFold2-Fast really does win more targets; what is not measurable at n = 29-109 is how much better it is on average. The mean-TM leaderboard difference is carried by a handful of large swings.

**The one gap that is statistically solid is measured against predictions, not experiments.** Long validation is AFDB, so its references are AlphaFold outputs. What that row demonstrates is that ESMFold2-Fast agrees with AlphaFold more than we do; it does not demonstrate that it is more accurate.

## 2. The reference is not a point
Two experimental structures of the same protein, superposed on their corresponding residues (Kabsch, TM-score with a fixed correspondence):

| set | pairs | A-vs-B TM | median | 10th pct | fraction < 0.9 |
|---|---|---|---|---|---|
| apo/holo | 86 | 0.767 | 0.817 | 0.509 | 0.70 |
| fold-switch (CoDNaS) | 68 | 0.542 | 0.602 | 0.101 | 0.88 |

The target itself moves 0.23 (apo/holo) to 0.46 (fold-switchers) between experiments. Single-reference TM reports a width-0.23 object as a single number, and we are arguing over 0.02 of it.

## 3. Being generative is NOT costing us (hypothesis refuted)
I expected a distribution-matching model to be penalised by a point-estimator metric, so that a conditional mean would beat a draw. It does not. K = 8 at the accuracy-optimal guidance weight w = 2:

| estimator | CASP <= 256 | held-out |
|---|---|---|
| single draw (what we always report) | 0.7192 | 0.7891 |
| mean of the 8 draws | 0.7194 | 0.7875 |
| medoid in latent space | 0.7214 | 0.7939 |
| **mean of the 8 ODE endpoints, projected** | **0.5682** | **0.6450** |
| **one-step conditional mean, projected** | **0.3698** | **0.4107** |
| best-of-8 (oracle) | 0.7571 | 0.8290 |
| within-protein sd across the 8 draws | 0.026 | 0.028 |

Two findings, both negative for the hypothesis:
- **At w = 2 the model is already effectively a point estimator.** The spread across eight draws is 0.026 TM, and the single draw equals the mean of draws to 0.0002. There is almost no sampling variance to be penalised for. (The generative behaviour lives at w = 1, where `generative_benchmarks.md` measures pairwise diversity 0.906 against 0.954 at w = 2.)
- **Averaging in the latent destroys structure.** The mean of eight valid latents decodes 0.15 worse than any one of them, and the one-step conditional mean is 0.35 worse. Per-residue layer_norm puts the average back on the manifold's sphere but not back in the data distribution; the latent is not a space where convex combinations of valid codes are valid. This also re-confirms CLAUDE.md section 7's warning against latent-space regression targets, by a different route.

So the 0.02 is a genuine accuracy deficit, not a scoring artifact of sampling. Selection remains worth +0.002 to +0.005 here (medoid), far less than the +0.01 measured at w = 2 on an earlier checkpoint, because diversity at this guidance weight is now low.

## 4. How much latent error the decoder tolerates
True latent plus isotropic noise, projected onto the manifold, decoded, scored:

| sigma | measured z_mse | TM | TM > 0.5 |
|---|---|---|---|
| 0.0 | 0.000 | 0.996 | 1.00 |
| 0.1 | 0.007 | 0.886 | 1.00 |
| 0.2 | 0.030 | 0.738 | 0.90 |
| 0.3 | 0.065 | 0.616 | 0.86 |
| 0.4 | 0.113 | 0.521 | 0.74 |
| 0.5 | 0.171 | 0.448 | 0.33 |
| 1.0 | 0.526 | 0.248 | 0.00 |

The decoder is faithful at zero error and very sharp away from it: a latent mean-squared error of 0.03, against latents with unit variance per dimension, already costs 0.26 TM.

**This exposes something useful.** Our model's own latent error is z_mse 0.74-0.80, which on this curve corresponds to TM below 0.25. We score 0.72. Our latent errors are therefore nothing like isotropic noise: almost all of their magnitude lies in directions the decoder ignores, and the structural error comes from a small component inside the decoder's sensitive subspace. Two consequences:
1. **z_mse is worthless as a progress measure**, and any latent-space regression loss spends its capacity on the 97 % of the error that does not matter. That is the mechanism behind the retired MSE head (TM 0.161) and behind why the auxiliary latent-MSE term is forbidden.
2. **The accuracy still on the table is a narrow, decoder-sensitive direction**, invisible to every latent-space signal we train on. Reaching it needs a signal computed after the decoder. The one attempt so far (`struct_loss.md`, lDDT through the decoder) lost 0.02 at 1.5x cost, but it was an auxiliary term during training rather than a targeted final stage, so the idea is not closed.

## Reading
1. Report CASP results as "statistically indistinguishable in mean, behind on target count" rather than "0.02 behind". Quote the CI and the sign test.
2. Stop treating the long-validation gap as the headline deficit: its reference is an AlphaFold prediction. The long-protein deficit on experimental structures is 0.031 on 29 domains with a CI spanning zero.
3. The generative-penalty excuse is dead. The remaining gap is accuracy.
4. Never average latents, and never regress them.
5. A per-target, paired view should be standard in every future comparison; means over 29 to 109 targets with sd 0.10 cannot support the conclusions we have been drawing from them.
