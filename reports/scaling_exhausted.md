# scaling_exhausted — on real, novel structures, neither size nor data moves the model

**Setting.** All numbers on `dataset_exp_val` (626 experimental chains, resolution <= 3 A, below 30 % identity to all 1.28M training sequences and to every benchmark; `experimental_heldout.md`). Sampling w = 2, 50 steps, coverage 100 % throughout. Paired per-target bootstrap over 20,000 resamples.

## Every model we have, on real structures
| model | params | training pool | all 626 | long 167 |
|---|---|---|---|---|
| 459M, pair 128x8 | 459M | 646k | **0.5921** | **0.5912** |
| 840M, pair 128x8 | 840M | 1.26M | 0.5904 | 0.5845 |
| 459M, pair-free | 461M | 646k | 0.5847 | 0.5832 |
| 840M + 2 epochs annealed | 840M | 1.26M | 0.5841 | 0.5844 |
| 840M + 40 % long composition | 840M | 1.26M | 0.5848 | 0.5830 |
| ESMFold2-Fast | 6.5B | — | 0.6200 | 0.6354 |

| comparison | all 626 | long 167 |
|---|---|---|
| pair 128x8 vs pair-free | **+0.0075**, CI [+0.0034, +0.0115] | **+0.0080**, CI [+0.0000, +0.0164] |
| 840M vs 459M | -0.0017, CI [-0.0052, +0.0018] | -0.0066, CI [-0.0138, +0.0002] |

## What this overturns
1. **Doubling the trunk and doubling the data buy nothing.** The 840M is 0.002 *behind* the 459M overall and 0.007 behind on long chains, both intervals straddling or below zero. The +0.005 to +0.008 it showed earlier was measured on AFDB held-out and on CASP; it does not survive on real structures the model has no relative of.
2. **The pair track's length advantage was an artifact.** I reported +0.022 on long chains against +0.008 on short, from the AFDB long-validation set and 29 long CASP domains. On 167 real long chains it is +0.0080, the same as its +0.0075 overall, with the long interval barely clearing zero. The pair track is a small, flat, real gain, not a length-scaled one.
3. **Therefore the length deficit is not architectural either.** `long_composition.md` had already ruled out data composition; this rules out the "long-range geometric capacity" reading that replaced it. We are 0.023 behind on short chains and 0.051 behind on long, and nothing we have tried touches either.

I launched a pair-depth experiment (128x16, 25 h on 8 H200) on the strength of point 2 and cancelled it 25 minutes in once these paired tests came back. The premise was measured on the metric the user has since retired.

## Where that leaves the levers
| lever | effect on real structures |
|---|---|
| pair track (3.5x step cost) | +0.008 |
| 2x parameters, 2x data | 0 |
| 40 % long composition | 0 |
| learning-rate anneal | 0 |
| earlier conditioner layer | -0.10 |
| conditional-mean readout | -0.15 |
| medoid selection over 8 samples | +0.002 to +0.005 |

Everything is at or below 0.008 against a 0.028 deficit. The model plateaus near 0.59 on novel proteins regardless of size, data or schedule, while the frozen decoder round-trips the same targets at 0.996 — so the ceiling is not the decoder and not trunk capacity. What is left is the conditioner (a frozen ESMC-6B final layer, historically the only +0.17 lever in this project) and the generative direction, which is untested.
