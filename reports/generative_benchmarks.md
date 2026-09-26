# generative_benchmarks — conformational ensembles (apo/holo, fold-switch) and CAMEO22

**Jobs** 48782235 (`code/gate23_ensemble_eval.py` + CAMEO22 scoring, one RTX, 22 min) and 48782237 (ESMFold2-Fast on CAMEO22). Model `last_pf_459M_p128x8_long512_scratch.ckpt` (the current best). Sets built by `code/gate22_build_ensembles.py` from the EigenFold splits (`data/ensembles/*.csv`): **apo/holo 86 of 90 pairs**, **fold-switch (CoDNaS) 68 of 77 pairs**, **CAMEO22 169 of 183 targets** (drops: non-standard residues or > 512 residues, one fold-switch state missing or unalignable). Every target: model input = the csv `seqres`; references = the experimental chains from RCSB, aligned to `seqres` so residues correspond across states.

## Protocol (EigenFold / SimpleFold Table 3)
K = 5 samples per sequence at 50 Euler steps; Foldseek TM-align (exhaustive, chunked) of each sample against each state. **TM-ens** = mean over targets of the mean over the two states of the best-of-5 TM (does the ensemble reach both states). **Residue flexibility r** = Pearson correlation between the per-residue RMSF across the 5 samples (Kabsch-aligned to the sample mean) and the per-residue deviation between the two experimental states (superposed on shared residues); "global" pools residues over all targets, "per-target" averages per-target correlations. **Diversity** = mean pairwise Kabsch TM-like score between samples (lower = more diverse). Two guidance weights; coverage 100 % for every (target, state).

## Results
| set | w | TM-ens | best-of-5 to A / B | single draw (nearest state) | flex r global / per-target | diversity |
|---|---|---|---|---|---|---|
| apo/holo (86) | 1.0 | **0.854** | 0.833 / 0.876 | 0.865 | **0.451 / 0.505** | 0.906 |
| apo/holo | 2.0 | 0.846 | 0.822 / 0.870 | 0.874 | 0.356 / 0.455 | 0.954 |
| fold-switch (68) | 1.0 | **0.743** | 0.772 / 0.715 | 0.776 | **0.198 / 0.392** | 0.701 |
| fold-switch | 2.0 | 0.737 | 0.766 / 0.708 | 0.790 | 0.148 / 0.340 | 0.801 |

Published numbers on the same benchmarks (SimpleFold paper Table 3; mean TM-ens, global / per-target flexibility): SimpleFold-100M 0.852, 0.492 / 0.500; SimpleFold-3B 0.893, 0.639 / 0.550; AlphaFlow (MSA) 0.856, 0.455 / 0.527; ESMFlow 0.856, 0.416 / 0.496; MSA subsampling 0.856, 0.398 / 0.404 on apo/holo. Fold-switch: SimpleFold-3B 0.734, 0.292 / 0.288; ESMFlow 0.700, 0.269 / 0.345; AlphaFlow 0.730, 0.385 / 0.384; EigenFold 0.614.

**CAMEO22 (169 targets, folding):** ours 0.781 / 92 % / 9.43 A (w = 2, best-of-4 0.795); ESMFold2-Fast on the identical inputs 0.804 / 92 % / 8.79 A (2.4 s per protein vs our 0.3-0.5 s). SimpleFold reports 0.803 (100M), 0.829 (700M), 0.837 (3B) and ESMFold 0.853 on the full 183.

## Reading
1. **The model captures conformational heterogeneity at the level of dedicated ensemble methods.** On apo/holo its TM-ens (0.854) and flexibility correlation (0.451 global / 0.505 per target) match SimpleFold-100M, AlphaFlow and ESMFlow; on fold-switching proteins its TM-ens (0.743) is at SimpleFold-3B's 0.734 and its per-target flexibility correlation (0.392) is the highest of the published numbers, with a weaker global correlation (0.198). These are 464M-parameter numbers from a model trained only for folding, with no ensemble-specific tuning, on subsets (86/90, 68/77) rather than the full sets.
2. **Guidance weight is a diversity dial:** w = 1 gives more diverse ensembles (pairwise TM 0.906 vs 0.954), better two-state coverage and better flexibility correlation, at a small cost in single-draw accuracy. SimpleFold obtains its ensemble numbers with a stochastic sampler (tau); ours is the deterministic ODE with different initial noise, so a stochastic variant is the obvious next knob.
3. **Folding on CAMEO22 sits 0.023 behind ESMFold2-Fast**, the same gap as on CASP, at 5-8x lower cost. ESMFold cannot produce ensembles at all; ESMFold2 samples but its per-protein cost is 2.4 s here.
4. Caveats: subsets, best-of-5 protocol favours any sampler with diversity, references are experimental states of possibly different constructs, and CAMEO22 targets longer than 512 residues are excluded (11 of 183, plus non-standard-residue chains).

## What it changed
The generative claim now has numbers: at 0.3-0.5 s per sample the model produces ensembles that reach both experimental states as well as SimpleFold-100M / AlphaFlow on apo/holo and as well as SimpleFold-3B on fold-switchers. Next: a stochastic sampler (tau) sweep for the diversity/accuracy curve, and the same evaluation for the 840M model when it finishes.

## Addendum — stochastic sampler sweep (job 48788095)
Euler-Maruyama sampling (SimpleFold's SDE form, `sample_sde` in gate7) at w = 1, K = 5:

| set | sampler | TM-ens | single draw | flex r global / per-target | diversity |
|---|---|---|---|---|---|
| apo/holo | ODE (above) | 0.854 | 0.865 | 0.451 / 0.505 | 0.906 |
| apo/holo | SDE tau 0.3 / 0.6 / 1.0 | 0.853 / 0.854 / 0.848 | 0.879 / 0.872 / 0.863 | 0.420 / 0.483, 0.451 / 0.506, 0.402 / 0.525 | 0.953 / 0.927 / 0.894 |
| fold-switch | ODE (above) | 0.743 | 0.776 | 0.198 / 0.392 | 0.701 |
| fold-switch | SDE tau 0.3 / 0.6 / 1.0 | 0.744 / 0.741 / 0.736 | 0.795 / 0.786 / 0.776 | 0.140 / 0.344, 0.141 / 0.357, 0.175 / 0.371 | 0.788 / 0.746 / 0.697 |

The noise level moves single-draw accuracy and diversity in the expected directions but leaves TM-ens and the flexibility correlations within +-0.01 of the deterministic sampler. In this latent the ensemble spread comes from the initial noise, not from the integrator; the guidance weight remains the useful dial. Coverage 100 % throughout.
