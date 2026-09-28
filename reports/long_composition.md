# long_composition — does training on more long proteins fix the long-protein deficit?

**Motivation.** On real structures our deficit to ESMFold2-Fast grows with length: 0.023 on chains up to 256 residues, 0.051 on 257-512 (`experimental_heldout.md`). The training pool is 86 % short (1.09M short against 173k long), so the obvious suspect was composition.

**Mechanism.** `LONG_FRAC` (gate 7) resamples each epoch so a chosen fraction of it is at least 257 residues: long chains drawn with replacement, short chains subsampled, **epoch size in proteins unchanged**. The trainer prints the realised composition, which for this run read `LONG_FRAC=0.4 -> 0.400 of each epoch is >= 257 residues (natural 0.136), epoch 158092 proteins`.

**Design.** Both arms warm-start from the same epoch-9 840M weights, run 2 epochs, lr 8e-5 decaying to zero over exactly 2 epochs with the measured step count, R = 8, identical data files. Control `pf_840M_p128x8_long512_anneal` (job 48973671), treatment `pf_840M_longfrac40` (job 48981624).

## Result on the experimental held-out set (real structures, coverage 100 %)
| | all 626 | long 167 (257-512) |
|---|---|---|
| 40 % long composition | 0.5848 | 0.5830 |
| control (natural 13.6 %) | 0.5841 | 0.5844 |
| paired difference | +0.0008, CI [-0.0026, +0.0041] | **-0.0014, CI [-0.0086, +0.0050]** |
| targets the treatment wins | 317 / 626 | 89 / 167 |

**Nothing.** On the long subset it was aimed at, the effect is -0.001 and the interval excludes any gain above 0.005. Win rates are chance.

## Two caveats, both pointing the same way
1. **The A/B was not compute-matched, and the confound favoured the treatment.** Holding the epoch at 158,092 proteins does not hold steps fixed: long chains make smaller batches under the residue-budget law, so the treatment ran 14,411 steps per epoch against the control's 8,699, and 13,652 s against 8,450. It had 1.6x the compute and still gained nothing, so the null is conservative.
2. **Both 2-epoch continuations were slightly worse than their own parent.** The epoch-9 840M scores 0.590 / 0.585 on this set; the control lands at 0.584 / 0.584 and the treatment at 0.585 / 0.583. The extra two epochs, annealed or not, cost about 0.005. `best_pf_840M_p128x8_long512_1p3M.pt` remains the best model.

## Reading
The long-protein deficit is not a data-composition problem. Combined with `pair_free_ablation.md`, where the pair track buys 0.022 on long chains against 0.008 on short, the length dependence looks architectural rather than distributional: what is missing on long chains is long-range geometric capacity, not more long examples. The AFDB long-validation metric would have called this a small win (0.611 against 0.609); the experimental set calls it zero, which is the second time in two days the AlphaFold-referenced metric disagreed with the real one.
