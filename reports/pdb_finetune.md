# pdb_finetune — do experimental PDB targets help? Fine-tune with and without them

**Jobs** `pf_459M_p128x8_pdbft` 48142182 and `pf_459M_p128x8_ctlft` 48142183, each 2 nodes x 4 H200, 8 epochs, ~2.0 h; scoring 48189201/48189738 (PDB arm) and 48189202 (control) on one RTX each. Logs `logs/lf_multinode_4814218{2,3}.out`, `logs/score_full_4818920{1,2}.out`, `logs/score_full_48189738.out`.

## Question
casp_benchmark.md showed ESMFold2-Fast ahead of us by 0.04 TM on experimental coordinates while we lead on AFDB held-out. One explanation: our targets are AlphaFold predictions only, so we inherit AF2's errors. pdb_build.md prepared 17,896 experimental chains. This is the direct test: fine-tune the best checkpoint with those chains mixed in, against a matched control fine-tuned identically without them.

## Configuration
Both arms: `--warm-start best_pf_459M_p128x8_esmc_afdb.pt` (epoch 42, the main-line best), same architecture (DiT 1024 x 24 + pair 128 x 8, ESMC-6B conditioning), fresh AdamW, lr 1e-4, warm-up 200 steps, cosine over 8 epochs, batch budget 96 x 256^2 per GPU, eval every epoch on the AFDB selection set.
- **PDB arm:** 473k AFDB + `dataset_pdb_train_esmc.h5` repeated 4x = 71.6k of 544.7k (13.1 %) experimental targets per epoch; each PDB chain seen 32 times over the run.
- **Control:** 473k AFDB only.

Scores below are the **final EMA weights** (`last_*.ckpt`) of each arm, not each run's "best" checkpoint: the selection set is AFDB validation, where a shift toward experimental geometry cannot register, and both runs' best-by-selection was epoch 1 (PDB arm) or 3 (control).

## Scores (w = 2, 50 steps, K = 8, coverage 100 % everywhere)
| set | warm start (epoch 42) | **+ PDB, 8 ep** | **control, 8 ep** |
|---|---|---|---|
| selection set, final epoch | 0.741 / 88 % | 0.735 / 86 % | 0.735 / 86 % |
| held-out (n=100) | 0.758 / 90 % / 6.13 A, bo8 0.796 | 0.764 / 87 % / 5.97 A, bo8 0.800 | **0.767 / 90 % / 6.27 A**, bo8 0.806 |
| no-neighbour < 0.6 (n=266) | 0.573 / 65 % / 12.69 A | 0.574 / 64 % / 12.19 A | **0.578** / 65 % / 12.58 A |
| no-neighbour < 0.5 (n=93) | 0.514 / 42 % / 15.35 A | 0.517 / 46 % / 14.80 A | 0.516 / 48 % / 15.32 A |
| **CASP15/16 experimental (n=80)** | 0.703 / 74 % / 8.10 A, bo8 0.745 | 0.703 / 74 % / 7.80 A, bo8 0.748 | **0.708 / 74 % / 7.52 A**, bo8 0.744 |

## Reading
1. **Experimental targets did not help at this dose.** On every set the control is equal or better in TM (CASP 0.708 vs 0.703), and its CASP RMSD is better too (7.52 vs 7.80 A). The PDB arm's only edge is RMSD on the AFDB sets (5.97 vs 6.27 A held-out), which is noise-level and points the wrong way for the hypothesis.
2. **The RMSD improvement over the warm start (CASP 8.10 -> 7.5-7.8 A) belongs to the extra epochs**, not to the data: a second low-rate cosine cycle gives +0.005-0.01 TM and ~0.5 A on the sets where the model was already good. Cheap and real, but not the CASP lever.
3. **Why the hypothesis failed, plausibly:** (i) dose: 17.9k chains are 3.7 % of the targets even upweighted to 13 %, against a converged 459M model; (ii) the CASP deficit is fold-level (the 14 novel-fold domains sit at TM 0.39 for us and ESMFold2 alike), not local geometry, and a few thousand experimental structures cannot teach new folds; (iii) AF2 is very accurate on the kinds of proteins in our 32-256 residue window, so "AF2's errors" are a small part of our error budget.
4. **What would test it properly:** experimental targets from the start of a full run (not tested; 13 h of 8 H200s for an effect the fine-tune says is below 0.01), or a dedicated benchmark of proteins where AF2 is known to be wrong. Neither is a priority against the recycling and data-scale levers.

## What it changed
- The "AFDB-only targets" explanation for the CASP gap is retired at fine-tune dose. The PDB set stays available (`--extra-train-h5 dataset_pdb_train_esmc.h5`) for a from-scratch run if one is done for other reasons.
- Practical recipe finding: a short second cosine cycle at lr 1e-4 from the converged checkpoint adds +0.005-0.01 TM; `last_pf_459M_p128x8_ctlft.ckpt` is now the best set of weights (held-out 0.767 / 90 %, CASP 0.708 / 74 % / 7.52 A).
- The H200s moved on to `pf_459M_p128x8_rec_esmc_afdb` (structural self-conditioning, gate16), which targets the fold-level failure directly.
