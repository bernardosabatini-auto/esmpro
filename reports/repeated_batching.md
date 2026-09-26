# repeated_batching — SimpleFold-style copies per protein: same accuracy, 2.9x cheaper per sample

**Job** `pf_174M_p64x6_esmc_r4` 48288533, 4 RTX (DDP), 2026-09-24 17:35-21:25, stopped by hand at epoch 50 on the plateau; scoring 48345063. Twin: `pf_174M_p64x6_esmc` (R = 1, batch 128, one RTX, 60 epochs, report pf_174M_p64x6_esmc.md). Code: `REPEAT_COPIES` in `code/gate10_pair_flow.py` (also `gate16`), profile in efficiency.md / papers_and_500aa.md.

## Mechanism
Each protein in a batch is expanded into R copies with independent flow time t, noise x0 and condition-drop mask; the pair track, which depends only on the sequence embedding, is computed once per protein and shared by the R copies. SimpleFold trains this way with B_c = 8-24 copies (their Table 6); SimpleDesign with 16.

## Cost (459M pair flow, one RTX, real bucketed batches)
| | samples/step | step | s per 1k sample-residues | peak GB |
|---|---|---|---|---|
| R = 1, budget 48 | 15.3k | 1.70 s | 0.111 | 51.9 |
| R = 2, budget 24 | 15.8k | 1.04 s | 0.066 | 54.6 |
| R = 4, budget 12 | 14.8k | 0.57 s | 0.039 | 48.5 |
At the 512 window: R = 4 at budget 12 gives 7.6k residues/step in 1.14 s vs 4.4k in 1.04 s for R = 1 at budget 24.

## Quality (174M, pair 64 x 6, ESMC-6B, 80k proteins)
Selection-set TM at equal epochs (R = 4 at budget 28 x 4 GPUs, 259-263 s/epoch, 60 GB; twin at fixed batch 128 on one GPU, 585 s/epoch):

| epoch | 6 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| R = 4 | 0.643 | 0.677 | 0.705 | 0.711 | 0.709 | **0.717** |
| twin R = 1 | 0.604 | 0.655 | 0.681 | 0.696 | 0.710 | 0.716 (0.719 at 60) |

Independent sets (`best_pf_174M_p64x6_esmc_r4.pt`, w = 2, K = 8, coverage 100 %):

| set | R = 4 | twin R = 1 |
|---|---|---|
| held-out (n=100) | **0.743 / 90 % / 6.16 A**, bo8 0.773 | 0.732 / 88 % / 6.38 A, bo8 0.767 |
| no-neighbour < 0.6 (n=266) | 0.548 / 62 % | 0.544 / 62 % |
| no-neighbour < 0.5 (n=93) | 0.497 / 45 % | 0.488 / 43 % |
| CASP15/16 (n=80) | 0.692 / 72 % / 7.60 A, bo8 0.723 | 0.684 / 74 % / 8.23 A, bo8 0.718 |

## Reading
1. Four copies per protein cost nothing in final accuracy (equal or +0.01 on every set) and reach the plateau in half the epochs. Gradient diversity from distinct proteins was not the binding constraint at this scale; diversity in t and noise per protein is at least as useful, as SimpleFold reports.
2. Per GPU-second of training the pair-track model is now 2.9x cheaper at R = 4; combined with the fused triangle update and the memory-filling budget, a 256-window epoch of the 459M main line should drop from ~800 s to roughly 350 s on 8 H200s.
3. Adopted: R = 2 in the running 512-residue fine-tune (memory-bound at that window), R = 4 for future 256-window runs. The 512 stage profile suggests R = 4 there too once the pair-bias memory fix (bf16, this evening) is measured.

## What it changed
The main-line recipe is `REPEAT_COPIES=4`, budget-filling batches, fused triangle projections, compiled DiT blocks. `best_pf_174M_p64x6_esmc_r4.pt` replaces `pf_174M_p64x6_esmc` as the 174M reference for further 80k ablations (held-out 0.743).

## Addendum 2026-09-26 — eight copies (`pf_174M_p64x6_esmc_r8_tlate`, job 48731107, 4 RTX, budget 14, late-t; scoring 48795868)
Selection curve: ahead of the R = 4 late-t run at every epoch (0.724 vs 0.708 at epoch 24), best 0.728 at epoch 38, early-stopped at 54; 375 s/epoch = 2x the samples of the R = 4 run's 261 s, i.e. 28 % less GPU time per sample.

| set | R = 8 | R = 4 (late-t reference) |
|---|---|---|
| held-out (n=100) | **0.758 / 89 % / 6.09 A**, bo8 0.784 | 0.747 / 89 % / 6.23 A |
| no-neighbour < 0.6 (n=266) | 0.561 / 65 % | 0.558 / 62 % |
| no-neighbour < 0.5 (n=93) | 0.505 / 48 % | 0.506 / 42 % |
| CASP15/16 (n=80) | **0.700 / 74 % / 7.16 A** | 0.695 / 72 % / 7.17 A |

Equal or better on every set at lower cost per sample. **R = 8 is the recipe** (the 840M main line runs with it); memory is governed by the trunk activations of the copies, hence the per-step residue cap (`TOKEN_CAP`). Best 174M weights are now `best_pf_174M_p64x6_esmc_r8_tlate.pt` (held-out 0.758).
