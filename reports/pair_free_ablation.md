# pair_free_ablation — does the triangular pair track earn its cost?

**Question.** Our model is a DiT over residues in the frozen ProteinAE 8-dim latent, plus a gated triangular-multiplication pair track that feeds an attention bias. The pair track is the one piece inherited from AlphaFold-style folding trunks. It is also the expensive piece. Is it worth it?

**Arms.** Identical except for the pair track: 460M parameters, 512-residue window, the same seven training files (646k proteins: 80k short AFDB, 160k long AFDB, 13k long PDB), ESMC-6B conditioning, late-t resampling, R = 4 repeated batching, 4e-4 peak learning rate, 30 epochs, selection on 1,000 held-out long proteins.

| | label | job | best long-val | epoch |
|---|---|---|---|---|
| pair 128x8 | `pf_459M_p128x8_long512_scratch` | 48398186 / 48437245 | 0.605 | 22 |
| pair-free | `lf_459M_nopair_long512_r4b` | 48898294 | 0.593 | 25 |

Two earlier attempts at the pair-free arm were discarded, both for measurement bugs rather than results: the first declared `REPEAT_COPIES=4` that its loss never read (a quarter of the flow samples per epoch, CLAUDE.md bug 4), the second ran its cosine to zero at epoch 15 of 30 because the schedule was sized from a nominal batch size (bug 5). Both are now guarded and printed at startup.

## Cost, same GPU, same data, same batching (job 48844654)
| | params | forward | backward | step | peak GPU |
|---|---|---|---|---|---|
| pair 128x8, fused, checkpointed | 464M | 0.273 s | 0.597 s | **0.899 s** | 49.5 GB |
| pair-free | 461M | 0.097 s | 0.131 s | **0.256 s** | 31.4 GB |

**3.5x the step time and 1.6x the memory for 3M of 464M parameters** (0.6 %). Triangular multiplication is O(L^2 d) and activation checkpointing forces a recompute, so the backward pass carries the cost.

## Quality, every benchmark, coverage 100 % throughout
| set | n | pair 128x8 | pair-free | gap |
|---|---|---|---|---|
| held-out AFDB (w 1.5) | 100 | 0.785 / 91 % | 0.775 / 90 % | 0.010 |
| no-neighbour < 0.6 | 24 | 0.591 | 0.585 | 0.006 |
| no-neighbour < 0.5 | 14 | 0.541 | 0.532 | 0.009 |
| CASP <= 256 | 80 | 0.715 / 74 % | 0.707 / 75 % | 0.008 |
| CASP <= 512 | 109 | 0.728 / 78 % | 0.720 / 80 % | 0.008 |
| **long val 257-512** | 1000 | **0.591 / 67 %** | **0.570 / 62 %** | **0.021** |
| **long CASP 257-512** | 29 | **0.763 / 90 %** | **0.740 / 90 %** | **0.023** |

## Reading
1. **The pair track's value is a function of length.** On proteins up to 256 residues it is worth 0.006 to 0.010 TM. On proteins of 257 to 512 residues it is worth 0.021 to 0.023, on both the 1,000-protein long validation set and the 29 long CASP domains independently. That is the expected shape: pairwise geometry buys long-range contacts, and short chains have few.
2. **At 3.5x the step time, it is a poor trade on short proteins and a defensible one on long proteins.** A compute-matched comparison is not "train the pair-free model longer" -- it converged inside 30 epochs, best at 25 -- but "train a larger pair-free model", which is the open experiment.
3. **The generative and cost claims do not depend on the pair track.** Sampling is 25 Euler steps in an 8-dim latent either way; removing the pair track makes inference cheaper still.
4. Caveats: one seed per arm; the two arms ran on different hardware (H200 vs RTX Pro 6000), which is why the cost claim comes from the single-GPU profile and never from the two runs' epoch times; the pair-free arm was selected at epoch 25 and the pair arm at epoch 22, both well inside their schedules.

## What it changed
The expensive inherited component is now priced. For the short-protein regime the project can drop it and spend the 3.5x on scale; for long proteins it pays for itself. The next experiment is a pair-free model at the pair model's compute budget rather than its parameter count.
