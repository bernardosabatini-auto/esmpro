# efficiency — where a training step goes, and what was changed (2026-09-24)

**Trigger.** The user observed low real GPU utilisation and made maximal GPU and memory efficiency an explicit goal. Tools: `code/step_profile.py` (phase-timed real batches + kernel table, A/B knobs), `slurm/step_profile.sbatch`. Jobs 48231646, 48233950, 48236662 (one RTX each, ~10 min).

## Measurements
**Live H200 job (nvidia-smi dmon, 30 s, one node of the 459M recycling run):** SM busy 81-100 % on all GPUs with brief dips, memory-bandwidth utilisation 0-83 %. The "utilisation" counter is high; the GPUs are not idle.

**Phase timing, 459M pair-flow (DiT 1024x24 + pair 128x8), real bucketed batches, budget 48x256^2, one RTX Pro 6000:**

| variant | assemble | h2d | fwd | bwd | opt | **step** | peak GB | tokens/step |
|---|---|---|---|---|---|---|---|---|
| as run on the H200s (eager DiT, unaligned L, sliced bias) | 0.010 | 0.007 | 0.54 | 1.33 | 0.028 | **1.92** | 60.0 | 15.1k |
| + compiled DiT blocks, L padded to 8, split bias, fused AdamW | 0.010 | 0.007 | 0.48 | 1.31 | 0.028 | 1.83 | 51.9 | 15.3k |
| + fused triangle projections (5 GEMMs -> 1) | 0.010 | 0.007 | 0.46 | 1.20 | 0.027 | **1.70** | 51.9 | 15.3k |
| fused, budget 64 | 0.013 | 0.008 | 0.51 | 1.26 | 0.027 | 1.81 | 69.3 | 18.2k |
| fused + recycling (p_rec 1) | 0.010 | 0.007 | 1.30 | 1.21 | 0.027 | 2.55 | 52.0 | 15.3k |
| fused, no pair checkpointing | | | | | | OOM at 95 GB | | |

Kernel table (unfused): matmuls ~400 ms of ~1050 ms GPU time; the compiled pair-track graphs (forward, checkpoint recompute, backward) ~700 ms; attention 88 ms forward; AdamW ~28 ms; SliceBackward of the pair bias 59 ms. The SDPA efficient kernel was already being selected (checked explicitly), so the "math fallback" suspicion was wrong. Data loading is 1 %.

## What the numbers say
1. **The pair track is the cost, and it is inherent.** Per protein at L = 256 each of the 8 pair blocks does ~47 GFLOP of L^2 x d^2 projections plus L^3 x d triangle contractions, ~75 TFLOP per step across forward, checkpoint recompute and backward, comparable to the whole 464M DiT (64 TFLOP). It also streams ~15 pair-sized tensors per block. Without checkpointing it does not fit (OOM at budget 48 on 95 GB), so the recompute stays.
2. **Achieved ~40 TFLOPS per RTX on the optimised path** (both trunks counted) — a low fraction of peak because the work is many medium GEMMs and memory-bound elementwise chains on (B, L, L, d) tensors, not because the GPU is idle.
3. **Kernel-level fixes bought 11 % (1.92 -> 1.70 s) and 13 % memory**, not the 2-3x I estimated before profiling; that estimate was wrong and is withdrawn. The remaining lever is throughput per GPU-second via larger residue budgets (budget 64: +19 % tokens for +6 % time) — the H200 runs now use budget 108 (was 96) to sit near 130 of 141 GB.
4. **Recycling costs +50 % per step** (2.55 vs 1.70 s): a no-grad decoder pass (3 ODE steps over 4L atoms), one extra no-grad pair pass and one extra DiT forward on the steps that recycle. On the H200s the unoptimised recycling run took 1,359 s/epoch vs 796 s. Whether that is worth it is what the two recycling runs measure; if it is, the decoder pass is the thing to make cheaper (fewer ODE steps for the hint, or recycling on a subset of steps).

## Changes shipped (all default-on, old checkpoints still load)
- `TriangleMultiplyFused`: one Linear(d, 5d) per triangle update; `fuse_triangle_state_dict` converts old <-> new layouts; verified equal to the original to 6e-7 on CPU, round trip exact. Recorded as `pair_fused` in every checkpoint's sidecar; `--pair-unfused` restores the old module.
- `torch.compile` on the DiT blocks (`DIT_COMPILE=1`), bucket lengths padded to multiples of 8 (`PAD8=1`), pair bias split once instead of 24 sliced copies, fused AdamW, compile-prefix-tolerant resume/warm-start/scoring (`adapt_state_dict`).
- `step_profile.py`: any future model change gets a step-time, tokens/step, peak-memory and kernel breakdown before it goes on the H200s.

## Runs launched under the new code
- `pf_459M_p128x8_rec_esmc_afdb` (job 48243251, 8 H200, budget 108) — the recycling main line, restarted from scratch (the unoptimised attempt was stopped after 2 epochs; its epoch-1 TM 0.186 vs 0.179 without recycling).
- `pf_174M_p64x6_rec_esmc` (job 48243266, 4 RTX) — recycling twin of `pf_174M_p64x6_esmc` (0.719 selection / 0.732 held-out), the cheap A/B on the 80k set. GPU total 12 of the allowed 16.

## Addendum 2026-09-26 — cracking the utilisation leaderboard (user: cluster metric 44 %, top users 45-55 %)

**What the GPUs actually do inside a step** (DCGM on one node of the running 512 main line, 40 one-second samples per GPU): SM active 0.60-0.95 (mean ≈ 0.80), SM occupancy ≈ 0.33, tensor-core active ≈ 0.10, DRAM active ≈ 0.33, nvidia-smi "utilisation" 100 %. The kernels are memory-bound elementwise work in the pair track; tensor cores idle 90 % of the time. The dashboard's 44 % is therefore dominated by *allocated-but-idle* time, not by slow kernels: the per-epoch evaluation on rank 0 while seven GPUs wait (~3 % of a run), start-up caching, and above all the scoring / comparison / data-build jobs whose CPU phases (Foldseek, downloads) hold a GPU for hours.

**Measured options (459M pair flow, one RTX, ~15k sample-residues per step):**
| variant | step | peak GB | verdict |
|---|---|---|---|
| R = 4, pair checkpointing on (recipe) | 0.556 s | 45.9 | baseline |
| R = 4, checkpointing off | 0.490 s | 100.7 | -12 % time for +55 GB: not worth it |
| **R = 8, checkpointing on** | **0.484 s** | **45.2** | -13 % at the same memory; quality A/B running (`pf_174M_p64x6_esmc_r8_tlate`) |
| R = 8, checkpointing off | 0.440 s | 76.7 | -21 %, 1.7x memory |
| pad 32 instead of 8 | 0.625 s | 48.8 | more padding, slower |
| CUDA graphs (`reduce-overhead`) | crash | | per-block graphs conflict with checkpointing; not pursued |
| `max-autotune-no-cudagraphs` | 12 s | | recompiles for every bucket shape; unusable with variable lengths |
| eager trunk (no compile) | 0.662 s | 55.6 | compile is worth 16 % |

**Shipped:**
1. **Distributed per-epoch evaluation** (`DIST_EVAL=1`, default in DDP): every rank samples and decodes its share of the evaluation proteins into a shared directory, rank 0 runs Foldseek once. Removes the seven-GPU idle window each epoch. Smoke-tested on 2 GPUs, identical metric format, coverage 100 %.
2. **Chunked exhaustive Foldseek** (`FOLDSEEK_CHUNK=64`): only matched pairs are needed, but exhaustive search aligned every prediction against every reference (N^2). Chunks of 64 give N x 64 alignments with identical scores (verified: max difference 0.00 on 109 domains, 2.6x faster there; ~15x on the 1,000-protein long validation set, which had held a GPU for 3 hours while the CPU aligned a million pairs).
3. Scoring jobs get 16 CPU threads for Foldseek.
4. Knobs for `DIT_COMPILE_MODE` and `PAD_MULT` remain for future tests.

**Recipe change pending the R = 8 A/B:** eight copies per protein with checkpointing (13 % faster per sample at equal memory), which also makes a 512-window budget of ~16 x 256^2 per H200 possible.
