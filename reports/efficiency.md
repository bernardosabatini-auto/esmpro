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
