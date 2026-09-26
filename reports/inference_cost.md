# inference_cost — per-protein cost of the latent flow vs ESMFold2-Fast on the same GPU class

**Job** 48780697 (`code/gate21_inference_cost.py`, one RTX Pro 6000 Blackwell, batch 1, 5 timed repetitions after warm-up). Model `last_pf_459M_p128x8_long512_scratch.ckpt` (464M), 25 Euler steps with CFG w = 2 (50 trunk forwards), pair track computed once, 3-step decoder; the ESMC-6B embedding is included. ESMFold2-Fast numbers are the per-protein wall-clock measured by `gate12_esmfold2_compare.py` on the same GPU class (3 recycles, 50 diffusion steps, one sample, batch 1) on real proteins of the stated sets.

| length | ESMC-6B | latent flow (50 fwd) | decoder | **ours, total** | ESMFold2-Fast |
|---|---|---|---|---|---|
| 64 | 21 ms | 270 ms | 28 ms | **0.32 s** | |
| 128 | 22 ms | 270 ms | 28 ms | **0.32 s** | |
| 256 | 27 ms | 269 ms | 30 ms | **0.33 s** | 0.84 s (CASP <= 256 set, mean length ~150) |
| 384 | 35 ms | 294 ms | 38 ms | **0.37 s** | 1.58 s (CASP <= 512 set) |
| 512 | 41 ms | 388 ms | 49 ms | **0.48 s** | 3.6-3.9 s (CASP 257-512 and long-val sets, mean length ~350) |

Peak GPU memory for inference: 15.8 GB (12 GB of it the language model).

## Reading
1. **2.6x faster than ESMFold2-Fast on short proteins and 7-8x on 300-500-residue proteins**, at batch 1, language model included. The flow is launch-bound at batch 1 (constant 270 ms up to 256 residues), so batched inference would widen the gap further; the decoder is 10 % of the total.
2. The absolute numbers are what the latent design buys: SimpleFold reports 10-26 s per 256-residue protein for its 100M-700M models at 200 steps on an M2 Max, and tens of seconds at 512; ESMFold needs 1-7 s on an H100 for 128-512 residues (their Table 7). We are at 0.3-0.5 s on a workstation-class GPU with a 464M trunk.
3. Accuracy context (casp_benchmark.md, pf_459M_p128x8_long512_scratch.md): 0.02-0.03 TM behind ESMFold2-Fast on CASP, 0.03 ahead on AFDB held-out. So the honest claim is "ESMFold2-class accuracy at 3-8x lower inference cost, with a sampler."

## Not done
SimpleFold itself was not timed here (no local install); their published H100 figures are quoted instead. An H200 measurement and batched-inference throughput can be added with the same script.
