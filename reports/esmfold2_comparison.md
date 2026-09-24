# esmfold2_comparison — ESMFold2-Fast on our evaluation sets (2026-09-23)

**Job** 48080943 (one RTX Pro 6000, 6.2 min for 351 proteins), `code/gate12_esmfold2_compare.py`,
`slurm/esmfold2_compare.sbatch`. Model: `biohub/ESMFold2-Fast` (6.54B params incl. its ESMC-6B
backbone), single-sequence mode, `num_loops=3`, `num_sampling_steps=50`, one diffusion sample,
fp32 (the sampler's SVD has no bf16 kernel). Loaded through transformers 5.17's native
`EsmFold2Model`; no MSA. ~1 s per protein.

## Same proteins, same pipeline
Predictions are reduced to alpha carbons, written as Ca pseudo-backbones and scored exactly like
our models: Foldseek TM-align with exhaustive search (coverage reported), Kabsch RMSD. Truth =
the AFDB model in `dataset_100k.h5` (pLDDT >= 80, <= 256 residues).

| set | n | ESMFold2-Fast TM / TM>0.5 / RMSD | `lf_174M_esmc` (80k, ESMC-6B) | `lf_459M_afdb` (473k, ESM-2) | `lf_459M_ddp6` (80k, ESM-2) |
|---|---|---|---|---|---|
| held-out slice | 100 | 0.753 / 85% / 7.18 A | 0.732 / 89% / 6.84 A | 0.660 / 74% / 9.01 A | 0.609 / 71% / 10.06 A |
| no-neighbour < 0.6 | 266 | 0.631 / 73% / 10.30 A | 0.535 / 62% / 12.89 A | 0.474 / 41% / 15.34 A | 0.439 / 29% / 16.38 A |
| no-neighbour < 0.5 | 93 | 0.563 / 60% / 12.12 A | 0.493 / 44% / 14.84 A | 0.430 / 24% / 17.25 A | 0.396 / 14% / 18.52 A |

Coverage 100% in every row. Our numbers are single samples at w=2 (w=1.5 for lf_174M_esmc);
best-of-8 sampling adds 0.03-0.06 TM to ours and is not applied here.

## Reading
- **On the plain held-out slice our 80k ESMC-conditioned model is at parity with ESMFold2-Fast**:
  0.02 behind on TM, ahead on correct-fold fraction (89% vs 85%) and mean RMSD (6.84 vs 7.18 A).
  With best-of-8 (0.763) it is ahead on TM as well. The 473k combined run
  (`pf_459M_esmc_afdb`, 0.738 on the selection set at epoch 55) should land above it.
- **On proteins with no close training relative, ESMFold2-Fast leads by 0.07-0.10 TM** and ~2.5 A
  RMSD. This is the generalisation gap and it is where the remaining work is (`PLAN.md`).
- **Cost.** ESMFold2-Fast: 6.5B parameters, ~1 s per protein for one structure. Ours: a 175M
  (or 461M) generator over a frozen ESMC-6B embedding; 25 Euler steps on 8-dim latents plus a
  3-step decoder, and the ESMC pass is shared by every sample of a protein.

## Caveats (read before quoting)
- "Truth" is an AlphaFold2 model, not an experimental structure. Both systems are being judged
  against AF2's opinion; the experimental benchmark (CAMEO/CASP) is in the plan.
- ESMFold2 was trained on AFDB-derived data; our validation proteins may be in its training
  set. The no-neighbour subsets control for *our* training set only.
- ESMFold2-Fast was run at modest inference compute (3 loops, 50 steps, 1 sample). Its authors
  report gains from more inference-time compute; a fairer ceiling would use more samples and
  pick by pTM. Likewise ours would use best-of-K with a selector (plan item A4).
- ESMFold2's own pTM (0.61 / 0.55 / 0.50 by set) tracks the measured TM ordering, which is a
  useful sanity check on the pipeline.
