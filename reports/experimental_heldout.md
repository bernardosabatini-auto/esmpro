# experimental_heldout — the first evaluation on real structures we have not seen

**Why** Every held-out, no-neighbour and long-validation number in this project before 2026-09-28 was scored against AFDB, i.e. against AlphaFold predictions, so it measured imitation of AlphaFold rather than accuracy (CLAUDE.md section 6b). This set replaces them.

**The set** `data/phase1_dataset/dataset_exp_val{,_esmc}.h5`, built by `code/gate27_select_exp_val.py` + `slurm/build_exp_val.sbatch` (job 49008141). PDB cluster representatives, X-ray or EM, resolution <= 3 A, kept as their observed residues (>= 60 % of SEQRES, >= 50 residues, like CASP evaluation units). Every chain is **below 30 % identity (MMseqs2, >= 50 % coverage either way) to all 1,282,670 training sequences and to all 512 CASP / CAMEO / apo / CoDNaS benchmark sequences**, so it is independent of what we train on and of what we test on.

**626 chains**: 459 at <= 256 residues, 167 at 257-512; length median 153, resolution median 2.40 A.
Autoencoder control on the set: round trip **TM 0.996 / 0.22 A / coverage 100 %**, so the target is representable and the error below belongs to the predictor, not the decoder.

## Result (840M, w = 2, 50 steps; ESMFold2-Fast on identical inputs; coverage 100 %)
| subset | n | ours | ESMFold2-Fast | gap | 95 % CI of the gap | targets we lose | our best-of-4 |
|---|---|---|---|---|---|---|---|
| all | 626 | 0.590 / 58 % / 12.6 A | 0.620 / 61 % / 12.1 A | **+0.030** | [+0.023, +0.037] | 417 / 626 | 0.619 |
| <= 256 residues | 459 | 0.592 | 0.614 | +0.023 | [+0.014, +0.031] | 286 / 459 | — |
| 257-512 residues | 167 | 0.585 | 0.635 | **+0.051** | [+0.039, +0.063] | 135 / 167 | 0.611 |

Every gap here is significant (sign test p < 1e-4 throughout); with 626 targets the metric resolves 0.02 comfortably, which the 29-109-target CASP sets could not.

## Reading
1. **Both models are far worse on genuinely novel proteins.** 0.59 and 0.62 here against 0.78 and 0.80 on CAMEO22 and 0.72 and 0.74 on CASP. The difference is the 30 % identity separation: CASP and CAMEO targets still have training relatives, this set does not. This is the honest operating point for a protein the model has never seen anything like.
2. **The deficit is real and it grows with length**: 0.023 on chains up to 256 residues, 0.051 from 257 to 512, both with intervals well clear of zero. The long-protein weakness, previously only suggestive on 29 CASP domains, is now measured on 167.
3. **At matched compute we are level, at matched samples we are behind.** Our best-of-4 on the whole set is 0.6186 against ESMFold2-Fast's single 0.6200, and four of our samples cost about 1.3-1.9 s against its 1.44 s per protein on this hardware. The same held on CAMEO22 (0.8005 against 0.8039). The honest claim is not "as accurate", it is "as accurate per unit of compute, with an ensemble instead of a point prediction".
4. The oracle headroom is 0.028, and `metric_resolution.md` shows our selection signals recover only a fifth of that, so best-of-K needs the samples, not a better ranker.

## What it changed
This is now the primary selection and reporting set for anything at 512 residues, replacing the 1,000-protein AFDB long validation. The long subset in particular gives the 257-512 regime an experimental metric it never had, and it is what the long-composition A/B (`LONG_FRAC`) and the conditioner-layer A/B will be judged on.
