# casp_benchmark — experimental CASP15/16 domains, all checkpoints vs ESMFold2-Fast

**Jobs** 48129390 (leakage audit, RTX, 29 min) and 48129633 (embeddings + scoring + ESMFold2-Fast, RTX, 34 min), 2026-09-24. Builder `code/gate13_build_casp.py` (CPU). Logs `logs/casp_leakage_48129390.out`, `logs/casp_bench_48129633.out`.

## Why
Every number so far was scored against AlphaFold-DB predictions of proteins drawn from the same distribution as the training data, and the leakage audit showed most of the absolute score is seen folds. This is the first score against **experimental coordinates** of proteins deposited after the training proteins were predicted, on the standard community benchmark, with ESMFold2-Fast run on the identical inputs and pipeline.

## The set
- CASP15 `TS-domains.public_12.20.2022` and CASP16 `monomer_trimmed2domains` evaluation units (official domain definitions, experimental coordinates). Residues lacking any of N/CA/C/O are dropped (the domain files already carry gaps), MSE mapped to M.
- Length 32-256 (our models' window): 34 CASP15 + 46 CASP16 = **80 unique domains** (CASP16 ships every unit under T0/T1/T2 phase prefixes; 56 exact-sequence duplicates removed). 68 domains longer than 256 residues are excluded, which biases the set toward the easier end of CASP.
- Sequence = observed residues, identical for both models. Truth Ca in the PCA-canonical frame (TM is frame-invariant). Files `data/phase1_dataset/dataset_casp{,_esmc,_esm2}.h5`; name lists `notes/casp*_domains_le256.txt`.

## Leakage against the 473k training set
- **Sequence** (MMseqs2, E < 1e-3): 36/80 have a hit; 4 have > 50 % identity over half the domain; **none > 90 %**. No CASP domain is in the training set.
- **Structure** (Foldseek TM-align, TM normalised by the CASP domain): mean nearest-training TM 0.700; 66/80 have a training fold with TM > 0.5, 13 with TM > 0.9. AFDB covers most of fold space, so most CASP domains are *fold recognition from sequence* tests, not novel folds. No-neighbour subsets: **TM < 0.6 to any training structure: 24 domains; < 0.5: 14** (notes/casp_noneighbour_lt0.6.txt, lt0.5.txt).

## Scores (w = 2, 50 Euler steps, K = 8 for best-of-8; coverage 100 % everywhere)
TM mean / TM > 0.5 fraction / best-of-8 mean. ESMFold2-Fast is deterministic (one sample).

| model | all (n=80) | casp15 (n=34) | casp16 (n=46) | nonb<0.6 (n=24) | nonb<0.5 (n=14) |
|---|---|---|---|---|---|
| pair64 + ESMC + 473k (459M) | 0.698 / 72% / bo8 0.743 | 0.674 / 65% / bo8 0.723 | 0.716 / 78% / bo8 0.757 | 0.414 / 25% / bo8 0.487 | 0.397 / 14% / bo8 0.464 |
| ESMC, 80k (174M) | 0.677 / 71% / bo8 0.707 | 0.655 / 62% / bo8 0.682 | 0.693 / 78% / bo8 0.726 | 0.392 / 25% / bo8 0.444 | 0.363 / 14% / bo8 0.399 |
| pair64 + ESMC, 80k | 0.684 / 74% / bo8 0.718 | 0.654 / 68% / bo8 0.700 | 0.706 / 78% / bo8 0.732 | 0.412 / 29% / bo8 0.471 | 0.382 / 21% / bo8 0.443 |
| ESM-2, 473k (459M) | 0.650 / 70% / bo8 0.706 | 0.608 / 59% / bo8 0.675 | 0.682 / 78% / bo8 0.728 | 0.381 / 21% / bo8 0.480 | 0.359 / 14% / bo8 0.450 |
| pair128 + ESM-2, 80k | 0.544 / 60% / bo8 0.591 | 0.513 / 53% / bo8 0.571 | 0.567 / 65% / bo8 0.606 | 0.334 / 12% / bo8 0.398 | 0.309 / 7% / bo8 0.368 |
| ESM-2, 80k (174M) | 0.509 / 52% / bo8 0.559 | 0.481 / 44% / bo8 0.524 | 0.530 / 59% / bo8 0.585 | 0.309 / 8% / bo8 0.364 | 0.276 / 7% / bo8 0.326 |
| **ESMFold2-Fast** (biohub, 3 recycles, 50 steps) | 0.741 / 79% / - | 0.682 / 68% / - | 0.784 / 87% / - | 0.473 / 42% / - | 0.386 / 21% / - |

Ca RMSD, all 80: ours 8.32 A (pair64 + ESMC + 473k), ESMFold2-Fast 7.15 A. Per protein our TM correlates 0.89 with ESMFold2's; we are better on 28/80. Both correlate ~0.7 with the nearest-training TM, i.e. both are mostly fold recognition on this set.

By length (ours vs ESMFold2-Fast): 32-99 aa 0.650 vs 0.678 (n=12); 100-159 aa 0.752 vs 0.786 (n=33); 160-256 aa 0.664 vs 0.719 (n=35). The gap grows with length, which is where a pair track and recycling matter most.

## Reading
1. **The ranking of our runs is preserved on experimental data.** ESMC-6B is the big lever again (+0.17 over ESM-2 at 80k), data adds ~+0.02-0.04 on top, pair tracks ~+0.01 at the plateau. The AFDB held-out slice was not lying about the ordering.
2. **Absolute scores drop ~0.06 from AFDB held-out to CASP** (0.758 -> 0.698 for the best model; ESMFold2-Fast 0.753 -> 0.741). Our drop is larger: our targets are AF2 predictions, and where AF2 was wrong on a CASP target we inherit the error, while ESMFold2 was trained with experimental structures in the mix. Part of the gap is therefore the AFDB-only training signal, not model capacity.
3. **ESMFold2-Fast leads by 0.04 TM here** (0.741 vs 0.698; 79 % vs 72 % correct folds), by 0.06 on the 24 no-neighbour domains, and is tied on the 14 hardest. Best-of-8 sampling (0.743) matches its single-shot score, so a working sample selector (PLAN.md A4) would close the gap on this set at 8x inference compute, which is still far cheaper than ESMFold2's trunk.
4. **Novel folds remain unsolved by everyone.** On the 14 domains with no training fold above TM 0.5, ESMFold2-Fast reaches 0.386 and we reach 0.397 (best-of-8 0.464). Neither model folds them; single-sequence methods are doing fold recognition through the language model.

## Caveats
- n = 80 (34 / 46), standard error on a mean TM ~0.025; the 0.04 gap on the full set is ~1.5 s.e., the subset differences are within noise.
- Only domains <= 256 residues; the harder long CASP targets are excluded. Domain-level truth from the multi-domain PDB (domains are cut out of their context; both models see only the domain sequence).
- ESMFold2-Fast could have seen some of these PDB entries (its training cutoff is unknown to us); we could not have (AFDB training targets only). The comparison is therefore, if anything, favourable to ESMFold2.
- ESMFold2 numbers are with 3 recycles and 50 sampling steps, one diffusion sample; the released default may use more.

## What it changed
The plan's Phase A3(b) is done: `gate13_build_casp.py` plus the `--h5`/`--names-file` scorer path and the `--query-h5` audit give a repeatable experimental benchmark with leakage subsets. Every future run gets a CASP row. The two actionable conclusions are (i) a sample selector is worth +0.04 on experimental data (Phase A4, next), and (ii) mixing experimental (PDB) structures into the training targets is now a concrete lever, since the AFDB-only signal costs us on exactly the targets AF2 gets wrong.
