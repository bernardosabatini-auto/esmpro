# pdb_build — experimental PDB chains as training targets (17,896 chains)

**Jobs** selection on CPU (`code/gate15_select_pdb.py`, 4 min), build 48140523 (kempner_rtx, 1 GPU, 20 min: download + parse + canonicalise + ProteinAE encode 7 min, ESMC-6B embeddings 5 min, two round-trip controls). Files `data/phase1_dataset/dataset_pdb_train.h5` (405 MB) and `dataset_pdb_train_esmc.h5` (13.9 GB, ESMC-6B embeddings under `esm2_emb`, training layout). Log `logs/build_pdb_48140523.out`.

## Why
casp_benchmark.md: on experimental CASP coordinates our best model trails ESMFold2-Fast by 0.04 TM while leading on AFDB held-out. Our targets are AlphaFold predictions only, so where AF2 is wrong we learn the error. This adds experimental structures to the targets. It is the first non-AFDB training signal in the project.

## Selection (RCSB indices of 2026-09-18)
- `pdb_seqres` protein chains, SEQRES length 32-256, standard amino acids only, X-ray or EM entries with resolution <= 3.0 A (resolu.idx, pdb_entry_type). 70,043 unique sequences.
- **CASP guard:** 278 chains within 30 % identity (coverage >= 0.5) of a CASP benchmark domain removed, so `casp_benchmark.md` stays a valid test after training on this set.
- MMseqs2 clustering at 50 % identity, coverage 0.8: 23,516 clusters; the best-resolution chain per cluster kept.
- **Val guard (after the build):** 308 chains with > 50 % identity to a `dataset_100k` validation protein deleted from both files, so held-out and no-neighbour scores remain comparable with earlier runs (the AFDB additions had no such guard; see leakage_audit.md).

## Build
- PDB-format entry from RCSB, author chain, residues with all of N/CA/C/O, MSE -> M; keep if >= 32 residues and >= 80 % of SEQRES observed. Gapped chains are stored as their observed residues (same convention as the CASP evaluation units). 18,229 built; 3,101 too few observed residues; 2,186 downloads failed (entries without a PDB-format file, or transient).
- PCA canonicalisation and ProteinAE encoding exactly as `gate8_build_afdb.py`; per-chain attrs `resolution`, `method`, `n_seqres`, `chain_breaks`.
- Final set 17,921 chains before the structural CASP guard below: median length 145 (23 % under 100 aa), median resolution 1.90 A (63 % <= 2 A), 16,876 X-ray + 1,045 EM; 78 % without a chain break, 5 % with two or more; median observed fraction 0.957.

## Controls
- **Round trip on experimental structures.** True latent through the frozen decoder: CASP set TM 0.996 / RMSD 0.20 A (80/80); PDB chains TM 0.999 / RMSD 0.13 A (200/200), coverage 100 %. The autoencoder, trained on AFDB, reproduces experimental backbones with gaps as well as it does AFDB models (0.997 / 0.17 A). The 8-dim latent is not a bottleneck on real structures either.
- **Structural CASP guard (added after the build).** Foldseek TM-align of the 80 CASP domains against the set: 54 have a chain with TM > 0.5 (the AFDB training set has 66, so this is ordinary fold-space coverage, not a leak), 28 above 0.7, 2 above 0.9. The 25 chains with TM > 0.7 to any CASP domain were deleted from both files, leaving **17,896 chains**, so the benchmark's difficulty relative to the training set is unchanged by the addition.

## How it will be used
The set is 3.7 % of the 473k AFDB targets. Two arms are planned once the H200 run finishes: (i) fine-tune `best_pf_459M_p128x8_esmc_afdb.pt` with `--warm-start` on 473k + PDB with the PDB file repeated to ~15 % of the mix, a few epochs, scored on CASP and held-out; (ii) if it helps, include it from the start in the next full run. The experimental structures are also the only targets in the project not produced by a single predictor, which matters for the "novel folds" reading in casp_benchmark.md.
