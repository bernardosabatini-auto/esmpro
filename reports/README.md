# Run reports

One report per finished (or deliberately stopped) training run, written when
the run ends. Each report records the configuration, the per-epoch curve,
the final scores on the CLAUDE.md section-4 gate set where available
(100 validation proteins, seed-42 permutation, Foldseek TM with exhaustive
search, coverage reported), and what the run changed in our thinking.

Reference points: inherited checkpoint TM 0.427 / 32% correct folds /
11.56 A on the gate set (0.420 / 29% / 12.67 A on the held-out slice);
true latent through the decoder 0.997 / 100% / 0.17 A.

Two evaluation sets are used. The **gate set** is the first 100 proteins of
the seed-42 validation permutation (CLAUDE.md section 4); every training run
selects its checkpoint on TM over a prefix that contains it, so gate-set
scores of a selected checkpoint are slightly optimistic. The **held-out
slice** is proteins 1000-1099 of the same permutation, never used for
selection (`--offset 1000` on either scorer); model comparisons use it.

See also [PLAN.md](PLAN.md) for the next steps and [STATUS.md](STATUS.md) for live progress.

| report | run | outcome |
|---|---|---|
| [h200_full_bs160](h200_full_bs160.md) | FAPE head, batch 160 | stopped at epoch 8, TM 0.462; tracked the batch-96 control |
| [pf_174M_p128x8](pf_174M_p128x8.md) | pair 128x8 + ESM-2, 80k, batch 64 | held-out 0.621 / 71%: +0.01 over the 64-dim track at the plateau |
| [pf_174M_p64x6_esmc](pf_174M_p64x6_esmc.md) | pair 64x6 + ESMC-6B, 80k | held-out 0.732 / 88% / 6.38 A: same TM as no-pair twin, better RMSD, 60 vs 88 epochs |
| [pf_174M_p64x6](pf_174M_p64x6.md) | pair 64x6 + ESM-2, 80k | held-out 0.613 / 71%: indistinguishable from no-pair twin at the plateau; 2x faster convergence |
| [pf_459M_esmc_afdb](pf_459M_esmc_afdb.md) | **pair track + ESMC-6B + 473k, 8 H200** | **held-out 0.758 / 88% / 6.51 A** (best-of-8 0.79), beats ESMFold2-Fast there; no-neighbour 0.567 / 63% (ESMFold2 0.631) |
| [esmfold2_comparison](esmfold2_comparison.md) | **external bar**: ESMFold2-Fast on our sets | held-out 0.753 / 85% / 7.18 A (ours 0.732 / 89% / 6.84); no-neighbour 0.631 vs ours 0.535 |
| [lf_174M_esmc](lf_174M_esmc.md) | **latent flow 174M, ESMC-6B conditioner, 80k** | **held-out 0.732 / 89% / 6.84 A**, best-of-8 0.763; no-neighbour 0.535 / 62%; conditioner is the biggest lever |
| [leakage_audit](leakage_audit.md) | **audit** | most of the score is seen folds: on no-neighbour proteins 80k 0.44 vs 473k 0.47 TM; data gain real but small |
| [lf_459M_afdb](lf_459M_afdb.md) | **latent flow 459M on 473k proteins, 8 H200s** | best of project: **held-out 0.660 / 74% / 9.01 A**, best-of-8 0.716; see leakage_audit |
| [lf_174M](lf_174M.md) | latent flow 174M, stored L33, 100 epochs | completed; held-out 0.609 / 69%, same ceiling as 459M |
| [lf_174M_mix](lf_174M_mix.md) | latent flow 174M, online ESM-2 all-layer mix | stopped ep 40; held-out 0.591 / 69%; mix -> 0.84 on L33, null result |
| [afdb_build](afdb_build.md) | Gate 8 data build | 393,531 new AFDB proteins encoded (train now 473k); val/test unchanged |
| [lf_459M_ddp6](lf_459M_ddp6.md) | latent flow 459M, 6 GPUs, batch 768 | 80k reference: held-out **0.609 / 71% / 10.06 A**, best-of-8 0.668 |
| [lf_459M_h200](lf_459M_h200.md) | latent flow 459M, one H200, batch 256 | stopped ep 43 at plateau; held-out 0.601 / 69%, best-of-8 0.659 |
| [lf_459M](lf_459M.md) | latent flow 459M, one GPU | stopped ep 19 as a duplicate; held-out 0.570 / 63% |
| [h200_full](h200_full.md) | FAPE head, section-5 job (control) | stopped ep 18; held-out 0.487 / 48% (inherited 0.420 / 29%) |
| [h200_16L512](h200_16L512.md) | FAPE head, 51M, fresh | stopped ep 18; held-out 0.510 / 53%, best FAPE head |
| [h200_24L768](h200_24L768.md) | FAPE head, 171M, fresh | stopped ep 17; held-out 0.492 / 47% |
| [h200_ca_bond](h200_ca_bond.md) | FAPE head, Ca-FAPE + bond penalty | stopped ep 15; held-out 0.479 / 45%; geometry fixed, no TM gain |
| [h200_cabb_bond](h200_cabb_bond.md) | FAPE head, Ca + true frames + bond | stopped ep 14; held-out 0.487 / 47%, same as control |
| [h200_bb_bond](h200_bb_bond.md) | FAPE head, true frames + bond penalty | stopped at epoch 11; held-out 0.469 / 43%, same as control; geometry fixed (N-CA dev 0.39 -> 0.10 A) |
| [h200_bb](h200_bb.md) | FAPE head, true frames alone | stopped at epoch 11; held-out 0.468 / 44%, same as control; true frames neither help nor hurt |
| [lf_base](lf_base.md) | latent flow 59M, first prototype | stopped at epoch 12; held-out TM 0.530 / 57% at w=2 (FAPE control 0.466 / 42%) |
| [casp_benchmark](casp_benchmark.md) | 80 CASP15/16 experimental domains <=256 aa, all checkpoints vs ESMFold2-Fast | ours 0.698 / 72%, ESMFold2-Fast 0.741 / 79%; ordering of runs preserved; best-of-8 ties ESMFold2 |
| [sample_selector](sample_selector.md) | training-free best-of-8 selection (consensus, re-encoding, decoder spread) | consensus recovers 15-27% of the oracle gap (+0.01 TM); the other signals nothing |
| [pdb_build](pdb_build.md) | 17,896 experimental PDB chains (<=3 A, 50% id clusters, CASP and val guards) as training targets | built in 20 GPU-min; ProteinAE round trip on experimental structures 0.996-0.999 TM |
