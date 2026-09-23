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

| report | run | outcome |
|---|---|---|
| [h200_full_bs160](h200_full_bs160.md) | FAPE head, batch 160 | stopped at epoch 8, TM 0.462; tracked the batch-96 control |
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
