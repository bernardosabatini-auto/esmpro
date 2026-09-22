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
| [lf_base](lf_base.md) | latent flow 59M, first prototype | stopped at epoch 12; held-out TM 0.530 / 57% at w=2 (FAPE control 0.466 / 42%) |
