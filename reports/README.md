# Run reports

One report per finished (or deliberately stopped) training run, written when
the run ends. Each report records the configuration, the per-epoch curve,
the final scores on the CLAUDE.md section-4 gate set where available
(100 validation proteins, seed-42 permutation, Foldseek TM with exhaustive
search, coverage reported), and what the run changed in our thinking.

Reference points: inherited checkpoint TM 0.427 / 32% correct folds /
11.56 A; true latent through the decoder 0.997 / 100% / 0.17 A.

| report | run | outcome |
|---|---|---|
| [h200_full_bs160](h200_full_bs160.md) | FAPE head, batch 160 | stopped at epoch 8, TM 0.462; tracked the batch-96 control |
| [lf_base](lf_base.md) | latent flow 59M, first prototype | stopped at epoch 12; gate set TM 0.540 / 56% at w=2 |
