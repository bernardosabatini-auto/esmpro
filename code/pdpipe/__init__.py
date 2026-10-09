"""pdpipe: a standard pipeline for chaperone pull-down screens.

Every screen goes through the same four steps, so results from different experiments can be laid
side by side:

  io      load a screen into one standard Dataset (log2 matrix + sample table + protein table)
  stats   per-protein linear models with limma-style variance moderation, contrasts, F-tests,
          and split-half reliability (how much of a contrast's spread across proteins is
          reproducible between disjoint biological replicates; the ceiling for any explanation)
  plots   QC, volcano, reproducibility and concordance scatters with fitted regressions,
          per-protein condition and dose plots
  report  render a markdown write-up to PDF and self-contained HTML with one house style

A screen is described by a YAML file in pdpipe/configs/. See run.py for the standard pass.
"""
