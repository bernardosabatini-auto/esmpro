# Live status

Updated 2026-09-23 07:04 (cluster time). Latest per-epoch evaluation of each job; these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. Reference: inherited checkpoint 0.427 / 32% / 11.56 A.

| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|---|---|
| lf_459M_afdb_rtx | latent flow | 47864287 | ended | 21 | 0.627 (w=2.0) | 76% | 9.07 | 100% |
| lf_174M_mix | latent flow | 47857391 | ended | 40 | 0.591 (w=2.0) | 64% | 9.24 | 100% |

## Notes (held-out checks of running models)

- 2026-09-23 04:50 — `best_lf_459M_afdb.pt` (epoch 42, 473k proteins, 8 H200): **held-out TM 0.657, TM>0.5 74%, RMSD 8.87 A, best-of-8 0.713**, coverage 100/100. 80k reference (`best_lf_459M_ddp6.pt`) on the same slice: 0.609 / 71% / 10.06 A / 0.668. Data scale-up is worth +0.05 TM and -1.2 A RMSD at equal model size; the run continues.
- 2026-09-23 07:10 — final: `best_lf_459M_afdb.pt` (epoch 55) held-out **0.660 / 74% / 9.01 A / best-of-8 0.716** (w=2). Report: `lf_459M_afdb.md`. No GPU jobs running; leakage audit next.
