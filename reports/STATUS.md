# Live status

Updated 2026-09-23 12:47 (cluster time). Latest per-epoch evaluation of each job; these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. Reference: inherited checkpoint 0.427 / 32% / 11.56 A.

| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|---|---|
| lf_174M_esmc | latent flow | 47955320 | running 2:21:13 on kempner_rtx | 27 | 0.670 (w=2.0) | 80% | 7.79 | 100% |

## Notes (held-out checks of running models)

- 2026-09-23 04:50 — `best_lf_459M_afdb.pt` (epoch 42, 473k proteins, 8 H200): **held-out TM 0.657, TM>0.5 74%, RMSD 8.87 A, best-of-8 0.713**, coverage 100/100. 80k reference (`best_lf_459M_ddp6.pt`) on the same slice: 0.609 / 71% / 10.06 A / 0.668. Data scale-up is worth +0.05 TM and -1.2 A RMSD at equal model size; the run continues.
- 2026-09-23 07:10 — final: `best_lf_459M_afdb.pt` (epoch 55) held-out **0.660 / 74% / 9.01 A / best-of-8 0.716** (w=2). Report: `lf_459M_afdb.md`. No GPU jobs running; leakage audit next.
- 2026-09-23 08:20 — **Leakage audit (first pass, lenient TM normalisation): the AFDB additions contain close structural relatives of the validation proteins.** Held-out proteins with a training neighbour at TM>0.7: 68% against the original 80k train, 96% with the AFDB additions; TM>0.9: 12% -> 34%. The 0.609 -> 0.660 gain is therefore partly explained by neighbours, and the original split was already lenient. Rerunning with the strict (query-length) normalisation and preparing a "no close neighbour" evaluation subset for both models.
- 2026-09-23 10:30 — **Leakage audit final** (`leakage_audit.md`): on validation proteins with no training neighbour above TM 0.6 (n=266) the 80k model scores 0.439/29% and the 473k model 0.474/41%; on the plain held-out slice 0.609 vs 0.660. Absolute scores are mostly seen folds; the data gain is real but ~+0.035. Next: pair-track generative model (Gate 10) for generalisation.
- 2026-09-23 15:10 — **Gate 10 / conditioner grid launched on the 80k set (174M DiT, all RTX):** `pf_174M_p64x6` and `pf_174M_p128x8` (pair track, ESM-2), `lf_174M_esmc` (no pair, ESMC-6B), `pf_174M_p64x6_esmc` (pair track, ESMC-6B). Baseline cell is `lf_174M` (no pair, ESM-2): held-out 0.609 / 69%, no-neighbour 0.44 / 29%. ESMC-6B embeddings precomputed (`dataset_100k_esmc.h5`, 61 GB). ESM3 7B is not available on HF; ESMC-6B (biohub/ESMC-6B) is.
- 2026-09-23 12:50 — **ESMC-6B conditioning is a large lever**: `lf_174M_esmc` (174M, no pair, 80k) reached TM 0.670 / 80% on the selection set at epoch 26 and is still rising, vs 0.611 for the ESM-2 twin at convergence and 0.656 for the 473k-protein ESM-2 model. Held-out and no-neighbour scoring queued. Pair-track runs are 11x slower per epoch than expected; profiling before reading their early (+0.06 TM) lead.
