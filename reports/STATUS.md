# Live status

Updated 2026-09-22 19:03 (cluster time). Latest per-epoch evaluation of each job; these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. Reference: inherited checkpoint 0.427 / 32% / 11.56 A.

| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|---|---|
| lf_174M | latent flow | 47811754 | running 2:52:59 on kempner_rtx | 34 | 0.572 (w=3.0) | 62% | 9.72 | 100% |
| lf_459M_h200 | latent flow | 47811759 | running 2:52:57 on kempner_h200 | 30 | 0.580 (w=3.0) | 61% | 9.41 | 100% |
| h200_ca_bond | FAPE head | 47785117 | running 5:53:08 on kempner_h200 | 14 | 0.489 | 47% |  | 100% |
| lf_459M_ddp6 | latent flow | 47816193 | running 2:16:52 on kempner_rtx | 61 | 0.604 (w=2.0) | 65% | 9.63 | 100% |
| lf_459M | latent flow | 47811751 | running 2:52:59 on kempner_rtx | 17 | 0.553 (w=3.0) | 55% | 10.21 | 100% |
| h200_full | FAPE head | 47758870 | running 7:25:56 on kempner_h200 | 18 | 0.501 | 51% |  | 100% |
| h200_cabb_bond | FAPE head | 47785127 | running 5:53:07 on kempner_h200 | 13 | 0.500 | 50% |  | 100% |
| lf_174M_mix | latent flow | 47844153 | ended | 2 | 0.269 (w=2.0) | 0% | 17.52 | 100% |
| h200_bb_bond | FAPE head | 47785121 | ended | 11 | 0.480 | 45% |  | 100% |
| h200_bb | FAPE head | 47785123 | ended | 11 | 0.478 | 46% |  | 100% |
| lf_459M_ddp6 | latent flow | 47813371 | ended | 5 | 0.110 (w=2.0) | 0% | 45.85 | 100% |
| lf_base | latent flow | 47809117 | ended | 12 | 0.534 (w=2.0) | 50% | 10.51 | 100% |
| h200_full_bs160 | FAPE head | 47766504 | ended | 8 | 0.462 | 41% |  | 100% |
