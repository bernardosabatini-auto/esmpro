# Live status

Updated 2026-09-22 20:21 (cluster time). Latest per-epoch evaluation of each job; these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. Reference: inherited checkpoint 0.427 / 32% / 11.56 A.

| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|---|---|
| lf_174M | latent flow | 47811754 | running 4:10:54 on kempner_rtx | 50 | 0.583 (w=3.0) | 64% | 9.39 | 100% |
| lf_174M_mix | latent flow | 47857391 | running 47:41 on kempner_rtx | 5 | 0.326 (w=2.0) | 5% | 14.93 | 100% |
| lf_459M_h200 | latent flow | 47811759 | ended | 43 | 0.593 (w=3.0) | 66% | 9.45 | 100% |
| lf_459M | latent flow | 47811751 | ended | 19 | 0.560 (w=3.0) | 59% | 9.95 | 100% |
| lf_459M_ddp6 | latent flow | 47816193 | ended | 71 | 0.614 (w=2.0) | 70% | 9.22 | 100% |
| h200_cabb_bond | FAPE head | 47785127 | ended | 14 | 0.497 | 48% |  | 100% |
| h200_full | FAPE head | 47758870 | ended | 18 | 0.501 | 51% |  | 100% |
| h200_ca_bond | FAPE head | 47785117 | ended | 15 | 0.490 | 48% |  | 100% |
| lf_174M_mix | latent flow | 47844153 | ended | 2 | 0.269 (w=2.0) | 0% | 17.52 | 100% |
| h200_bb_bond | FAPE head | 47785121 | ended | 11 | 0.480 | 45% |  | 100% |
| h200_bb | FAPE head | 47785123 | ended | 11 | 0.478 | 46% |  | 100% |
| lf_459M_ddp6 | latent flow | 47813371 | ended | 5 | 0.110 (w=2.0) | 0% | 45.85 | 100% |
| lf_base | latent flow | 47809117 | ended | 12 | 0.534 (w=2.0) | 50% | 10.51 | 100% |
| h200_full_bs160 | FAPE head | 47766504 | ended | 8 | 0.462 | 41% |  | 100% |
