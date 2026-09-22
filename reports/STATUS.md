# Live status

Updated 2026-09-22 18:19 (cluster time). Latest per-epoch evaluation of each job; these are selection-set numbers (100 or 200 val proteins), see README for the held-out caveat. Reference: inherited checkpoint 0.427 / 32% / 11.56 A.

| label | kind | job | status | epochs | TM | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|---|---|
| lf_174M | latent flow | 47811754 | running 2:09:33 on kempner_rtx | 26 | 0.543 (w=1.0) | 58% | 10.38 | 100% |
| lf_459M | latent flow | 47811751 | running 2:09:33 on kempner_rtx | 13 | 0.537 (w=3.0) | 52% | 10.55 | 100% |
| lf_459M_h200 | latent flow | 47811759 | running 2:09:31 on kempner_h200 | 23 | 0.563 (w=3.0) | 57% | 9.70 | 100% |
| h200_ca_bond | FAPE head | 47785117 | running 5:09:42 on kempner_h200 | 12 | 0.486 | 47% |  | 100% |
| lf_459M_ddp6 | latent flow | 47816193 | running 1:33:26 on kempner_rtx | 41 | 0.593 (w=2.0) | 69% | 9.35 | 100% |
| h200_cabb_bond | FAPE head | 47785127 | running 5:09:41 on kempner_h200 | 11 | 0.491 | 49% |  | 100% |
| h200_full | FAPE head | 47758870 | running 6:42:30 on kempner_h200 | 16 | 0.498 | 51% |  | 100% |
| h200_bb_bond | FAPE head | 47785121 | ended | 11 | 0.480 | 45% |  | 100% |
| h200_bb | FAPE head | 47785123 | ended | 11 | 0.478 | 46% |  | 100% |
| lf_459M_ddp6 | latent flow | 47813371 | ended | 5 | 0.110 (w=2.0) | 0% | 45.85 | 100% |
| lf_base | latent flow | 47809117 | ended | 12 | 0.534 (w=2.0) | 50% | 10.51 | 100% |
| h200_full_bs160 | FAPE head | 47766504 | ended | 8 | 0.462 | 41% |  | 100% |
