# h200_cabb_bond — FAPE head, Ca-FAPE + true-frame FAPE + bond-length penalty

**Job** 47785127, kempner_h200, 2026-09-22. Stopped by hand after 14 epochs (6 h); tracking the control within noise. Log `logs/esm_proae_train_47785127.out`.

## Configuration
Head 10L d256 (8.2M), warm-started; sum of Ca-FAPE (`--fix-end-frames`) and true N-CA-C-frame FAPE with all backbone atoms as points (`--loss ca+bb`), plus bond-length penalty weight 1.0. Peak GPU 65.9 GB, ~1520 s/epoch. `code/gate6_fape_train.py`; batch 96; 3 ODE steps; lr 1e-4 cosine over 60 epochs; clamp 20 A, 10% unclamped; `--normalize-out`; 79,653 train / 10,355 val; TM on 200 val proteins every epoch, selection on TM, patience 6; kempner_h200, one GPU.

## Curve (TM on 200 val proteins each epoch)
| epoch | train | val FAPE | TM | TM>0.5 | coverage | N-CA dev |
|---|---|---|---|---|---|---|
| 1 | 3.3389 | 0.8634 | 0.416 | 30% | 100% | 0.236 |
| 2 | 3.2783 | 0.8570 | 0.429 | 34% | 100% | 0.216 |
| 3 | 3.2238 | 0.8501 | 0.441 | 36% | 100% | 0.194 |
| 4 | 3.1824 | 0.8446 | 0.455 | 40% | 100% | 0.183 |
| 5 | 3.1481 | 0.8417 | 0.454 | 39% | 100% | 0.165 |
| 6 | 3.1106 | 0.8342 | 0.470 | 44% | 100% | 0.159 |
| 7 | 3.0799 | 0.8309 | 0.472 | 44% | 100% | 0.152 |
| 8 | 3.0528 | 0.8265 | 0.480 | 45% | 100% | 0.144 |
| 9 | 3.0202 | 0.8248 | 0.476 | 46% | 100% | 0.137 |
| 10 | 2.9993 | 0.8196 | 0.486 | 46% | 100% | 0.136 |
| 11 | 2.9747 | 0.8171 | 0.491 | 49% | 100% | 0.131 |
| 12 | 2.9564 | 0.8159 | 0.492 | 48% | 100% | 0.126 |
| 13 | 2.9402 | 0.8118 | 0.500 | 50% | 100% | 0.126 |
| 14 | 2.9192 | 0.8100 | 0.497 | 48% | 100% | 0.122 |

Best TM 0.499773 at epoch 13; best val FAPE 0.8100301788912879 at epoch 14.
Stopped by hand (see above).

## Scores
Best checkpoint `best_h200_cabb_bond.pt` (epoch 13, selected on the 200-protein set): selection-set TM 0.500 / 50%.
**Held-out slice (offset 1000, selection-free):** TM 0.487, TM>0.5 47%, TM>0.3 88%, RMSD 11.72 A, coverage 100/100. Same slice: inherited checkpoint 0.420 / 29% / 12.67 A; latent flow 59M (epoch 12) 0.530 / 57% / 11.14 A.

## Outcome
Best selection-set TM 0.500 / 50% at epoch 13, held-out 0.487 / 47% / 11.72 A: identical to the control on the held-out slice. It was the best of the four loss arms on the selection set throughout, but never by more than noise. Geometry improved as in the other penalised arms (N-CA scatter 0.39 -> 0.15 A). Coverage 100% throughout.

## What it changed
The hedged combination of every loss term does not beat the plain Ca-FAPE loss. With `h200_bb`, `h200_bb_bond` and `h200_ca_bond`, this closes the loss-function axis: Ca frames, true frames, their sum, with or without a geometry penalty, all land on the control's curve.
