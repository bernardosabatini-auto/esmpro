# lf_174M_mix — latent flow 174M, online ESM-2 with learnable all-layer mix (ablation)

**Job** 47857391, kempner_rtx (one RTX Pro 6000), 2026-09-22 19:40 to 2026-09-23 01:40 (6 h). Stopped by hand at epoch 40, level with its stored-embedding twin at equal samples, to let the eight-H200 job start. Two earlier attempts were restarted for bugs found on the way (bf16 ESM weights producing NaNs; a frozen mix initialisation; uninitialised padding in the online cache). Log `logs/latent_flow_47857391.out`.

## Configuration
As `lf_174M` (d768 x 16L, 173.6M) except: `--esm online` runs frozen ESM-2 650M per batch (fp32 weights under bf16 autocast) and conditions on a learned softmax mix over all 34 hidden layers (init 0.2 on layer 33, mix logits at 20x lr, no weight decay); batch 256 (memory rule: 70.6 GB peak on a 96 GB RTX), lr 2e-4, warm-up 1000, EMA with warm-up, eval every 2 epochs at w=2 with 25 Euler steps. 523 s/epoch (311 steps): ESM-2 roughly doubles the per-sample cost. Train = 79,653 proteins, sequences only in RAM.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 1.3423 | 0.8258 | | | | |
| 2 | 0.4891 | 0.3128 | 0.246 (2.0) | 0% | 18.36 | 100% |
| 3 | 0.2792 | 0.2434 | | | | |
| 4 | 0.2326 | 0.2022 | 0.326 (2.0) | 5% | 14.93 | 100% |
| 5 | 0.2003 | 0.1762 | | | | |
| 6 | 0.1795 | 0.1579 | 0.387 (2.0) | 21% | 13.35 | 100% |
| 7 | 0.1652 | 0.1452 | | | | |
| 8 | 0.1564 | 0.1364 | 0.455 (2.0) | 34% | 12.00 | 100% |
| 9 | 0.1487 | 0.1304 | | | | |
| 10 | 0.1436 | 0.1257 | 0.501 (2.0) | 46% | 11.15 | 100% |
| 11 | 0.1396 | 0.1220 | | | | |
| 12 | 0.1362 | 0.1193 | 0.519 (2.0) | 49% | 10.71 | 100% |
| 13 | 0.1330 | 0.1168 | | | | |
| 14 | 0.1310 | 0.1147 | 0.529 (2.0) | 50% | 10.52 | 100% |
| 15 | 0.1290 | 0.1129 | | | | |
| 16 | 0.1262 | 0.1114 | 0.536 (2.0) | 51% | 10.26 | 100% |
| 17 | 0.1246 | 0.1098 | | | | |
| 18 | 0.1235 | 0.1085 | 0.542 (2.0) | 57% | 10.12 | 100% |
| 19 | 0.1216 | 0.1076 | | | | |
| 20 | 0.1199 | 0.1066 | 0.546 (2.0) | 56% | 10.17 | 100% |
| 21 | 0.1188 | 0.1056 | | | | |
| 22 | 0.1175 | 0.1047 | 0.553 (2.0) | 57% | 9.92 | 100% |
| 23 | 0.1160 | 0.1037 | | | | |
| 24 | 0.1152 | 0.1029 | 0.555 (2.0) | 57% | 9.90 | 100% |
| 25 | 0.1137 | 0.1023 | | | | |
| 26 | 0.1130 | 0.1016 | 0.560 (2.0) | 59% | 9.86 | 100% |
| 27 | 0.1119 | 0.1010 | | | | |
| 28 | 0.1116 | 0.1005 | 0.567 (2.0) | 60% | 9.68 | 100% |
| 29 | 0.1103 | 0.1000 | | | | |
| 30 | 0.1094 | 0.0998 | 0.569 (2.0) | 60% | 9.67 | 100% |
| 31 | 0.1086 | 0.0993 | | | | |
| 32 | 0.1080 | 0.0988 | 0.574 (2.0) | 60% | 9.65 | 100% |
| 33 | 0.1070 | 0.0984 | | | | |
| 34 | 0.1064 | 0.0978 | 0.579 (2.0) | 62% | 9.45 | 100% |
| 35 | 0.1058 | 0.0973 | | | | |
| 36 | 0.1051 | 0.0969 | 0.580 (2.0) | 62% | 9.33 | 100% |
| 37 | 0.1038 | 0.0967 | | | | |
| 38 | 0.1036 | 0.0961 | 0.588 (2.0) | 62% | 9.35 | 100% |
| 39 | 0.1026 | 0.0958 | | | | |
| 40 | 0.1018 | 0.0955 | 0.591 (2.0) | 64% | 9.24 | 100% |

Best TM 0.590929 at epoch 40.

## Scores
Selection set (w=2): best TM 0.591 / 64% / RMSD 9.24 A at epoch 40.
**Held-out slice (offset 1000, selection-free):** TM 0.591, TM>0.5 69%, RMSD 10.06 A, best-of-8 0.638, coverage 100/100.
Twin (`lf_174M`, stored layer 33) at the same epoch 40: ~0.59 on the selection set; at completion 0.609 held-out.

## Outcome
The learned mix moved decisively toward the last layer: from 0.20 at init to 0.72 by epoch 4 and 0.84 by epoch 40, with 3% on layer 32 and 2-3% each on layers 1-4, everything else near zero. Per optimizer step the run led its twin (0.501 at 3,110 steps versus ~0.45), but at batch 256 each step sees twice the samples; per sample seen the two curves coincide within noise (0.591 vs ~0.59 at epoch 40). No non-finite proteins were reported after the fp32/zero-init fixes. Coverage 100% throughout.

## What it changed
All-layer mixing of ESM-2 is a null result for this task: the model chooses layer 33 almost exclusively, and the gain, if any, is inside the noise. CLAUDE.md section 7 listed it as the cheap lever to try before fine-tuning ESM-2; it has now been tried. The online conditioning path itself is the lasting product: it removes the 52 GB embedding cache and is what makes the 473k-protein runs possible.
