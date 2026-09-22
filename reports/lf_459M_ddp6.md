# lf_459M_ddp6 — latent flow 459M, six RTX GPUs, batch 768 (the 80k reference model)

**Job** 47816193, kempner_rtx (6 x RTX Pro 6000, `slurm/train_latent_flow_ddp.sbatch`), 2026-09-22 17:35-21:15. Stopped by hand at epoch 70 (3.7 h) after twenty epochs within noise of its plateau, to free GPUs for the AFDB-scale run. Log `logs/latent_flow_ddp_47816193.out`.

## Configuration
`code/gate7_latent_flow.py`: flow matching from Gaussian noise to the stored ProteinAE latents, conditioned on stored ESM-2 layer-33 embeddings; DiT-style transformer d1024 x 24 layers x 16 heads (459.2M params), adaLN-Zero on time + pooled condition, relative-position attention bias, self-conditioning, condition dropout 0.1 (CFG), EMA 0.999 with warm-up, AdamW, bf16 autocast, logit-normal t. Training data: the 79,653-protein train split, cached in host RAM. Eval every N epochs: 100 val proteins (the gate set), 50 Euler steps, guidance w=2, frozen 3-step decoder, Foldseek TM. Data-parallel over 6 GPUs, batch 128 per GPU = 768 effective, lr 4e-4, warm-up 1000 steps, cosine over 300 epochs (so the lr was still ~85% of peak when stopped), eval every 5 epochs. 120 s/epoch (104 steps), peak GPU 69.6 GB per card. A first attempt without EMA warm-up evaluated near-random averaged weights and was restarted; this is the restart.

## Curve
| epoch | train | val | TM (best w) | TM>0.5 | RMSD | coverage |
|---|---|---|---|---|---|---|
| 1 | 1.6627 | 1.4577 | | | | |
| 2 | 0.9945 | 0.8148 | | | | |
| 3 | 0.6200 | 0.5299 | | | | |
| 4 | 0.3147 | 0.2947 | | | | |
| 5 | 0.2648 | 0.2483 | 0.273 (2.0) | 0% | 17.00 | 100% |
| 6 | 0.2316 | 0.2156 | | | | |
| 7 | 0.2074 | 0.1914 | | | | |
| 8 | 0.1888 | 0.1728 | | | | |
| 9 | 0.1757 | 0.1578 | | | | |
| 10 | 0.1645 | 0.1449 | 0.433 (2.0) | 32% | 12.58 | 100% |
| 11 | 0.1537 | 0.1351 | | | | |
| 12 | 0.1469 | 0.1284 | | | | |
| 13 | 0.1412 | 0.1236 | | | | |
| 14 | 0.1360 | 0.1196 | | | | |
| 15 | 0.1325 | 0.1163 | 0.524 (2.0) | 48% | 10.72 | 100% |
| 16 | 0.1311 | 0.1137 | | | | |
| 17 | 0.1278 | 0.1117 | | | | |
| 18 | 0.1267 | 0.1098 | | | | |
| 19 | 0.1243 | 0.1077 | | | | |
| 20 | 0.1219 | 0.1060 | 0.546 (2.0) | 57% | 10.21 | 100% |
| 21 | 0.1211 | 0.1044 | | | | |
| 22 | 0.1175 | 0.1033 | | | | |
| 23 | 0.1172 | 0.1021 | | | | |
| 24 | 0.1164 | 0.1011 | | | | |
| 25 | 0.1148 | 0.1001 | 0.568 (2.0) | 61% | 9.74 | 100% |
| 26 | 0.1137 | 0.0995 | | | | |
| 27 | 0.1125 | 0.0986 | | | | |
| 28 | 0.1101 | 0.0977 | | | | |
| 29 | 0.1108 | 0.0970 | | | | |
| 30 | 0.1092 | 0.0964 | 0.573 (2.0) | 61% | 9.78 | 100% |
| 31 | 0.1088 | 0.0960 | | | | |
| 32 | 0.1073 | 0.0955 | | | | |
| 33 | 0.1079 | 0.0950 | | | | |
| 34 | 0.1059 | 0.0945 | | | | |
| 35 | 0.1041 | 0.0939 | 0.582 (2.0) | 65% | 9.37 | 100% |
| 36 | 0.1051 | 0.0934 | | | | |
| 37 | 0.1034 | 0.0931 | | | | |
| 38 | 0.1030 | 0.0928 | | | | |
| 39 | 0.1022 | 0.0925 | | | | |
| 40 | 0.1024 | 0.0924 | 0.593 (2.0) | 69% | 9.35 | 100% |
| 41 | 0.1021 | 0.0920 | | | | |
| 42 | 0.1003 | 0.0917 | | | | |
| 43 | 0.0997 | 0.0914 | | | | |
| 44 | 0.0989 | 0.0911 | | | | |
| 45 | 0.0982 | 0.0909 | 0.596 (2.0) | 67% | 9.48 | 100% |
| 46 | 0.0989 | 0.0906 | | | | |
| 47 | 0.0976 | 0.0903 | | | | |
| 48 | 0.0974 | 0.0900 | | | | |
| 49 | 0.0963 | 0.0899 | | | | |
| 50 | 0.0954 | 0.0897 | 0.600 (2.0) | 69% | 9.69 | 100% |
| 51 | 0.0959 | 0.0895 | | | | |
| 52 | 0.0941 | 0.0893 | | | | |
| 53 | 0.0952 | 0.0891 | | | | |
| 54 | 0.0939 | 0.0891 | | | | |
| 55 | 0.0940 | 0.0891 | 0.605 (2.0) | 67% | 9.60 | 100% |
| 56 | 0.0926 | 0.0890 | | | | |
| 57 | 0.0919 | 0.0890 | | | | |
| 58 | 0.0907 | 0.0890 | | | | |
| 59 | 0.0913 | 0.0889 | | | | |
| 60 | 0.0912 | 0.0888 | 0.604 (2.0) | 65% | 9.63 | 100% |
| 61 | 0.0907 | 0.0887 | | | | |
| 62 | 0.0899 | 0.0889 | | | | |
| 63 | 0.0888 | 0.0889 | | | | |
| 64 | 0.0880 | 0.0890 | | | | |
| 65 | 0.0869 | 0.0891 | 0.611 (2.0) | 68% | 9.46 | 100% |
| 66 | 0.0865 | 0.0892 | | | | |
| 67 | 0.0863 | 0.0893 | | | | |
| 68 | 0.0871 | 0.0892 | | | | |
| 69 | 0.0858 | 0.0893 | | | | |
| 70 | 0.0860 | 0.0894 | 0.614 (2.0) | 70% | 9.22 | 100% |
| 71 | 0.0836 | 0.0896 | | | | |

Best TM 0.6140569999999999 at epoch 70.

## Scores
Selection set (gate proteins, w=2): TM 0.614 / 70% / RMSD 9.22 A at epoch 70 (best).
**Held-out slice (offset 1000, selection-free, `gate7_score.py`, 25 Euler steps):** TM 0.609, TM>0.5 71%, RMSD 10.06 A, best-of-8 TM 0.668, coverage 100/100. Same slice: inherited 0.420 / 29% / 12.67 A; best FAPE head (`h200_16L512`) 0.510 / 53% / 11.62 A; flow 59M epoch 12 0.530 / 57%.

## Outcome
The best model of the day on 80k proteins. The curve rose fast to 0.55 by epoch 20 (40 min), then crept: 0.593 (ep 40), 0.600 (45), 0.605 (50), 0.604 (55), 0.611 (60), 0.614 (70). On the held-out slice it scores 0.609 / 71% / 10.06 A, so selection bias is negligible for this model, and best-of-8 sampling adds 0.06 TM as it did for the small model. RMSD is now within 6 A of ESMFold's 4.33 A on this set. Coverage was 100% at every one of the 14 evaluations. It was stopped before its cosine schedule annealed; the final anneal usually adds a little, but the last five evaluations spanned 0.02 and the run had clearly reached the regime where more epochs on 80k buy almost nothing.

## What it changed
It fixes the 80k ceiling for this model class at ~0.61 TM / ~70% correct folds, independent of model size from 174M up (`lf_174M` plateaued at 0.59, the batch-256 H200 run at 0.60). Three model sizes converging on the same number is the evidence that the data (79,653 proteins) or the conditioning (a single ESM-2 layer) is now the limit, and it is what justified building the 470k-protein AFDB set (`code/gate8_build_afdb.py`) and the online all-layer ESM-2 conditioning (`--esm online`). This checkpoint (`best_lf_459M_ddp6.pt`) is the reference the AFDB-scale run has to beat.
