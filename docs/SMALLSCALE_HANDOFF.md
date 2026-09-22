# Small-scale result and H200 handoff
**2026-09-17** — sequence → frozen ESM-2 → head → 8-dim latent → frozen ProteinAE decoder → structure

## Result

40k training proteins, 8.2M-parameter head, everything else frozen. Converged
at epoch 28; stopped manually because both the loss and the fold metric had
flattened.

| | TM-score | correct folds (TM>0.5) | Ca RMSD |
|---|---|---|---|
| True latent through decoder | 0.997 | 100% | 0.17 Å |
| **This run** | **0.427** | **32%** | **11.56 Å** |
| MSE-trained head (previous approach) | 0.161 | 0% | 14.81 Å |
| Mean-latent baseline | 0.061 | 0% | 20.17 Å |
| ESMFold reference | — | — | 4.33 Å |

Roughly a third of proteins fold correctly, against none before. Not
competitive with ESMFold, but a working predictor rather than a blob.

## Convergence

| epoch | FAPE | TM | TM>0.5 | RMSD |
|---|---|---|---|---|
| 9 | 0.8735 | 0.379 | 20% | 12.41 Å |
| 16 | 0.8577 | 0.411 | 25% | 12.02 Å |
| 22 | 0.8512 | 0.422 | 33% | 11.70 Å |
| 28 | 0.8494 | 0.427 | 32% | 11.56 Å |

Epoch 22→28 moved everything by less than one standard error (the fold fraction
has SE ≈ 4.6 points on 100 proteins). Loss and fold quality plateaued *together*,
so this is genuine saturation of the configuration, not the surrogate failing.
**40k has saturated this head size. Scale both data and head.**

## Three bugs that would silently recur — read before scaling

1. **FAPE must use each structure's OWN frames.** The first implementation put
   predicted and true points in the *true* frames. The frame index then cancels
   algebraically and it collapses to un-superposed point distance: a perfect
   structure rotated 90° scored 0.95. Days of results were void.
   *Guard:* score a true structure against rotations (30/90/180°) and
   translations (1/5/20 Å). All must read ~0. Real 1 Å noise must not.

2. **Foldseek needs `--exhaustive-search 1`.** Without it the 3Di k-mer
   prefilter silently drops dissimilar structures, they return no row, and
   naive code imputes zero. Coverage was 10–20 of 100 for every arm except the
   positive control, making TM a function of 3Di rather than independent of it.
   It manufactured a fake "three metrics agree" and nearly killed a line of work
   that was succeeding.
   *Guard:* always report coverage next to any Foldseek-derived statistic.

3. **Stop on TM, not on FAPE.** FAPE tracks fold quality in direction but not
   magnitude. Already wired: `--tm-every N --select-on tm`. Costs 27 s against a
   4384 s epoch (0.6%), so run it every epoch. Patience is scaled by `tm_every`.

Also: 3Di identity is useless below TM ≈ 0.4 — unrelated real proteins score
~0.142, so 0.146–0.167 is all floor. Use TM.

## Recommended scale-up

Start from `smallscale_final_40k.pt`.

```
--n-train 79653 --n-val 10355 --batch-size 96 --ode-steps 3
--lr 1e-4 --clamp 20.0 --frac-unclamped 0.1 --normalize-out
--tm-every 1 --tm-n 200 --select-on tm --patience 6
```

- **ODE steps: 3 is correct, not a shortcut.** Head output is identical at 3, 5,
  10 and 20 steps (FAPE 0.9217 at all four). Do not pay for 20.
- **Head size is the untested axis.** 8.2M params is far below any published
  trunk reaching non-trivial TM. Sweep depth/width before assuming more data.
- **Keep ESM-2 frozen.** ESMFold freezes its language model and trains only the
  trunk on a learned softmax-weighted mix over *all* layers. We use layer 33
  only, so all-layer mixing is a cheap untried lever (33 extra parameters vs
  118M for fine-tuning).

## Ready but deliberately OFF

- `backbone_100k.h5` — 99,832 structures of N/CA/C, verified to match stored Ca
  exactly. `fape_loss_backbone` implements AF2 Gram-Schmidt frames, verified
  invariant to 1e-5.
- **Do not enable yet.** Decoded backbone from the head's latents has bond
  lengths of 1.38 ± 0.52 Å (N-CA) where truth is 1.46 ± 0.01 and the *true*
  latent decodes to 1.46 ± 0.03. True frames use 1.5 Å baselines and would be
  dominated by that scatter; the Ca pseudo-frames use 3.8 Å and are robust to it.
  **Fix local geometry first** — a bond-length penalty on decoded output is
  cheap and rotation-invariant, unlike an auxiliary latent-MSE whose target is
  PCA-frame-committed and pulls back toward the failed MSE solution.

## Files

| path | what |
|---|---|
| `data/phase1_dataset/smallscale_final_40k.pt` | the result, archived |
| `notes/gate6_smallscale_final.json` | full history + TM evaluations |
| `gate6_fape_train.py` | training; TM selection/stopping wired |
| `gate6_score_checkpoint.py` | `--ckpt <name> --bs 10`, safe alongside training |
| `gate6_fold_validation.py` | TM + 3Di vs baselines |
| `data/phase1_dataset/backbone_100k.h5` | N/CA/C, ready, unused |
