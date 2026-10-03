---
title: "Additional text 6: does reading several ESMC layers at once predict better?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

The best single ESMC layer differs between responses: layer 50 for salt and staurosporine, the final
layer (80) for heat and the DNAJB11-versus-DNAJA1 preference. Does combining layers predict better than
the best single layer, and which layers does each response draw on?

# Method

Nine mean-pooled hidden states (layers 0, 10, ..., 80) for every protein, eight responses, the usual
outer folds grouped by sequence cluster. Within each training fold, a second, inner grouped split
makes every choice, so the outer test fold is never used to choose anything:

- **each layer alone:** ridge regression, penalty chosen on the inner split;
- **chosen single layer:** the layer that scores best on the inner split, a fair stand-in for "the
  best layer", which would otherwise be picked by looking at the test results;
- **stacking:** the nine single-layer models combined with non-negative weights fitted to their inner
  out-of-fold predictions;
- **concatenation:** all nine layers in one ridge model, each layer standardised and scaled so that
  each contributes equally.

Penalties come from a grid of seven values between 300 and 300,000; the staurosporine responses are
sensitive to this choice (their R2 ranges from -20 % to +6 % across the grid), so a coarse grid
understates them. Intervals resample sequence clusters. 2.1 minutes on one RTX card for all eight responses.

# Results

| Response | L0 | L10 | L20 | L30 | L40 | L50 | L60 | L70 | L80 | Chosen | Stacked | Concat. |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Salt 75 mM | 6.2 | 13.2 | 13.9 | 14.7 | 14.5 | 16.0 | 14.7 | 14.8 | 15.8 | 16.0 | **18.0** | 17.8 |
| Salt 150 mM | 7.4 | 14.8 | 16.0 | 16.6 | 16.3 | 18.2 | 16.2 | 17.0 | 18.1 | 17.7 | **20.4** | 19.5 |
| Heat 43 °C | 8.9 | 20.2 | 20.7 | 21.1 | 20.9 | 22.5 | 20.8 | 21.1 | 25.7 | 25.7 | **27.4** | 26.4 |
| Heat 37 °C | 3.4 | 9.3 | 9.4 | 9.4 | 9.6 | 10.7 | 10.2 | 10.2 | 11.4 | 11.4 | **12.7** | 12.3 |
| Stau. 35 °C | 1.6 | 5.0 | 6.1 | 6.4 | 6.3 | 6.0 | 4.9 | 4.6 | 5.0 | 5.9 | **6.6** | 6.6 |
| Stau. 37 °C | 0.4 | 5.8 | 6.2 | 6.1 | 6.1 | 5.9 | 6.5 | 5.8 | 4.3 | 5.7 | **6.6** | 6.5 |
| Stau. 43 °C | 0.6 | 2.8 | 3.1 | 3.8 | 3.7 | 4.1 | 3.2 | 2.9 | 2.0 | 2.4 | 3.5 | **3.7** |
| DNAJB11 pref. | 30.6 | 45.2 | 46.0 | 46.2 | 45.7 | 43.5 | 36.2 | 38.6 | 45.6 | 46.1 | **49.1** | 48.6 |

*Percent of variance explained on held-out sequence clusters.*

| Response | Stacked minus chosen layer (95 % interval) | Layers stacking relies on (mean weight) |
|:--|--:|:--|
| Salt 75 mM | +2.0 [+1.3, +2.7] | 80 (0.35), 50 (0.24), 60 (0.20) |
| Salt 150 mM | +2.6 [+1.8, +3.5] | 80 (0.35), 50 (0.23), 60 (0.17) |
| Heat 43 °C | +1.6 [+1.0, +2.2] | 80 (0.59), 60 (0.15), 10 (0.14) |
| Heat 37 °C | +1.2 [+0.7, +1.7] | 80 (0.51), 50 (0.21), 60 (0.17) |
| Stau. 35 °C | +0.7 [+0.1, +1.3] | 20 (0.39), 80 (0.21), 60 (0.16) |
| Stau. 37 °C | +1.0 [+0.5, +1.4] | 60 (0.34), 20 (0.31), 80 (0.19) |
| Stau. 43 °C | +1.1 [+0.6, +1.8] | 60 (0.59), 20 (0.23), 0 (0.13) |
| DNAJB11 pref. | **+3.0 [+2.3, +3.6]** | 80 (0.36), 40 (0.22), 10 (0.17) |

# Conclusions

- **Combining layers helps every response.** Stacking beats the chosen single layer on all eight, by
  0.7 to 3.0 points, with every interval above zero; concatenation does about as well (within 1 point
  of stacking). The best figures are now 49.1 % for the co-chaperone preference, 27.4 % for heat at
  43 °C and 20.4 % for salt at 150 mM.
- **Different responses read different depths.** Heat draws mostly on the final layer; salt on the final
  layer together with layers 50-60; staurosporine on layer 20 and layer 60; the co-chaperone preference on layers
  spread across the network (80, 40, 10).
- **Why the sparse autoencoder lost on the preference.** For the DNAJB11 preference, layer 60 is the
  weakest of the middle layers (36.2 %, against 45-46 % at layers 10-40 and 80). The sparse autoencoder
  reads layer 60, which is why its features explained 7 points less of the preference than the
  embedding did, while matching it for the other responses.
- **Size of the gain.** One to three points of variance is a 6.5-15 % relative improvement for salt, heat and
  the preference (largest for salt at 150 mM, 17.7 to 20.4 %). For staurosporine, the absolute gain is under a point.

# Critical evaluation

- **Penalty sensitivity.** With a coarser grid the staurosporine single layers scored 1-2 points lower and
  stacking appeared to gain 1.2-1.6 points; the finer grid is used here, and the staurosporine gains
  shrink to 0.7-1.1. Choosing the penalty on the inner split remains noisier for staurosporine than
  for the other responses, so its single-layer figures still sit up to 1.6 points below a fit whose
  penalty was chosen on the test folds (the optimistic figure used in some earlier tables).
- **Nine of 81 layers.** Only every tenth hidden state was pooled; finer sampling could move the best
  single layer and the stacking weights slightly.
- **Linear readouts only.** Stacking was not combined with the neural-network readouts, which gained a
  similar 1-2 points on heat and salt; whether the two gains add is untested.
- **The weights are descriptive.** They show which layers carry non-redundant information under ridge, not
  where in the network a property is computed.

# Methods files

`code/ad06_layers.py`, `slurm/ad06_layers.sbatch`; results in `data/ga_data/ad06_results.json`.
