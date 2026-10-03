---
title: "Additional text 5: proteins that appear or vanish between conditions"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

The fold-change targets keep a protein only when it is detected in most replicates of both conditions
(at least 4 of 6 in GA_33, 7 of 10 in the salt titration). Proteins detected in one condition and
essentially absent in the other are left out, though they may be the strongest responses of all. How
many are there, are they real, are they the extremes of the same responses the models predict, and
what are they?

# Definitions

A protein **appears** when it is detected in at least 4 of 6 replicates of the condition and at most 1
of 6 of the reference, and **vanishes** in the reverse case (salt: at least 7 of 10 against at most 2 of
10). In GA_33 the DNAJA1 and DNAJB11 pull-downs are independent samples, so an event seen in both is a
reproducible one, and agreement beyond chance tests whether on/off calls reflect biology or flicker at
the detection floor.

# Results

**1. Two large, reproducible sets.**

| GA_33 contrast | Event | DNAJA1 | DNAJB11 | Both | Expected by chance | Enrichment |
|:--|:--|--:|--:|--:|--:|--:|
| Heat 43 vs 35 °C | appears | 323 | 262 | **131** | 8.4 | 15.6x |
| Heat 43 vs 35 °C | vanishes | 10 | 14 | 1 | 0.0 | |
| Heat 37 vs 35 °C | appears | 77 | 99 | 4 | 0.7 | 5.6x |
| Staurosporine, 35/37/43 °C | appears or vanishes | 5-45 | 5-32 | 0-2 | | |

| Salt titration (HSPB1, one pull-down) | Appears | Vanishes |
|:--|--:|--:|
| 150 vs 0 mM | 5 | **114** |
| 75 vs 0 mM | 4 | 13 |

At 43 °C, 131 proteins appear in both co-chaperone pull-downs independently, 16 times more than chance,
and hardly any vanish. At 150 mM salt, 114 proteins leave the HSPB1 pull-down and 5 enter it. Both sets
point in the direction of the corresponding quantitative response (heat brings proteins into the
co-chaperone pull-downs; salt removes them from HSPB1's). The other contrasts have too few events to use.

The heat events are not a loading effect. The 43 °C pull-downs carry more material overall (run medians
+0.9 log2), which could push faint proteins over the detection threshold in both pull-downs at once. But
the appearing proteins reach a median of 12.3 log2 at 43 °C, at least 1.3-1.4 log2 above the lowest
intensities detected at 35 °C (a lower bound, since their 35 °C level is below detection), and kept
proteins of the same 43 °C intensity, which were detectable at 35 °C, *fell* on average (-0.30).

**2. They are the extremes of the sequence-determined responses.** A ridge model of the quantitative
response, trained only on kept proteins and with no relative (30 % identity) of any event protein in its
training set, was applied to the event proteins and compared with kept proteins matched on intensity
on the side where the event proteins are detected (five nearest each):

| | Heat 43 °C, appearing (131) | Salt 150 mM, vanishing (114) |
|:--|--:|--:|
| Predicted response, event proteins | **+0.35** | **-0.26** |
| Predicted response, matched kept proteins | -0.01 | -0.02 |
| AUC, event vs matched (95 % interval) | **0.70 [0.65, 0.74]** | **0.66 [0.60, 0.72]** |
| Classifier trained on events directly, sequence (AUC) | **0.76** | 0.66 |
| Same, intensity alone (AUC) | 0.50 | 0.47 |

*Classifiers: logistic regression on the ESMC embedding (final layer for heat, layer 50 for salt), folds
grouped by sequence cluster.*

The model predicts the appearing proteins to rise and the vanishing ones to fall, without having seen
them or their relatives, and intensity alone does not separate them from the matched kept proteins.
The stricter the event, the better sequence picks it out: requiring 5 of 6 at 43 °C and none at 35 °C
(33 proteins) raises the AUC to 0.80, and 6 of 6 against none (20 proteins) to 0.83.

**3. What they are.** Compared with intensity-matched kept proteins:

- **Appearing at 43 °C:** depleted of membrane proteins (transmembrane 5/131 against 96/583, odds ratio
  0.2, p = 4e-5) and enriched in cytoplasmic proteins (odds ratio 1.7, p = 0.01), as the sparse-autoencoder
  analysis of the heat response found (folded cytoplasmic proteins rise, membrane proteins fall). Their
  melting temperature is no different from that of kept proteins (median 51.1 against 51.0 °C), as in
  additional text 1: the heat response is not a readout of thermal stability.
- **Vanishing at 150 mM salt:** depleted of membrane proteins (8/114 against 98/517, p = 0.001), the class
  whose HSPB1 association survives salt, and enriched in coiled-coil proteins (odds ratio 2.0, p = 0.009).

# Interpretation

The on/off proteins are not a separate phenomenon. They are the far ends of the quantitative heat and
salt responses, carrying the same protein classes, and the sequence models trained without them place
them correctly. Leaving them out of the fold-change targets truncates the responses at their extremes; the models,
trained without them, still place them correctly. The heat set is worth listing as a candidate set
of heat-induced co-chaperone clients: 131 proteins detectable in both pull-downs only at 43 °C, listed
with their detection counts, intensities and predicted response in
`reports/additional/heat43_appearing_proteins.tsv`.

# Critical evaluation

- **Thresholds.** The definitions are arbitrary; tightening them shrinks the sets but strengthens every
  result (AUC 0.70 to 0.83).
- **The sensitivity figures for stricter definitions** use a model trained on all kept proteins, relatives
  included; the strict out-of-cluster AUC is the 0.70 for the main definition.
- **Keyword tests** are not corrected for the number of keywords examined; the membrane depletions
  (p = 4e-5 and 0.001) would survive any reasonable correction, the weaker ones may not.
- **No input lysate.** Whether appearing proteins are newly bound or newly abundant cannot be told from
  pull-downs alone.

# Methods files

`code/ad05_onoff.py` (event counts and reproducibility) and `code/ad05b_onoff_models.py` (generalisation,
classification, characterisation). Results: `data/ga_data/ad05_counts.json`, `ad05b_results.json`.
