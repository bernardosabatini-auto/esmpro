---
title: "Additional text 3: is there one axis shared by all the responses?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

The main report found recurring classes: membrane proteins rise with salt in the HSPB1 pull-down,
prefer DNAJB11 over DNAJA1, and fall with heat; disordered, charged proteins tend the other way.
Is there one axis of protein properties behind all the responses, and how much of each response
is left once it is removed?

# Data and method

6,331 proteins measured in both experiments and kept for all eight reproducible responses: salt
(75 and 150 mM, HSPB1), heat (37 and 43 vs 35 °C), staurosporine (35, 37, 43 °C), and the DNAJB11-versus-DNAJA1
preference; the GA_33 responses averaged over the two co-chaperones.

Correlations between responses need care. The salt and GA_33 experiments are independent, so their
correlation is corrected for measurement noise with each response's reliability in the usual way.
Within GA_33 the responses reuse the same control samples (heat at 43 °C is control 43 minus
control 35; staurosporine at a temperature is treated minus control at that temperature), so their
measurement noise is shared and would bias their correlation, in a direction set by that arithmetic.
Within GA_33, therefore, each response was computed separately from biological replicates 1-3 and
4-6, and the correlation of one response from one half with the other from the other half was used,
corrected for the reliability of a three-replicate estimate. The two salt doses also share
their 0 mM samples, but measurement noise is about 1 % of the salt variance, so this is negligible
there. Intervals resample sequence clusters.

# Results

**1. The responses are largely independent of one another.** Correlations, free of shared noise and
corrected for measurement noise:

| | salt 75 | salt 150 | heat 37 | heat 43 | stau 35 | stau 37 | stau 43 | DNAJB11 pref. |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| salt 75 | 1 | | | | | | | |
| salt 150 | +0.92 | 1 | | | | | | |
| heat 37 | -0.17 | -0.13 | 1 | | | | | |
| heat 43 | -0.27 | -0.24 | +0.78 | 1 | | | | |
| stau 35 | +0.01 | +0.03 | +0.10 | -0.04 | 1 | | | |
| stau 37 | -0.04 | -0.04 | -0.11 | -0.01 | +0.92 | 1 | | |
| stau 43 | +0.02 | +0.02 | -0.02 | -0.07 | +0.79 | +0.76 | 1 | |
| DNAJB11 pref. | -0.05 | -0.04 | +0.31 | +0.29 | -0.23 | -0.11 | -0.08 | 1 |

The strong correlations are within a perturbation: the two salt doses, the two heat temperatures,
and the three staurosporine temperatures. Between perturbations the only appreciable links are salt
with heat (-0.24 [-0.27, -0.21] for 150 mM and 43 °C) and heat with the DNAJB11 preference (+0.28
[+0.26, +0.30] from disjoint halves). Staurosporine is unrelated to salt and to heat. The
principal components of the corrected matrix are, in order, the staurosporine response (34 %), salt
against heat (29 %), and a mixture of salt, heat and preference (19 %).

**2. The other experiment explains little of each response.** Cross-validated over sequence
clusters, the responses of one experiment predicting each response of the other:

| Response | Explained by the other experiment | ESMC on the response | ESMC on what is left |
|:--|--:|--:|--:|
| Salt 75 mM | 7.4 % | 17.0 % | 13.4 % |
| Salt 150 mM | 6.4 % | 18.5 % | 14.7 % |
| Heat 37 °C | 2.8 % | 11.2 % | 9.5 % |
| Heat 43 °C | 7.1 % | 21.8 % | 18.3 % |
| Staurosporine 35/37/43 °C | 0 % | 5.0-7.1 % | 4.9-7.0 % |
| DNAJB11 preference | 0 % | 44.3 % | 44.3 % |

Most of what sequence predicts for each response is specific to that response.

**3. The salt-heat link is explained by each protein's level in the two pull-downs, not by a shared protein property.** The link does follow protein
class: membrane proteins sit at one end (rise with salt, fall with heat; class mean +0.83 on the
standardised salt-minus-heat score, p = 2e-103), proteins found only in the cytoplasm at the other
(-0.38), and it persists without membrane proteins (-0.21). But each response depends on how much of
a protein its own pull-down contains, and the two pull-downs' contents are correlated (+0.59 across
proteins). What drives the link is a protein's level in the HSPB1 pull-down relative to the DNAJ
pull-downs, which correlates -0.53 with the salt response and +0.53 with the heat response:

| Salt 150 mM vs heat 43 °C | Correlation |
|:--|--:|
| Raw | -0.241 |
| Each adjusted for its own pull-down's level | -0.107 |
| Both adjusted for both pull-downs' levels | **-0.013** |

Proteins relatively enriched in the HSPB1 pull-down lose HSPB1 association with salt and gain DNAJ
association with heat, and proteins relatively enriched in the DNAJ pull-downs do the reverse. Each
perturbation erodes what is specifically enriched in its own pull-down, and that, not a shared
sequence property, is what links the two experiments.

**4. The class effects within each experiment are not abundance effects.** Removing each response's
dependence on its own pull-down level leaves the sparse-autoencoder readings of the main report in
place:

| Response | Feature or class | Raw | Abundance removed |
|:--|:--|--:|--:|
| Salt 150 mM | Transmembrane helix exit signature (SAE 9620) | +0.190 | +0.186 |
| | Hydrophobic membrane helices (SAE 11444) | +0.181 | +0.174 |
| | Transmembrane proteins vs others (log2) | +0.46 | +0.39 |
| Heat 43 °C | Long low-complexity IDRs (SAE 654) | -0.189 | **-0.291** |
| | Mixed-charge low-complexity IDRs (SAE 10715) | -0.164 | -0.237 |
| | Basic/polar N-terminal segments, enzymes (SAE 5338) | +0.195 | +0.224 |
| | Alpha-helical transmembrane segments (SAE 5659) | -0.177 | -0.128 |
| | Transmembrane proteins vs others (log2) | -0.52 | -0.34 |

The disorder effect on the heat response is stronger once abundance is removed; part of the
membrane effect on heat is abundance, but most of it remains.

**5. A sequence-only membrane-versus-disorder axis tracks one response.** Scoring each protein by the
mean activation of 107 transmembrane features minus that of 90 disorder and low-complexity features
(chosen by their Biohub category and label, not by the data), the score correlates +0.43 with the
DNAJB11 preference, +0.10-0.12 with salt, and about 0 with heat and staurosporine. With heat both
membrane and disordered proteins fall, so a membrane-minus-disorder contrast cancels.

# Interpretation

There is no single protein axis behind the four perturbations. Each response has its own sequence
determinants (membrane helices for salt, folded-versus-disordered for heat, kinase domains for
staurosporine, secretory-versus-nuclear for co-chaperone preference), and they overlap little. The
one link between experiments, salt against heat, is explained by how each protein is distributed
between the two pull-downs: a perturbation removes what its own bait specifically enriched. That is
worth knowing for any comparison across these pull-downs, because it will reappear wherever two
pull-downs with different baits are compared.

# Critical evaluation

- **Shared samples.** The GA_33 responses reuse the same control samples. Using disjoint replicate
  halves removes the shared noise but leaves only three replicates per estimate, and the correction
  for that divides by a half-replicate reliability (0.4-0.98), which inflates the noisier
  staurosporine correlations. The intervals quoted for the GA_33 pairs are before that correction.
- **Abundance as an explanation.** "Level in the pull-down" mixes cellular abundance with binding to
  the bait. The input lysate would separate the two; until then, "each perturbation erodes what its
  bait enriched" is the most direct reading, not a demonstrated mechanism.
- **The abundance adjustment is quadratic.** A more flexible adjustment could remove a little more,
  but the salt-heat correlation is already at -0.01.
- **Coverage.** Only proteins measured in both experiments and kept for all eight responses enter;
  this excludes the least abundant and least reproducibly measured proteins.

# Methods files

`code/ad03_shared_axis.py` (correlations, components, the sequence-only axis),
`code/ad03b_shared_axis.py` (shared-noise-free correlations, cross-experiment prediction, the
salt-heat axis and its abundance analysis); results in `data/ga_data/ad03_results.json` and
`ad03b_results.json`.
