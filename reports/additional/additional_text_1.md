---
title: "Additional text 1: does the heat response follow measured protein thermal stability?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

The sequence and sparse-autoencoder analyses of the GA_33 heat response (co-chaperone pull-downs at
43 vs 35 °C, DNAJA1 and DNAJB11 averaged) found that folded, globular enzymes rise and disordered and
membrane proteins fall. The main report read this as heat drawing in "the proteins that partly
unfold when heated". That reading makes a testable prediction: proteins with lower measured thermal
stability should rise more.

# Data

Melting temperatures (Tm) from the Meltome atlas (Jarzab et al. 2020, *Nature Methods*), human part:
ten cell lines and cell types (HepG2, Jurkat, K562, U937, HAOEC, HL60, HEK293T, colon-cancer
spheroids, HaCaT, primary T cells), as redistributed by the FLIP benchmark from the atlas server.
For each protein, replicate values were combined within a cell line, each cell line was centred on
its own median, and the median across cell lines was taken. This gives 9,792 proteins (median 51.6 °C,
interquartile range 48.7-55.4 °C). The consensus Tm is highly reproducible: two consensus values
built from disjoint halves of the cell lines correlate at 0.92, implying a reliability of 0.96 for
all ten. 87 % of the GA_33 proteins have a Tm.

These Tm values come from thermal proteome profiling, which measures the temperature at which half of
a protein becomes insoluble after a three-minute heat pulse. That is closer to heat-induced
aggregation than to unfolding as such.

# Results

**1. Thermal stability barely predicts the 43 °C response.**

| Response | Proteins with Tm | Spearman rho with Tm | p |
|:--|--:|--:|--:|
| Heat, 43 vs 35 °C | 7,131 | -0.04 | 8e-4 |
| Heat, 37 vs 35 °C | 7,130 | **-0.13** | 5e-29 |
| Staurosporine, 37 °C (control) | 7,291 | +0.06 | 5e-7 |
| DNAJB11 vs DNAJA1 preference | 7,113 | +0.19 | 1e-56 |

**2. At 43 °C the relation is an inverted U.** Mean heat response by decile of Tm:

| Tm (°C) | <46.7 | -47.9 | -48.9 | -49.9 | -51.0 | -52.2 | -53.6 | -55.5 | -58.4 | >58.4 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 43 vs 35 °C | +0.11 | +0.24 | +0.41 | +0.38 | +0.37 | +0.34 | +0.27 | +0.21 | +0.10 | +0.02 |
| 37 vs 35 °C | +0.07 | +0.07 | +0.08 | +0.03 | +0.03 | +0.02 | +0.01 | -0.01 | -0.03 | -0.04 |

The rise is largest for proteins melting at 48-51 °C and smaller both for the least and the most
stable. At 37 °C the relation is monotonic, as originally predicted: the least stable proteins rise
most.

| Model of the response on Tm | Linear term | Quadratic term |
|:--|--:|--:|
| 43 vs 35 °C | -0.013 [-0.046, +0.023] | **-0.073 [-0.095, -0.050]** |
| 43 vs 35 °C, adjusted for disorder and membrane features | -0.038 [-0.074, -0.002] | **-0.056 [-0.077, -0.036]** |
| 43 vs 35 °C, adjusted for abundance | +0.025 [-0.005, +0.056] | **-0.072 [-0.094, -0.051]** |
| 37 vs 35 °C | **-0.038 [-0.046, -0.028]** | +0.002 [-0.004, +0.008] |
| 37 vs 35 °C, adjusted for abundance | **-0.032 [-0.041, -0.022]** | +0.002 [-0.004, +0.008] |

*Tm standardised; 95 % intervals resampling sequence clusters. The fitted 43 °C peak lies at 51 °C
(50 °C after adjustment).*

The inverted U appears equally with Tm from HEK293T, K562 or Jurkat alone, so it does not depend on
the cell line. Part of the raw curve comes from protein class (membrane proteins fall with heat at
every Tm; nuclear proteins hardly change), but the curvature remains after adjusting for the
disorder and membrane features.

**3. Thermal stability does not explain what sequence predicts.** On the 7,115 proteins with both,
cross-validated over sequence clusters:

| Predictor of the 43 °C response | Variance explained |
|:--|--:|
| Tm (cubic) | 1.2 % |
| Sequence model (ESMC, MLP + ridge, out of fold) | 25.2 % |
| Both | 26.0 % (Tm adds +0.8 [+0.3, +1.3]) |

The sequence model's predictions are uncorrelated with Tm (r = +0.01), so what sequence predicts does
not run through thermal stability. The SAE features that carried
the interpretation keep their effect unchanged once Tm is in the model: disorder features (slopes
-0.17 and -0.16 without Tm, -0.19 and -0.20 with), the membrane feature (-0.21, unchanged), and the
enzyme-core features (+0.16 and +0.20, unchanged or larger).

**4. The proteins Meltome does not cover fall with heat.** The 13 % of GA_33 proteins without a Tm
have a mean 43 °C response of -0.43, against +0.24 for those with one (p = 4e-95). They are
over-represented for homeobox, DNA-binding and developmental proteins, cilium proteins and secreted
proteins, and 89 % carry the long-disorder feature (76 % of the others): mostly disordered
transcription factors that thermal profiling cannot melt.

**5. Abundance: a real dependence, but not what sequence predicts.** As for salt, the heat response
depends on how much of a protein the 35 °C pull-downs contain: proteins scarce at 35 °C rise most.
The dependence is real rather than shared noise (correlation -0.39 with abundance and response taken
from disjoint replicates, against -0.40 from the same ones), and abundance explains 16.5 % of the
43 °C response. It does not carry the sequence signal. The embedding predicts abundance, but with
abundance projected out of the response it explains more of what remains:

| ESMC final layer, cross-validated over sequence clusters | 43 vs 35 °C | 37 vs 35 °C |
|:--|--:|--:|
| Abundance alone (disjoint replicates) | 16.5 % | 1.9 % |
| ESMC predicting the response | 25.2 % | 8.2 % |
| ESMC predicting abundance | 24.6 % | 25.0 % |
| ESMC predicting the response with abundance projected out | **28.9 %** | **8.8 %** |

Abundance does not explain the Tm curvature either (quadratic term unchanged after adjustment).

# Interpretation

The heat response is not, in the main, a readout of thermal stability. Measured stability explains
about 1 % of it, and the sequence model's 25 % does not run through stability. The categories the model
uses (folded cytoplasmic enzymes up; disordered and membrane proteins down) operate independently of
how readily a protein melts.

Stability does matter in a narrower, temperature-dependent way. At 37 °C the least stable proteins
gain the most co-chaperone association, as expected if mild heat destabilises them first. At 43 °C
the gain shifts to proteins melting around 48-51 °C, 5-8 °C above the heat-shock temperature,
and falls off for the least stable. One reading is a window: proteins close to their melting point
at 43 °C leave the soluble pool or are captured by other chaperones, while proteins somewhat more
stable are partly destabilised and become co-chaperone clients. The data cannot yet distinguish this
from other explanations. The least stable proteins are still detected at 43 °C (97-99 % of them, as
often as at 35 °C), so they are not lost outright, and only the input lysate can show whether their
total amount falls.

**Correction to the main report.** Its sentence that heat draws in "the proteins that partly unfold
when heated" is not supported by measured thermal stability, and has been replaced there by a
statement of what the features show and a pointer to this text.

# Critical evaluation

- **Is the test fair?** The prediction was tested with a direct, independent and highly reproducible
  measurement (reliability 0.96) on 87 % of the proteins, in three single cell lines as well as the
  consensus, with sequence-cluster intervals. The negative result for the main interpretation is
  robust.
- **What the Tm measures.** Thermal proteome profiling reports loss of solubility after a brief heat
  pulse in cultured cells. The GA_33 heat treatment (its duration is not recorded in the data file) is unlikely to match a
  three-minute pulse in cultured cells. A protein's Tm is a stability ranking, not a prediction of its state in these cells.
- **Cell line.** The GA_33 cell line is not stated here; the shape is the same in HEK293T, K562 and
  Jurkat, which limits but does not remove this concern.
- **Coverage bias.** The proteins without a Tm are not a random 13 %: they fall strongly with heat and
  are mostly disordered. Conclusions about Tm apply to measurably melting proteins.
- **Size of the effects.** The curvature is statistically clear but small: two standard deviations
  of Tm from the peak lowers the expected response by about 0.2-0.3 log2 units, against a response
  spread of 1.1. The window interpretation is a hypothesis, not a finding.
- **Abundance.** The heat response depends on abundance as strongly as the salt response did. This
  was checked here (point 5) and does not change the sequence result; the main report's heat numbers
  stand, and are if anything conservative.
- **What would settle it.** The input lysate (does total soluble protein fall for the least stable
  proteins at 43 °C?) and the insoluble fraction, if collected.

# Methods files

`code/ad01_meltome.py`, `code/ad01b_meltome_checks.py`, `code/ad01c_heat_abundance.py`; results in
`data/external/meltome/ad01_results.json`, `ad01b_results.json`, `ad01_shape.json`.
