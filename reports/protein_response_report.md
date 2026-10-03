---
title: "Predicting chaperone-interactome responses from protein sequence: salt, heat and staurosporine"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \renewcommand{\arraystretch}{1.15}
---

# Summary

Two mass-spectrometry pull-down experiments were used to ask how well a protein's own sequence
predicts how its association with a chaperone changes under a perturbation. Each protein is
represented by the ESMC-6B protein language model, and its log2 fold change is predicted on
proteins whose sequence families were held out of training.

| Response | Pull-down | Ceiling | Best sequence model | Composition alone |
|:--|:--|--:|--:|--:|
| Salt, 150 vs 0 mM | HSPB1 | 99.3 % | **20.4 %** | 7.5 % |
| Salt, 75 vs 0 mM | HSPB1 | 98.6 % | **17.2 %** | 6.2 % |
| Heat, 43 vs 35 °C | DNAJA1, DNAJB11 | 99.2 % | **27.5 %** | 8.9 % |
| Heat, 37 vs 35 °C | DNAJA1, DNAJB11 | 86.5 % | **12.2 %** | 3.5 % |
| Staurosporine, 37 °C | DNAJA1, DNAJB11 | 75.5 % | **6.5 %** | 0.3 % |

*All values are percent of variance explained in held-out proteins. The ceiling is the share of
the variance that is reproducible between independent sets of biological replicates, and so the
most any predictor could explain.*

- **Sequence predicts all three responses, and the heat response best.** The embedding explains
  2.3 times or more what amino-acid composition does, and already contains composition.
- **The responses are far from saturated.** The best models reach 6-28 % of the variance against
  ceilings of 75-99 %, and the gap is not measurement noise.
- **A nonlinear readout helps where the signal is broad.** An ensemble of small neural networks
  averaged with linear regression beats linear regression alone on salt and on heat, but not on
  staurosporine, whose signal is concentrated in a few kinase families.
- **Staurosporine acts on protein kinases.** Kinases are specifically depleted from both
  co-chaperone pull-downs (p = 2e-20), while some, notably the PKC family, rise.

# Data and approach

**Salt titration.** HSPB1 pull-downs from 7,184 proteins at 0, 75 and 150 mM salt, each as 2
technical x 5 biological replicates of log2 intensities. The fold change is the difference of
biological-replicate means, kept when both conditions have at least 7 of 10 values (6,989 proteins
at 75 mM, 6,695 at 150 mM).

**GA_33.** DNAJA1 and DNAJB11 pull-downs at 35, 37 and 43 °C, with and without 10 µM
staurosporine, 6 biological replicates each (9,871 protein groups). Each run is median-normalised,
so fold changes are relative to the pull-down as a whole. Heat responses are measured without
staurosporine; staurosporine responses are measured at each temperature. Proteins are kept when
both groups have at least 4 of 6 values. The two co-chaperones show essentially the same heat and
staurosporine responses (correlations 0.83-1.0 after correcting for noise), so their average is
used as the main target: the same quantity measured with twice the replicates.

**Ceiling.** For each fold change, the biological replicates are split into two disjoint halves,
the fold change is computed from each, and the agreement is corrected to the full replicate
count. This reliability is the largest fraction of variance any predictor could explain.

**Prediction.** Each protein's sequence (all but two digestion enzymes resolved from UniProt) is
embedded with ESMC-6B and averaged over residues; long proteins are embedded in overlapping
windows. Proteins are clustered at 30 % sequence identity, and every model is scored by 5-fold
cross-validation in which whole clusters are held out, so predictions are never made from a close
relative. Models: ridge regression, and an ensemble of 16 two-layer neural networks selected on an
inner held-out set of clusters. Intervals are 95 % bootstrap intervals over sequence clusters.

# Salt sensitivity (HSPB1)

Salt weakens electrostatic interactions, so the fold change measures how salt-sensitive each
protein's association with HSPB1 is. A typical protein moves 1.6-fold at 75 mM and 1.9-fold at
150 mM, and these changes are highly reproducible: 99 % of the variance is explainable in
principle.

**What sequence explains.** Each row adds features to the one above.

| Features | 75 mM | 150 mM |
|:--|--:|--:|
| Protein length | 0.0 % | 0.0 % |
| Amino-acid composition (20 fractions) | 6.2 % | 7.5 % |
| + length | 6.4 % | 8.0 % |
| + charge, hydrophobicity, charge patterning | 6.6 % | 8.2 % |
| ESMC embedding (final layer) | 15.3 % | 17.0 % |
| ESMC embedding + composition | 15.3 % | 17.1 % |
| *Measurement ceiling* | *98.6 %* | *99.3 %* |

Composition alone explains 6-8 %. Charge and hydrophobicity add almost nothing beyond it, which
for a salt-disruption experiment is notable: the obvious electrostatic descriptors carry no
information that the raw composition lacks. The embedding adds a further +9.1 points at 75 mM
[7.7, 10.5] and +9.6 at 150 mM [8.2, 10.9], and appending composition to it gains nothing, so the
embedding contains composition and more. Length explains nothing. Nearest-neighbour lookup in the
same embedding explains only 6-7 %, so the signal is not simply family resemblance.

**Which layer.** Of nine evenly spaced layers of the 81, layer 50 predicts salt sensitivity best
(16.0 % and 18.2 %), about one point above the final layer; the deepest layers track the protein's
abundance rather than its salt response.

**Linear versus nonlinear readout** (ESMC layer 50):

| | 75 mM | 150 mM |
|:--|--:|--:|
| Ridge regression | 16.0 % | 18.2 % |
| Neural-network ensemble | 16.5 % | 20.3 % |
| Ensemble and ridge averaged | **17.2 %** | **20.4 %** |
| Gain of the average over ridge | +1.2 [+0.6, +1.7] | +2.2 [+1.6, +2.8] |

The ensemble alone gains +0.5 [-0.7, +1.7] points at 75 mM and +2.2 [+1.0, +3.4] at 150 mM.
Averaging it with ridge is positive at both doses. Fitting the two doses jointly, although they
correlate at 0.91, gains nothing over fitting them separately.

# Heat response (DNAJA1 and DNAJB11)

Raising the temperature from 35 to 43 °C changes what both co-chaperones pull down by about
2.2-fold for a typical protein, almost free of noise (2 % of the variance), and the two
co-chaperones respond alike (correlation 0.93).

| Target | Ceiling | Composition | ESMC, layer 50 | ESMC, final layer | Ensemble + ridge |
|:--|--:|--:|--:|--:|--:|
| 43 vs 35 °C, averaged | 99.2 % | 8.9 % | 22.5 % | 25.7 % | **27.5 %** |
| 43 vs 35 °C, DNAJA1 | 97.9 % | 10.1 % | 23.0 % | 26.0 % | **27.7 %** |
| 43 vs 35 °C, DNAJB11 | 98.3 % | 8.3 % | 21.4 % | 25.0 % | **27.2 %** |
| 37 vs 35 °C, averaged | 86.5 % | 3.5 % | 10.8 % | 11.4 % | **12.2 %** |

This is the most sequence-predictable response in either experiment. The embedding adds +13.6
points over composition for the averaged 43 °C response [12.3, 14.9], and here the final layer is
better than layer 50, by 3-4 points: the best layer depends on the property being predicted.
Averaging a neural-network ensemble with ridge beats ridge on all four heat targets, by +1.8
[+1.2, +2.4], +2.1 [+1.6, +2.6], +2.5 [+1.9, +3.0] and +0.8 [+0.5, +1.1] points. Which proteins a
co-chaperone binds more of at 43 °C plausibly depends on thermal stability and aggregation
propensity, both encoded in sequence and structure.

# Staurosporine response

10 µM staurosporine changes what the co-chaperones pull down by only about 1.15-fold, and half
of that is replicate noise. The ceiling for a single co-chaperone is 49-64 %; averaging the two
raises it to 71-76 %.

| Target (both co-chaperones) | Ceiling | Composition | ESMC, layer 50 |
|:--|--:|--:|--:|
| 35 °C | 71.7 % | 1.6 % | 6.0 % |
| 37 °C | 75.5 % | 0.3 % | 6.5 % |
| 43 °C | 71.4 % | 0.5 % | 5.7 % |

**Staurosporine acts on protein kinases.** Staurosporine is an ATP-competitive inhibitor of most
protein kinases, and the 346 kinases detected shift as a class. The 673 other ATP-binding proteins
do not, and the heat response shows no kinase signal at all.

| | Kinases | Other ATP-binding | All others | Kinase shift | p | Depleted 5 % |
|:--|--:|--:|--:|--:|--:|--:|
| 35 °C | -0.15 | +0.01 | +0.00 | -0.73 sd | 2e-7 | 6.0x |
| 37 °C | -0.19 | -0.01 | +0.02 | -1.05 sd | 2e-20 | 6.5x |
| 43 °C | -0.20 | -0.01 | +0.00 | -1.12 sd | 5e-13 | 5.1x |
| *Heat (control)* | *+0.11* | *+0.33* | *+0.14* | *-0.03 sd* | *0.56* | *1.3x* |

*Mean log2 fold change. Kinase shift: kinases minus all others, in standard deviations of the fold
change; p from a Mann-Whitney test. Depleted 5 %: how over-represented kinases are among the 5 % most
depleted proteins. Heat (control): the same comparison for the 43 vs 35 °C heat response.*

The most depleted kinases include FER, CAMK1, CAMKK2, CSK, CDK5, CHEK2, CDK2 and STK4 (down 2- to
6-fold). Kinases are also over-represented among the most enriched proteins: the PKC isoforms, PRKG1
and the PKN kinases rise. Staurosporine therefore shifts kinase-chaperone association in both
directions, kinase by kinase.

The embedding already encodes which proteins are kinases, and most of what it predicts at 35 and
37 °C lies outside them (6.0 % and 6.1 % on non-kinases alone). Which kinases go down and which go
up is only weakly predictable from sequence (1-6.5 % across about 300 kinases).

**No nonlinear gain.** Unlike salt and heat, the neural-network readout does not beat ridge
reliably here (-13, -2 and +7 points at the three temperatures). Because the signal sits in a few
kinase families, which families fall in the selection set decides which model looks best. The
43 °C gain comes almost entirely from the PKC family, whose held-out isoforms the network predicts
from their relatives. Ridge regression remains the model to use for this response.

# Using individual samples instead of replicate means

Each protein's measurements were also modelled sample by sample, with a separate noise level for
each run and a noise level that rises for low-abundance proteins (4-fold from the most to the least
abundant fifth). Three ways of letting that noise discount samples were compared against the
plain replicate mean, in the same networks:

| | Salt | Heat | Staurosporine |
|:--|--:|--:|--:|
| Measurement noise, share of variance | 0.4-0.8 % | 2 % | 15 % |
| Weight by measurement noise alone | -4.1 to -1.3 | -1.9 to -0.2 | -3.7 to +7.7 |
| Full likelihood (model error + noise) | -0.3 to +0.2 | -0.2 to +0.3 | -2.1 to +10.8 |
| Network predicts its own uncertainty | -0.2 to +0.6 | -0.2 to +0.5 | -1.8 to +15.1 |

*Change in percent of variance explained, against the same network trained on the replicate mean.*

Weighting by measurement noise alone hurts, because it gives the best-measured proteins up to 16
times the influence and those are not the easiest to predict. Under the full likelihood the
model's own error (27-48 % of the variance on the salt data) dwarfs measurement noise, so the weights come out
nearly uniform and the result equals the plain mean. Staurosporine, where noise is largest, shows a
slightly positive median effect, but its spread is far wider than the effect.

# Conclusions and next steps

Protein sequence, read through a language-model embedding, predicts how chaperone association
responds to salt, heat and staurosporine, in each case well beyond amino-acid composition and in
proteins with no close relative in the training data. Heat is the most predictable (27.5 % of a
99 % ceiling); staurosporine is the least, and its signal is dominated by protein kinases.

1. **Input lysate.** Measuring each pull-down against its own input would separate a change in a
   protein's amount from a change in its binding, the open question for the depleted kinases.
2. **Staurosporine affinity across the kinome.** Measured binding constants are the natural test of
   which kinases go down and which go up.
3. **Several layers at once.** The best embedding layer differs by target by 3-4 points.
