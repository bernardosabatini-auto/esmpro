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
  co-chaperone pull-downs (p = 2e-20), while some, notably the PKC family, rise. The 3.6 % of
  proteins that are kinases carry about half of the response's variance.
- **One nonlinear model serves several temperatures.** For heat, a network trained at 43 °C needs
  only a new linear readout to predict 37 °C. For staurosporine, a network trained at another
  temperature does as well as or better than the temperature's own.
- **Part of what is predicted is the family.** Half to three quarters of the model's success on
  proteins in large families is getting the family's average response right.

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

**Kinases dominate the response.** The 295 kinases are 3.6 % of the proteins but, because their
changes are so large, carry most of the variance, and most of what the model explains comes from
them:

| Staurosporine | Variance in kinases | Explained, from kinases | R2 within kinases | R2, all others |
|:--|--:|--:|--:|--:|
| 35 °C | 44 % | 50 % | 3.5 % | 5.8 % |
| 37 °C | 54 % | 62 % | 1.5 % | 6.6 % |
| 43 °C | 63 % | 81 % | 1.3 % | 3.2 % |

The model captures the kinases mainly by knowing that kinases as a class go down. Which kinases go
down and which go up it barely predicts (1-3.5 % within the kinases); among all other proteins it
explains 3-7 % of a much smaller variance.

**No nonlinear gain.** Unlike salt and heat, the neural-network readout does not beat ridge
reliably here (-13, -2 and +7 points at the three temperatures). Because the signal sits in a few
kinase families, which families fall in the selection set decides which model looks best. The
43 °C gain comes almost entirely from the PKC family, whose held-out isoforms the network predicts
from their relatives. Ridge regression remains the model to use for this response.

# The same proteins at every temperature?

For each pair of responses on the same proteins: how similar the true responses are, how similar
the model's predictions are, and whether the same proteins are well explained. A protein's share of
the explained variance is how much the model reduces its squared deviation from the mean; these
shares add up to R2.

| Pair | r, responses | r, predictions | r, explained | Top-5 % overlap | R2 | R2 from other |
|:--|--:|--:|--:|--:|--:|--:|
| Salt, 75 / 150 mM | 0.91 | 0.91 | 0.76 | 14x | 20.4 % | 17.8 % |
| Heat, 37 / 43 °C | 0.68 | 0.80 | 0.49 | 6.3x | 27.1 % | 17.0 % |
| Staurosporine, 35 / 37 °C | 0.54 | 0.40 | 0.12 | 6.9x | 8.3 % | 1.6 % |
| Staurosporine, 37 / 43 °C | 0.44 | 0.40 | 0.07 | 6.5x | 7.1 % | 1.9 % |
| Staurosporine, 35 / 43 °C | 0.47 | 0.72 | 0.18 | 7.4x | 7.1 % | 6.1 % |

*r, explained: correlation of the proteins' shares of explained variance. Top-5 % overlap: how
often the 5 % best-explained proteins coincide, relative to chance. R2 is for the second response
of each pair; "R2 from other" scores the first response's predictions, rescaled linearly, against
it.*

- **Salt:** the same proteins at both doses. The 75 mM predictions alone explain 17.8 % of the
  150 mM response.
- **Heat:** largely the same proteins. The predictions at 37 and 43 °C correlate at 0.80, the
  best-explained 5 % overlap six-fold beyond chance, and the 37 °C predictions explain 17 % of the
  43 °C response, against 27 % for its own model.
- **Staurosporine:** different proteins at each temperature, apart from a small shared core. The
  responses correlate at 0.44-0.54, but which proteins the model explains barely carries over
  (0.07-0.18), and one temperature's predictions explain only 1.6-6 % of another's.

# One nonlinear model, different linear readouts?

Two tests, on the proteins kept at every temperature, with a fixed network configuration:

- **One network with a separate linear output per temperature**, trained jointly, against separate
  networks.
- **A network trained on one temperature, frozen, with a new linear readout fitted for another.**
  Each is compared with the second temperature's own network, read out the same way.

| | Shared vs separate networks | Network from the other temperature vs own |
|:--|--:|--:|
| Heat, 37 vs 35 °C | +0.1 [-0.2, +0.5] | from 43 °C: +0.4 [-0.6, +1.4] |
| Heat, 43 vs 35 °C | -0.4 [-0.8, +0.0] | from 37 °C: **-3.4 [-4.7, -2.1]** |
| Staurosporine, 35 °C | -0.8 [-1.8, +0.3] | from 37 °C: -0.4 [-1.8, +1.3]; from 43 °C: +1.5 [-0.4, +3.6] |
| Staurosporine, 37 °C | +0.2 [-0.5, +1.0] | from 35 °C: **+1.6 [+0.4, +3.1]**; from 43 °C: **+3.5 [+0.9, +6.4]** |
| Staurosporine, 43 °C | +2.8 [-0.5, +6.2] | from 35 °C: -0.5 [-5.2, +4.9]; from 37 °C: +0.0 [-3.8, +4.5] |

*Change in percent of variance explained, with 95 % intervals over sequence clusters.*

**For heat, yes, one nonlinear model serves both temperatures.** A single network with two linear
outputs matches separate networks to within 0.4 points. A network trained only on the 43 °C
response, frozen and given a new linear readout, predicts the 37 °C response as well as a network
trained on 37 °C. The reverse fails: the 37 °C network loses 3.4 points on the 43 °C response.
The 43 °C response contains the 37 °C pattern and more.

**For staurosporine, a network from another temperature is never worse and sometimes better.** For
37 °C, networks trained at 35 or 43 °C beat 37 °C's own, by 1.6 and 3.5 points. That is the
pattern expected if one sequence determinant underlies all three temperatures and each temperature
measures it with independent noise. The per-protein differences between temperatures above then
reflect that noise more than different biology.

# Protein families

Proteins were grouped by UniProt family (top level; families of at least 15 detected members), and
by subcellular location.

| | Salt, 150 mM | Heat, 43 vs 35 °C | Staurosporine, 37 °C |
|:--|--:|--:|--:|
| Proteins in families of 15 or more | 972 (23 families) | 1,327 (34) | 1,308 (34) |
| Share of the true response that lies between families | 16 % | 20 % | 6 % |
| Model, R2 on these proteins | 20.6 % | 29.3 % | 6.5 % |
| R2 from each family's average prediction alone | 13.6 % | 14.7 % | 4.9 % |
| Correlation within families | 0.29 | 0.43 | 0.13 |

For proteins in large families, half to three quarters of what the model explains comes from
getting each family's average response right; the rest comes from ranking proteins within a family,
which works best for heat (r = 0.43).

**Best and worst predicted families** (share of their deviation from the overall mean explained):

| Response | Best predicted | Worst predicted |
|:--|:--|:--|
| Salt, 150 mM | mitochondrial carriers (88 %, up), KRAB zinc fingers (49 %, up), class I aminoacyl-tRNA synthetases (52 %, down), ABC transporters (42 %) | small GTPases, DEAD-box and other helicases, myosins and kinesins (below 0 %) |
| Heat, 43 vs 35 °C | mitochondrial carriers (77 %, down), MFS and P-type transporters (67 %, down), deubiquitinases (57 %, up; within-family r = 0.79), dynamins (53 %, up), class II aminoacyl-tRNA synthetases (48 %, up) | tubulins (true down, predicted up), cyclins, actins, ABC transporters |
| Staurosporine, 37 °C | protein kinases (3.6 % of proteins, 62 % of what is explained), ATP-dependent AMP-binding enzymes, translation GTPases, AAA ATPases | dynamins, SAM methyltransferases, actins |

Membrane transporters stand out in both salt and heat, in opposite directions: they rise in the
HSPB1 pull-down with salt and fall in the co-chaperone pull-downs with heat, and in both the model
predicts the family average well. Cytoskeletal structural proteins (tubulin, actin) are the most
consistently mispredicted.

**By compartment** the heat response is predicted about equally well everywhere (20-30 %, lowest for
the cytoskeleton). For staurosporine, endoplasmic-reticulum proteins are predicted best (17 %, within
r = 0.41) and secreted proteins worst (about 0 %). For salt, mitochondrial, Golgi and membrane
proteins (25-28 %) are predicted better than cytoplasmic and cytoskeletal ones (17 %).

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
