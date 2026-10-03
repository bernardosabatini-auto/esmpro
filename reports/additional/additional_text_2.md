---
title: "Additional text 2: is kinase depletion by staurosporine explained by drug binding?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

10 µM staurosporine depletes protein kinases from both co-chaperone pull-downs (main report): the
346 kinases detected shift by -0.7 to -1.1 standard deviations, while the other ATP-binding proteins
do not move. Some kinases, notably the PKC family, rise instead. If the depletion reflects the drug
binding kinases and releasing them from the chaperone machinery, then (i) kinases that staurosporine
does not bind should not be depleted, and (ii) tighter binders should be depleted more.

# Data

Staurosporine dissociation constants (Kd) from ChEMBL for three kinome-wide competition-binding
studies: Davis et al. 2011 (371 kinases), Karaman et al. 2008 (278) and Fabian et al. 2005 (102).
Measurements on mutant or variant constructs (89 records) were excluded, and each kinase was given
the median log10 Kd across studies. The studies agree closely (Spearman 0.98 Davis vs Karaman on 277
kinases, 0.95 and 0.93 for the comparisons with Fabian), giving 374 kinases with a consensus Kd
(median 62 nM; 30 not bound at 10 µM). 239-255 of the kinases in each GA_33 staurosporine target
have a Kd.

Chaperone client strength comes from Taipale et al. 2014 (*Cell* 158:434, "A quantitative chaperone
interaction network reveals the architecture of cellular protein homeostasis pathways"; supplementary
table NIHMS605110-supplement-02, provided by hand): LUMIER interaction scores of 713 client proteins
against 60 chaperones and co-chaperones, including CDC37, HSP90AB1 and HSP90AA1 (scores above 7 are
significant). 512 clients match a UniProt accession by gene symbol. The kinome-wide client survey of
Taipale et al. 2012 could not be obtained (its supplement is served behind a browser check).

# Results

**1. Tighter binders are depleted more.** Spearman correlation between log10 Kd and the staurosporine
response (positive = tighter binding, stronger depletion):

| Target | Kinases with Kd | rho | p | Binders, Kd < 1 µM | Non-binders, Kd >= 10 µM | Non-kinases |
|:--|--:|--:|--:|--:|--:|--:|
| 35 °C | 239 | +0.16 | 0.01 | -0.21 (195) | +0.04 (18) | +0.00 |
| 37 °C | 245 | +0.14 | 0.03 | -0.25 (201) | -0.04 (18) | +0.02 |
| 43 °C | 254 | **+0.28** | 7e-6 | -0.31 (208) | +0.01 (19) | +0.00 |

*Mean log2 fold change, staurosporine vs control, the two co-chaperones averaged (number of kinases).*

The kinases staurosporine does not bind behave like non-kinases (p = 0.68, 0.07 and 0.93 for the difference), unlike the binders. But there are only 18-19 of them, and the difference
between binders and non-binders reaches significance only at 43 °C (p = 0.07, 0.2 and 0.02). The
control points the right way without proving that binding is required.

**2. The relation holds within kinase groups, and binding is graded in the cell.** Correcting for
kinase group (AGC, CAMK, CMGC, STE, TKL, tyrosine kinases, and others), the partial rank correlation of
Kd with depletion is +0.12 [+0.00, +0.24] at 35 °C, +0.11 [-0.01, +0.23] at 37 °C and
**+0.24 [+0.13, +0.34]** at 43 °C (bootstrap over kinases). At 10 µM nearly every binder below 1 µM
would be saturated in a test tube, which predicts a step near micromolar; the data instead follow Kd
across the nanomolar range. Occupancy computed with the test-tube Kd correlates more weakly with the
response (r = -0.15 at 43 °C) than occupancy computed with Kd raised 1,000-fold (r = -0.26), as
competition with cellular ATP would do.

**3. The AGC exception.** By Kd bin (37 °C), the tightest binders are depleted most, but the
relation is broken by the AGC kinases; without them it falls off steadily with affinity:

| Kd | < 1 nM | 1-10 nM | 10-100 nM | 100 nM-1 µM | 1-10 µM | >= 10 µM |
|:--|--:|--:|--:|--:|--:|--:|
| All kinases | -0.71 (19) | -0.09 (43) | -0.25 (81) | -0.21 (58) | -0.13 (26) | -0.04 (18) |
| Without AGC kinases | **-0.82** (16) | -0.24 (35) | -0.29 (66) | -0.22 (52) | -0.13 (26) | -0.04 (18) |
| AGC kinases | -0.12 (3) | **+0.54** (8) | -0.07 (15) | -0.10 (6) | | |

The AGC kinases that rise sit mostly in the 1-10 nM bin:

| AGC kinase | Kd (nM) | Response | AGC kinase | Kd (nM) | Response |
|:--|--:|--:|:--|--:|--:|
| PRKG1 | 32 | **+4.36** | PRKCD | 1.5 | +0.37 |
| PRKCQ | 8.5 | +2.24 | PRKCA | 50 | +0.33 |
| PRKCH | 4.8 | +1.85 | PRKCE | 0.2 | +0.16 |
| PKN2 | 2.5 | +1.52 | RPS6KB1 | 1.3 | +0.12 |
| PKN1 | 1.3 | +1.24 | AKT2 | 44 | +0.00 |

These kinases bind staurosporine as tightly as any, yet gain co-chaperone association. Whatever
staurosporine does to PKC, PKN and PKG, it is not release. Group by group, CAMK kinases fall most
(mean -0.41, median Kd 16 nM) and AGC kinases rise (+0.07, median Kd 27 nM) at similar affinities.

**4. Kinases most present in the control pull-downs lose the most.** Among kinases, the level in the
control pull-down predicts depletion (Spearman -0.22 to -0.34 with level and response taken from
disjoint replicates, so not shared noise). In a joint rank model of the kinases:

| Target | Kd | Level in control pull-down | R2: group | + Kd | + level |
|:--|--:|--:|--:|--:|--:|
| 35 °C | +0.19 [+0.04, +0.32] | -0.39 [-0.51, -0.26] | 3.9 % | 5.3 % | 18.8 % |
| 37 °C | +0.17 [+0.02, +0.33] | -0.37 [-0.50, -0.25] | 3.2 % | 4.4 % | 17.5 % |
| 43 °C | +0.27 [+0.15, +0.40] | -0.32 [-0.44, -0.21] | 9.7 % | 14.8 % | 25.0 % |

*Standardised rank coefficients with 95 % bootstrap intervals; R2 in sample.*

Affinity and pull-down level contribute independently, and the level is the stronger. Across all
proteins the level does not predict the staurosporine response (rho -0.07; 0.00 from disjoint
replicates), and the kinase depletion is unchanged after adjusting for it (-0.60, -0.90 and -0.98 sd).


**5. Chaperone client strength: underpowered, and the trend runs the other way.** Only about 40
GA_33 kinases have Taipale 2014 scores, because that network focused on other client classes. Among
them, stronger CDC37 and HSP90 clients lose *less* co-chaperone association under staurosporine,
opposite to what release from the HSP90-CDC37 system would predict:

| Spearman rho with the response (n) | CDC37 | HSP90AB1 | HSP90AA1 | HSPA8 | HSPB1 |
|:--|--:|--:|--:|--:|--:|
| 35 °C (41) | +0.22 (p 0.16) | +0.21 (0.19) | +0.23 (0.15) | -0.02 (0.91) | +0.10 (0.56) |
| 37 °C (40) | +0.25 (0.11) | +0.26 (0.11) | +0.24 (0.14) | +0.09 (0.58) | +0.10 (0.54) |
| 43 °C (42) | +0.29 (0.06) | +0.31 (0.049) | +0.23 (0.16) | +0.18 (0.27) | +0.10 (0.55) |

None survives correction for the 15 tests. The kinase groups show the same direction: TKL and
tyrosine kinases are the strongest CDC37 clients (median scores 33 and 10) and barely change, while
CMGC, CAMK and STE kinases are weak clients (0.4-2.7) and the most depleted. CDC37 client strength is
also unrelated to each kinase's level in the control pull-downs (rho +0.09, +0.10, -0.15), so it gives
no support to reading that level as chaperone engagement. A joint model with Kd, level and CDC37 score
cannot be estimated reliably on 32-34 kinases.

# Interpretation

The kinase depletion behaves as drug-driven release from the co-chaperones would: it follows binding
affinity, within kinase groups as well as between them, more strongly at 43 °C, along a graded curve
that fits ATP competition in the cell better than a simple saturation model (not formally tested); and the few kinases staurosporine does not bind are not
depleted, as far as 18 of them can show. Two further facts sharpen the picture. Kinases that are most present in the co-chaperone
pull-downs to begin with lose the most, which is what release would predict if that level reflects
how strongly each kinase engages the chaperone machinery. And the AGC kinases with lipid- and
nucleotide-sensing regulatory domains go the other way despite tight binding, which marks them as
mechanistically different.

# Critical evaluation

- **Binding data.** Three independent studies agree at Spearman 0.93-0.98, so the Kd values are not
  the weak point. They are measured on isolated kinase domains in vitro, not in cells.
- **The non-binder control is small.** 18-19 kinases; it is consistent with release but does not
  establish that binding is necessary.
- **Effect sizes.** Affinity explains a few percent of the variance among kinases (partial rho 0.11-0.24).
  Most of the kinase-to-kinase variation lies elsewhere, notably in the pull-down level and group.
- **What the pull-down level means.** It mixes how much of a kinase the cell contains with how strongly
  it engages the co-chaperone. Reading it as chaperone engagement is a hypothesis; the input lysate
  would separate the two (level in the pull-down relative to level in the lysate).
- **AGC kinases.** Their rise is clear but unexplained. One possibility to test: ATP-site inhibitors are
  known to change the regulatory state of some AGC kinases (for example by protecting activation-loop
  phosphorylation), which could alter their chaperone engagement in the opposite direction.
- **Client strength.** The Taipale 2014 scores cover only ~40 of the kinases. They do not support
  release from the HSP90-CDC37 system specifically (the trend is the opposite, not significant), nor
  the reading of pull-down level as chaperone engagement. The co-chaperones measured here, DNAJA1 and
  DNAJB11, act upstream of HSP90, so the two need not move together. The 2012 kinome-wide survey
  (about 300 kinases) would give a powered test.

# Next steps

1. HSP90-CDC37 client strength for the full kinome (Taipale et al. 2012, *Cell* 150:987): the 2014
   table covers too few of these kinases to decide.
2. The input lysate: depletion from the pull-down relative to depletion from the lysate.

# Methods files

`code/ad02_kinase_affinity.py`, `code/ad02b_taipale.py`; ChEMBL extracts in `data/external/kinase/`; results in
`data/external/kinase/ad02_results.json`.
