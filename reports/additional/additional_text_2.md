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

Chaperone client strength comes from Taipale et al. 2012 (*Cell* 150:987, "Quantitative analysis of
Hsp90-client interactions reveals principles of substrate recognition"; Table S1, sheet "Kinases",
provided by hand): for 439 kinases, the LUMIER HSP90 interaction score, dissociation from HSP90 after
1 h of the HSP90 inhibitor ganetespib, protein level after 20 h of ganetespib (low = stability depends
on HSP90), client class (98 strong, 95 weak, 121 non-clients) and CDC37 luminescence. 211 of the 346
GA_33 kinases are in the table; the others (CDC7, JAK2, MET, KIT and so on) were not part of that
survey. The follow-up network of Taipale et al. 2014 (*Cell* 158:434) covers only about 40 of these
kinases and is used as a consistency check.

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


**5. Staurosporine depletes the kinases that are NOT HSP90 clients.** Release from the HSP90-CDC37
machinery predicts that strong, HSP90-dependent clients lose the most. The opposite holds:

| Spearman rho with the response | 35 °C | 37 °C | 43 °C |
|:--|--:|--:|--:|
| HSP90 interaction score | +0.18 (p 0.01) | +0.16 (0.03) | **+0.24 (6e-4)** |
| Dissociation from HSP90 after ganetespib | -0.16 (0.03) | -0.15 (0.04) | **-0.23 (1e-3)** |
| Protein level after ganetespib | -0.18 (0.02) | -0.17 (0.03) | **-0.20 (8e-3)** |
| CDC37 luminescence | +0.08 (0.3) | +0.08 (0.3) | +0.18 (0.01) |
| Kinases | 184 | 189 | 198 |

*Positive rho with the HSP90 score: stronger clients lose less. Dissociation and protein level are
lower for stronger, more HSP90-dependent clients, so negative rho says the same.*

| Mean staurosporine response by client class | Strong | Weak | Non-client | Kruskal p |
|:--|--:|--:|--:|--:|
| 35 °C | -0.06 (47) | -0.07 (50) | **-0.29** (87) | 0.06 |
| 37 °C | -0.12 (48) | -0.13 (53) | **-0.32** (88) | 0.2 |
| 43 °C | -0.03 (51) | -0.20 (55) | **-0.37** (92) | 0.006 |

The kinase groups agree: CAMK kinases, only 7 % strong clients, are depleted most (-0.51 at 37 °C),
while TKL and tyrosine kinases, 50-60 % strong clients, barely change (-0.08 and -0.12). In a joint rank
model with kinase group, Kd and pull-down level, the HSP90 score adds a small independent positive term
(+0.17 [+0.02, +0.32] at 35 °C, +0.16 [-0.01, +0.30] at 37 °C, +0.11 [-0.05, +0.26] at 43 °C), while Kd and
level keep their effects. The smaller Taipale 2014 table points the same way (CDC37 and HSP90 scores
+0.22 to +0.31 with the response, about 40 kinases, not significant).

HSP90 client strength is unrelated to each kinase's level in the control pull-downs (rho +0.03, -0.00,
-0.19), so the level is not a measure of HSP90 engagement.

# Interpretation

The kinase depletion follows drug binding: it tracks binding affinity, within kinase groups as well as
between them, more strongly at 43 °C, along a graded curve that fits ATP competition in the cell better
than a simple saturation model (not formally tested); and the few kinases staurosporine does not bind
are not depleted, as far as 18 of them can show.

It is not release from HSP90. The kinases depleted are mostly *not* HSP90 clients: non-clients lose
most, strong HSP90-CDC37 clients least, consistently across four client measures and three
temperatures. One reading is that drug binding stabilises the kinase fold and releases kinases that
DNAJA1 and DNAJB11 hold as folding intermediates at the Hsp40/Hsp70 stage, while strong HSP90 clients
are held in HSP90-CDC37 complexes and have less to lose from these co-chaperones. This is a hypothesis.

Two further facts are independent of affinity and client strength. Kinases most present in the control
co-chaperone pull-downs lose the most; that level does not track HSP90 client strength, and whether it
reflects cellular abundance or engagement with these co-chaperones needs the input lysate. And the AGC
kinases with lipid- and nucleotide-sensing regulatory domains go the other way despite tight binding,
which marks them as mechanistically different.

# Critical evaluation

- **Binding data.** Three independent studies agree at Spearman 0.93-0.98, so the Kd values are not
  the weak point. They are measured on isolated kinase domains in vitro, not in cells.
- **The non-binder control is small.** 18-19 kinases; it is consistent with release but does not
  establish that binding is necessary.
- **Effect sizes.** Affinity explains a few percent of the variance among kinases (partial rho 0.11-0.24).
  Most of the kinase-to-kinase variation lies elsewhere, notably in the pull-down level and group.
- **What the pull-down level means.** It mixes how much of a kinase the cell contains with how strongly
  it engages the co-chaperones. It does not track HSP90 client strength. The input lysate would
  separate abundance from engagement.
- **AGC kinases.** Their rise is clear but unexplained. One possibility to test: ATP-site inhibitors are
  known to change the regulatory state of some AGC kinases (for example by protecting activation-loop
  phosphorylation), which could alter their chaperone engagement in the opposite direction.
- **Client strength.** Measured by LUMIER in HEK293 cells on tagged constructs, a different system from
  these pull-downs; 211 of 346 kinases covered. The effect is consistent across all four measures and
  three temperatures but modest (rho 0.15-0.24). The stabilisation reading is a hypothesis.

# Next steps

1. Ligand-induced stabilisation: kinase-domain thermal shifts with staurosporine (as measured by
   Fedorov et al. 2007) should predict depletion if stabilisation drives release.
2. The input lysate: depletion from the pull-down relative to depletion from the lysate.

# Methods files

`code/ad02_kinase_affinity.py`, `code/ad02b_taipale.py` (2014), `code/ad02c_taipale2012.py` (2012); ChEMBL extracts in `data/external/kinase/`; results in
`data/external/kinase/ad02_results.json`.
