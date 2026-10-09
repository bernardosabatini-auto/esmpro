---
title: "Predicting how chaperone binding responds to salt, heat and staurosporine from protein sequence"
date: "8 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{float}
  - \renewcommand{\arraystretch}{1.15}
---

# Summary

Recombinant, tagged chaperones were added to a human cell lysate and pulled down with the proteins
they bind, under different conditions applied to the lysate: salt (HSPB1), and temperature and the
kinase inhibitor staurosporine (DNAJA1 and DNAJB11). Mass spectrometry measured how much of each
protein each pull-down contained. The question is how well a protein's sequence predicts how its
binding changes, and what the predictions reveal.

| Response | Bait | Ceiling | Sequence | Composition |
|:--|:--|--:|--:|--:|
| Co-chaperone preference | DNAJB11 vs DNAJA1 | 96 % | **49 %** | 32 % |
| Heat, 43 vs 35 °C | DNAJA1, DNAJB11 | 99 % | **27 %** | 9 % |
| Heat, 43 vs 35 °C, selective part | DNAJA1, DNAJB11 | 98 % | **13 %** | |
| Salt, 150 vs 0 mM | HSPB1 | 99 % | **20 %** | 8 % |
| Salt, 75 vs 0 mM | HSPB1 | 99 % | **18 %** | 6 % |
| Heat, 37 vs 35 °C | DNAJA1, DNAJB11 | 87 % | **13 %** | 4 % |
| Staurosporine, 37 °C | DNAJA1, DNAJB11 | 76 % | **7 %** | 0 % |

*Percent of variance explained on proteins whose sequence families were held out of training, by the
best sequence model and by the 20 amino-acid fractions alone. The ceiling is the part of the variance
that is reproducible between independent replicates, and so the most any predictor could explain.
The selective part of the heat response is what remains once the shift of the 43 °C pull-downs
toward the lysate composition is removed (section 4).*

- **Sequence predicts every response, well beyond amino-acid composition.** The best results come
  from reading several layers of the ESMC-6B protein language model at once.
- **Each response has its own, interpretable determinants.** Salt spares proteins with transmembrane
  helices. Heat recruits folded cytoplasmic proteins and loses disordered and membrane proteins.
  Staurosporine removes kinases, in proportion to how tightly it binds them. DNAJB11 prefers
  secretory and membrane proteins, DNAJA1 nuclear proteins with charged disordered regions.
- **At 43 °C the pull-downs become half lysate-like.** Measuring the input lysate shows that about half of
  the 43 °C pull-down is in proportion to the lysate, the same for both co-chaperones. This accounts
  for 45 % of the 43 °C response, including the earlier starting-level effect.
- **Less stable proteins gain more with heat.** At matched lysate abundance and 35 °C enrichment,
  the heat gain rises as stability falls. The least stable proteins respond by 37 °C, those of intermediate stability
  between 37 and 43 °C, and the most stable lose share. Stability explains about 5 % of the change.
- **The responses are largely independent of one another.** There is no single protein property
  behind them all.

# 1. The experiments

**Design.** All manipulations were made in vitro. A tagged chaperone was added to cell lysate,
incubated under the condition of interest, and recovered with its bound proteins.

| Experiment | Bait | Conditions (in lysate) | Replicates | Proteins |
|:--|:--|:--|:--|--:|
| Salt titration | HSPB1 | 0, 75, 150 mM salt | 5 x 2 technical | 7,184 |
| GA_33 | DNAJA1, DNAJB11 | 35/37/43 °C, +/- 10 µM staurosporine | 6 | 9,871 |

Because everything happens in lysate, there is no cellular stress response, synthesis, degradation
or relocalisation: a change in a pull-down reflects a change in binding, or in how much of a protein
stays soluble during the incubation. The two co-chaperones meet the same mixture of proteins, so any
difference between them is a difference in what they bind, not in where they normally live
(DNAJB11 is an ER-lumenal protein, DNAJA1 a cytosolic and nuclear one).

**What is measured.** For each protein, the change in its log2 intensity in the pull-down between two
conditions (a log2 fold change), averaged over biological replicates. A protein is kept when it is
detected in most replicates of both conditions. GA_33 runs are median-normalised; the main
conclusions are unchanged under four other normalisations. For GA_33 the input lysate of each
reaction was also measured, which separates how much of a protein the lysate contains from how
strongly it is captured (additional text 8). In GA_33 the heat and staurosporine
responses are averaged over the two co-chaperones, which respond almost identically (correlations
0.83-1.0 after correcting for noise).

**Measurement ceiling.** Each fold change was recomputed from two disjoint halves of the biological
replicates; their agreement, corrected to the full replicate count, is the reliability. It is
99 % for salt and for heat at 43 °C, so these responses are measured almost without noise.
Staurosporine changes binding only about 1.15-fold, and a quarter of its variance is replicate
noise (ceiling 72-76 %).

# 2. Predicting a response from sequence

Each protein's sequence is embedded with ESMC-6B, a protein language model, and averaged over
residues. Ridge regression predicts the response from the embedding. Proteins are grouped into
families at 30 % sequence identity, and every model is scored on families it has never seen, so a
prediction is never made from a close relative. Intervals resample those families.

- **Composition first.** The 20 amino-acid fractions explain 6-9 % of the salt and heat responses and
  32 % of the co-chaperone preference. Charge, hydrophobicity and charge patterning add almost nothing
  beyond composition. The embedding explains 2-3 times as much, and contains composition.
- **Several layers beat one.** The most predictive layer differs by response (the final layer for
  heat and the preference, layer 50 for salt). Combining nine layers with learned weights beats the
  best single layer on every response, by 0.7 to 3 points, and gives the figures in the summary table.
- **Linear is nearly enough.** A small neural network averaged with ridge adds 1-2 points for heat
  and salt; it does not help for staurosporine, whose signal sits in a few kinase families.
- **Sparse-autoencoder features make the predictions readable.** Biohub's sparse autoencoder for
  ESMC-6B (layer 60) decomposes each residue into 64 active features out of 16,384, each with an
  annotated meaning. These features keep 90-114 % of the embedding's predictive power, and the
  features that predict each response are reported below. Each label was checked against the
  annotations of the proteins it actually fires on in these data.

# 3. Salt

Salt weakens electrostatic interactions and strengthens hydrophobic ones. A typical protein's HSPB1
binding changes 1.6-fold at 75 mM and 1.9-fold at 150 mM.

| Features (each row adds to the one above) | 75 mM | 150 mM |
|:--|--:|--:|
| Protein length | 0.0 % | 0.0 % |
| Amino-acid composition | 6.2 % | 7.5 % |
| + charge, hydrophobicity, charge patterning | 6.6 % | 8.2 % |
| ESMC embedding, one layer | 16.0 % | 18.2 % |
| ESMC embedding, nine layers combined | **18.0 %** | **20.4 %** |

**Transmembrane helices resist salt.** Of the 100 sparse-autoencoder features most associated with
the salt response, 94 describe membrane helices ("transmembrane helix exit signature", "multipass
membrane helices"; 13 % of random features; p = 6e-58), all on the side of proteins whose HSPB1 binding
survives or grows with salt. Proteins with a transmembrane helix gain 0.46 log2 relative to others;
long disordered regions go the other way. This is what salt would do to the two kinds of
interaction: binding through hydrophobic helices strengthens, binding through charged disordered
regions weakens.

114 proteins leave the HSPB1 pull-down entirely at 150 mM (5 enter). A model trained without them
or their relatives predicts them to fall (AUC 0.66 against proteins of matched abundance).

# 4. Heat

Raising the temperature of the lysate from 35 to 43 °C changes co-chaperone binding about 2.2-fold
for a typical protein, and overall the pull-downs contain about 1.9 times more protein (run medians
+0.9 log2), which with equal inputs means more total binding at 43 °C. The normalised responses below
describe how each protein's share of the binding changes.

**At 43 °C about half of each pull-down is lysate-like.** Measured against the input lysate, the
pull-downs are selective: the most abundant lysate proteins (glycolytic enzymes, proteasome,
translation factors) are strongly under-represented, and pull-down level rises with lysate level with
a slope of only 0.35. At 43 °C the slope rises to 0.6. Each protein's 43 °C level is close to
0.42 x its 35 °C level + 0.46 x its lysate level. At 37 °C the corresponding fit is 0.93 and 0.07.
Both co-chaperones give the same mixture. Staurosporine, at any temperature, leaves the composition
unchanged (coefficient 1.0 and 0.0). This lysate-like part, predicted for each protein from its lysate
abundance and its 35 °C enrichment, accounts for 45 % of the variance of the 43 °C response; the
rest is called the *selective* response below.

| Heat response | 43 vs 35 °C | 37 vs 35 °C |
|:--|--:|--:|
| Amino-acid composition | 8.9 % | 3.5 % |
| ESMC, one layer (final) | 25.7 % | 11.4 % |
| ESMC, nine layers combined | **27.4 %** | **12.7 %** |
| Lysate-like part, nine layers | 33.4 % | |
| Selective part, nine layers | **13.1 %** | **9.1 %** |

Sequence predicts the lysate-like part well, because it predicts lysate abundance (51 %) and
capture at fixed abundance (23-26 %). The change in capture itself is predicted at 13 %.

**Folded cytoplasmic proteins are recruited; disordered and membrane proteins are lost.** The
features that rise are those of folded enzyme cores (catalase, isocitrate and aldehyde dehydrogenases,
enolase, TIM barrels, Rossmann folds; structural-motif features over-represented, odds ratio 5.5). Those that fall
describe long disordered and low-complexity regions and transmembrane segments (membrane features over-represented,
odds ratio 11.6). These class effects hold in the selective response, at a third to a half of their
size: enzymes +0.14, soluble cytoplasmic proteins +0.09, transmembrane proteins -0.11, strongly
disordered proteins -0.17 log2 relative to the rest.

**The starting-level effect is this loss of selectivity.** Proteins plentiful in the 35 °C
pull-down tend to lose share at 43 °C (r = -0.42). Split into its two parts, that level acts through
enrichment, not abundance. Proteins captured well above their lysate level at 35 °C lose share
(r = -0.68, 46 % of the variance). Abundant lysate proteins, under-represented at 35 °C, gain
(r = +0.33). The 35 °C quantities and the response come from different replicates, so this is not
shared measurement noise.

## 4.1 Less stable proteins gain more

If heat recruits proteins as they begin to unfold, the less stable a protein, the more it should gain.
Melting temperatures for 9,792 human proteins come from the Meltome atlas (Jarzab et al. 2020;
thermal proteome profiling, consensus over ten cell lines, reliability 0.96); 87 % of the heat-response
proteins have one. Proteins were sorted by melting temperature into ten equal groups:

| Melting temperature (°C) | <46.8 | -47.9 | -48.9 | -49.9 | -51.0 | -52.2 | -53.6 | -55.5 | -58.5 | >58.5 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Lysate abundance | -0.54 | -0.32 | -0.20 | -0.10 | +0.09 | +0.01 | +0.07 | +0.45 | +0.60 | +0.76 |
| 35 °C enrichment | +0.51 | +0.21 | +0.18 | +0.01 | -0.11 | -0.09 | -0.05 | -0.14 | -0.34 | -0.53 |
| 43 vs 35 °C, as measured | +0.13 | +0.25 | +0.41 | +0.39 | +0.38 | +0.35 | +0.28 | +0.22 | +0.12 | +0.02 |
| 43 vs 35 °C, selective | +0.17 | +0.19 | **+0.28** | +0.21 | +0.17 | +0.09 | +0.06 | -0.04 | -0.15 | **-0.37** |
| 35 to 37 °C, selective | **+0.10** | +0.08 | +0.09 | +0.03 | +0.03 | +0.00 | +0.00 | -0.02 | -0.04 | -0.07 |
| 37 to 43 °C, selective | +0.06 | +0.11 | **+0.19** | +0.18 | +0.15 | +0.09 | +0.06 | -0.02 | -0.11 | -0.30 |

*Lysate abundance and enrichment: medians relative to all proteins. Responses: means, log2.*

![Mean change in co-chaperone binding by melting temperature. A: the 43 vs 35 °C response as measured
(grey) is the sum of a lysate-like part (tan), which favours stable, abundant proteins, and a
selective part (red), which falls steadily with stability; the selective 37 vs 35 °C response (blue)
does the same, more weakly. B: in the selective 43 °C response, complex subunits (purple) gain more
than other proteins (grey) at intermediate stability. Error bars, one standard error.](figures/heat_tm_selective.png){width=100%}

**As measured, the relation is an inverted U.** The gain peaks for proteins melting at 48-52 °C and
falls for both the most and the least stable.

**The downturn for the least stable proteins comes from the lysate shift.** The least stable proteins
are scarce in the lysate (-0.54) but already enriched at 35 °C (+0.51), and the lysate-like part of the
43 °C pull-down favours abundant, unenriched proteins. With that part removed, the gain falls steadily
as stability rises. The least stable proteins gain about +0.2 and the most stable lose -0.37: a slope of
-0.18 per standard deviation of melting temperature [-0.20, -0.16], against -0.07 as measured. A mild
curvature remains (-0.05 [-0.07, -0.03]).

**Each protein's binding rises over a temperature range set by its stability.** The two steps show
it, and the 35 to 37 °C step is hardly touched by the lysate shift:

- the least stable proteins gain most between 35 and 37 °C, and little after;
- proteins melting near 48-52 °C gain most between 37 and 43 °C;
- the most stable lose share.

**Stability explains part of the selective response.** Melting temperature explains 5.5 % of the
selective 43 °C response, against 1.0 % as measured. The protein classes above act on top of it.
Proteins that the Meltome atlas could not melt, mostly disordered transcription factors, fall with heat
(-0.25, against +0.06 for the rest).

## 4.2 Protein complexes

Heat can also dissociate complexes, turning their subunits into clients whatever their own stability.
Using 2,167 curated human complexes (EBI Complex Portal), 2,471 of the heat-response proteins are
subunits.

- **Subunits gain more at 43 °C**, at matched melting temperature and location: +0.10 log2
  [+0.06, +0.14] in the selective response, against +0.02 at 37 °C. The effect is heat-dependent.
  In the response as measured it is hidden (-0.02), because the lysate-like part runs against
  subunits.
- **The gain sits at intermediate stability** (figure, B). It is clearest for proteins melting at
  50-53 °C.
- **Subunits of a complex move together.** About a fifth of the variance of the selective response is
  shared within a complex (22 % against random proteins, 10 % against subunits of other complexes).
  It is at least as strong at 37 °C, so it is not caused by heat: pull-downs carry intact complexes
  along with whichever subunit the chaperone holds.
- **Which complexes.** Translation initiation factor eIF3 (+1.6, 13 subunits), the COPII vesicle coat
  and GINS gain most. CDK-activating kinase, the SET complex and VCP-NPL4-UFD1 lose most.

Membership accounts for 0.5 % of the selective response, and melting temperature for 5.5 %: both
point to real mechanisms, neither is the main driver.

## 4.3 Proteins that appear at 43 °C

131 proteins are detected in both co-chaperone pull-downs only at 43 °C, 16 times more than chance
would give. They are the extreme of the same response: a model trained without them or their
relatives predicts them to rise (AUC 0.70 against proteins of matched intensity; 0.83 for the
strictest definition), they are cytoplasmic and rarely membrane proteins, and their melting
temperature is ordinary (median 51 °C). 84 % of them are in the input lysate, at below-median
abundance (36th percentile), so they are soluble proteins captured selectively at 43 °C, not the
lysate-like shift, which favours abundant proteins. They are listed as candidate heat-recruited
clients in `reports/additional/heat43_appearing_proteins.tsv`.

# 5. Staurosporine

10 µM staurosporine changes co-chaperone binding only about 1.15-fold, identically for DNAJA1 and
DNAJB11. Sequence explains 4-7 % of it, under a tenth of the ceiling, and almost all of what it explains is
about one class of proteins.

**Staurosporine removes protein kinases.** Kinases are 3.6 % of the proteins but carry 44-63 % of
the variance of the response; other ATP-binding proteins do not move, and the heat response shows no
kinase signal.

| | Kinases | Other ATP-binding | All others | Kinase shift | p |
|:--|--:|--:|--:|--:|--:|
| 35 °C | -0.15 | +0.01 | +0.00 | -0.73 sd | 2e-7 |
| 37 °C | -0.19 | -0.01 | +0.02 | -1.05 sd | 2e-20 |
| 43 °C | -0.20 | -0.01 | +0.00 | -1.12 sd | 5e-13 |

**Depletion follows drug binding.** Staurosporine dissociation constants for 374 kinases (ChEMBL;
three kinome-wide studies agreeing at Spearman 0.93-0.98) show that tighter binders lose more,
within kinase groups as well as between them (partial correlation +0.24 [+0.13, +0.34] at 43 °C).
Leaving aside the AGC group (below), the mean change at 37 °C falls off steadily with affinity:

| Dissociation constant | < 1 nM | 1-10 nM | 10-100 nM | 0.1-1 µM | 1-10 µM | > 10 µM (not bound) |
|:--|--:|--:|--:|--:|--:|--:|
| Mean change | **-0.82** | -0.24 | -0.29 | -0.22 | -0.13 | -0.04 |

Since staurosporine acts in the lysate, there is no cellular turnover: these are changes in binding
(or in solubility). The reading that fits is that the drug, sitting in the ATP site, stabilises the kinase fold, and a
stabilised kinase is no longer held by the co-chaperone.

**The more of a kinase is bound, the more it loses.** Kinases abundant in the control pull-down
lose most. With the lysate measured, both parts of that level predict the loss: lysate abundance
(rank slope -0.2 to -0.4) and enrichment over the lysate (-0.4 to -0.55), from replicates disjoint
from those of the response. Across all proteins the drug response is unrelated to either, so this is
specific to kinases: the drug releases bound kinase.

**Not through HSP90.** Kinome-wide HSP90 client data (Taipale et al. 2012; 211 of these kinases) show
that the kinases that lose most are *not* HSP90 clients: non-clients fall most (-0.29 to -0.37) and
strong clients least (-0.03 to -0.12), and stronger HSP90 binding predicts a smaller loss (rho +0.24,
p = 6e-4 at 43 °C), even though strong clients are the most enriched kinases in the control
pull-downs (+0.54 log2 over the median kinase). In the lysate, strong HSP90 clients may remain held
by the endogenous HSP90-CDC37 machinery, and so be released less by the drug.

**The AGC exception.** The PKC isoforms, PKN and PKG bind staurosporine as tightly as any kinase
(dissociation constants 0.2-50 nM) yet *gain* binding (PKG1 +4.4, PKCtheta +2.2, PKN2 +1.5). The
sparse-autoencoder features that separate them mark AGC kinases with lipid- or
cyclic-nucleotide-sensing regulatory domains (PKC, PKN, PKD, PKG) on the gaining side, and, on the
losing side, CAMK2, DAPK and the STE kinases, through features labelled as disordered acidic tails,
basic regulatory linkers and coiled-coil segments. These features separate rising from
falling kinases well within known subfamilies (38 % of the variance at 43 °C, against 20 % for the
embedding), but not for a kinase group never seen in training: direction is a subfamily property.

# 6. How the two co-chaperones differ

Which of the two co-chaperones a protein prefers is the most reproducible (96 %) and the most
predictable response here: 49 % from sequence, a third from composition alone. Because both baits
met the same lysate, the preference is intrinsic binding specificity.

| Prefers DNAJB11 | | Prefers DNAJA1 | |
|:--|--:|:--|--:|
| Signal peptide | +0.35 (p 6e-88) | SR splicing factors | -0.62 |
| Transmembrane | +0.32 (p 1e-157) | Ribosomal proteins | -0.45 |
| Glycoprotein | +0.31 | Ribonucleoproteins | -0.42 (p 3e-58) |
| Disulfide bond | +0.29 | KRAB zinc fingers | -0.46 |
| Lysosome, Golgi, ER | +0.17 to +0.21 | Nucleus | -0.15 (p 1e-158) |

*Mean preference, log2; > 0 toward DNAJB11.*

DNAJB11 binds secretory and membrane proteins, the clients it meets in the ER, even when offered the
whole proteome. DNAJA1 binds nuclear RNA-processing proteins; its sparse-autoencoder features are all
charge and disorder features ("charged disordered low-complexity tracts", "KR-rich NLS-like motifs",
"RS/SR phospho-regulated regions"). Family averages account for 60 % of what the model explains, and
within families prediction still correlates at 0.56.

**Different preferences, the same responses.** Two different quantities are involved, and they
behave differently.

- *Preference* is how much of a protein one co-chaperone holds relative to the other. The two
  pull-downs do not contain separate sets of proteins: 8,265 proteins are measured in both at 35 °C,
  and they differ in proportion, typically by 1.3- to 1.5-fold. The preferences above are tendencies
  of this kind, not exclusive client lists.
- *Response* is how a protein's binding changes with a condition, measured separately in each
  pull-down. Across proteins, the heat response measured with DNAJA1 and the heat response measured
  with DNAJB11 correlate at 0.93; for staurosporine the two agree completely once measurement noise
  is accounted for (0.83-1.0). A protein that gains binding with heat gains it with both
  co-chaperones, and one that loses, loses with both. The response is therefore a property of the
  protein, of how it behaves when the lysate is heated or treated, rather than of the co-chaperone
  that captures it.

The two co-chaperones thus differ in *which proteins they favour*, but heat and staurosporine change
those proteins in the same way for both. The one systematic difference is that heat weakens the
preferences. The spread of preferences across proteins falls from 0.54 log2 at 35 °C to 0.36 at
43 °C, and each protein keeps about 40 % of its 35 °C preference on the log scale: a protein 1.5-fold
more abundant in the DNAJB11 pull-down at 35 °C is about 1.2-fold more abundant at 43 °C. Nothing
reproducible remains beyond this uniform shrinkage. It is what the lysate shift predicts: if each
43 °C pull-down is 0.42 x its 35 °C self plus a lysate-like part common to both co-chaperones, each
preference keeps 42 % of its 35 °C size.

# 7. Across experiments

The responses are largely independent. The salt experiment predicts at most 7 % of any GA_33 response
and vice versa, and staurosporine and the co-chaperone preference share nothing with salt. The one
cross-experiment link, salt against heat (r = -0.24), disappears (-0.01) once each protein's starting
level in the two pull-downs is accounted for: proteins relatively enriched in one pull-down lose share
in it under that experiment's perturbation. There is no single protein property behind all the
responses; each has its own determinants.

# 8. What would test these readings next

- **A pull-down without bait at 35 and 43 °C.** It would show whether the lysate-like half of the
  43 °C pull-downs is bound by the co-chaperones or captured without them (soluble aggregates, bead
  binding). Western blots of the soluble proteins cannot decide this, because the pull-downs take up
  very little protein.
- **Binding at more temperatures.** Pull-downs at, say, 39 and 41 °C would trace each protein's
  transition directly, and test whether its midpoint follows its stability.
- **Melting temperatures measured in this lysate.** The Meltome values come from other cells and a
  three-minute heat pulse; lysate-specific values would sharpen the stability relation.

# Notes on methods

- Folds group proteins at 30 % sequence identity (MMseqs2). Out-of-family intervals resample families.
- The salt workbook had been damaged by a find-and-replace (missing values had become zeros, and eight
  accessions were mangled); targets were rebuilt from the replicates and checked against the file's
  own statistics.
- Weighting proteins by their measurement noise was tested and does not help: noise is too small a
  part of the variance for the salt and heat responses.
- Detailed analyses are in `reports/additional/additional_text_1` to `_8`: thermal stability (1),
  kinases and staurosporine (2), cross-experiment structure (3), normalisation (4), proteins that
  appear or vanish (5), combining layers (6), complexes (7), the input lysate (8). Where texts 1
  and 7 describe the heat response as measured, text 8 gives the selective version.
