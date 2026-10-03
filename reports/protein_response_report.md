---
title: "Predicting how chaperone binding responds to salt, heat and staurosporine from protein sequence"
date: "3 October 2026"
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
| Salt, 150 vs 0 mM | HSPB1 | 99 % | **20 %** | 8 % |
| Salt, 75 vs 0 mM | HSPB1 | 99 % | **18 %** | 6 % |
| Heat, 37 vs 35 °C | DNAJA1, DNAJB11 | 87 % | **13 %** | 4 % |
| Staurosporine, 37 °C | DNAJA1, DNAJB11 | 76 % | **7 %** | 0 % |

*Percent of variance explained on proteins whose sequence families were held out of training, by the
best sequence model and by the 20 amino-acid fractions alone. The ceiling is the part of the variance
that is reproducible between independent replicates, and so the most any predictor could explain.*

- **Sequence predicts every response, well beyond amino-acid composition.** The best results come
  from reading several layers of the ESMC-6B protein language model at once.
- **Each response has its own, interpretable determinants.** Salt spares proteins with transmembrane
  helices. Heat recruits folded cytoplasmic proteins and loses disordered and membrane proteins.
  Staurosporine removes kinases, in proportion to how tightly it binds them. DNAJB11 prefers
  secretory and membrane proteins, DNAJA1 nuclear proteins with charged disordered regions.
- **Thermal stability shapes the heat response through a window.** Proteins that melt a few degrees
  above 43 °C gain the most binding; less stable and more stable proteins gain less. Stability
  matters, but explains little of the response overall.
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
conclusions are unchanged under four other normalisations. In GA_33 the heat and staurosporine
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

| Heat response | 43 vs 35 °C | 37 vs 35 °C |
|:--|--:|--:|
| Amino-acid composition | 8.9 % | 3.5 % |
| ESMC, one layer (final) | 25.7 % | 11.4 % |
| ESMC, nine layers combined | **27.4 %** | **12.7 %** |

**Folded cytoplasmic proteins are recruited; disordered and membrane proteins are lost.** The
features that rise are those of folded enzyme cores (catalase, isocitrate and aldehyde dehydrogenases,
enolase, TIM barrels, Rossmann folds; structural-motif features over-represented, odds ratio 5.5). Those that fall
describe long disordered and low-complexity regions and transmembrane segments (membrane features over-represented,
odds ratio 11.6).

**A protein's starting level matters.** Proteins plentiful in the 35 °C pull-down tend to lose share
at 43 °C, and scarce ones to gain (correlation -0.43; the starting level explains 16.5 % of the
response). This is not shared measurement noise: taking the starting level and the response from
different replicates gives the same correlation. It is regression to the mean in the biological
sense: the 35 °C and 43 °C binding profiles agree only partly (r = 0.6), so proteins that dominate
one tend to dominate the other less. In vitro there is also a direct reason: a fixed amount of
chaperone means that new clients at 43 °C compete with the old. Sequence does not predict the
response through the starting level: with the level removed, the embedding explains more of what is
left (28.9 %).

## 4.1 Thermal stability acts through a window

If heat recruits proteins as they begin to unfold, the less stable a protein, the more it should gain.
Melting temperatures for 9,792 human proteins come from the Meltome atlas (Jarzab et al. 2020;
thermal proteome profiling, consensus over ten cell lines, reliability 0.96); 87 % of the heat-response
proteins have one.

**At 43 °C the relation is an inverted U.** Proteins were sorted by melting temperature into ten equal
groups:

| Melting temperature (°C) | <46.7 | -47.9 | -48.9 | -49.9 | -51.0 | -52.2 | -53.6 | -55.5 | -58.4 | >58.4 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Change at 43 vs 35 °C | +0.11 | +0.24 | **+0.41** | +0.38 | +0.37 | +0.34 | +0.27 | +0.21 | +0.10 | +0.02 |
| Change at 37 vs 35 °C | +0.07 | +0.07 | +0.08 | +0.03 | +0.03 | +0.02 | +0.01 | -0.01 | -0.03 | -0.04 |

![Mean change in co-chaperone binding by melting temperature. A: at 43 °C binding rises with
decreasing stability down to a melting temperature of about 48 °C, then falls for the least stable
proteins (shaded: the binding window); at 37 °C it rises steadily as stability decreases. B: at 43 °C,
complex subunits (purple) show the window more sharply than other proteins (grey). Error bars, one
standard error.](figures/heat_tm_window.png){width=100%}

Reading the 43 °C curve from right to left: the most stable proteins (melting above 58 °C) are hardly
affected by 43 °C and gain nothing (+0.02). As stability decreases, the gain grows, and it peaks for
proteins that melt at 48-52 °C (+0.41), 5-8 °C above the incubation temperature. For the least stable
proteins, melting below 47 °C, the gain falls again (+0.11). So the relation rises and then falls,
and a straight-line correlation, which averages the two halves, is close to zero (rho = -0.04). The
curvature is statistically clear (quadratic term -0.073 [-0.095, -0.050], peak at 51 °C), appears with
melting temperatures from HEK293T, K562 or Jurkat alone, and survives adjustment for protein class
and starting level. At 37 °C, where nothing is near its melting point, the relation is the simple one:
the least stable proteins gain most (slope -0.038 [-0.046, -0.028]).

**A window of partial unfolding.** In the lysate at 43 °C, very stable proteins stay folded and are
not bound. Moderately stable proteins loosen, expose hydrophobic segments, and become co-chaperone
clients while staying soluble. The least stable proteins pass their melting point, aggregate, and
leave the soluble reaction (or are captured by the lysate's own chaperones) before the added
co-chaperone can hold them. Binding therefore peaks in a window just above the incubation
temperature. The least stable proteins are still detected at 43 °C (97-99 % of them), so they are
reduced rather than lost, which a partial loss to aggregation would produce.

**But stability explains little of the heat response.** Melting temperature, with its curvature,
explains 1.2 % of the 43 °C response and adds 0.8 points to the sequence model; the sequence model's
predictions do not correlate with melting temperature at all (r = 0.01). The protein classes above
(folded cytoplasmic versus disordered or membrane) act independently of stability. Proteins that the
Meltome atlas could not melt, mostly disordered transcription factors, fall with heat (-0.43, against
+0.24 for the rest).

## 4.2 Protein complexes

Heat can also dissociate complexes, turning their subunits into clients whatever their own stability.
Using 2,167 curated human complexes (EBI Complex Portal), 2,471 of the heat-response proteins are
subunits.

- **Subunits gain more at 43 °C**, at matched melting temperature, starting level and location:
  +0.17 log2 [+0.12, +0.22], against +0.04 at 37 °C. The effect is heat-dependent.
- **They show the window more sharply** (figure, B): in the middle third of melting temperatures
  subunits gain +0.45 against +0.28 for other proteins; among the least stable, +0.21 against +0.30.
- **Subunits of a complex move together**: 24 % of the variance left after removing level, stability
  and location is shared within a complex. It is at least as strong at 37 °C, so it is not caused by heat;
  pull-downs carry intact complexes along with whichever subunit the chaperone holds.
- **Which complexes**: translation initiation factor eIF3 (+2.7, 13 subunits), the COPII vesicle coat,
  the COP9 signalosome and GINS gain most; chromatin-bound nuclear complexes (nucleosomes, the
  chromosomal passenger complex, CDK-activating kinase) lose most.

Membership accounts for under 1 % of the response, like melting temperature: both point to real
mechanisms, neither is the main driver.

## 4.3 Proteins that appear at 43 °C

131 proteins are detected in both co-chaperone pull-downs only at 43 °C, 16 times more than chance
would give. They are the extreme of the same response: a model trained without them or their
relatives predicts them to rise (AUC 0.70 against proteins of matched intensity; 0.83 for the
strictest definition), they are cytoplasmic and rarely membrane proteins, and their melting
temperature is ordinary (median 51 °C). They are listed as candidate heat-recruited clients in
`reports/additional/heat43_appearing_proteins.tsv`.

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

**Not through HSP90.** Kinome-wide HSP90 client data (Taipale et al. 2012; 211 of these kinases) show
that the kinases that lose most are *not* HSP90 clients: non-clients fall most (-0.29 to -0.37) and
strong clients least (-0.03 to -0.12), and stronger HSP90 binding predicts a smaller loss (rho +0.24,
p = 6e-4 at 43 °C). In the lysate, strong HSP90 clients may remain held by the endogenous
HSP90-CDC37 machinery rather than by the added co-chaperone, and so have less to lose.

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

The two co-chaperones respond the same way to staurosporine and nearly the same way to heat. Their
one difference under heat is uniform convergence: at 43 °C each protein keeps about 40 % of its 35 °C
preference, and nothing reproducible remains beyond that shrinkage.

# 7. Across experiments

The responses are largely independent. The salt experiment predicts at most 7 % of any GA_33 response
and vice versa, and staurosporine and the co-chaperone preference share nothing with salt. The one
cross-experiment link, salt against heat (r = -0.24), disappears (-0.01) once each protein's starting
level in the two pull-downs is accounted for: proteins relatively enriched in one pull-down lose share
in it under that experiment's perturbation. There is no single protein property behind all the
responses; each has its own determinants.

# 8. What would test these readings next

- **One input lysate.** Since every reaction started from the same lysate, measuring it once separates
  a protein's amount in the lysate from how strongly it binds, and turns the starting-level effect into
  a binding measure.
- **The soluble and insoluble fractions after incubation at 43 °C.** The window predicts that the least
  stable proteins move into the insoluble fraction.
- **Melting temperatures measured in this lysate.** The Meltome values come from other cells and a
  three-minute heat pulse; lysate-specific values would sharpen the window.

# Notes on methods

- Folds group proteins at 30 % sequence identity (MMseqs2). Out-of-family intervals resample families.
- The salt workbook had been damaged by a find-and-replace (missing values had become zeros, and eight
  accessions were mangled); targets were rebuilt from the replicates and checked against the file's
  own statistics.
- Weighting proteins by their measurement noise was tested and does not help: noise is too small a
  part of the variance for the salt and heat responses.
- Detailed analyses are in `reports/additional/additional_text_1` to `_7`: thermal stability (1),
  kinases and staurosporine (2), cross-experiment structure (3), normalisation (4), proteins that
  appear or vanish (5), combining layers (6), complexes (7).
