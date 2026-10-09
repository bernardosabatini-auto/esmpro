---
title: "Additional text 8: the GA_33 pull-downs read against the input lysate"
date: "8 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

A protein's level in a pull-down mixes two things: how much of it the lysate contains, and how
strongly it is captured. The input lysate of each GA_33 reaction was measured, so the two can now
be separated. Several readings in the main report depended on that split:

- the starting-level effect on the heat response;
- the inverted U against thermal stability;
- the kinase pull-down level under staurosporine;
- which proteins are bound beyond background.

# Data and checks

**Input.** `GA_33_Proteome_report_out.tsv`. It covers 12 reactions (2 co-chaperones x 2 drug
conditions x 3 temperatures), each run in 3 technical (MS) replicates. There are no biochemical
replicates. Values are log2, not normalised and not imputed.

- **Outliers.** Two reactions deviate from the rest and are left out: DNAJA1 / no drug / 35 °C, and
  DNAJB11 / staurosporine / 35 °C, which is heavily contaminated with keratins.
- **One reference input.** The input does not change reproducibly with temperature. The other 10
  reactions are therefore averaged into one reference input (9,450 proteins).
  - Its two halves agree at a reliability of 0.997.
  - 8,837 of the 9,871 pull-down proteins have an input level.
  - Contaminants (keratins, immunoglobulins, lysozyme and similar, 65 proteins) are excluded throughout.
- **Matched inputs agree.** Using each reaction's own input instead gives the same enrichments
  (r 0.983-0.996 across the 10 reactions) and the same heat response (r 0.986).

**Enrichment.** Enrichment is the log2 pull-down level minus the log2 input level; both are
median-normalised. The pull-down and the input are separate runs on separate scales, so each
enrichment profile is centred on its median. Zero is the typical protein, not "no binding". Only
differences between proteins carry meaning.

**Replicates.** Wherever the same pull-down measurement could appear on both sides of a comparison,
the predictor and the response come from disjoint biological replicates: BR1-3 against BR4-6, and
the reverse. Intervals resample 30 % sequence clusters.

# 1. What the pull-down contains, relative to the lysate

**The pull-down is not a scaled-down lysate.** Across proteins, pull-down level rises with input
level, but with a slope of only 0.34 (DNAJA1) and 0.37 (DNAJB11), against 1 for a proportional sample.
The correlation is 0.38-0.40. Enrichment therefore correlates at -0.6 with abundance: the most
abundant lysate proteins are strongly under-represented, and scarce proteins over-represented.

| Class (35 °C, no drug) | Input vs typical protein | Enrichment, DNAJA1 | Enrichment, DNAJB11 |
|:--|--:|--:|--:|
| Glycolytic enzymes (12) | +4.5 | -3.7 | -3.6 |
| Proteasome (35) | +2.5 | -2.0 | -1.6 |
| Translation factors (56) | +1.9 | -1.6 | -1.6 |
| HSP70 / HSP90 (15) | +3.7 | -1.7 | -0.9 |
| Cytosolic ribosome (85) | +3.4 | +0.4 | -0.4 |
| Tubulins (14) | +3.7 | +0.4 | +0.6 |
| Mitochondrial ATP synthase (15) | +1.9 | +1.0 | +1.5 |

*Medians, log2.*

**By compartment**, at matched abundance:

- membrane and ER proteins are enriched, more so with DNAJB11 (+0.6) than with DNAJA1 (+0.25);
- soluble cytoplasmic proteins are depleted with both (-0.25).

**The baits are higher in their own pull-down, but do not dominate it.** Each bait is 2.8 (DNAJA1)
and 2.4 (DNAJB11) log2 higher in its own pull-down than in the other's.

| Bait in its own pull-down | 35 °C | 37 °C | 43 °C |
|:--|--:|--:|--:|
| DNAJA1: rank by pull-down level | 8 | 17 | 22 |
| DNAJA1: enrichment over input | +3.2 | +2.7 | +2.1 |
| DNAJB11: rank by pull-down level | 257 | 249 | 100 |
| DNAJB11: enrichment over input | +0.6 | +0.4 | +1.1 |

The pull-downs capture very little protein in total. The most intense proteins in them include tubulins,
small subunits of the mitochondrial ATP synthase, and two proteins absent from the input (DNASE2B,
KBTBD3).

**There is no separate population of "clients".** Enrichment forms one broad distribution. Its
long low tail consists of abundant, under-represented proteins. Only 4 % of proteins lie more than
2 log2 above the median. Capture is therefore treated below as a continuous quantity, not as a
client-versus-background call.

**The two baits differ even among the least enriched proteins.** The DNAJB11-versus-DNAJA1
preference is largest among the most enriched proteins. Its reproducible spread across proteins
rises from 0.38 log2 in the least enriched quarter to 0.63 in the most enriched. But it is
reproducible in every quarter: reliability is 0.76 in the least enriched quarter and 0.98 in the most
enriched. No part of the pull-down behaves as bait-independent background at 35 °C.

**Pull-down-only proteins.** 406 (DNAJA1) and 476 (DNAJB11) proteins are detected in the pull-down
but not in the input. Most likely they sit below the input's detection limit and are concentrated
by capture. They are over-represented among nuclear proteins (35 % against 20 %).

# 2. At 43 °C the pull-down becomes partly lysate-like

The slope of pull-down level on input changes little from 35 to 37 °C. It rises sharply at 43 °C:

| Slope on input | 35 °C | 37 °C | 43 °C |
|:--|--:|--:|--:|
| DNAJA1 | 0.34 | 0.38 | 0.59 |
| DNAJB11 | 0.37 | 0.42 | 0.63 |

**Mixing fit.** A fit of each protein's level at the higher temperature as a mixture of its 35 °C
level (from disjoint replicates, corrected for their noise) and its input level gives:

- at 37 °C: level = 0.93 x (35 °C level) + 0.07 x input;
- at 43 °C: level = 0.42 x (35 °C level) + 0.46 x input.

Both co-chaperones give the same coefficients to within 0.04. So about half of the 43 °C pull-down
composition is the 35 °C pull-down, and the rest is in proportion to the lysate.

**Staurosporine is the control, and does not do this.** At every temperature the
staurosporine pull-down is its own control pull-down: coefficient 0.98-1.00, input coefficient under
0.02, slope on input unchanged. The shift toward the lysate is specific to heat.

**An independent check.** The main report found that each protein keeps about 40 % of its
DNAJB11-versus-DNAJA1 preference at 43 °C. The input cancels from a preference, so a mixture with
coefficient 0.42 on the 35 °C pull-down predicts exactly that. The lysate-like component is
therefore the same for both co-chaperones.

# 3. The starting-level effect is loss of selectivity

The main report found that proteins plentiful in the 35 °C pull-down lose share at 43 °C. With the 35 °C
level split into its two parts, the effect belongs to enrichment, not to abundance. In the table
below, the 35 °C quantities come from BR1-3 and the response from BR4-6; the reverse split gives
correlations within 0.01 of these at 43 °C and within 0.06 at 37 °C.

| Predictor of the response | 43 vs 35 °C: r | 43 vs 35 °C: variance explained | 37 vs 35 °C: r | 37 vs 35 °C: variance explained |
|:--|--:|--:|--:|--:|
| 35 °C pull-down level | -0.42 | 18 % | -0.17 | 3 % |
| Input abundance | **+0.33** | 11 % | +0.17 | 3 % |
| 35 °C enrichment | **-0.68** | **46 %** | -0.31 | 9 % |
| Both | | 47 % | | 9 % |

Proteins captured well above their lysate level at 35 °C lose share at 43 °C. Abundant proteins,
under-represented at 35 °C, gain. This is what the mixing in section 2 predicts. Because predictor
and response come from different replicates, it is not shared measurement noise.

# 4. The heat response with the lysate-like part removed

The lysate-like part is defined as the part of the response predicted from input abundance and
35 °C enrichment, fitted on disjoint replicates. Removing it explains 45 % of the variance of the
43 °C response, and 11 % at 37 °C. The remainder, called the **selective response** here, is
the change in capture that the lysate shift does not account for.

**The protein classes keep their direction, at a third to a half of the size:**

| Class minus the rest, 43 vs 35 °C | Raw | Lysate-like part | Selective |
|:--|--:|--:|--:|
| Enzymes (EC keywords) | +0.30 | +0.16 | **+0.14** [+0.10, +0.18] |
| Soluble cytoplasmic | +0.31 | +0.23 | **+0.09** [+0.04, +0.13] |
| Transmembrane | -0.55 | -0.44 | **-0.11** [-0.15, -0.08] |
| Nuclear only | -0.27 | -0.14 | **-0.13** [-0.17, -0.08] |
| SAE disorder feature, top quarter | -0.36 | -0.19 | **-0.17** [-0.21, -0.13] |
| SAE membrane-helix feature, top quarter | -0.28 | -0.12 | **-0.16** [-0.20, -0.12] |

**Thermal stability: less stable proteins gain more.** Response by Meltome melting-temperature decile,
7,013 proteins:

| Melting temperature (°C) | <46.8 | -47.9 | -48.9 | -49.9 | -51.0 | -52.2 | -53.6 | -55.5 | -58.5 | >58.5 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Input abundance | -0.54 | -0.32 | -0.20 | -0.10 | +0.09 | +0.01 | +0.07 | +0.45 | +0.60 | +0.76 |
| 35 °C enrichment | +0.51 | +0.21 | +0.18 | +0.01 | -0.11 | -0.09 | -0.05 | -0.14 | -0.34 | -0.53 |
| 43 vs 35 °C, raw | +0.13 | +0.25 | +0.41 | +0.39 | +0.38 | +0.35 | +0.28 | +0.22 | +0.12 | +0.02 |
| 43 vs 35 °C, lysate-like part | -0.03 | +0.06 | +0.13 | +0.19 | +0.21 | +0.26 | +0.22 | +0.26 | +0.28 | +0.39 |
| **43 vs 35 °C, selective** | +0.17 | +0.19 | **+0.28** | +0.21 | +0.17 | +0.09 | +0.06 | -0.04 | -0.15 | **-0.37** |

*Medians (input, enrichment) and means (responses), log2.*

- **The least stable proteins are scarce and already enriched.** They are scarce in the lysate (-0.54)
  and already enriched at 35 °C (+0.51). Their 35 °C pull-down level looks ordinary only because the
  two cancel. The lysate-like component at 43 °C therefore favours them least.
- **That is what made the inverted U.** In the selective response, binding falls steadily with
  stability. The least stable proteins gain about +0.2, and the most stable lose -0.37. The slope is
  -0.18 per standard deviation of melting temperature [-0.20, -0.16], against -0.07 for the raw response.
- **A mild curvature remains.** It is -0.05 [-0.07, -0.03] against -0.07 raw. The peak sits in the third
  decile (melting near 48-49 °C).

**The two steps.** The 35-37 °C step is hardly affected by mixing (coefficient 0.93), so it keeps its
earlier reading.

| Melting temperature decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Selective, 35 to 37 °C | **+0.10** | +0.08 | +0.09 | +0.03 | +0.03 | +0.00 | +0.00 | -0.02 | -0.04 | -0.07 |
| Selective, 37 to 43 °C | +0.06 | +0.11 | **+0.19** | +0.18 | +0.15 | +0.09 | +0.06 | -0.02 | -0.11 | -0.30 |

- The least stable proteins gain most by 37 °C and little after.
- Proteins of intermediate stability gain most between 37 and 43 °C.
- The most stable proteins lose share.

The "timing of binding" reading therefore survives in the steps. The 35-43 °C inverted U itself
was mostly the lysate shift.

**Complex subunits gain, once the lysate shift is removed.** Subunits minus other proteins, at matched
stability and location:

| Response | Subunits minus other proteins |
|:--|--:|
| 43 °C, raw | -0.02 [-0.07, +0.05] |
| 43 °C, selective | **+0.10** [+0.06, +0.14] |
| 37 °C, selective | +0.02 [+0.01, +0.04] |

The lysate shift had masked the effect: at matched stability and location, its lysate-like part runs
against subunits (-0.12), cancelling their selective gain in the raw response.

**The two co-chaperones still share their heat response.** Their selective responses correlate at
0.95; their raw responses at 0.93.

# 5. What sequence predicts

These are the nine-layer stacked ESMC models of the main report, on held-out sequence families.

| Target | Ceiling | Sequence |
|:--|--:|--:|
| Input abundance (lysate level) | 99.7 % | **50.7 %** |
| 35 °C enrichment, average of the two co-chaperones | 99.5 % | 38.3 % |
| 35 °C enrichment at fixed abundance, DNAJA1 | 98.3 % | **25.8 %** |
| 35 °C enrichment at fixed abundance, DNAJB11 | 98.8 % | **23.1 %** |
| Heat 43 vs 35 °C, raw (as in the main report) | 98.9 % | 27.3 % |
| Heat 43 vs 35 °C, lysate-like part | | 33.4 % |
| **Heat 43 vs 35 °C, selective** | 98 % | **13.1 %** |
| Heat 37 vs 35 °C, selective | 84 % | 9.1 % |

*The ceiling of the selective response is approximate.*

- **Sequence predicts lysate abundance** (half its variance) and **capture at fixed abundance** (a
  quarter).
- **Much of the 27 % for heat is the lysate shift.** Sequence predicts that part well, through
  abundance and enrichment. The selective part, the change in capture itself, is predicted at 13 %.

# 6. Kinases and staurosporine

**Across all proteins, staurosporine does not change what kind of protein is captured.** Its
response is unrelated to input abundance or to control enrichment (|r| < 0.08).

**Among kinases, the more is bound, the more is lost.** The joint rank model of additional text 2
(kinase group, staurosporine affinity, HSP90 client score) is refitted with the control pull-down
level split into abundance and enrichment. Control quantities come from BR1-3 and the response from
BR4-6, and the reverse.

| Kinases (n 163-166) | 35 °C | 37 °C | 43 °C |
|:--|--:|--:|--:|
| Input abundance | -0.39 / -0.41 | -0.32 / -0.28 | -0.25 / -0.20 |
| Control enrichment | **-0.49 / -0.55** | **-0.46 / -0.45** | **-0.39 / -0.43** |
| Staurosporine affinity (higher = weaker binding) | +0.07 / +0.06 | +0.12 / +0.14 | +0.19 / +0.19 |
| HSP90 client score | +0.10 / +0.23 | +0.18 / +0.17 | +0.06 / +0.12 |

*Standardised rank slopes, the two replicate splits. Intervals are about 0.17 on either side; every enrichment
and abundance term excludes zero.*

- **Kinases lose in proportion to how much was bound.** Those held most strongly in the control,
  relative to their lysate level, lose most under the drug, and the abundant ones also lose more.
  This is release of bound kinase, not an abundance artefact.
- **HSP90 clients are the exception.** Strong HSP90 clients (Taipale et al. 2012) are the most
  enriched kinases in the control pull-down (+0.54 against the median kinase), yet they lose least:
  the HSP90 score enters with the opposite sign to enrichment, as additional text 2 found.

# 7. Proteins that appear at 43 °C

Of the 131 proteins detected in both co-chaperone pull-downs only at 43 °C:

- **Most are in the lysate.** 110 (84 %) are in the input, against 90 % of all pull-down proteins,
  typically in all 10 input samples.
- **They are not abundant.** Their median lysate abundance is the 36th percentile of pull-down
  proteins. The lysate-like component favours abundant proteins, so it does not explain them.

They are soluble lysate proteins captured selectively at 43 °C.

# Critical evaluation

- **What the lysate-like component is cannot be decided from these data.** It is the same for both
  co-chaperones and specific to heat. Two readings fit:
  1. the co-chaperones bind broadly, roughly in proportion to abundance, once many proteins begin to
     unfold;
  2. heat-induced material that is captured or co-sediments without the co-chaperone, such as small
     soluble aggregates or bead binding.

  Western blots, which show no change in the soluble proteins, cannot exclude the second, because the
  pull-downs take up very little protein. A pull-down without bait, or with an inactive bait, at 35
  and 43 °C would decide it.
- **The DNAJB11 bait is barely enriched over its input.** That suggests inefficient capture of the
  recombinant DNAJB11. Yet the DNAJB11 pull-down has its own reproducible preferences (section 1).
- **The selective response depends on a linear mixing model.** The mixing model fits well (the coefficients sum to 0.88-0.89, and both baits agree).
  Removing it is equivalent to comparing proteins at matched abundance and 35 °C enrichment.
  Stability is correlated with abundance, so the selective curve is a conditional statement: at
  matched abundance and 35 °C enrichment, less stable proteins gain more. It does not separate a
  stability effect that acts through abundance.
- **Limits of the input.** It was measured after the bait was added, from 10 of 12 reactions, without
  biochemical replicates. The temperature and drug arms do not differ reproducibly, and matched and
  reference inputs agree, so this does not affect the results.

# Consequences for the main report

- **Starting level.** The starting-level effect on the heat response is a loss of selectivity at
  43 °C, not regression to the mean or competition for a fixed amount of chaperone.
- **Stability.** The inverted U against stability is mostly the lysate shift. At matched abundance and
  enrichment, less stable proteins gain more, with mild curvature. The 35-37 / 37-43 °C step
  pattern stands.
- **Complexes.** The complex-subunit gain is clearer on the selective response (+0.10), not weaker.
- **Sequence.** About half of the sequence-predictable heat response is the lysate shift.
- **Kinases.** The kinase result is release of bound kinase: enrichment and abundance both count.
