---
title: "Additional text 9: why abundant proteins are under-represented in the pull-downs"
date: "9 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

In GA_33, a protein's level in the pull-down rises with its level in the input lysate, but much
less than proportionally. The slope on the log2 scale is 0.35: a protein ten times more abundant in
the lysate is only about 2.2 times more abundant in the pull-down. Enrichment (pull-down minus
input) therefore falls as abundance rises.

Part of that negative correlation is built in. Enrichment contains the input with a minus sign, so
*any* capture that is less than proportional makes enrichment fall with abundance. The real
question is why the slope is 0.35 and not 1.

A slope of 1 is the natural default. Under simple mass action, the amount of a protein captured is
its concentration times a capture factor (its affinity for the bait, or for the beads). Doubling the
protein doubles what is captured. A slope below 1 therefore means that something which goes with
abundance reduces capture.

All numbers below are for the control pull-downs (no staurosporine), against the GA_33 reference
input (additional text 8). Contaminants are excluded, and proteins without an input level are not
imputed. Intervals are 95 % and resample 30 % sequence clusters unless stated otherwise. Code:
`code/ad15_abundance.py`, `code/ad15b_assembly.py`, `code/fig_abundance_slope.py`.

# Hypotheses

| | Hypothesis | What it predicts here |
|:--|:--------------------|:--------------------|
| H1 | **Intrinsic.** Abundant proteins are, as proteins, poorer clients. Highly expressed proteins are known to be selected for stability and solubility. | The slope moves toward 1 at matched stability, disorder, location and so on. The part of abundance that is specific to this lysate is captured proportionally. |
| H2 | **In the tube.** Capture of a protein grows less than proportionally with its own amount in the tube. | The same slope for every source of abundance. No change at matched properties. |
| H3 | **Measurement.** The pull-down samples are quantified on a compressed scale relative to the input. | The slope follows sample load, and is the same for all kinds of protein. |

Two simpler explanations were already ruled out:

- **Detection bias.** If scarce proteins were seen in the pull-down only when strongly enriched,
  their enrichment would be inflated. But the pull-down detects 90-99 % of proteins in every decile
  of input abundance, and the slope is the same (0.33-0.36) among proteins detected in all six
  replicates.
- **Noise in the input.** Noise in the predictor pulls a slope toward zero. The input's
  reliability is 0.997, which is far too high for that.

# 1. The relation is a straight line on the log scale

| Slope of pull-down level on input | 35 °C | 37 °C | 43 °C |
|:--|--:|--:|--:|
| DNAJA1 | 0.33 [0.31, 0.36] | 0.38 | 0.59 |
| DNAJB11 | 0.36 [0.34, 0.39] | 0.41 | 0.63 |
| Correlation, both baits averaged | r 0.40 | r 0.46 | r 0.67 |

At 35 °C the input explains only 16 % of the variance of pull-down level. The pull-down's own
ceiling is 98 %: its two halves of replicates agree at r 0.97.

![Pull-down level against input abundance. **A**: means of 20 bins of input, both baits averaged, with
a slope-1 line for proportional capture. **B**: each pull-down run's slope on input against its total
raw intensity (load); the dotted line is the input runs. **C**: slope on the typical part of abundance
(predicted from PaxDb or from sequence) and on the residual that is specific to this lysate, with 95 %
intervals.](../figures/abundance_slope.png){width=100%}

The binned relation is close to a straight line (panel A). Its curvature at 35 °C is +0.03 [0.00,
+0.06] per squared standard deviation of input. On the original scale, pull-down level grows roughly
as input to the power 0.35.

**This shape rules out the simplest mixture model.** Suppose the pull-down were a sum of two parts:
selective capture unrelated to abundance, plus background carried over in proportion to the
lysate. On the log scale that sum bends upward. Scarce proteins would be dominated by the selective
part (slope near 0) and abundant ones by the background (slope near 1). I simulated such mixtures
using the measured input distribution and tuned them to reproduce the observed slope (0.35) and
correlation (0.40). The best-fitting mixtures all predicted a curvature of +0.11 to +0.12, three to four times the
observed value and outside its interval. A mixture could still fit if the selective part itself
fell with abundance, but that brings back the original question.

# 2. Known client properties do not explain it (against H1)

If abundant proteins are poor clients because of what they are, then comparing proteins of equal
stability, disorder, location and so on should bring the slope toward 1. It does not:

| Added to input (35 °C, 7,024 proteins with a Meltome Tm) | Slope | Gap to 1 closed |
|:------------------------------------------------|---------:|-------:|
| Input only | 0.35 [0.32, 0.38] | - |
| + melting temperature (Tm, linear and quadratic) | 0.36 | 1 % |
| + disorder (SAE feature 654) | 0.34 | -2 % |
| + membrane helix (SAE feature 10715) and transmembrane keyword | 0.38 | 4 % |
| + location (nucleus, cytoplasm, membrane, ER/Golgi, mitochondrion, secreted) | 0.36 | 2 % |
| + enzyme (EC keywords) | 0.36 | 1 % |
| + length | 0.34 | -1 % |
| **+ all of these** | **0.39 [0.36, 0.43]** | **6 %** |

These properties matter for capture: together they explain 19.5 % of enrichment. But they are
nearly unrelated to abundance, so they leave the slope where it was. Without the Tm requirement
(7,880 proteins), the full set closes 1 % of the gap.

# 3. Typical and lysate-specific abundance are captured alike (against H1)

A protein's abundance in this lysate has two parts:

- **A typical part.** Its abundance in general, which can be predicted from other data.
- **A lysate-specific part.** How far this cell type and preparation depart from the typical value.

If H1 were right, only the typical part would reduce capture, since it reflects what the protein
is. The lysate-specific part changes only the amount in the tube, so it would be captured
proportionally, with a slope near 1. Under H2 or H3, only the amount in the tube matters, and both
parts give the same slope.

The input was split using four predictors of typical abundance:

- PaxDb protein abundance in 8 cell lines (consensus of HEK293, U2OS, Jurkat, LNCaP, HeLa, K562,
  A549 and RKO);
- PaxDb whole-organism abundance;
- PaxDb HEK293 alone;
- the ESMC sequence model of input abundance (out-of-fold predictions, 51 % of variance).

For each, the input was fitted on the predictor. The fitted value is the typical part and the
residual is the lysate-specific part.

| Predictor of typical abundance | % of input | 35 °C, typical | 35 °C, lysate-specific | 43 °C, typical | 43 °C, lysate-specific |
|:----------------|-----:|------------:|------------:|------:|------:|
| PaxDb, 8 cell lines | 39 % | 0.42 [0.38, 0.47] | 0.30 [0.27, 0.34] | 0.64 | 0.58 |
| PaxDb, whole organism | 56 % | 0.34 [0.31, 0.38] | 0.35 [0.32, 0.39] | 0.68 | 0.53 |
| PaxDb, HEK293 | 26 % | 0.35 [0.31, 0.38] | 0.36 [0.32, 0.40] | 0.71 | 0.58 |
| ESMC sequence | 54 % | 0.33 [0.29, 0.38] | 0.36 [0.33, 0.40] | 0.65 | 0.57 |
| All four together | 66 % | 0.35 [0.31, 0.38] | 0.37 [0.33, 0.41] | 0.63 | 0.58 |

At 35 °C the two parts give the same slope with every predictor (panel C). The lysate-specific
part is not captured proportionally; it is held to the same 0.35 as the typical part. Under H1 it
would have been captured with a slope near 1.

At 43 °C the typical part has a somewhat higher slope (0.63-0.71) than the lysate-specific part
(0.53-0.58). Part of the heat-induced lysate-like capture (additional text 8) therefore goes with
the kind of protein, not only with its amount in the tube.

**The limit of this test.** The predictors are imperfect. Some of the residual is typical abundance
they missed, not abundance specific to this lysate. Under H1, the residual's slope would be 0.35 +
0.65 x (the share of the residual that is truly lysate-specific). The observed 0.37 [0.33, 0.41]
caps that share at about 9 %. H1 survives only if at least 91 % of the residual, about a third of
all input variance, is typical abundance that all four predictors missed. That is possible, given
how noisy the older PaxDb data are, but unlikely. The test leans against H1; it does not exclude it.

# 4. The slope does not follow sample load (against H3)

The pull-down samples are not low-load samples for the instrument. Their total raw intensity
(median 27.3 log2) is close to that of the input runs (27.5 log2), and they detect a similar number
of proteins (8,400-9,150 per condition, against 9,120-9,280 per input run).

| Condition (control / staurosporine pooled) | Load, log2 total intensity | Slope on input |
|:--|--:|--:|
| DNAJA1, 35 °C | 26.5 / 26.8 | 0.34 / 0.34 |
| DNAJA1, 37 °C | 26.9 / 27.0 | 0.38 / 0.38 |
| DNAJA1, 43 °C | 27.6 / 27.6 | 0.59 / 0.59 |
| DNAJB11, 35 °C | 27.0 / 27.1 | 0.36 / 0.37 |
| DNAJB11, 37 °C | 27.5 / 27.4 | 0.41 / 0.42 |
| DNAJB11, 43 °C | 28.0 / 28.1 | 0.63 / 0.62 |

- **Within a condition.** Runs vary in load by 0.14 log2 (standard deviation), and the slope does
  not rise with load: -0.013 per log2 of load [-0.024, -0.002], resampling the 73 runs.
- **Between baits.** DNAJB11 runs carry 0.3-0.6 log2 more load than DNAJA1 runs at the same
  temperature, yet their slope is only +0.03 higher. With bait and temperature in the model, load
  adds +0.004 per log2.
- **With heat.** From 35 to 43 °C, load rises by 0.98 log2 and the slope by 0.26. The
  within-condition rate predicts a change of -0.01. The rise in slope with heat is not a load effect.

If equal volumes were injected, the higher load at 43 °C means the heated pull-downs recovered about
twice as much protein, consistent with the main report.

# 5. Capture of the free form of oligomers: rejected

Under mass action, capture is proportional to a protein's own concentration unless the capturable
form is not. One concrete way that can happen: suppose a co-chaperone binds only the free,
unassembled form, and a protein forms a homo-oligomer of n copies that holds together at its
concentration in the lysate. Then the free form grows only as concentration to the power 1/n: 0.5
for dimers, 0.33 for trimers, 0.25 for tetramers. This predicts that monomers are captured nearly
proportionally, and oligomers less so, falling with order.

| Assembly class (UniProt "Subunit structure", first statement) | n | Slope, 35 °C | Slope, 43 °C |
|:--|--:|--:|--:|
| Monomer | 307 | 0.14 [0.01, 0.28] | 0.60 |
| Homodimer | 844 | 0.25 [0.20, 0.31] | 0.61 |
| Homotrimer | 60 | 0.25 [0.00, 0.47] | 0.41 |
| Homotetramer | 126 | 0.25 [0.12, 0.38] | 0.50 |
| Hetero-complex subunit | 1,973 | 0.48 [0.43, 0.52] | 0.65 |
| All proteins | 7,880 | 0.35 [0.32, 0.37] | 0.61 |

The prediction fails: monomers have the *lowest* slope, not the highest. The free-oligomer mechanism
is rejected.

The table holds a second observation. The slope depends on the kind of protein: 0.14 for monomers
against 0.48 for hetero-complex subunits, with intervals that do not overlap. A compression in the
measurement (H3) would act on all proteins alike. Class-dependent slopes therefore argue against a
purely technical explanation. The classes are coarse, though, and differ in other ways too.

\newpage

# What this leaves

| Explanation | Status | Evidence |
|:------------------------|:----------|:--------------------------|
| Detection bias in the pull-down | Excluded | Detection 90-99 % at every abundance; same slope in fully detected proteins |
| Noise in the input | Excluded | Reliability 0.997 |
| Selective capture plus proportional background | Excluded in its simple form | Predicts 3-4 times the observed curvature |
| Sample load (instrument) | Excluded | Slope does not follow load within or between conditions |
| Intrinsic, through stability, disorder, membrane, location, enzyme class or length | Excluded | Closes 1-6 % of the gap to 1 |
| Intrinsic, through unmeasured properties | Disfavoured | The lysate-specific part has the same slope as the typical part |
| Free form of homo-oligomers | Rejected | Monomers have the lowest slope |
| Compression of the measurement that does not depend on load | Not testable with these data | Class-dependent slopes argue against a generic compression |
| Less-than-proportional capture in the tube, mechanism unknown | Consistent with everything | No mechanism identified |

These data say more about what the effect is not than about what it is. The two candidates still
standing, an effect in the tube (H2) and a load-independent compression of the measurement (H3),
predict the same result in every test these data allow. Separating them, and settling H1, needs
experiments in which the amount of a protein in the tube is changed while the protein stays the same.

**Why H2 is physically demanding.** Under simple mass action, doubling one protein doubles its
capture. Many proteins compete for the same bait sites, but that competition scales everyone's
capture by the same factor, and normalisation removes it. A slope below 1 for a protein's own amount
therefore needs one of two things:

- the protein saturates the particular sites it uses; or
- its capturable form does not scale with its total amount.

The obvious version of the second, free subunits of oligomers, fails (section 5).

# Proposed controls

In order of how much each settles.

**1. Two-species mixing (settles H1 vs H2 vs H3).**

- **Design.** Mix the human lysate with a lysate from another species (yeast or *E. coli*) at two
  ratios, for example 1:1 and 1:4 at constant total protein. Run the pull-down from each mixture,
  and measure the inputs and the pull-downs.
- **What it does.** Between the two mixtures, every human protein changes its amount in the tube by
  the same known factor relative to every foreign protein. Each protein stays the same protein.
  Read the shift of the human block against the foreign block, within each sample.
- **Predictions.**
  - H1, or plain mass action: the pull-down shifts by the full factor (log2 4 = 2).
  - H2: it shifts by about 0.35 x 2 = 0.7.
  - H3: the inputs measured at the pull-downs' composition shift less than 2 as well.

  The inputs also give the instrument's own ratio compression directly.

**2. Spike-in titration (direct per-protein capture curve).**

- **Design.** Add 3-5 purified foreign proteins to the lysate before the pull-down, at 4-5
  concentrations spanning about 100-fold.
- **Which proteins.** Include a likely client, for example heat-labile firefly luciferase, and a
  likely non-client, for example GFP.
- **Reading.** The slope of log pull-down on log spike-in is the per-protein capture exponent
  itself. The client/non-client pair separates selective capture from background.
- **Trade-off.** Cheaper than mixing, but it covers only a few proteins.

**3. Pull-down without bait, at 35 and 43 °C.**

- **Design.** Beads plus lysate, no bait.
- **Reading.** This measures background capture and its slope on input. If background capture is
  proportional (slope near 1), the shortfall belongs to capture by the bait. If it also has a slope
  near 0.35, the cause lies in the beads, the washes or the measurement, not in the chaperone.
- **Bonus.** The same experiment settles what the lysate-like half of the 43 °C pull-down is
  (additional text 8).

**4. Bait titration (tests saturation).**

- **Design.** Run the pull-down at 0.25, 1 and 4 times the usual bait.
- **Predictions.** If abundant clients saturate the sites they use, more bait raises their capture
  more than that of scarce clients, so the slope rises with bait. Under H1 or H3 the slope does not
  change.

**5. Information we need.** Were the pull-down samples injected at equal volume or at equal protein
amount? This decides whether the +1 log2 load at 43 °C is more captured protein or only a change in
composition.

**From experiments already planned.** Once the GA_22 and GA_24 pull-downs arrive, salt provides a
free version of control 1. It changes the unbound amount of individual proteins by up to 3 log2
(supernatants, checked 9 October). The ratio of each protein's change in the pull-down to its
change in the supernatant is a within-protein capture exponent. The catch is that salt also changes
binding, so this complements the mixing experiment but does not replace it.

# Critical evaluation

- **Proxies for typical abundance.** The PaxDb data are older and noisy: HEK293 alone explains only
  26 % of the input. The test in section 3 is only as strong as the predictors, and the files do not
  state the lysate's cell type.
- **Load.** It is measured as total raw intensity, which reflects composition as well as amount.
  With equal-amount injection, section 4 tests composition, not quantity.
- **Mixture simulation.** It assumed the selective part was normally distributed and unrelated to
  abundance. Mixtures in which the selective part falls with abundance were not excluded, but they
  restate the question.
- **Matched properties.** Section 2 covers only the properties tested. Sequence enters through the
  ESMC abundance predictor in section 3, not as a direct covariate of capture.
- **Assembly classes.** They come from the first sentence of UniProt text, a rough classification.
  Monomer annotations are few (307), and the classes differ in abundance (monomers +0.46 log2
  above the median).
- **Scope.** All of this is GA_33, two baits, one lysate. Whether the 0.35 slope holds for other
  baits (HSPB1 in GA_20/22/24) is open.
