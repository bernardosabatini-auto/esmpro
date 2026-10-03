# GA_33: DNAJA1 and DNAJB11 pull-downs across temperature, with and without staurosporine

**Status 2026-10-02.** Sequence predicts the **heat-induced change** in what DNAJA1 and DNAJB11
pull down far better than anything in the salt data: the ESMC-6B embedding explains **22.5 %** of
the variance in the 43 °C vs 35 °C log2 fold change with ridge on layer 50, 25.7 % from the final
layer, and **27.5 %** with an MLP ensemble averaged with ridge, against a measurement ceiling of 99 %, and adds +13.6 points over amino-acid composition. The effect of
**10 µM staurosporine** is small, half of its variance is replicate noise, it is the same for both
co-chaperones, and it falls hardest on **protein kinases**, which are depleted from both
pull-downs (p = 2e-20 at 37 °C); sequence explains 6–7 % of the co-chaperone-averaged effect against a ceiling of
71–76 %.

## 1. The experiment

`pd_data/GA_33_report_out.tsv`: 9,871 protein groups, raw log2 intensities, missing values as NaN
(10.0 % of cells). Column names encode a full factorial design:

| factor | levels |
|---|---|
| co-chaperone bait | `21A` = DNAJA1, `24` = DNAJB11 |
| staurosporine | `STAU_0` = none (control), `STAU_10` = 10 µM |
| temperature | 35, 37, 43 °C |
| biological replicate | BR1–BR6 |

2 × 2 × 3 × 6 = 72 samples. One sample (`24_STAU_10_T_43_BR2`) was injected twice; the two
injections (correlation 0.962) are averaged into that replicate.

**Normalisation.** Run medians rise with temperature: the mean run offset is −0.45 log2 at
35 °C, −0.06 at 37 °C and +0.43 at 43 °C. That is either more material in the pull-down or a real
global increase in binding, and the data cannot tell the two apart. Each run is median-normalised
over the 8,132 proteins detected in at least 90 % of runs, so fold changes describe how a protein
changes *relative to the pull-down as a whole*; the per-run offsets are saved in
`data/ga_data/run_offsets.tsv`.

**Sequences.** 9,869 of 9,871 resolved; the two without are the digestion enzymes (bovine trypsin,
Lys-C). 7,034 overlap the salt-titration proteins and reuse their embeddings; 2,835 were embedded
new (1.7 min on one RTX card, 16.3k residues/s). MMseqs2 at 30 % identity gives 6,864 clusters,
with 48 % of proteins having a relative, so cross-validation folds are grouped by cluster.

## 2. Targets and their ceilings

Log2 fold changes, each a difference of biological-replicate means, kept when both groups have at
least 4 of 6 real values. Ceiling from split-half agreement over all 10 ways of dividing the six
replicates into two triples, Spearman-Brown corrected to six.

| target | n | sd (log2) | noise share of variance | reliability = max R2 |
|---|---|---|---|---|
| staurosporine vs control, per co-chaperone and temperature (6) | 8,252–9,041 | 0.21–0.24 | 37–52 % | 49–64 % |
| staurosporine vs control, averaged over the two co-chaperones (3) | 8,168–8,807 | 0.19–0.20 | — | 71–76 % |
| 37 vs 35 °C, without staurosporine (2) | 8,300 / 8,607 | 0.31 / 0.33 | 21–28 % | 72–80 % |
| 37 vs 35 °C, averaged (1) | 8,224 | 0.29 | — | 87 % |
| **43 vs 35 °C, without staurosporine (2)** | 8,309 / 8,589 | **1.08 / 1.13** | **2 %** | **98 %** |
| **43 vs 35 °C, averaged (1)** | 8,217 | **1.09** | — | **99 %** |

The two effects are very different in kind. **Heat to 43 °C moves proteins ~2.2-fold**, almost
noise-free. **staurosporine moves them ~1.15-fold**, and half of that is replicate scatter.

**Both effects are shared by the two co-chaperones.** Corrected for measurement noise on identical
rows, the staurosporine differences for DNAJA1 and DNAJB11 correlate at 0.95 (35 °C), 0.83
(37 °C) and ≈1.0 (43 °C); the 43-vs-35 °C effects correlate at 0.93 uncorrected. The same
difference appears through two different baits, so the co-chaperone average is the same quantity
measured with twice the replicates — which is why the averaged targets have higher ceilings. Replicate noise is
not shared between the two pull-downs at matched replicate index (correlation ≈0.00), so the
biological replicates were not split across baits. Nor are replicates paired across the staurosporine arms or
temperature (matched vs mismatched replicate differences 0.312 vs 0.333).

staurosporine differences and temperature effects are essentially unrelated (correlations −0.08 to +0.26).

## 3. What sequence explains

Ridge, 5-fold cross-validation with folds grouped by sequence cluster, predictive R2 on held-out
proteins; increments with paired bootstrap intervals (2,000 resamples). "Share of ceiling" is
ESMC's R2 divided by the target's reliability.

| target | ceiling | composition | physicochemical | **ESMC L50** | ESMC − composition | share of ceiling |
|---|---|---|---|---|---|---|
| staurosporine vs control, DNAJA1, 35 °C | 53.0 % | 2.3 % | 4.0 % | 4.8 % | +2.4 [+1.3, +3.6] | 9 % |
| staurosporine vs control, DNAJA1, 37 °C | 53.7 % | 0.9 % | 1.5 % | 4.6 % | +3.7 [+1.8, +5.7] | 9 % |
| staurosporine vs control, DNAJA1, 43 °C | 48.8 % | 0.0 % | 0.1 % | 2.8 % | +2.8 [+0.4, +5.0] | 6 % |
| staurosporine vs control, DNAJB11, 35 °C | 55.8 % | 0.3 % | 0.8 % | 3.1 % | +2.8 [+0.6, +4.8] | 6 % |
| staurosporine vs control, DNAJB11, 37 °C | 64.4 % | 0.9 % | 1.9 % | 5.5 % | +4.6 [+2.3, +6.7] | 9 % |
| staurosporine vs control, DNAJB11, 43 °C | 56.0 % | 0.7 % | 1.2 % | 5.4 % | +4.7 [+2.0, +6.9] | 10 % |
| staurosporine vs control, averaged, 35 °C | 71.7 % | 1.6 % | 2.8 % | 6.0 % | +4.4 [+2.3, +6.5] | 8 % |
| staurosporine vs control, averaged, 37 °C | 75.5 % | 0.3 % | 1.5 % | 6.5 % | +6.2 [+3.2, +8.8] | 9 % |
| staurosporine vs control, averaged, 43 °C | 71.4 % | 0.5 % | 0.9 % | 5.7 % | +5.1 [+2.3, +7.6] | 8 % |
| 37 vs 35 °C, DNAJA1 | 71.9 % | 6.5 % | 6.7 % | 11.7 % | +5.2 [+4.1, +6.4] | 16 % |
| 37 vs 35 °C, DNAJB11 | 79.5 % | 2.1 % | 2.3 % | 9.2 % | +7.1 [+5.9, +8.3] | 12 % |
| 37 vs 35 °C, averaged | 86.5 % | 3.5 % | 3.8 % | 10.8 % | +7.3 [+6.3, +8.3] | 12 % |
| **43 vs 35 °C, DNAJA1** | 97.9 % | 10.1 % | 10.9 % | **23.0 %** | **+13.0 [+11.6, +14.3]** | 24 % |
| **43 vs 35 °C, DNAJB11** | 98.3 % | 8.3 % | 9.3 % | **21.4 %** | **+13.1 [+11.8, +14.5]** | 22 % |
| **43 vs 35 °C, averaged** | 99.2 % | 8.9 % | 9.9 % | **22.5 %** | **+13.6 [+12.3, +14.9]** | 23 % |

Shuffled-label control (43 vs 35 °C averaged): R2 = −0.022.

- **The heat response is the most sequence-predictable quantity in either experiment.** ESMC
  explains 21–23 % of it — 2.5× composition — and reaches about a quarter of the way to a ceiling
  of ~99 %. Which proteins a co-chaperone pulls down more of at 43 °C plausibly depends on thermal
  stability and aggregation propensity, which are encoded in sequence and structure, and the
  language model carries far more of that than composition does.
- **staurosporine vs control is barely sequence-predictable.** 3–7 % of the variance, under a tenth of
  its ceiling, and composition alone explains almost none of it (0–2 %). Whatever separates them in these
  pull-downs is not written in the protein's own sequence in a form a linear readout can find.
- **Averaging the co-chaperones helps the staurosporine targets** as expected: ceiling 49–64 % → 71–76 %,
  ESMC 2.8–5.5 % → 5.7–6.5 %.
- **The embedding contains composition.** Appending the 20 amino-acid fractions to ESMC changes R2
  by −0.02 to +0.16 points across all fifteen targets.
- **Charge and hydrophobicity add little over composition** (+0.1 to +1.7 points), as in the salt
  data.

## 3b. Staurosporine acts on protein kinases

Staurosporine is an ATP-competitive inhibitor of most protein kinases. Proteins carrying a UniProt
"Protein kinase" domain (346 detected) shift as a class; the 673 other ATP-binding proteins do not,
so this is not a general nucleotide-pocket effect, and the heat response shows no kinase signal at
all. Mean log2 fold change, staurosporine vs control, averaged over the two co-chaperones:

| | kinases | other ATP-binding | everything else | kinase shift | Mann-Whitney p | kinases among the 5 % most depleted |
|---|---|---|---|---|---|---|
| 35 °C | −0.146 | +0.014 | +0.003 | −0.73 sd | 2e-7 | 6.0× expected |
| 37 °C | −0.185 | −0.011 | +0.020 | **−1.05 sd** | **2e-20** | **6.5×** |
| 43 °C | −0.204 | −0.007 | +0.002 | −1.12 sd | 5e-13 | 5.1× |
| *heat, 43 vs 35 °C* | *+0.105* | *+0.327* | *+0.143* | *−0.03 sd* | *0.56* | *1.3×* |

The same holds separately for each co-chaperone (shift −0.44 to −1.00 sd, p from 1e-4 to 2e-21).
The most depleted kinases at 37 °C are FER (−2.6), CAMK1, CAMKK2, CSK, CDK5, CHEK2, PRKX, CDK2,
STK4 and CAMK2D (−2.0 to −2.5). Kinases are also over-represented 2–3× among the most *enriched*
5 %: the PKC family (PRKCB +2.6, PRKCA +2.3, PRKCE +1.9, PRKCQ +1.8, PRKCD +1.5 at 43 °C), PRKG1
(+4.4 at 37 °C) and the PKN kinases rise. Staurosporine therefore changes kinase–co-chaperone
association in both directions, kinase by kinase. Whether the depleted kinases have left the
chaperone or left the lysate is not decidable from pull-downs alone; the input lysate would settle
it.

How much of the sequence signal is this?

| target (averaged) | kinase / ATP indicators | ESMC | ESMC + indicators | ESMC on non-kinases | ESMC within kinases |
|---|---|---|---|---|---|
| 35 °C | 0.3 % | 6.0 % | 6.0 % | 6.0 % | 2.0 % |
| 37 °C | 3.3 % | 6.5 % | 6.5 % | 6.1 % | 1.1 % |
| 43 °C | 2.7 % | 5.7 % | 5.7 % | 3.1 % | 6.5 % |

The embedding already encodes kinase identity (the indicators add nothing to it), and most of what
it predicts at 35 and 37 °C lies outside the kinases. Which kinases go down and which go up is only
weakly predictable from sequence (1–6.5 % across ~300 kinases); staurosporine's measured affinity
across the kinome is the obvious external variable for that question.

## 4b. Nonlinear readout and per-sample likelihoods

The protocol from the salt data: each candidate is a 16-member MLP ensemble on the full
standardised embedding, trained in one vectorised pass; an inner split holding out 15 % of each
training fold's sequence clusters drives early stopping and chooses among 33 candidates (24
architecture/regularisation settings with plain squared error, then three per-sample losses on the
best three); the outer folds are the ga05 folds. Intervals below resample **sequence clusters**,
not proteins: when a signal is concentrated in a few protein families, resampling proteins treats a
family of correlated kinases as many independent observations and understates the uncertainty.
Heat from the final layer, staurosporine from layer 50; 1.8 GB peak on one RTX card per run.

**Heat response — the nonlinear readout helps on every target.**

| target | ceiling | ridge | selected MLP | MLP − ridge | **50/50 MLP + ridge** | blend − ridge |
|---|---|---|---|---|---|---|
| 43 vs 35 °C, averaged | 99.2 % | 25.7 % | 25.9 % | +0.2 [−1.0, +1.4] | **27.5 %** | +1.8 [+1.2, +2.4] |
| 43 vs 35 °C, DNAJA1 | 97.9 % | 25.6 % | 27.3 % | +1.6 [+0.7, +2.6] | **27.7 %** | +2.1 [+1.6, +2.6] |
| 43 vs 35 °C, DNAJB11 | 98.3 % | 24.7 % | 26.6 % | +1.9 [+0.9, +2.9] | **27.2 %** | +2.5 [+1.9, +3.0] |
| 37 vs 35 °C, averaged | 86.5 % | 11.5 % | 12.1 % | +0.7 [+0.0, +1.2] | **12.2 %** | +0.8 [+0.5, +1.1] |

An untuned 50/50 average of the MLP ensemble and ridge is the best and most dependable form: it
beats ridge on all four targets with cluster-resampled intervals well clear of zero. The heat
signal is spread over thousands of unrelated proteins, so protein- and cluster-resampled intervals
agree almost exactly. **The best sequence model of the heat response explains 27.5–27.7 % of its
variance.**

**Staurosporine — no dependable gain over ridge.**

| target | ridge | selected MLP | MLP − ridge, cluster interval |
|---|---|---|---|
| averaged, 35 °C | 6.0 % | −7.5 % | −13.4 [−40.9, +0.8] |
| averaged, 37 °C | 6.5 % | 4.2 % | −2.3 [−5.4, +0.8] |
| averaged, 43 °C | 5.1 % | 12.4 % | +7.4 [+0.4, +14.0] |

(The same targets with mean rather than median ensembling and no variance cap gave −5.2, −2.0 and
+7.8 points.) The staurosporine signal is concentrated in a few kinase families, and that defeats
model selection: across the 33 candidates, inner-validation score and outer score correlate at
**−0.68** at 35 °C and about 0 at 37 °C, against +0.5 to +0.7 for the heat targets. Which families
land in the inner split decides which model looks best. The failures are family extrapolations —
at 35 °C the three PKA regulatory subunits (PRKAR1A, 1B, 2A) are predicted at +3.1 to +3.6 when
their true change is about zero, while the strongly depleted CDK5, CDK2 and CSK are missed. The
43 °C gain is real out of fold, and it comes from the PKC family: held-out PRKCD and PRKCQ are
predicted at +1.55 and +1.65 (true +1.54 and +1.77) from the other PKC isoforms, which the
30 %-identity clustering places in a different cluster. That is legitimate transfer between
subfamilies, but it is family-level knowledge, and it does not generalise across temperatures.
Ridge remains the model to use for the staurosporine response.

**Per-sample likelihoods.** Change in R2 when each per-sample loss replaces plain squared error in
the same three best architectures:

| | measurement variance | naive (1 / noise) | replicate likelihood | heteroscedastic |
|---|---|---|---|---|
| heat | 2.3 % of var(y) | −1.9 to −0.2 (median −1.4) | −0.2 to +0.3 (median −0.1) | −0.2 to +0.5 (median +0.1) |
| staurosporine | 15.2 % of var(y) | −3.7 to +7.7 (median +0.2) | −2.1 to +10.8 (median +0.6) | −1.8 to +15.1 (median +0.7) |

As on the salt data, weighting by measurement noise alone hurts where noise is small, and the
correct likelihood comes out at the plain fit. Where noise is larger, as for staurosporine, the
median effect is slightly positive but the spread is far wider than the effect, for the same
family-concentration reason; it does not establish a gain.

## 4. Which layer

Layer 50 of 81 was the best layer for the salt titration and is used above. For the heat response
the final layer is better:

| target | layer 50 | layer 80 (final) |
|---|---|---|
| 43 vs 35 °C, averaged | 22.5 % | **25.7 %** |
| 43 vs 35 °C, DNAJA1 | 23.0 % | **26.0 %** |
| 43 vs 35 °C, DNAJB11 | 21.4 % | **25.0 %** |
| 37 vs 35 °C, averaged | 10.8 % | 11.4 % |
| staurosporine vs control, averaged, 37 °C | **6.5 %** | 4.7 % |

The best layer depends on the property: the final layer for the heat response (+3 to +4 points),
layer 50 for staurosporine vs control and for salt sensitivity. A single fixed layer is therefore a
compromise; reading several layers is the natural next step for these data.

## 5. Next

1. **Input lysate.** Expressing each pull-down relative to its own input separates a change in a
   protein's amount in the lysate from a change in its association with the co-chaperone — the
   open question for the kinases that staurosporine depletes.
2. **Staurosporine affinity across the kinome.** Which kinases go down and which go up is barely
   predictable from sequence; measured staurosporine binding constants are the natural predictor
   to test against.
3. **More than one layer.** The best layer differs by target by 3–4 points.

### Code

| file | role |
|---|---|
| `code/ga01_parse.py` | parse the design from column names, median-normalise runs, targets, split-half ceilings, replicate-pairing tests |
| `code/pd02_fetch_seqs.py` | UniProt fetch, reusing cached sequences |
| `code/pd03b_layers.py` | nine ESMC hidden states pooled in one pass (new proteins only) |
| `code/pd04_cluster.py` | MMseqs2 clustering at 30 % identity |
| `code/pd_folds.py` | the scikit-learn folds written to disk |
| `code/ga05_fit.py` | composition, physicochemical, ESMC layer 50 and 80, nested increments, controls |
| `code/ga06_kinase.py` | kinase identity versus the embedding for the staurosporine response |
| `code/ga_prep_readout.py` | merged embeddings and the exact ga05 folds for the GPU env |
| `code/pd_readout.py` | shared ridge, vectorised ensemble MLP and per-sample losses |
| `code/ga09_mlp.py` | GA_33 readout: ensemble MLPs, per-sample noise model, nested selection |
