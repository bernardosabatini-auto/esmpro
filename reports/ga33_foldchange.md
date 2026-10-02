# GA_33: co-chaperone pull-downs across temperature and STAU treatment

**Status 2026-10-02.** Sequence predicts the **heat-induced change** in what DNAJA1 and DNAJB11
pull down far better than anything in the salt data: the ESMC-6B embedding explains **22.5 %** of
the variance in the 43 °C vs 35 °C log2 fold change (25.7 % from the final layer), against a
measurement ceiling of 99 %, and adds +13.6 points over amino-acid composition. The **STAU
treatment effect** is small, half of its variance is replicate noise, and it is the same for both
co-chaperones; sequence explains 6–7 % of the co-chaperone-averaged effect against a ceiling of
71–76 %.

## 1. The experiment

`pd_data/GA_33_report_out.tsv`: 9,871 protein groups, raw log2 intensities, missing values as NaN
(10.0 % of cells). Column names encode a full factorial design:

| factor | levels |
|---|---|
| co-chaperone bait | `21A` = DNAJA1, `24` = DNAJB11 |
| treatment | `STAU_0` = control, `STAU_10` |
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
| STAU_10 vs control, per co-chaperone and temperature (6) | 8,252–9,041 | 0.21–0.24 | 37–52 % | 49–64 % |
| STAU effect averaged over the two co-chaperones (3) | 8,168–8,807 | 0.19–0.20 | — | 71–76 % |
| 37 vs 35 °C, controls (2) | 8,300 / 8,607 | 0.31 / 0.33 | 21–28 % | 72–80 % |
| 37 vs 35 °C, averaged (1) | 8,224 | 0.29 | — | 87 % |
| **43 vs 35 °C, controls (2)** | 8,309 / 8,589 | **1.08 / 1.13** | **2 %** | **98 %** |
| **43 vs 35 °C, averaged (1)** | 8,217 | **1.09** | — | **99 %** |

The two effects are very different in kind. **Heat to 43 °C moves proteins ~2.2-fold**, almost
noise-free. **STAU moves them ~1.15-fold**, and half of that is replicate scatter.

**Both effects are shared by the two co-chaperones.** Corrected for measurement noise on identical
rows, the STAU effects for DNAJA1 and DNAJB11 correlate at 0.95 (35 °C), 0.83 (37 °C) and ≈1.0
(43 °C); the 43-vs-35 °C effects correlate at 0.93 uncorrected. A treatment effect seen identically
through two different baits most likely reflects a change in the input proteome rather than in
binding to one co-chaperone, and it makes the co-chaperone average the same quantity measured with
twice the replicates — which is why the averaged targets have higher ceilings. Replicate noise is
not shared between the two pull-downs at matched replicate index (correlation ≈0.00), so the
biological replicates were not split across baits. Nor are replicates paired across treatment or
temperature (matched vs mismatched replicate differences 0.312 vs 0.333).

STAU effects and temperature effects are essentially unrelated (correlations −0.08 to +0.26).

## 3. What sequence explains

Ridge, 5-fold cross-validation with folds grouped by sequence cluster, predictive R2 on held-out
proteins; increments with paired bootstrap intervals (2,000 resamples). "Share of ceiling" is
ESMC's R2 divided by the target's reliability.

| target | ceiling | composition | physicochemical | **ESMC L50** | ESMC − composition | share of ceiling |
|---|---|---|---|---|---|---|
| STAU, DNAJA1, 35 °C | 53.0 % | 2.3 % | 4.0 % | 4.8 % | +2.4 [+1.3, +3.6] | 9 % |
| STAU, DNAJA1, 37 °C | 53.7 % | 0.9 % | 1.5 % | 4.6 % | +3.7 [+1.8, +5.7] | 9 % |
| STAU, DNAJA1, 43 °C | 48.8 % | 0.0 % | 0.1 % | 2.8 % | +2.8 [+0.4, +5.0] | 6 % |
| STAU, DNAJB11, 35 °C | 55.8 % | 0.3 % | 0.8 % | 3.1 % | +2.8 [+0.6, +4.8] | 6 % |
| STAU, DNAJB11, 37 °C | 64.4 % | 0.9 % | 1.9 % | 5.5 % | +4.6 [+2.3, +6.7] | 9 % |
| STAU, DNAJB11, 43 °C | 56.0 % | 0.7 % | 1.2 % | 5.4 % | +4.7 [+2.0, +6.9] | 10 % |
| STAU, averaged, 35 °C | 71.7 % | 1.6 % | 2.8 % | 6.0 % | +4.4 [+2.3, +6.5] | 8 % |
| STAU, averaged, 37 °C | 75.5 % | 0.3 % | 1.5 % | 6.5 % | +6.2 [+3.2, +8.8] | 9 % |
| STAU, averaged, 43 °C | 71.4 % | 0.5 % | 0.9 % | 5.7 % | +5.1 [+2.3, +7.6] | 8 % |
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
- **The STAU effect is barely sequence-predictable.** 3–7 % of the variance, under a tenth of its
  ceiling, and composition alone explains almost none of it (0–2 %). Whatever STAU does to these
  pull-downs is not written in the protein's own sequence in a form a linear readout can find.
- **Averaging the co-chaperones helps the STAU targets** as expected: ceiling 49–64 % → 71–76 %,
  ESMC 2.8–5.5 % → 5.7–6.5 %.
- **The embedding contains composition.** Appending the 20 amino-acid fractions to ESMC changes R2
  by −0.02 to +0.16 points across all fifteen targets.
- **Charge and hydrophobicity add little over composition** (+0.1 to +1.7 points), as in the salt
  data.

## 4. Which layer

Layer 50 of 81 was the best layer for the salt titration and is used above. For the heat response
the final layer is better:

| target | layer 50 | layer 80 (final) |
|---|---|---|
| 43 vs 35 °C, averaged | 22.5 % | **25.7 %** |
| 43 vs 35 °C, DNAJA1 | 23.0 % | **26.0 %** |
| 43 vs 35 °C, DNAJB11 | 21.4 % | **25.0 %** |
| 37 vs 35 °C, averaged | 10.8 % | 11.4 % |
| STAU, averaged, 37 °C | **6.5 %** | 4.7 % |

The best layer depends on the property: the final layer for the heat response (+3 to +4 points),
layer 50 for the STAU effect and for salt sensitivity. A single fixed layer is therefore a
compromise; reading several layers is the natural next step for these data.

## 5. Next

1. **The nonlinear readout on the heat target.** A regularised ensemble MLP beats ridge by 2.2
   points on the salt data at 150 mM; the heat response has a larger, cleaner signal and is the
   best place to apply it, from the final layer.
2. **Per-sample weighting for the STAU targets.** Replicate noise is 37–52 % of their variance, so
   unlike the salt data there is real room for the per-sample likelihood to act.
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
