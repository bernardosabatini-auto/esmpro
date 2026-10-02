# Predicting the salt sensitivity of an HSPB1 interactome from sequence

**Status 2026-10-02.** Sequence alone explains **15.6 % and 16.3 % of the variance** in the log2
fold change of a protein's association with HSPB1 between 0 mM and 75/150 mM salt, on
homology-grouped held-out folds (Pearson r = 0.397 and 0.416). About 99 % of that variance is
reproducible and so explainable in principle, which makes this about **one sixth of the way**.
Expressed as a correlation against the 0.99 ceiling the same result reads as 40 %; the variance
figure is the one to use.

---

## 1. The experiment and the target

`pd_data/GA_20_results_bs.xlsx`, one sheet, 7,184 proteins. Every replicate column is named
`HSPB1_<salt>_<tech>_BR<n>`: a pull-down against an HSPB1 bait across a salt titration.

| columns | condition | n after filtering |
|---|---|---|
| F–O | 0 mM salt (baseline) | — |
| Q–Z | **150 mM** salt | 6,695 |
| AB–AK | **75 mM** salt | 6,989 |

The column order is not the dose order — the second block is the high dose. Each block is
**2 technical groups × 5 biological replicates**, a nested design rather than 10 independent
measurements, so technical pairs are averaged within a biological replicate first.

F–AL are **already log2 intensities**, so the target is

```
y = mean_BR(log2 intensity at salt) - mean_BR(log2 intensity at 0 mM) = log2(ratio)
```

a log2 fold change. A protein is kept for a comparison when both blocks have at least 7 of 10
real values. Typical |y| is 0.67 log2 units at 75 mM and 0.95 at 150 mM, i.e. 1.6- to 1.9-fold.

Salt disrupts electrostatic interactions, so y measures how electrostatically dependent each
protein's association with HSPB1 is — plausibly sequence-encoded, which is why amino-acid
composition and charge are serious competitors here rather than formalities.

## 2. The source file needs repair before use

A global case-insensitive replace of `NA` with `0` had been applied to the sheet, hitting text as
well as numbers. Consequences, all silent:

- **Missing values became literal `0`**, impossible for a log2 intensity. Zeros run 0.90 % of
  cells at 0 mM, 5.25 % at 150 mM, 2.26 % at 75 mM — condition-dependent, as below-detection
  missingness should be. Every `0` in F–AL is therefore read as missing.
- **The sheet's own block means in P/AA/AL are unusable.** Recomputing the naive 10-value mean
  *including* the zeros reproduces those columns to 7.1e-15, which is what confirms they were
  computed over the zeros. They are not used; all targets are built from the replicates.
- **Eight UniProt accessions were mangled** (`Q8072` for `Q8NA72`), with 426 gene symbols and 367
  entry names also damaged. Each accession was repaired by reinserting `NA` at each `0` and
  confirming the candidate against the gene symbol UniProt returned. All eight resolved, giving
  100 % sequence coverage.

**Parse check.** On rows with no missing values the freshly computed fold changes reproduce the
file's own limma `logFC` columns at r = 1.00000. One property of those columns is worth
recording: limma's `logFC` is **exactly 2× the log2 fold change** for 100 % of rows in both
comparisons (checked by hand on P24752: 16.8546 → 12.4544, difference −4.4002, limma −8.8004),
the signature of a ±1 rather than 0/1 contrast coding. Whether the p-values carry the same factor
is undetermined, since limma's moderated variance legitimately differs from a per-protein
standard error.

## 3. The measurement ceiling

Ten columns per block are 5 biological replicates measured twice, so a reproducibility estimate
must split **by biological replicate**. Splitting technical pairs apart would measure instrument
noise and report a reassuringly high ceiling that says nothing about biological reproducibility.

Two disjoint biological-replicate pairs give two independent fold-change estimates; their
agreement across proteins, Spearman-Brown corrected to all five replicates (k = 2.5):

| | 75 mM | 150 mM |
|---|---|---|
| half-vs-half agreement | 0.966 | 0.981 |
| implied reliability of the target | 0.986 | 0.993 |
| maximum r any predictor could reach | 0.993 | 0.996 |
| **maximum variance explainable** | **98.6 %** | **99.3 %** |

The target is highly reproducible — sd 0.67–0.95 against a median per-protein standard error of
0.042. **Measurement noise is not what limits the models below**, and every number here is read
against ~99 %.

Features derived from the assay itself are excluded from the models: the baseline level measures
the pull-down's dynamic range rather than any property of the protein, and the `XIC` column is a
total summed across all runs, which combined with the baseline reconstructs the condition level
algebraically. Only sequence-derived features are used.

## 4. What sequence explains

5-fold cross-validation, folds grouped by MMseqs2 cluster at 30 % identity / 50 % coverage (5,369
clusters over 7,184 proteins), ridge penalty chosen over a fixed grid, bootstrap intervals over
proteins (2,000 resamples). `R2` is the predictive R-squared on held-out predictions,
`1 - SSE/SS_tot`, which unlike r-squared also penalises a mis-scaled prediction.

| model | r (75 mM) | r (150 mM) | R2 (75) | R2 (150) | var. explained |
|---|---|---|---|---|---|
| null (global mean) | −0.029 | −0.040 | −0.000 | −0.001 | 0 % |
| k-NN in embedding space | 0.284 | 0.292 | 0.061 | 0.071 | 6–7 % |
| physicochemical, 31-d | 0.254 | 0.296 | 0.065 | 0.088 | 6.5–8.8 % |
| PLS, 12 components | 0.385 | 0.405 | 0.130 | 0.146 | 13–15 % |
| **ESMC-6B mean-pool, 2560-d, ridge** | **0.397** [0.375, 0.418] | **0.416** [0.392, 0.439] | **0.156** | **0.163** | **15.6–16.3 %** |
| ESMC + physicochemical | 0.398 | 0.418 | 0.158 | 0.166 | 15.8–16.6 % |
| *measurement ceiling* | *0.993* | *0.996* | — | — | *98.6–99.3 %* |

For the ESMC row, r-squared (0.158 / 0.173) and out-of-fold R2 (0.156 / 0.163) agree to within
0.01, so predictions are correctly scaled rather than shrunk toward the mean. Where the two
diverge the model is miscalibrated: k-NN is r-squared 0.081 against R2 0.061, PLS 0.148 against
0.130.

**In fold-change units.** The target's spread is 0.673 log2 units at 75 mM (1.59-fold typical
deviation); after prediction the residual spread is 0.618, or 1.54-fold. At 150 mM, 0.946
(1.93-fold) becomes 0.865 (1.82-fold). A 16 % variance reduction is an 8 % reduction in spread,
and that is what r = 0.4 buys.

Three further points:

- **The embedding beats explicit biophysics by about a factor of two in variance** (15.6 % against
  6.5 % at 75 mM; 16.3 % against 8.8 % at 150 mM). Charge and hydrophobicity are the
  mechanistically obvious features for a salt-disruption experiment, and the language model
  carries something past them. Adding them to ESMC changes R2 by +0.002.
- **It is not family lookup.** k-NN in the same embedding space reaches 6–7 % against ridge's
  15.6–16.3 %, so the fitted map does work beyond "this protein resembles that one".
- **Homology inflation is small here.** Random folds score only +0.004 / +0.013 in r above
  grouped folds, despite 39.6 % of proteins having a relative in the set — worth having measured
  rather than assumed.

Controls: shuffled labels give R2 = −0.011 and −0.074, negative as they must be. Inverse-variance
weighting *hurts* (R2 0.119 / 0.088), because the noisiest proteins carry the largest fold
changes, so down-weighting them discards signal.

## 5. Composition versus representation

A nested decomposition, each level a superset of the last, scored identically throughout. The
increments carry a paired bootstrap over proteins on the out-of-fold predictions, so "the
embedding adds X" is an interval rather than a difference of two point estimates. (Penalty here
is selected on R2 rather than r, which moves the ESMC row by ~0.005 against §4.)

| feature set | dim | R2 (75 mM) | R2 (150 mM) | increment over previous (150 mM) |
|---|---|---|---|---|
| length | 1 | −0.000 | −0.000 | — |
| **amino-acid composition** | 20 | **0.062** | **0.075** | +0.075 [+0.062, +0.088] |
| composition + length | 21 | 0.064 | 0.080 | +0.005 [+0.002, +0.009] |
| physicochemical (+ charge, GRAVY, patterning, MW) | 31 | 0.066 | 0.082 | +0.002 [−0.001, +0.005] |
| **ESMC-6B mean-pool** | 2560 | **0.153** | **0.170** | **+0.088 [+0.075, +0.101]** |
| ESMC + composition | 2580 | 0.153 | 0.171 | +0.001 [+0.000, +0.001] |

- **Composition alone explains 6.2 % (75 mM) and 7.5 % (150 mM) of the variance.** Twenty numbers
  per protein — no charge, no length, no structure — recover nearly half of what the 6-billion
  parameter model does.
- **Length explains nothing at all** (R2 = −0.000), which is a useful negative: the signal is not
  a size effect.
- **Hand-built biophysics adds almost nothing to composition.** Net charge, charge per residue,
  K/R and D/E fractions, GRAVY, aromaticity, disorder-prone fraction, charge segregation and
  molecular weight together add +0.002 over composition-plus-length, with an interval that
  includes zero. For a salt-disruption experiment that is a surprise worth stating plainly: the
  obvious electrostatic descriptors carry no information that the raw composition does not
  already have.
- **The embedding adds +9.1 points at 75 mM [+7.7, +10.5] and +9.6 at 150 mM [+8.2, +10.9]** over
  composition — a factor of 2.5 and 2.3 in variance explained. This is the clearest statement of
  what the language model is worth here.
- **The embedding already contains composition.** Appending the 20 fractions to the 2560-d
  embedding gains +0.001. So the embedding's advantage is not that it also knows composition; it
  is a genuinely different and larger signal that subsumes it.

## 6. Nonlinear and joint models do not help

Same rows (6,681 proteins passing both comparisons), same folds; gradient boosting on 256 PCA
components, and a torch MLP (512-128) early-stopped on an inner split carved out **by cluster** —
a random inner split would put paralogs on both sides and stop late.

| features | ridge | gradient boosting | MLP |
|---|---|---|---|
| ESMC, 75 mM (r) | **0.401** | 0.357 | 0.385 |
| ESMC, 150 mM (r) | **0.416** | 0.373 | 0.390 |

Ridge wins wherever the embedding is involved. On 2,560 correlated dimensions and ~6,700
proteins, a linear map is the right capacity.

y75 and y150 correlate at **0.913**, so joint fitting should have helped. It does not: a
multi-output ridge reproduces the separate fits exactly, a multi-output MLP is worse, and the
mean-and-shape reparameterisation — predict `(y75+y150)/2` and `y150−y75`, then reconstruct —
reproduces the separate fits to three decimals. Since the dose-averaged target is *less noisy*
than either dose and gains nothing, **the limit is not measurement noise**, consistent with the
99 % ceiling. The dose shape is separately predictable at r = 0.433, so the two doses differ in
content even though joint fitting extracts nothing extra. Shuffled-label control through the
boosted path: r = 0.006.

## 7. Which ESMC layer to read

Nine evenly spaced hidden states of 81 were pooled in one forward pass and scored separately.
Pearson r against an assay-independent target (baseline taken from disjoint biological
replicates):

| hidden state | 75 mM | 150 mM | same layer → baseline level |
|---|---|---|---|
| 0 (embeddings) | 0.258 | 0.310 | 0.309 |
| 10 | 0.362 | 0.418 | 0.382 |
| 20 | 0.373 | 0.433 | 0.383 |
| 30 | 0.385 | 0.443 | 0.401 |
| 40 | 0.382 | 0.439 | 0.405 |
| **50** | **0.390** | **0.454** | 0.426 |
| 60 | 0.379 | 0.425 | 0.426 |
| 70 | 0.377 | 0.431 | 0.430 |
| 80 (last, the default) | 0.380 | 0.436 | 0.454 |

The profile is smooth, rises steeply out of the token embeddings, and peaks around layer 50 of
81 rather than at the end — as expected if the final layers specialise for token prediction. The
right-hand column is the same layers predicting the assay's own dynamic range, and it moves the
other way, peaking at the last layer: **the deepest layers are relatively more about expression
level and less about salt sensitivity.**

Scored on the main target for comparability with §4:

| | 75 mM | 150 mM |
|---|---|---|
| layer 50 | r 0.400, R2 0.160 (16.0 %) | r 0.426, R2 0.182 (18.2 %) |
| layer 80 (default) | r 0.392, R2 0.153 (15.3 %) | r 0.415, R2 0.170 (17.0 %) |
| gain, paired bootstrap | +0.7 points [−0.4, +1.8] | +1.1 points [−0.2, +2.4] |

So layer 50 is worth about a point of variance, consistently in both comparisons and in the same
direction, but the paired interval includes zero in both. **Suggestive, not established** — it is
free to adopt (the vectors already exist) and the layer profile is coherent, but it does not on
its own close any meaningful part of the gap.

## 8. Compute

| stage | hardware | wall | throughput | peak GPU |
|---|---|---|---|---|
| ESMC-6B embeddings, last layer | 1 RTX (96 GB) | 4.9 min | 17.1k residues/s | 13.3 GB |
| ESMC-6B, 9 hidden states pooled | 1 A100 (40 GB) | 19.0 min | 4.4k residues/s | 18.0 GB |
| nonlinear + joint models | 1 GPU + 16 CPU | 6.8 min | — | — |

**Padding efficiency, not memory fill, is the lever for this workload.** Sweeping the token budget
from 8,000 to 160,000 padded residues made throughput fall monotonically, 17.0k → 12.4k
residues/s, because batch length-homogeneity fell with it (padding efficiency 98.0 % → 80.6 %).
Length-sorted token-budget batching at 99.3 % efficiency on a small budget beats filling the card.
The 254 proteins over 2,048 residues (up to 34,350) are embedded in overlapping 2,000-residue
windows pooled by residue weight, rather than truncated. The layer sweep runs at a quarter of the
single-layer throughput because keeping 81 hidden states resident multiplies activation memory and
forces a smaller token budget.

## 9. Where this stands

Sequence explains 16–18 % of the variance in salt sensitivity against ~99 % being explainable.
Amino-acid composition alone gets 6–8 % of it; the embedding roughly 2.4× that, and it subsumes
composition rather than adding to it. Hand-built electrostatic descriptors add nothing over raw
composition. The signal is not nearest-neighbour lookup, and the gap to the ceiling is large and
is not measurement noise.

Most likely reasons for the gap, cheapest to test first:

1. **Representation — tested, small.** Layer 50 of 81 beats the final layer by about one point
   of variance, consistently but within the paired bootstrap interval (§7). Worth adopting, not
   worth much.
2. **Pooling.** One mean vector per protein discards where in the sequence the signal sits. An
   attention pool, or residue-level features restricted to predicted-disordered or
   charge-segregated regions, is the natural next step.
3. **The biology may not be in the sequence alone.** How a protein responds to salt in a
   co-immunoprecipitation depends on the complex it sits in, not only on its own charge, so there
   is a real sequence-only ceiling somewhere below 99 %. A useful probe: how well one dose
   predicts the other, which bounds the protein-intrinsic component empirically.

---

### Code

| file | role |
|---|---|
| `code/pd01_parse.py` | stream the sheet (stdlib `zipfile` + `iterparse`), zeros to missing, nested-design targets and standard errors, limma cross-check |
| `code/pd02_fetch_seqs.py` | UniProt fetch with backoff, repair the 8 mangled accessions, confirm each by gene symbol |
| `code/pd03_embed.py` | ESMC-6B mean/max pools, token-budget batching, overlapping windows |
| `code/pd03b_layers.py` | nine hidden states pooled in one pass |
| `code/pd04_cluster.py` | MMseqs2 clustering at 30 % identity |
| `code/pd05_fit.py` | ceiling, baselines, ridge, PLS, weighted variants, controls |
| `code/pd05b_abundance.py` | target decomposition and detection-floor diagnostics |
| `code/pd05c_disjoint.py` | disjoint-replicate construction of the assay-independent target |
| `code/pd06_nonlinear.py` | gradient boosting, GPU MLP, multi-output and dose-shape joint models |
| `code/pd07_layers.py` | score each hidden state |
| `code/pd07b_bestlayer.py` | the chosen layer on the main target, paired bootstrap |
| `code/pd08_nested.py` | nested decomposition: length, composition, physicochemical, embedding |
