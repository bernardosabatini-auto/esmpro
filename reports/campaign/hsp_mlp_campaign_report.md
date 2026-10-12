---
title: "What predicts the HSP pull-downs: sequence, structure, and the complex"
date: "11 October 2026"
geometry: margin=2.1cm
fontsize: 10pt
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \DeclareUnicodeCharacter{2265}{$\geq$}
  - \DeclareUnicodeCharacter{2264}{$\leq$}
  - \DeclareUnicodeCharacter{2212}{$-$}
  - \DeclareUnicodeCharacter{2248}{$\approx$}
  - \DeclareUnicodeCharacter{00D7}{$\times$}
  - \DeclareUnicodeCharacter{00B5}{$\mu$}
  - \DeclareUnicodeCharacter{00B2}{$^2$}
  - \DeclareUnicodeCharacter{00B1}{$\pm$}
  - \DeclareUnicodeCharacter{00B0}{$^\circ$}
  - \DeclareUnicodeCharacter{2192}{$\rightarrow$}
  - \DeclareUnicodeCharacter{03B1}{$\alpha$}
  - \DeclareUnicodeCharacter{03B2}{$\beta$}
---

# Summary

**In one sentence.** About a quarter of what these pull-downs measure can be read off a protein's own
amino acid sequence; most of the rest is not a property of the protein at all but of the complex it sits
in, and that part turns out to be how the complex behaves in the lysate rather than anything about how it
binds the chaperone.

**The question.** A pull-down gives a number per protein: how much of it came down with the bait, and how
that changed with salt, temperature or Mg^2+^. How much of that number is determined by the protein
itself? Whatever is *not* determined by the protein is a pointer to what else is going on in the tube, so
the leftover is as interesting as the part that is predicted. This report answers both halves, and spends
most of its length on the leftover.

**The approach.** A single neural network reads each protein's sequence -- via ESMC-6B, a 6-billion-parameter
protein language model, and its annotated sparse autoencoder (SAE) -- and predicts every non-drug quantity
measured across the HSP pull-downs at once:

| Kind | Targets | Runs |
|---------------|------------------|---------------------------|
| Baseline pull-down, no perturbation | pull-down level | GA_20, GA_22, GA_24, FS73, FS76 (HSPB1); GA_33 (DNAJA1, DNAJB11) |
| Abundance | lysate level, supernatant level | GA_22, GA_24, GA_33 |
| Abundance-free binding | bound:free partition; enrichment over input | GA_22, GA_24 (partition); GA_33 (enrichment) |
| Effects | salt | GA_20, GA_22, GA_24 |
| | temperature | GA_33 |
| | Mg^2+^ | FS76 |

That makes 33 targets over 11,541 proteins. Drug effects are excluded.

Every number is out-of-fold. Folds are grouped by 30% sequence identity, so no homolog of a test
protein is ever trained on. Each number is read against its own split-half ceiling.

| Question | Answer |
|-----------|-------------------------------------------------|
| How much does sequence predict? | Mean R² / ceiling 0.27-0.28 over all 33 targets. Abundance 0.51; abundance-free binding 0.33; baseline pull-down 0.26; temperature 0.24; salt 0.23; Mg 0.17 |
| Is it better than a linear read-out? | Yes, slightly: +0.028 [0.023, 0.034] over ridge on the best layer; most on salt (+0.039) |
| Does reading residue by residue help? | No. A learned attention-pooling front end over 7.4 M residue states equals the mean-pooled model: +0.000 [−0.003, +0.004] |
| Does one shared model help? | Marginally. Multi-task beats 33 single-task models on 22 of 32 targets, but by 0.003 R² on average |
| Do conditions share one representation? | Yes. A trunk trained without a run predicts that run nearly as well as one trained with it. The exceptions are the condition types no other run contains: heat (0.26 → 0.19) and Mg (0.15 → 0.10) |
| What predicts the pull-down with no perturbation? | Mostly how much protein there is. Measured abundance plus Tm beats sequence for every baseline. Sequence plus those measurements is best: R² 0.30-0.54 |
| Fold changes from sequence alone, no input information? | R2 0.12-0.27 per perturbation (rho 0.35-0.52), calibrated; top-10 % responders recovered 2.5-4x above chance; same accuracy as the full model (-0.002 [-0.007, +0.002]) |
| What does sequence add beyond abundance? | Salt and Mg responses: +0.056 and +0.045 R² / ceiling over measured abundance and Tm alone |
| Regressing out the baseline | Salt and Mg responses are almost untouched (83-99% of their variance remains). Heat is not: the baseline accounts for about half of the 43 °C response and half of what sequence predicts about it |
| Is the plateau a ceiling on predictability? | No, a ceiling on what a *single protein's* sequence can say. Subunits of the same complex miss the prediction together and in the same direction (r 0.22-0.58 between a protein's error and the mean error of its complex-mates; r ≈ 0.00 under permutation). Telling the model how a protein's complex-mates came out lifts mean R² / ceiling 0.309 → 0.410, far more than any architecture change |
| Does AlphaFold structure help? | No. Surface vs buried ion pairs, charged surface patches, Debye-Huckel screening at the real ionic strengths and Mg^2+^ carboxylate clusters all add 0.000 over the embedding; the largest residual correlation of any structural feature is 0.092 |
| Does the native partner's interface help? | No. Measured on 29,816 experimental assemblies covering 6,306 proteins: how much exposed hydrophobic and aggregation-prone surface a native partner buries adds +0.002 (salt and heat: −0.000). Three pre-stated directional predictions all fail |
| Does a *predicted* chaperone-client interface help? | No, and the measurement is half noise. 4,787 Boltz-2 co-folds of the HSPB1 ACD dimer with tiled 16-mer client windows: all four pre-stated predictions fail, including real windows versus their own scrambles (-0.026, p = 0.955). The same window re-run moves by 0.140 ipTM, so half the window score is run-to-run noise |
| What *is* the complex effect, then? | Mostly not the pull-down. Run the same correlation on the unbound supernatant, on the pull-down, and on their ratio: **0.55-0.61, 0.38-0.40, 0.26-0.31** -- highest where the beads were never involved. Complex members already behave alike in the lysate. About a third is simply that they are present at similar amounts (a correction to 3c). Only 0.20-0.23 could be about binding; co-capture is the leading guess, supported in 3 of 4 targets |
| What the SAE features say | Pull-down and abundance: structured enzymes up, phosphorylated disordered Ser/Thr-rich regions down. Abundance-free binding, salt and Mg: membrane-helix features. Heat: structured α/β enzyme cores |

**The bottom line.** Sequence predicts about a quarter of what these measurements reliably contain, and
an MLP over 94 architectures, attention pooling and multi-task sharing all hit the same plateau. That
plateau is not a limit on predictability, but the way past it is not more structure: AlphaFold surface
electrostatics, ion pairs and mask-pooled residue states all add nothing, because the embedding already
contains them. What the model misses is instead shared among the subunits of a complex, which no
sequence model can reach because the folds separate homologs and complex-mates are not homologs. One
complex-context term takes mean R² / ceiling from 0.31 to 0.41 for the quarter of proteins in a curated
complex, though it fades to +0.007 across the broad STRING physical network. That complex effect is not
an interface effect: on 29,816 experimental structures, how much surface a native partner buries explains
none of the residual and only +0.014 of the complex-shared part, so what a complex contributes is
something other than steric competition for the chaperone's binding surface. Predicting the missing
structure does not rescue it either: 4,787 co-folds of the HSPB1 ACD dimer with tiled client windows fail
every pre-stated test, and cannot tell a real window from its own scramble, so three independent
structural approaches now land in the same narrow band just above nothing (largest residual correlation
0.092, 0.087 and 0.134). The
no-perturbation pull-down is mostly abundance and stability. The salt and Mg responses are where
sequence carries information that the measured lysate properties do not.

Section 3g then asks what that complex effect is, and most of it turns out not to be about the chaperone
at all. The key is that GA_22 and GA_24 measure every protein three times in the same reaction: in the
**supernatant** (what stayed unbound, which never touches the beads), in the **pull-down** (what bound),
and as the **partition**, their ratio. Running the complex-mate correlation separately on each of the
three asks at which point in the protocol the subunits of a complex start behaving alike.

They already behave alike before the bait is involved. The correlation is **0.55-0.61 in the supernatant,
0.38-0.40 in the pull-down made from it, and 0.26-0.31 in the ratio** -- highest where there is no
pull-down at all, and falling at every stage after. Every difference interval excludes zero, and the
gradient steepens when each measurement is divided by its own reliability. Had the correlation been
manufactured by the pull-down -- for instance by its non-proportional capture, which is the obvious
suspect -- the order would have run the other way.

So most of what sequence misses is not chaperone biology. About a third of it is simply that complex
members are present at similar amounts, which 3c wrongly dismissed and which is corrected here. Most of
the remainder is that the members of a complex go into and out of solution together in the lysate, before
any bait is added; the pull-down then inherits a damped copy of that. Only the 0.20-0.23 that survives
dividing the lysate out could be about binding at all. That is also why three sections hunting an
interface property found nothing: an interface could only ever have explained the smallest of the four
parts.

**Status.** This is the final version of the campaign report. Every analysis these data support has been
run. What remains is five questions for the lab and two proposed experiments, set out in section 9.

# How to read this report

Six terms carry most of the argument. They are defined here rather than where they first appear, because
every later section uses all of them.

**Target.** One measured quantity, for every protein at once. For example, *how much of each protein HSPB1
pulled down with no KCl added* is one target; *how much that changed at 120 mM KCl* is another. There are
33, drawn from six experiments.

**Fold, and "out-of-fold".** Proteins are split into five blocks, grouped so that no protein in one block
shares more than 30 % sequence identity with any protein in another. Each block is predicted by a model
trained only on the other four. So every number quoted for a protein comes from a model that saw neither
that protein nor any of its close relatives. This matters: without it a model can score well simply by
recognising a family it has already been shown.

**Ceiling.** How well the experiment agrees with itself: split the replicates into two halves, compute each
protein's value from each half, and correlate the halves across proteins. Nothing can be predicted better
than the measurement reproduces, so this is the bar. Accuracy is reported as **R² / ceiling**, where 1.0
would mean "as accurate as repeating the experiment" and 0.27 means "about a quarter of the way there".

**Residual.** Measured minus predicted, for one protein and one target. It is the part of the measurement
that the protein's sequence did not account for. The whole second half of this report is about what the
residuals contain.

**Complex-mate.** Another protein annotated as a subunit of the same assembly in Complex Portal, a curated
catalogue of stable protein complexes. Of the 11,570 measured proteins, 2,858 have at least one measured
complex-mate.

**The complex effect** (used throughout; the statistic behind it is the *complex-mate correlation*). Take
one protein and compute its residual. Separately, take the other subunits of its complex and average
*their* residuals. The complex effect is the correlation between those two numbers, taken across proteins.

> If it were zero, the model's errors would be independent from protein to protein: knowing that the other
> proteasome subunits came out higher than predicted would tell you nothing about this subunit. It is not
> zero. Across the 33 targets it averages **r = 0.34**, and for some measurements reaches 0.6. In plain
> terms: **subunits of the same complex miss the prediction together, and in the same direction.** If the
> rest of your assembly came out well above what sequence predicted, so did you.

This cannot be the model recognising protein families, because subunits of one complex are not sequence
relatives — they land in different blocks and are never fitted together. A permutation that shuffles
residuals within a block, preserving every complex's size, puts the null at r ≈ 0.00.

Why it matters: the complex effect measures how much of a pull-down measurement is a property of the
**assembly** a protein belongs to rather than of the **molecule itself** — and therefore how much of it no
sequence model, however good, can ever reach. Sections 3c to 3g are about how large that is, and what it
turns out to be.

# 1. Data, targets and features

**Proteins.** The union of every run is 11,570 proteins, of which 11,541 have a sequence and all
features. Embeddings and SAE features already existed for 9,879. I computed the rest in one pass of
ESMC-6B (1,662 proteins, under 1 minute on one RTX card).

A control re-embedding of 60 stored proteins reproduced them: median r 1.0000. One protein moved
slightly with batch context (r 0.98). Embedded alone it sits between its two batched versions, so this
is bf16 numerics, not a pipeline error.

**Targets** (`cm02`). 33 per-protein quantities, each with a split-half reliability across proteins
(the ceiling):

| Kind | n | Ceiling range |
|---------------------------------------------------|---|--------|
| Abundance (GA_33 input, GA_22/24 supernatant) | 3 | 0.98-0.995 |
| Baseline pull-down (no perturbation) | 7 | 0.98-0.998 |
| Abundance-free binding (GA_22/24 partition, GA_33 enrichment) | 4 | 0.99 |
| Salt (GA_20 75/150 mM, GA_22/24 60/120 mM in pull-down, supernatant and partition) | 14 | 0.76-0.99 |
| Temperature (GA_33, 37 and 43 vs 35 °C, both baits) | 4 | 0.64-0.98 |
| Mg^2+^ (FS76) | 1 | 0.98 |

Drug arms were excluded throughout, with one exception: FS76's Mg effect is averaged over the lithium
doses, because lithium has no proteome-wide effect (reliability 0.14). GA_22 and GA_24 effects use only
their drug-free arms and carry the drift correction of the GA_22/24 report.

**Inputs.**

- ESMC-6B hidden states 0, 10, ..., 80, mean-pooled.
- The layer-60 SAE: 16,384 features, max over residues.
- Per-residue final-layer states for the attention model: 7.36 M residues, 37 GB.
- Sequence covariates: length, charge, pI, hydropathy, composition.
- Optionally, measured per-protein properties: lysate abundance here, PaxDb abundance, Meltome Tm.

# 2. The model and how it was trained

One MLP:

- an input branch per layer (a learned projection) and for the SAE;
- a shared trunk ending in a 64-dimensional latent;
- one linear head per target, reading that shared latent.

The loss is masked MSE over all observed targets, each standardised on the training fold.

**Efficient training.** Each architecture is trained as a population of 72-144 members, differing in
learning rate, weight decay, dropout and seed. All members are one batched computation: every weight
carries a member axis, and a vectorised AdamW applies per-member hyperparameters.

All inputs sit on the GPU. Each member is early-stopped on an inner validation fold, and the outer test
fold is predicted by the mean of the eight best members (chosen on validation only).

**GPU choice.** RTX PRO 6000 (96 GB) for everything:

- data plus a population is 2-50 GB;
- the attention model holds 37 GB of residue states plus 15-25 GB of work;
- nothing needed an H100 or H200.

| Job type | SM active | Occupancy | Tensor active | DRAM active | Peak memory |
|------------|------|------|---------|--------|--------------------|
| MLP populations | 0.84-0.95 | 0.49-0.58 | 0.04-0.05 | 0.76-0.87 | 3-50 GB |
| Attention MLP | 0.95 | 0.48 | 0.21 | 0.66 | 54-66 GB |
| ESMC-6B embedding | | | | | 18.5 GB at 12-17 k residues/s |

The populations are memory-bandwidth-bound: the matrices per member are small. Packing more members per
step was the lever, and it kept SM activity near the maximum.

**Scale.** About 10 GPU-hours on at most 8 RTX cards, in roughly three hours of wall clock:

- 94 MLP architectures;
- 7 attention configurations;
- 33 single-task models;
- 6 leave-one-run-out trunks;
- 6 residualisation and covariate models.

I stopped four sweep jobs early. Their SAE-input groups cost 10-20 minutes each and had already shown
that the SAE adds nothing to the dense layers.

# 3. How much sequence predicts

![Out-of-fold R² / ceiling for every target: ridge on ESMC layer 80; the best MLP; the attention-pooling MLP; the MLP with measured abundance and Tm added; measured abundance and Tm alone.](../figures/campaign/overview.png){width=100%}

![Predicted (out-of-fold) against measured for one target of each kind, with the least-squares line and 95% band.](../figures/campaign/scatter.png){width=100%}

Mean R² / ceiling, all targets, with paired cluster-bootstrap intervals (1,000 resamples of sequence
clusters):

| Model | All | Abund. | Base. | Enrich. | Salt | Temp. | Mg |
|----------------------------|----|-----|----|------|----|----|----|
| Ridge, ESMC layer 80 | 0.245 | 0.504 | 0.240 | 0.302 | 0.191 | 0.213 | 0.149 |
| MLP, layers 50+80 | 0.273 | 0.506 | 0.261 | 0.333 | 0.229 | 0.238 | 0.168 |
| Attention MLP | 0.273 | 0.505 | 0.257 | 0.336 | 0.230 | 0.239 | 0.172 |
| Abundance + Tm only (32 t.) | 0.286 | 0.729 | 0.336 | 0.403 | 0.186 | 0.246 | 0.131 |
| Sequence + abundance + Tm (32 t.) | 0.316 | 0.703 | 0.348 | 0.416 | 0.243 | 0.258 | 0.176 |
| Sequence only (32 t.) | 0.263 | 0.497 | 0.257 | 0.329 | 0.228 | 0.233 | 0.167 |

The models marked "32 t." are trained on 32 targets; the MLP row is the multi-task model with 144 members; the lysate abundance target is
dropped because it is the abundance input.

| Comparison | Difference in R² / ceiling, 95% interval |
|-------------------|-----------------------------------------|
| MLP − ridge | all +0.028 [0.023, 0.034]; salt +0.039 [0.028, 0.050]; enrichment +0.031; abundance +0.002 (n.s.) |
| Attention − mean-pooled MLP | all +0.000 [−0.003, +0.004]; no kind differs |
| Sequence + measurements − measurements alone | salt +0.056 [0.042, 0.072]; Mg +0.045 [0.024, 0.065]; baseline +0.012 (n.s.); abundance −0.026 |
| Measurements − sequence | abundance +0.233; baseline +0.079; enrichment +0.074; salt −0.042; Mg −0.036 |

**A plateau.** All 94 MLP architectures lie within 0.274-0.276 of the best. That holds across:

- layer sets (50 + 80 or all nine);
- projection widths of 32-256;
- latents of 16-256;
- trunks of 256-2,048;
- 150 or 300 epochs.

The SAE as an input is worse than the dense layers: 0.18 alone, nothing added on top. Attention pooling
over residues, at 4, 8 or 16 heads (seven configurations, 0.268-0.273 of ceiling), equals mean pooling. The ensemble's members are nearly
interchangeable (best member vs top-8 mean: r 0.9996).

The limit is what ESMC-6B's representation carries about these measurements, not the read-out.

**The one weakness.** Adding sequence to the measured covariates lowers abundance (0.729 → 0.703) and a
few individual baselines. The combined model's regularisation does not fully adapt when one input block is far
more informative than the other. A residual design (sequence predicting what the covariates leave) would
avoid this.

# 3b. Fold changes alone: each perturbation from sequence, with no input information

This model knows nothing about the input / lysate distribution. One MLP is trained only on the 11 pull-down fold
changes, with no abundance, supernatant, partition or baseline targets and no measured covariates: ESMC-6B layers
50 + 80 and sequence covariates in, 11 fold changes out (`cm11`, run `s5`).

![Predicted against measured log2 fold change for every perturbation (sequence only, out-of-fold), with the least-squares line; last panel: the same targets from four models.](../figures/campaign/fold_changes.png){width=100%}

| Perturbation | Ceiling | R2 | R2/ceil. | rho | Top 10 % | Bottom 10 % | Sign |
|---------------------|------|----|-------|---|-------|---------|---|
| HSPB1, salt 75 mM (GA_20) | 0.98 | 0.197 | 0.20 | 0.43 | 37 % | 30 % | 79 % |
| HSPB1, salt 150 mM (GA_20) | 0.99 | 0.220 | 0.22 | 0.44 | 38 % | 29 % | 80 % |
| HSPB1, KCl 60 (GA_22) | 0.90 | 0.136 | 0.15 | 0.36 | 29 % | 27 % | 76 % |
| HSPB1, KCl 120 (GA_22) | 0.96 | 0.193 | 0.20 | 0.40 | 38 % | 27 % | 78 % |
| HSPB1, KCl 60 (GA_24) | 0.91 | 0.152 | 0.17 | 0.38 | 33 % | 27 % | 76 % |
| HSPB1, KCl 120 (GA_24) | 0.97 | 0.201 | 0.21 | 0.41 | 38 % | 25 % | 79 % |
| DNAJA1, 37 vs 35 C | 0.64 | 0.136 | 0.21 | 0.36 | 25 % | 28 % | 79 % |
| DNAJA1, 43 vs 35 C | 0.97 | 0.274 | 0.28 | 0.52 | 37 % | 33 % | 89 % |
| DNAJB11, 37 vs 35 C | 0.74 | 0.117 | 0.16 | 0.35 | 24 % | 25 % | 77 % |
| DNAJB11, 43 vs 35 C | 0.98 | 0.268 | 0.27 | 0.51 | 36 % | 27 % | 89 % |
| HSPB1, Mg2+ 2.5 vs 0.25 mM | 0.98 | 0.176 | 0.18 | 0.40 | 20 % | 37 % | 77 % |

"Top / Bottom 10 %": of the proteins measured in the top (or bottom) 10 %, the share the prediction also places in its
top (bottom) 10 %; chance is 10 %. "Sign": among the 10 % of proteins with the largest measured change,
the share whose predicted change is on the same side of the median.

* **Sequence alone predicts every fold change, modestly**: R2 0.12-0.27 (0.15-0.28 of ceiling), rank correlation
  0.35-0.52. The predictions are calibrated (out-of-fold slopes 0.92-1.06; Mg 1.20), so the size of a predicted
  change means what it says.
* **As a screen it is 2.5-4 times better than chance**: a quarter to over a third of the strongest responders in
  each direction land in the predicted top or bottom 10 %, and the direction is right for 76-89 % of the strongest.
  Heat at 43 C is the most predictable perturbation; KCl 60 mM and heat 37 C, the smallest effects, the least.
  Mg is asymmetric: proteins that fall with Mg are found far better (37 %) than those that rise (20 %).
* **Leaving out the input information costs nothing**: the fold-change-only model equals the full multi-task model
  (difference in mean R2 / ceiling -0.002 [-0.007, +0.002]), matches one MLP per perturbation (+0.004 [-0.000,
  +0.008]) and beats ridge (+0.030 [0.022, 0.037]). The abundance and baseline targets did not help the network
  learn the fold changes, consistent with section 5.3: salt and Mg responses are largely independent of the baseline.

# 3c. What the model misses is largely a property of the complex, not the protein

Section 3 reports a plateau: 94 architectures, attention pooling over every residue, single- against multi-task, all
land within 0.274-0.276 of ceiling. A plateau can mean the representation is exhausted, or that the remaining
variance is not a function of the protein's sequence at all. The second possibility is testable (`cm12`).

A pull-down measures a protein that may be held in a multi-subunit assembly, so its level, and its response to a
perturbation, can be a property of the assembly rather than of the molecule. No model that reads one protein's
sequence can reach that.

**The test, in words.** Take a protein and ask by how much the measurement exceeded or fell short of what the
model predicted from its sequence: its *residual*. Separately, take the other subunits of the same Complex Portal
assembly and average *their* residuals. Then ask, across proteins, whether the two agree. If they do, proteins in
the same complex are being mispredicted together, which means the missing information is a property of the
assembly. This correlation is what the report calls the **complex effect**.

Two features of the design make it interpretable. The blocks used for prediction are separated at 30 % sequence
identity, and subunits of one complex are not sequence relatives, so complex-mates fall in different blocks and
are never fitted together -- a complex-level effect is unlearnable from sequence by construction, and therefore
lands entirely in the residual. And a protein's partners are taken **from its own block**, so both sides of the
correlation are held out of the same model.

The null is not zero, because residuals share abundance and other structure, so it is obtained by shuffling
residuals within a block, 500 draws, which preserves every complex's size and each protein's block while
destroying the pairing. It comes out at r ≈ 0.00.

![How much a protein's measured value, and its residual, agree with the mean of its complex-mates, against the permutation null (black dashes). Right: two worked examples, one point per protein.](../figures/campaign/complex_residual.png){width=100%}

| Target kind | n | Residual r | Null | R2/ceil. | + complex | Gain |
|-------------|---|------------|------|----------|-----------|------|
| baseline | 7 | 0.27 | 0.007 | 0.384 | 0.440 | +0.055 |
| abundance | 3 | 0.30 | 0.004 | 0.557 | 0.584 | +0.027 |
| enrichment | 4 | 0.27 | -0.001 | 0.250 | 0.299 | +0.049 |
| salt | 14 | 0.40 | 0.003 | 0.281 | 0.445 | +0.164 |
| temperature | 4 | 0.32 | 0.004 | 0.193 | 0.266 | +0.073 |
| Mg | 1 | 0.28 | -0.001 | 0.113 | 0.213 | +0.100 |
| **all 33** | **33** | **0.34** | **0.004** | **0.309** | **0.410** | **+0.101** |

Scored on the 2,858 proteins that are in a Complex Portal complex with at least two measured members; "+ complex"
adds one cross-fitted term, the complex-mate mean residual, with its coefficient fitted on the other outer folds.

* **It happens for every target**: r 0.22-0.58 against a null of ~0.00, z = 6.3-16.9. Concretely, if the other
  measured subunits of a protein's complex came out above what sequence predicted, that protein usually did too,
  and by a related amount. The effect is largest where the sequence model is weakest.
* **It is worth more than everything architecture gave**: mean R2 / ceiling rises 0.309 to 0.410. On the salt
  responses it roughly doubles R2 (GA_22 KCl 60: 0.19 to 0.37; GA_24 KCl 60: 0.17 to 0.35), and on heat at 43 C it
  rises from 0.22-0.24 to 0.31-0.35. For comparison, the entire MLP-over-ridge gain is +0.028.
* **It is not abundance** -- *this claim was too strong and is corrected in 3g.* Projecting out
  `abundance_here`, PaxDb abundance and length barely moves it (mean r 0.336 to 0.326), and a flexible,
  decile-and-interaction version of the same projection only reaches 0.304. But a projection removes a
  protein's own abundance while leaving the information its partners carry. Restricting the partner mean
  to complex-mates in a *different* abundance quintile, which is the test that actually bears on it, takes
  the mean to 0.214. **About a third of the effect is co-abundance.** Two thirds is not, and the rest of
  this section stands.
* **It is not shared peptides or paralogues.** The deployable variant, in which a protein's partner mean is taken
  only from **training-fold** proteins -- measurements already in hand, never in the test protein's own 30 %
  identity cluster -- performs identically (0.410 against 0.403). That version is also the honest one: it uses no
  held-out measurement.

Two limits. This is measured on the 25 % of proteins Complex Portal annotates with a measured partner; Complex
Portal is incomplete, so the coverage is a floor, not a statement about the other 75 %. And the term consumes
measured partner data, so it improves annotation of a *measured* proteome, not the sequence-only prediction of an
unmeasured one -- for a complex with no measured member it offers nothing.

**How far does it reach?** Complex Portal annotates only a quarter of the measured proteins, so the test was
repeated on the STRING v12 *physical* subnetwork at three confidence cutoffs, which buys coverage at the cost of
specificity (`cm16`, same training-fold-only design).

| Graph | Proteins | Median degree | Residual r | R2/ceil. | + partner | Gain |
|-----------------|------|----------|--------|------|-------|-----|
| Complex Portal | 2,858 | 5 | 0.346 | 0.300 | 0.401 | +0.101 |
| STRING physical >= 900 | 6,014 | 4 | 0.187 | 0.291 | 0.323 | +0.032 |
| STRING physical >= 700 | 7,341 | 6 | 0.145 | 0.286 | 0.306 | +0.019 |
| STRING physical >= 400 | 10,165 | 12 | 0.085 | 0.280 | 0.287 | +0.007 |

![The complex-mate correlation computed on four interaction graphs. Left: per target, ordered by coverage. Right: R2 / ceiling with the partner term against sequence alone; points above the diagonal are gains.](../figures/campaign/network_residual.png){width=100%}

The effect is **specific to curated stable complexes and decays fast as the graph widens**. STRING reaches 10,165
of 11,570 proteins, but at that coverage a protein's partner mean is an average over a dozen loose associations
and is worth +0.007. This tempers the claim above: complex context is a large lever for the quarter of the
proteome that sits in a defined assembly, and a small one for the rest. It is not a general route to the other
three quarters, and the honest summary of the campaign-wide number is that 0.41 applies to 2,858 proteins.

The conclusion for the campaign is that the ~0.27 plateau is the ceiling on *per-protein* sequence information,
not on predictability. Interaction context is a larger lever than any change to the network.

**What makes complex-mates behave alike, this section does not say**, and the four sections that follow were
written to find out.
Each takes one mechanistic reading of "complex membership" and tests it directly: the client's own folded surface
(3d), the surface a native partner buries (3e), and the chaperone's own predicted binding interface (3f). All
three return nothing; 3g then finds the answer outside structure altogether.

# 3d. Structure, tested directly: it adds nothing

A simple probe motivates this section. Of the sequence covariates, net charge per residue is the single strongest
predictor of the salt response, reproducibly across three independent experiments (Spearman rho 0.20-0.23 for
GA_20, GA_22 and GA_24). But the *capacity* to form ion pairs is not: 2 x min(D+E, K+R) gives rho -0.04 to -0.10,
and the total charged fraction -0.02 to -0.09. And every one of these is orthogonal to the sequence model's
residual (|rho| <= 0.07), so the network has already absorbed the compositional charge signal completely. For
salt its prediction is close to a pure charge axis: rho(prediction, measured) 0.36-0.44 against rho(prediction,
net charge) 0.38-0.41.

Whatever is left, if it is a protein property at all, therefore has to be spatial -- *where* the charge sits,
whether an ion pair is buried or solvent-facing, how large a contiguous charged patch is. None of that survives
averaging over residues. So the AlphaFold human proteome was folded in (`cm13`): 20,256 models parsed, none
failed, and the model sequence is byte-identical to the embedded sequence for 20,001 of them, so per-residue
quantities cannot be silently misaligned. Features per protein: relative SASA and burial; ion pairs split into
buried and surface; surface charge, buried charge, the most charged 10 A surface neighbourhood, dipole; the change
in intramolecular Debye-Huckel energy between the two ionic strengths **actually used in each experiment**
(kappa = 0.329 sqrt(I) /A, with MgCl2 contributing I = 3c); carboxylate clusters as the geometric signature of a
Mg^2+^ site; exposed and buried hydrophobic patches; pLDDT disorder and domain counts; crude secondary structure.

The test is not whether these predict the response -- they do, because they correlate with composition -- but
whether they predict the **residual** of the out-of-fold sequence prediction (`cm14`, cross-fitted ridge, alpha
chosen inside the training folds).

![Left: R2 / ceiling for each target with each block added to the sequence model. Right: Spearman of every structural feature with the residual, on a +-0.2 scale.](../figures/campaign/structure.png){width=100%}

| Added to the sequence model | Mean R2 / ceiling, 33 targets |
|------------------------------------|------------------------|
| nothing (the sequence model) | 0.273 |
| composition, 25 covariates (negative control) | 0.271 |
| **AlphaFold structure, 28 features** | **0.272** |
| named properties (abundance, Tm, keywords) | 0.320 |
| complex context | 0.304 |
| structure + named | 0.322 |
| everything | 0.348 |

* **Structure adds nothing.** The bootstrap interval straddles zero on almost every target, and structure over
  the named properties is likewise null. The composition control behaves identically, which is the evidence that
  the test can detect "already absorbed" rather than being insensitive.
* **Resolving ion pairs in 3D does not rescue the hypothesis.** Averaged over the 14 salt targets, every charge
  and ion-pair feature correlates with the measured response and with nothing in the residual:

| Feature | rho with measured | rho with residual |
|-------------------------------|-------|-------|
| surface charge | +0.167 | +0.027 |
| buried charge | +0.152 | +0.016 |
| largest positive surface patch | +0.075 | +0.010 |
| salt bridges per residue | -0.041 | +0.004 |
| salt bridges, buried only | -0.006 | +0.003 |
| salt bridges, surface only | -0.048 | +0.006 |
| Debye-Huckel dE at 150 mM | -0.080 | -0.035 |
| carboxylate clusters (Mg^2+^) | -0.042 | +0.012 |

The largest residual correlation of **any** structural feature with **any** target is 0.092.

**Pooling the embedding over structural masks does not help either.** Instead of averaging the ESMC states over
  the whole protein, they were pooled separately over surface, buried, ordered, disordered, charged-surface,
  hydrophobic-surface and buried-hydrophobic residues (`cm15`), giving the network the masks rather than making it
  learn them. On the 11 fold changes: sequence alone 0.2053, + structural features 0.2068, + mask pooling 0.2078,
  + both 0.2066. The internal check that the "all" mask reproduces the stored mean-pooled layer 80 gives r = 1.0000.

**The caveat that matters.** An AlphaFold model is itself predicted from sequence, and protein language models are
known to encode structure implicitly, so this is close to asking whether one sequence-derived view adds to
another. The result says that nothing these summary statistics extract from a predicted **single-chain, unbound,
unmodified** structure is missing from the embedding. It does not say 3D geometry is irrelevant to salt
sensitivity. The quantity the salt-bridge hypothesis is really about is the electrostatics of the chaperone-client
*interface*, which needs a structure of the complex, not of the client alone.

*Resolved in 3e and 3f.* That escape route was followed, twice: once with experimental complexes on 6,306
proteins (3e) and once with a predicted chaperone-client complex (3f). Both return nothing. The caveat stated
here is therefore no longer an open one, and the sentence above should be read as the question that the next two
sections go on to answer, not as an outstanding limitation of this one.

# 3e. The native partner, tested on experimental structures: it adds nothing either

Section 3d left one escape route open: an AlphaFold model is predicted from the same sequence the network
reads, so a null there is half-expected. A **complex** is not. It is a function of two sequences and their
pairing, and the pairing is outside the model's input entirely -- which is exactly why the complex-mate term in
3c worked. So the next question is whether the missing variance is an *interface* property.

The hypothesis is competition. HSPB1 binds exposed hydrophobic and aggregation-prone surface on non-native
clients. A protein that spends its life with that surface buried against a native partner should be a poor
client, and a perturbation that loosens the native interface should make it a better one. That predicts a
specific thing about the residual, and it is testable with structures that already exist -- no prediction, no
GPU, 17 minutes of CPU.

**What was measured** (`cm17`, `cm18`). Every PDB biological assembly containing at least one measured protein,
capped at 15 structures per protein, is 29,816 assemblies. For each measured chain the solvent-accessible
surface was computed twice -- alone, and in the presence of its partners -- and the difference resolved by
residue class: total buried area, the fraction of *exposed hydrophobic* surface a partner occludes, the same for
aggregation-prone regions, the composition and ion-pair density of the interface itself, and occlusion by
nucleic acid. That is 15 features, each aggregated over structures as median, max and min, for **6,306 of the
11,570 proteins** (1,816 of which have no interface in any structure -- the informative zero, deliberately kept).

Entries holding at least *one* measured protein, not two: requiring a measured partner would condition on
having a partner, which is close to the variable under test. Whether a partner chain was itself measured is
irrelevant to how much surface it buries.

**The guards.** Chains were aligned to their UniProt sequence (median identity 1.000, p05 0.987; 546 of 110,832
chains fell below 0.80 and were dropped). RCSB names symmetry copies `A-2` where SIFTS calls them `A`; matching
on the raw name would have silently discarded exactly the symmetry-generated partners that matter most, such as
a homodimer absent from the asymmetric unit. The 10,462 chains with no partner chain in their assembly come out
with a mean interface fraction of 0.00008, as they must.

**The positive control.** Before asking whether the features explain anything, they have to agree with things
already known. Complex Portal members bury 0.176 of their surface against 0.090 for non-members, and burial
correlates with Meltome T~m~ at rho +0.170 (n = 5,407). The measurement is real.

**The result is a null, and a flat one** (`cm19`, mean R² / ceiling over 33 targets):

| Block added to the sequence model | R² / ceiling |
|---------------------------------------------------|---------|
| sequence alone | 0.278 |
| + study-depth control (structure count, chain size, partner count) | 0.278 |
| + composition control | 0.278 |
| **+ native-partner interface** | **0.280** |
| + named properties (abundance, T~m~, keywords) | 0.326 |
| + complex-mate term (3c) | 0.319 |
| + everything | 0.353 |

By kind, the interface block adds +0.002 to baselines, +0.001 to enrichment, +0.007 to abundance, and
**−0.000 to both salt and temperature** -- the two response types the hypothesis was built for. Every
per-target bootstrap interval for the interface gain over the depth control includes zero.

**The three directional predictions, fixed before looking, all fail.**

| Prediction | Feature | Targets | rho with residual |
|---------------------|------------------|-----|---------------|
| D1 salt releases an ionically held interface | interface ion pairs per 1000 Å² | 14 salt | −0.009 to +0.025, every CI includes 0 |
| D2 heat unmasks occluded aggregation-prone surface | fraction of APR surface occluded | 4 temperature | −0.022 to +0.008 |
| D3 leftover hydrophobic surface is what HSPB1 binds | exposed hydrophobic area in the bound state | 7 baselines | −0.011 to +0.014 |

D3 is worth a second look because it fails in an interesting way: on the *measured* scale it correlates
**negatively** with the HSPB1 baselines (−0.050 to −0.081 for FS73, FS76, GA_20, GA_22, GA_24), the opposite of
the prediction. Proteins with more hydrophobic surface left exposed in their native complexes are pulled down
slightly *less*. That sign is consistent with those proteins being small, abundant and stable rather than with
any chaperone-competition story, and it vanishes entirely in the residual (+0.009), so the sequence model
already accounts for it.

The largest residual correlation of **any** of the 46 interface features with **any** of the 33 targets is
0.087 -- essentially the same ceiling as the 0.092 found for AlphaFold monomer features in 3d.

**One near-miss, and why it is not a result.** Interface features predict the complex-mate mean residual of 3c
at r = +0.080. But a control block built only from study depth and chain size reaches +0.066, so the interface
contributes +0.014. In an earlier version of this analysis that gap looked like +0.04; the difference was an
artefact of aggregating over however many structures a protein happens to have, since a maximum over 311
structures is mechanically larger than one over 3. Subsampling to a fixed cap before aggregating removed most
of the apparent effect. The honest reading is that the complex-level signal of 3c is **not** explained by how
much surface the partner buries.

**What this does and does not close.** It closes the version of the competition hypothesis that can be answered
with static, unbound-versus-bound surface burial in native complexes, measured on more than half the proteome.
It does not touch the chaperone side: none of these structures contains HSPB1, and the quantity the hypothesis
is ultimately about is the interface between the chaperone and a *non-native* client -- a state that is
essentially absent from the PDB, because it is transient, low-affinity and disordered. Testing that needs a
model of the bound complex, which is the next tier and a genuinely different experiment.

*That tier was run; it is 3f.* 4,787 Boltz-2 co-folds of the HSPB1 α-crystallin-domain dimer with tiled client
windows fail every pre-stated test. The escape route this section leaves open is therefore closed as far as a
co-folding model can close it -- with the qualification, developed in 3f, that half of the co-fold score is
run-to-run noise.

![Native-partner occlusion against the sequence model's residual. Left: R² / ceiling per target; the interface
bars sit on the sequence-only bars, while named properties and the complex-mate term rise above them. Right:
Spearman correlation of each of the 46 interface features with the residual, on the same colour scale as the
structure heat map in 3d.](../figures/campaign/interface.png)

# 3f. The chaperone's own interface, predicted: a null, and a measurement too noisy to rest on

3e closed the version of the competition hypothesis that experimental structures can answer, but it left
the question the hypothesis is really about untouched: the interface between HSPB1 and a *non-native*
client. No such structure exists in the PDB. This tier predicted it (`cm20`-`cm25`).

**Design A.** The HSPB1 alpha-crystallin domain (ACD, residues 84-171) as a dimer -- the client-binding
unit, and exactly the construct crystallised in 9ef3 and 3q9p -- co-folded with one short window of a
client, using Boltz-2 in single-sequence mode. A short window is the point: HSPB1 binds locally unfolded
segments, so a full-length co-fold would dock the chaperone onto a *folded* client surface, which is the
state 3e already showed is the wrong one. 200 proteins, 20 from each decile of the measured bound:free
partition, every sequence tiled into non-overlapping 16-mers, 4,787 co-folds in 3.98 h on one RTX PRO
6000 at 97.7 % mean utilisation and 13.0 GB of 96 GB. No job failed.

Windows were tiled **uniformly**, not chosen by hydrophobicity or aggregation propensity. Choosing them
would have confounded the structural score with the composition that selected them, so that the score
would "predict" binding for the same reason a hydrophobicity scale does -- which the sequence model
already contains.

**The readout** is the bound:free partition of GA_22 and GA_24, because the partition divides abundance
out experimentally rather than statistically: a protein scores high because it binds HSPB1, not because
there is a lot of it. The two runs are independent measurements of the same quantity and agree at
r = 0.914, which is the ceiling this tier could reach.

## The validation gate, passed before any GPU time was committed to the pilot

9ef3 is the HSPB1 ACD with its own C-terminal IxI motif peptide bound in the beta4/beta8 groove. It was
**released 2026-06-24, after Boltz-2 was trained**, so it is a genuine held-out test rather than a recall
exercise. The groove was not asserted from the literature; it was measured as the 20 residues of 9ef3
chain A within 4.5 A of the peptide.

| Check | Result | |
|---|---|---|
| ACD fold, holo | monomer core Ca RMSD 1.54-2.44 A | pass |
| ACD fold, apo | 3.36 A | marginal |
| Held-out peptide placement | **1.27 A**, centroid off by 0.77 A | pass |
| Groove overlap | **all 14 contacts among the 20 experimental groove residues** | pass |
| Contrast | held-out positive ipTM 0.899 vs best homopolymer negative 0.746 | pass |

One real defect: the peptide is threaded **backwards**. Forward-order RMSD is 16.49 A, reverse-order
1.27 A, and the per-residue deviations (25.5, 19.7, 13.3, 7.5, 1.2, 5.8, 12.7, 18.7, 25.0 A) are symmetric
about the centre -- the signature of an extended peptide lying in the right channel the wrong way round.
Site, position and conformation are right; polarity is wrong. Neither quantity the pilot uses depends on
direction, so the gate stands, but a model that cannot orient a peptide it has placed correctly is not a
model to build a mechanism on.

A second symmetry trap had to be cleared first, the same one as in 3e: the ACD dimer has a two-fold axis,
so its two grooves are equivalent by construction, and scoring against the wrong one reports the ~40 A
distance between them as if it were an error.

## Four predictions, fixed before the run finished; all four fail

| | Prediction | Result |
|---|---|---|
| P1 | interface score correlates positively with measured partition | **fails.** Every aggregation of every score lies between -0.17 and +0.02. The most reliable score reaches rho = -0.127 [-0.267, +0.021] -- the wrong sign |
| P2 | real windows beat their own composition-matched shuffles | **fails.** Real 0.612, shuffled 0.638, difference **-0.026** [-0.059, +0.006]; Wilcoxon p = 0.955 in the predicted direction |
| P3 | structure adds over window composition | **fails.** Cross-validated rho: composition -0.059, structure -0.093, both -0.035 |
| P4 | structure adds over the ESMC-6B + SAE sequence model | **fails.** Sequence model alone rho 0.335; adding structure *lowers* it to 0.232. Largest correlation of any structural score with the sequence model's residual is 0.134, wrong-signed, CI including zero |

P2 is the one that matters most. If the co-fold cannot tell a real 16-mer from its own scramble, it is not
reading sequence order, and a score that does not read sequence order cannot rank clients. For context, the
largest residual correlation here (0.134) sits alongside 0.092 for AlphaFold monomer features (3d) and
0.087 for native-partner interfaces (3e) -- three independent structural approaches landing in the same
narrow band just above nothing.

## The controls say what the score is actually measuring

| Control | ipTM | Percentile among 4,577 real client windows |
|---|---|---|
| 9ef3 IxI 9-mer (held out) | 0.899 | 97.9 |
| IxI 16-mer (held out) | 0.790 | 77.9 |
| **poly-G 16-mer (negative)** | **0.746** | **69.2** |
| 6gjh N-terminal 5-mer (in training) | 0.650 | 51.1 |
| poly-S (negative) | 0.614 | 44.7 |
| 6gjh N-terminal 16-mer (in training) | 0.549 | 34.4 |
| poly-EK (negative) | 0.497 | 27.5 |
| poly-K (negative) | 0.299 | 5.4 |
| poly-E (negative) | 0.227 | 1.3 |

The evolved IxI motif does score at the top. But poly-glycine -- a sequence with no side chains at all --
outscores both in-training positives and 69 % of real client windows, while the negatives that do sit at
the floor are the charged ones. On this evidence ipTM mostly separates "charged homopolymer" from
"everything else", which is a far weaker axis than the gate alone suggested.

## How much of the score is noise

Boltz-2 defaults to no seeding, so the pilot was one unseeded draw. Re-running 548 of the same windows is a
second independent draw from the same model on the same input (`cm25`).

A window's ipTM moves by **0.140 on average between two runs of identical input**, against a spread across
different windows of 0.185. Test-retest r = **0.497**: half the variance in the pilot's window scores is
run-to-run noise.

That matters less than it sounds for a per-protein score built from ~22 windows, and more than it sounds
for the aggregations that take extremes. Measured by split-half over each protein's own windows:

| Per-protein score | Reliability | Ceiling on \|rho\| | Observed \|rho\| |
|---|---|---|---|
| groove_recall_top3 | 0.461 | 0.679 | 0.127 |
| iptm_top3 | 0.359 | 0.599 | 0.047 |
| groove_recall_mean | 0.300 | 0.548 | 0.120 |
| iptm_fhi | 0.170 | 0.413 | 0.108 |
| iptm_max | 0.169 | 0.411 | 0.022 |
| groove_recall_max | 0.134 | 0.366 | 0.054 |
| iptm_mean | 0.115 | 0.340 | 0.135 |
| groove_recall_fhi | 0.103 | 0.321 | 0.168 |

Three of the eight have ceilings of 0.55-0.68 and observe 0.05-0.13, so for those the null is real: the
instrument had room to show an effect three to five times larger than it did. The other five are too noisy
to interpret in either direction, and that includes `max`, which is the most biologically natural
aggregation -- "the best binding site on the protein" -- because a maximum over noisy values is mostly a
maximum over noise.

**A selection trap worth recording.** The score with the largest correlation, `groove_recall_fhi` at
-0.168, is the single *least* reliable of the eight. Choosing a headline by effect size would have picked
the noisiest feature; the figure selects on reliability instead, which does not depend on the outcome.

## What this closes

The honest summary is a null with a caveat, not a clean null. Where the measurement is reliable enough to
interpret, a predicted HSPB1-ACD / client-window interface carries no information about which proteins
HSPB1 actually binds -- it does not beat composition, does not beat a scramble, and subtracts from the
sequence model. Where the measurement is not reliable enough, the tier is uninformative rather than
negative.

Scaling Design A from 200 to 2,000 proteins would not fix this. The limit is not sample size; the limit is
that the per-window score is half noise and does not respond to sequence order, and ten times more proteins
leaves both properties unchanged. The failure of P2 in particular is a property of the model, not of the
sample.

Two things could change the answer, and neither is a bigger version of this experiment. First, the ACD
dimer alone omits the N-terminal region, which carries much of sHSP client binding and is disordered --
the part a co-folding predictor is worst at, and the part no crystal structure constrains. Second, a
co-folding model trained on native crystallised interfaces is being asked about a transient,
low-affinity, non-native complex; the reversed threading of a peptide it otherwise placed to 1.27 A is a
small visible symptom of that mismatch. Both point away from more structure prediction, and the complex
effect of 3c remains real, remains unexplained by the client's own surface (3d), by native-partner
occlusion (3e), or by a predicted chaperone interface (3f). Section 3g finds the explanation elsewhere, and
in doing so explains this null: most of the complex effect is not an interface property and never was.

An earlier draft of this paragraph ended by proposing that the next step was to separate abundance from
binding strength in the measurements. That step is already taken and is reported above, not ahead: GA_22
and GA_24 supply a bound:free partition against their own matched supernatants, GA_33 supplies enrichment
over a reference input, and section 4 reads the result off those abundance-free targets directly. The
separation did not dissolve the complex effect; the enrichment targets carry it as strongly as the raw
pull-downs do (residual r 0.27, section 3c). With that correction, the remaining question was what the complex
effect *is*, which 3g answers.

![Design A pilot. A: the validation gate against 9ef3, released after Boltz-2 was trained. B: where every
control falls among 4,577 real client windows. C, D: the most reliable per-protein score against the
measured partition, as a scatter and by decile. E: real windows against their own composition-matched
shuffles. F: cross-validated prediction from each feature block. G: the same window re-run, showing half
the score is run-to-run noise. H: observed correlation against the ceiling each score's reliability
allows.](../figures/campaign/pilot.png)


# 3g. What the complex effect actually is: mostly the lysate, partly co-abundance, not a capture nonlinearity

3c established that what the sequence model misses is shared among the members of a complex, and left the
cause open. Sections 3d-3f tested three structural readings of that and found nothing. This section tests
three non-structural readings, and it is the one that identifies the effect.

The question came from a specific objection, and it is worth stating carefully because it would, if true,
dissolve the whole of 3c.

Capture in these pull-downs is not proportional. A protein ten times more abundant in the lysate comes out
only about 2.2 times more abundant in the pull-down: the level grows as input^0.35 (`ad15`), so the
fraction of a protein that gets captured *falls* as there is more of it. Where a protein sits on that curve
is set by how abundant it is. Subunits of a complex are present at similar amounts, so they sit at similar
points on the curve, so they would be distorted by capture in similar ways -- and their residuals would
correlate for that reason alone, with nothing about chaperones in it. 3c's defence against this was a
**linear** projection of log abundance, which is not obviously adequate against a nonlinear mechanism.

`cm26` tests four candidate explanations: that the capture curve manufactures it; that it is shared
abundance; that it is how complexes behave in the lysate; and that complexes are pulled down as units.

## The measurement stage settles the nonlinearity question

The decisive comparison was already in the data and had not been read.

In GA_22 and GA_24, every protein is quantified three times from the same reaction. The **supernatant** is
what was left unbound; it is essentially the lysate, and it never passes over the beads, so no property of
capture can touch it. The **pull-down** is what bound, and does pass through capture. The **partition** is
their ratio, in which how much protein was present is divided out experimentally, leaving something closer
to how tightly it bound.

Running the complex-mate correlation separately on each of the three asks *at which point in the protocol*
the subunits of a complex begin to behave alike. If the capture curve were responsible, the correlation
would be small in the supernatant and appear in the pull-down. It is the other way round, on an identical
set of proteins:

| Run, perturbation | n | supernatant | pull-down | partition | sup − ip | ip − part |
|-------------|-----|------|------|------|---------------|---------------|
| GA_22, KCl 60 | 1,401 | **0.607** | 0.401 | 0.305 | +0.136, +0.282 | +0.052, +0.142 |
| GA_22, KCl 120 | 1,392 | **0.584** | 0.375 | 0.292 | +0.124, +0.276 | +0.045, +0.125 |
| GA_24, KCl 60 | 1,473 | **0.554** | 0.398 | 0.269 | +0.088, +0.233 | +0.087, +0.172 |
| GA_24, KCl 120 | 1,442 | **0.580** | 0.381 | 0.264 | +0.140, +0.264 | +0.077, +0.159 |
| GA_22, baseline | 1,426 | 0.340 | 0.310 | 0.257 | −0.037, +0.087 | −0.008, +0.112 |
| GA_24, baseline | 1,485 | 0.352 | 0.283 | 0.249 | +0.007, +0.130 | −0.032, +0.105 |

The last two columns are paired cluster-bootstrap intervals on the difference, both correlations computed
on the same resampled proteins; per-stage intervals are the error bars in panel A of the figure. For the
salt **response**, every one of the eight difference intervals excludes zero. For the baseline **level**,
none of them does.

**Reliability runs against this result, not for it.** The supernatant salt responses are the least reliable
targets in the campaign (ceilings 0.757-0.913) and the pull-down salt responses among the most reliable
(0.896-0.971), so attenuation shrinks the supernatant's correlation harder. Dividing each by its own
target's reliability makes the gradient steeper, not shallower:

| Run, perturbation | supernatant | pull-down | partition |
|---------------|--------|--------|--------|
| GA_22, KCl 60 | 0.801 | 0.448 | 0.350 |
| GA_22, KCl 120 | 0.689 | 0.393 | 0.316 |
| GA_24, KCl 60 | 0.659 | 0.439 | 0.295 |
| GA_24, KCl 120 | 0.636 | 0.392 | 0.274 |
| GA_22/GA_24, baseline | 0.342 / 0.354 | 0.313 / 0.285 | 0.261 / 0.252 |

The unbound pool shows roughly **twice** as much complex-mate agreement as the pull-down made from it.
Passing a sample over the beads *reduces* the tendency of complex members to move together; it does not
create it. Whatever makes subunits of a complex behave alike is already present in the lysate.

## A nonlinear abundance control removes almost nothing

The linear block of 3c was replaced by decile indicators of each abundance source and their full
interaction, 155 columns, which absorbs any monotone or non-monotone function of abundance, including a
saturating capture curve and an abundance x response interaction:

| Abundance control | Mean r over 33 targets |
|-------------------------------------------|------|
| none | 0.336 |
| linear, as in 3c | 0.326 |
| flexible: deciles of each source + interactions | **0.304** |

This is conservative in the right direction: projecting 155 columns out of ~1,500 proteins removes some
variance by chance, which biases the result *downward*, and it still moves 0.336 only to 0.304.

## But abundance does mediate about a third of it, and 3c was wrong to say otherwise

Projection is the weak test, because it removes a protein's own abundance but leaves the information its
partners carry. The strong test asks whether the effect needs abundance-**matched** partners: recompute
each protein's partner mean using only complex-mates in a *different* abundance quintile.

| Partners used | Mean r over 33 targets |
|-------------------------------|------|
| all | 0.336 |
| same abundance quintile | 0.408 |
| **different abundance quintile** | **0.214** |

The method's positive control fires exactly as it must. `GA33_input` *is* the lysate abundance, so its
complex-mate correlation has to be pure co-abundance, and when partners are drawn from a different
abundance quintile it falls from 0.239 to **−0.029**. A test that drives a known-abundance target to zero while leaving the others at 0.21 is
discriminating, not insensitive.

That control is weaker in one specific way than it first appears, and the same caveat touches two other
targets. The abundance covariate here is the measured `GA33_input` level. Three of the 33 targets share a
measurement with it: `GA33_input` itself, and the two enrichments `GA33_A1_enrich` and `GA33_B11_enrich`,
which are defined as pull-down minus input. For those three, residual and covariate carry the same
measurement noise, so the discordant-partner test is pulled down by shared noise as well as by genuine
co-abundance, and it cannot separate the two. Dropping all three leaves 30 targets whose abundance
covariate comes from an independent run, and the table above becomes 0.342 / 0.404 / **0.225** — the same
conclusion, and the fraction surviving is 66 % rather than 64 %. The positive control still shows the test
is not insensitive; it does not, on its own, show the collapse is co-abundance rather than shared noise.
The 30-target recomputation is what carries that.

So the claim in 3c that the effect "is not abundance" rested on a control too weak to support it.
**About a third of the complex effect is co-abundance** that the linear projection did not remove. The
correction does not overturn 3c -- two thirds survives partners deliberately drawn from a different
abundance stratum -- but the original statement was too strong and is withdrawn.

What survives is also the part that matters. The partition targets, in which abundance is divided out
experimentally, are almost untouched by the discordance test: `GA22_part_base` 0.257 → 0.226 and
`GA24_part_base` 0.249 → 0.200. That residue is capture-specific and abundance-independent.

## Co-capture is the remaining candidate, and the evidence for it is weak

Stoichiometric co-capture is a nonlinearity too, but of a different kind: an OR over subunits rather than
a saturating function of concentration. Pull one subunit and the assembly follows, so a complex's capture
is set by its best-binding member. That predicts an asymmetry additive sharing does not -- a weak binder in
a complex with a strong binder is carried up, while a strong binder is not dragged down as much -- so the
partner effect should be larger for proteins that are themselves poor binders. Each abundance-free target
was stratified on the protein's binding measured in an **independent run**, so the stratifier never enters
the residual under test:

| Target | Stratifier | Weakest third | Middle | Strongest third | low − high |
|-------------------|---------------------|------|------|------|-------------|
| GA33_A1_enrich | measured GA33_B11_enrich | 0.207 | 0.277 | 0.092 | **+0.009, +0.229** |
| GA33_B11_enrich | measured GA33_A1_enrich | 0.109 | 0.163 | 0.135 | −0.148, +0.101 |
| GA22_part_base | measured GA24_part_base | 0.189 | 0.157 | 0.112 | −0.052, +0.214 |
| GA24_part_base | measured GA22_part_base | 0.177 | 0.139 | 0.093 | −0.050, +0.217 |

Three of the four run in the predicted direction, DNAJB11 runs against it, and one interval excludes zero.
The means are 0.171 for the weakest third against 0.108 for the strongest. That is a hint, not a result.

The max-versus-mean comparison adds nothing, as was stated before running it. The maximum over partners
does consistently *worse* than the mean (0.276 against 0.354 averaged over targets, every interval on the
difference negative). The mean is a lower-variance estimator of any shared factor, so it should win under
additive sharing and under a noisy OR alike; only "max wins" would have discriminated, and it did not
happen.

## What this section settles

| Explanation of the complex effect | Verdict |
|------------------------------|-------------------------------------------|
| **Nonlinear capture** (complex-mates sit at similar points on the input^0.35 curve) | **Rejected.** The correlation is highest in the one measurement that never passes over the beads, and a flexible abundance control moves the mean only 0.336 → 0.304 |
| **Co-abundance** (complex-mates are simply present at similar amounts) | **Partly true, about a third.** 3c's linear control was too weak; the correlation falls 0.336 → 0.214 when partners are drawn from a different abundance quintile |
| **Co-behaviour in the lysate** (members go into and out of solution together, before any bait) | **The largest single part.** In the unbound supernatant the complex-mate correlation reaches 0.55-0.61, roughly twice the pull-down's once reliability is accounted for |
| **Stoichiometric co-capture** (pull one subunit, the assembly follows) | **Suggestive, unproven.** The predicted asymmetry appears in 3 of 4 targets, with 1 interval excluding zero |

The complex effect is not one thing. A third is co-abundance that 3c failed to remove. Most of the rest,
for the salt responses at least, is how the complex behaves in the lysate before any bait is added --
members' free concentrations move together with salt, and the pull-down inherits a damped copy of that.
Only the part that survives dividing the lysate out, around 0.20-0.23 in the partition targets, is a
candidate for anything about chaperone binding at all.

That also explains why 3d, 3e and 3f found nothing. All three were looking for an interface property, and
an interface property could only ever have accounted for the smallest of these four components. The right
reading of the campaign is not that the complex effect is mysterious, but that most of it was never about
the chaperone.

![What the complex effect is made of. **A**: the complex-mate correlation at three stages of the same
reaction -- the supernatant never passes through capture, the pull-down does, the
partition divides the lysate out; black dashes are the same values divided by each target's own
reliability, which steepens the gradient because the supernatant targets are the noisiest.
**B**: replacing 3c's linear abundance control with decile indicators and their interactions.
**C**: the same statistic using only complex-mates from a different abundance quintile; the red diamond is
`GA33_input`, which is lysate abundance itself and must collapse, and
does. **D**: the co-capture asymmetry, by the protein's own binding measured in an independent
run.](../figures/campaign/nonlinearity.png){width=100%}

# 4. What predicts the pull-down with no perturbation

A pull-down level mixes how much protein is in the tube (abundance) with how strongly it binds. Both
parts were tested with named, readable properties, alone and with the sequence model (`cm07`).

![A: Spearman rho of each named property with each baseline. B: out-of-fold R² / ceiling as property blocks are added, the sequence MLP alone, and the two combined.](../figures/campaign/baseline.png){width=100%}

| Predictor set | HSPB1 pull-down | DNAJA1/B11 pull-down | Binding over abundance |
|-----------------------|----------|-------------|--------------|
| Abundance (here + PaxDb) | 0.23-0.32 | 0.15-0.16 | 0.19-0.40 |
| + Tm, sequence physics, location | 0.28-0.36 | 0.26 | 0.25-0.49 |
| All named properties | 0.29-0.37 | 0.27-0.28 | 0.26-0.50 |
| Named properties without abundance | 0.09-0.17 | 0.15-0.16 | 0.16-0.25 |
| Sequence MLP alone | 0.21-0.32 | 0.27-0.28 | 0.26-0.40 |
| **Sequence MLP + named properties** | **0.31-0.42** | **0.34-0.35** | **0.30-0.54** |

Values are out-of-fold R². The ceilings here are 0.98-0.998, so R² / ceiling differs by under 0.01.

What the properties show:

- **Abundance dominates the raw pull-down.** It correlates with every HSPB1 baseline (rho 0.46-0.57).
- **It reverses for enrichment.** Abundant proteins are less enriched over input (rho −0.57 to −0.60
  in GA_33). Capture rises less than proportionally with amount, as found before (slope about 0.35).
- **Abundance-free binding** rises with transmembrane segments and with positive charge (DNAJA1:
  rho +0.30 with pI), and falls with stability (Tm, rho −0.21 to −0.27).
- **Ribonucleoprotein and complex membership** raise the raw HSPB1 pull-down slightly.

Sequence predicts the abundance-free binding about as well as all the named properties together, and
the two combined are best. So sequence carries most of what the named properties do, and somewhat more.

# 5. One representation across conditions

## 5.1 Leave-one-run-out transfer

Six trunks were each trained without one run's targets. That run's targets were then read out from the
frozen latent by ridge, within each fold, since fold models' latents are not aligned with each other.

![For each target: read-out from the latent of the model trained with the run (red), from a trunk that never saw the run (blue), and, for scale, the prediction from the other runs' MEASURED targets with no sequence (orange).](../figures/campaign/transfer.png){width=100%}

| Run left out | Its targets, read from a trunk that never saw it (with the run) |
|-----------------------|-------------------------------------|
| GA_20 (HSPB1, salt) | baseline 0.20 (0.19); salt 75 / 150: 0.16 / 0.18 (0.16 / 0.18) |
| GA_22, GA_24 (HSPB1, salt; with supernatant and partition) | within 0.00-0.03 of trained-with for all 18 targets |
| FS73 (HSPB1) | baseline 0.28 (0.30) |
| FS76 (HSPB1, Mg) | baseline 0.28 (0.29); **Mg 0.10 (0.15)** |
| GA_33 (DNAJA1/B11, heat) | abundance 0.47 (0.49); enrichment 0.32-0.34 (0.36-0.38); **heat 43 °C 0.18-0.19 (0.25-0.26)** |

One shared representation covers every HSPB1 baseline and salt condition: no run teaches it something
the others do not. What it loses when a run is left out is a condition type found nowhere else. Heat
exists only in GA_33 and Mg only in FS76. The two DNAJ baits also lose a little on their baselines.

The measured cross-run prediction (orange) shows how much the conditions share as data:

- salt responses are 70-93% predictable from the other salt experiments' measurements;
- heat is only 9-34% predictable from anything else measured.

Heat is the condition least like the others.

## 5.2 Which conditions the model represents along the same axes

![Cosine between targets' read-out directions in the shared latent, clustered.](../figures/campaign/geometry.png){width=85%}

The latent organises the conditions into a few axes:

- **Salt, pull-down and partition:** one axis shared by all three salt experiments (GA_20, GA_22,
  GA_24).
- **Salt, supernatant:** a separate axis. Salt changes the free pool along different directions from
  binding, as the GA_22/24 report found from the data.
- **Abundance and baselines:** an abundance axis carrying the supernatant and input levels, with the
  baselines close to it.
- **Enrichment and partition:** pointing opposite to abundance.
- **Heat:** an axis of its own.
- **Mg:** aligned with nothing.

## 5.3 Regressing the baseline out

Two versions, which agree.

**(a) In the model.** Within each fold, the subspace of the latent that predicts abundance, baselines
and enrichment (rank 14) was projected out, and each effect read again. Effects barely change:

- GA_20 salt: 0.16 → 0.15;
- heat 43 °C: 0.25 → 0.24;
- Mg: 0.15 → 0.13.

The effect signal sits in directions separate from the baseline signal.

![Effect targets read from the full latent, from the latent minus its first baseline axis, and minus the whole baseline / abundance subspace.](../figures/campaign/regress_out.png){width=100%}

**(b) In the data.** Each effect was residualised on its own run's measured baseline and abundance,
fitted on training proteins only, and a fresh MLP was trained on the residuals.

Effect and baseline in one run share their control samples, and therefore their measurement noise. So
I repeated it with baselines from an independent experiment: GA_24's for GA_22, FS73's for GA_20 and
FS76, the other bait's for GA_33.

| Effect | Variance left after removing baseline (own run / independent run) | Sequence R², raw → residual (own / independent) |
|---------------|--------------------------|-------------------|
| Heat 43 °C, DNAJA1 | 0.52 / 0.68 | 0.27 → 0.13 / 0.20 |
| Heat 43 °C, DNAJB11 | 0.52 / 0.52 | 0.26 → 0.13 / 0.10 |
| Heat 37 °C | 0.86-0.88 / 0.85-0.97 | 0.11-0.13 → 0.07-0.11 |
| Salt, pull-down (GA_20, GA_22, GA_24) | 0.83-0.94 / 0.90-0.94 | essentially unchanged |
| Salt, supernatant | 0.83-0.99 / 0.84-0.99 | unchanged or higher (0.25 → 0.28-0.31) |
| Mg | 0.92 / 0.97 | 0.17 → 0.16 / 0.15 |

The heat response at 43 °C is about half a starting-level effect. Proteins' 43 °C change depends on
their 35 °C level, even when that level comes from another experiment. About half of what sequence
predicts about heat goes through that level.

Salt and Mg responses are condition-specific. Almost none of their variance is explained by the
baseline, and what sequence predicts about them is untouched by removing it.

# 6. What the SAE features say

Gradient attribution through an SAE-input model was unstable from fold to fold (r 0.19-0.29). The
16,384 features are highly redundant, and different fold models credit different members of correlated
groups. So the features are ranked by a stable statistic instead: Spearman rho with the family-averaged
target. Its agreement between two halves of the proteins split by sequence cluster:

| Family | Halves r | Top-50 overlap |
|---|---|---|
| Abundance | 0.93 | 0.82 |
| Pull-down baseline | 0.84 | 0.78 |
| Bound:free and enrichment | 0.90 | 0.56 |
| Salt | 0.73 | 0.72 |
| Mg | 0.79 | 0.68 |
| Temperature | 0.87 | 0.24 |

Labels are Biohub's LLM-generated annotations, which are hypotheses. Each is checked against the
UniProt keywords of the proteins on which the feature is active.

![The 8 features most positively and most negatively associated with each family, with their Biohub labels.](../figures/campaign/sae_features.png){width=88%}

| Family | Higher with (label; keyword check) | Lower with |
|------|----------------------------------|---------------------|
| Pull-down baseline, abundance | low-complexity IDRs typical of eukaryotic enzymes; NTPase catalytic residues (nucleotide-binding); basic N-terminal tails and ribosomal RNA-binding motifs (ribosomal proteins) | Ser/Thr-rich, S/T-Pro-rich and SLiM disordered regions (59-78% phosphoproteins); secreted disordered loops |
| Bound:free / enrichment | transmembrane helices, juxtamembrane loops (62-74% membrane proteins); long low-complexity IDRs | enzyme ligand-pocket β-strands and active-site loops (hydrolases); basic N-terminal mitochondrial segments |
| Salt | hydrophobic and juxtamembrane transmembrane helices (72-82% membrane) | phosphoregulated trafficking / endocytic IDR scaffolds |
| Temperature (43 °C) | α/β enzyme cores, TIM barrels, active-site helix caps (nucleotide-binding, hydrolases) | membrane, secreted, signal-peptide and disordered features |
| Mg | transmembrane-helix features (61-75% membrane) | acidic tails, charged coiled coils, nuclear interfaces |

**Two cautions.**

- The membrane axis appears in salt, Mg and enrichment alike. It may describe how membrane proteins
  behave in a detergent-free lysate (aggregation, insolubility) as much as chaperone binding.
- The heat features (structured α/β enzyme cores) fit thermal unfolding, but the temperature ranking's
  top features are the least stable of any family (top-50 overlap 0.24). Read them as a class, not as
  individual features.

# 7. A proteome-wide prediction table

The five fold models were applied to the 8,836 reviewed human proteins that no run measured; each fold
model was trained on 80% of the measured proteins. This adds predictions for every target to the
out-of-fold predictions for the 10,777 measured proteins: 19,613 proteins in
`data/campaign/proteome_predictions.tsv`.

The expected accuracy is each target's out-of-fold R² (0.10-0.49). The table is a prior for picking
candidates, not a substitute for measurement.

![Predicted distributions for measured and never-measured proteins, for six targets.](../figures/campaign/proteome.png){width=100%}

# 8. Critical evaluation

1. **One protein-level representation.** ESMC-6B mean-pooled states carry about a quarter of the
   reliable variance, and nothing tried here moved that -- not 94 architectures, not attention over
   7.4 M residue states, not sharing one trunk across targets, and not any of the three structural tiers
   (3d, 3e, 3f).
   The rest is either not in the sequence (lysate
   state, complex membership, post-translational state, local concentration) or not in this
   representation. The measured covariates show the first is substantial for baselines.
2. **Abundance as input or target.** For baselines, measured abundance is the strongest predictor, and
   sequence partly predicts it. The fair question is what sequence adds beyond abundance. That is small
   for baselines (+0.012, interval includes 0) and real for salt and Mg.
3. **Shared measurement noise.** In-run residualisation shares control samples between effect and
   baseline. The independent-baseline version removes that and gives the same answer.
4. **Not every run measures abundance.** HSPB1's FS73 / FS76 and GA_20 have no supernatant or input,
   so abundance comes from GA_33's lysate and PaxDb. Different lysate preparations add noise.
5. **The SAE is not an accurate input.** It is the most interpretable one, but on its own it predicts
   less (0.18) than the dense layers. Its feature associations describe the measurements, not the
   model's predictions.
6. **Bugs and analysis errors found and fixed during the campaign:**
   - a shared row-index file overwritten by later specs, which zeroed the first latent read-out;
   - a drift-covariate aliasing in the linear model on sparse proteins (from the GA_22/24 work);
   - variable shadowing in the per-residue writer, caught by its own completeness assertion;
   - in 3e, aggregating interface features over however many structures a protein happens to have:
     a maximum over 311 structures is mechanically larger than one over 3, which inflated an apparent
     interface effect on the complex-mate residual from +0.014 to +0.04 until a fixed subsampling cap
     was imposed;
   - in 3e, matching chains on their raw RCSB name, which calls symmetry copies `A-2` where SIFTS calls
     them `A`. That would have silently dropped exactly the symmetry-generated partners that matter
     most, such as a homodimer absent from the asymmetric unit;
   - in 3f, a first estimate of aggregate reliability that simulated a protein's windows as i.i.d.
     draws. That builds in the absence of a protein-level component, so it measures reliability *under
     the null* rather than a ceiling. Replaced by an empirical split-half over each protein's own
     windows with a Spearman-Brown correction;
   - in 3g, an abundance covariate that can be a copy of the target. Three of the 33 targets share a
     measurement with `GA33_input`, so their abundance control strips shared measurement noise rather than
     abundance. Recomputing on the 30 independent targets moves the discordant-partner mean 0.214 to 0.225
     and changes nothing, but the defect is real and it bit harder in the reusable pipeline, where a study
     with no pre-IP lysate falls back to the supernatant and then six of nine targets are self-referential.
     That is guarded now: such targets are rerun against PaxDb, or excluded and named;
   - in 3c, concluding "it is not abundance" from a linear projection of log abundance. A projection
     removes a protein's own abundance but leaves what its partners carry, so it is the wrong test; the
     right one restricts partners to a different abundance quintile, and it shows about a third of the
     effect is co-abundance after all (3g). No internal check caught this -- it surfaced only when someone
     asked whether a nonlinear capture curve could do the same job;
   - in 3f, a figure that chose its headline score by the largest |rho|. Selecting on effect size
     reliably finds the noisiest feature: it picked `groove_recall_fhi`, the least reliable of the eight
     (ceiling 0.321). Selection now runs on split-half reliability, which does not depend on the outcome.

   Each was caught by a check before any number was reported. The last two are the instructive ones,
   because neither would have crashed, and both would have produced a more interesting-looking result.

7. **The discipline this campaign added: measure the probe, not only the sample.** Three of the four
   structural tests are nulls, and a null is only as strong as the resolution of the instrument that
   produced it. In 3f that resolution turned out to be poor -- a window's ipTM moves 0.140 between two
   runs of identical input, against a between-window spread of 0.185, so half the score is run-to-run
   noise and five of the eight per-protein aggregations are too noisy to interpret at all. The nulls in
   3d and 3e do not carry that weakness: their features are deterministic functions of a fixed structure,
   and their positive controls fire (burial separates Complex Portal members at 0.176 vs 0.090 and
   tracks Meltome T~m~ at rho +0.170). Stating a null without its ceiling would have made all three look
   equally conclusive, and they are not.

8. **Three nulls in a row invite a fair objection**, which is that the wrong thing was tested each time
   rather than that there is nothing there. The defence is specific rather than rhetorical: each tier
   carried a positive control that fired (3d's composition control behaves as "already absorbed" rather
   than as insensitive; 3e's burial control reproduces known complex biology; 3f's gate placed a
   genuinely held-out peptide, from a structure released after the model was trained, to 1.27 Å with all
   14 contacts in the experimental groove). An instrument that demonstrably works and still reports
   nothing is reporting a null, not a failure. Section 3g supplies the constructive half of the answer:
   the three tiers were looking for an interface property, and an interface property could only ever have
   explained the smallest of the four components the complex effect is made of.

# 9. What is left

This campaign has run out of analysis that its own data support. Sequence predicts about a quarter of what
these measurements reliably contain; most of the remainder is a property of the complex rather than of the
protein; three structural attempts
to say *what* the complex contributes returned nothing, and 3g then showed why -- most of that effect is a
property of the lysate rather than of the chaperone interface. The one piece that could still be chaperone
biology, the ~0.20-0.23 surviving in the partition targets, has a leading candidate (stoichiometric
co-capture) that the present data support only weakly.

**Blocked on information from the lab, not on analysis.** Five questions, carried from the GA_22/24 report
and `additional_text_9`, each of which currently forces a caveat into a stated result:

| | Question | What it unblocks |
|---|---------------------------|-------------------------------|
| 1 | Are the NADP 0.6 / 15 and NAD 11 / 300 values in µM? | The ligand dose-response curves are drawn without a unit on the x-axis |
| 2 | Was **equal volume or equal protein amount** injected? | Whether the global fall in pull-down with salt is real, and whether the +1 log2 load at 43 °C is more captured protein or only a change in composition. This one affects two reports |
| 3 | Was anything different about GA_22 BR5 at KCl 60 and GA_24 BR5 at KCl 120? | The two failed supernatant blocks are currently dropped without a reason |
| 4 | Were the NAD 300 / KCl 120 reactions processed separately? | If not, the ribonucleoprotein effect needs a repeat before it is reported |
| 5 | Can future runs randomise injection order across arms? | The cofactor arms were injected in blocks, so MS drift had to be *modelled* rather than designed out |

**Needs a new experiment, not a new analysis.** Two are specified in full elsewhere and both follow from
results this campaign considers solid:

- **A drug dose series with an inactive analogue** (for example methotrexate against DHFR, or a selective
  kinase inhibitor against its kinase). This tests whether the HSPB1 pull-down works as a target-engagement
  assay. It follows from the strongest positive result in the whole programme: each ligand removes its own
  binders from HSPB1 in a dose-graded way, at 46-54x for NADP enzymes, 26x for NAD enzymes and 13x for
  kinases, across all salts, while the supernatant barely moves (`reports/screens/ga22_ga24_report.md`).
- **A bait titration at 0.25, 1 and 4 times the usual bait**, to settle whether abundant clients saturate
  the sites they use. Under saturation, more bait raises the capture of abundant clients more than that of
  scarce ones; the competing explanations predict no change in slope (`additional_text_9`).

- **A co-capture test, which is the one 3g calls for.** Stoichiometric co-capture predicts a complex is
  pulled down as a unit, so its weak-binding members are carried by its strong-binding ones. The designed
  version pairs a bait titration with complexes whose subunits differ in predicted binding: under
  co-capture the weak members track the strong one's capture, under independent binding they do not. The
  asymmetry test in 3g is the observational shadow of this and reaches only 3 of 4 targets with one
  interval excluding zero.

The wet-lab proposals attached to the heat response -- a pull-down without bait at 35 and 43 °C, binding at
39 and 41 °C, and melting temperatures measured in this lysate -- remain open and are listed in
`reports/protein_response_report.md` section 8.

**One optional computation.** 3f's null carries a caveat only because a single unseeded Boltz-2 draw has a
test-retest reliability of 0.497. Averaging eight draws per window would raise that to about 0.89 by
Spearman-Brown, and lift the per-protein ceilings with it -- by how much cannot be stated in advance,
because the split-half figure also carries genuine window-to-window heterogeneity within a protein, which
averaging over draws does not remove. The cost is
seven further passes over 4,777 windows: about 28 GPU-hours on one card, or three and a half hours on a
full eight-GPU node. It would buy rigour, not a new result -- P2, the test that real windows score no higher
than their own scrambles, is paired across 4,577 windows and already well powered, so the finding that the
score ignores sequence order will not move.

**What is deliberately not proposed.** Scaling the co-fold pilot from 200 to 2,000 proteins, for the reason
given in 3f: the limit is the score's insensitivity to sequence order, which sample size does not touch.
Full-length co-folds (10-20x the cost on the same model that threaded a peptide backwards). And predicting
the complex effect from the *sequences* of a protein's partners, which is close to circular -- the complex
effect is defined as the part a sequence model misses, so a sequence model of the partners is being asked
for exactly what sequence has already been shown not to carry.

# 10. Files

- **Code:**
  - `cm00_union.py`, `cm01_embed.py`, `cm01b_residues.py` (embeddings, SAE, per-residue states);
  - `cm02_targets.py`, `cm02b_features.py` (targets, features, folds);
  - `cm03_mlp.py` (population MLP, covariates, residualisation), `cm03b_ridge.py`;
  - `cm04_latent.py` (transfer, regress-out, geometry, measured cross-run);
  - `cm05_attrib.py`, `cm05b_assoc.py`, `cm05c_annot.py` (SAE);
  - `cm06_attn.py` (attention pooling), `cm07_baseline.py` (named properties);
  - `cm08_figures.py`, `cm09_stats.py` (bootstrap), `cm10_proteome.py`;
  - `cm12_complex.py`, `cm16_network.py` (complex and network context, 3c);
  - `cm13_structure.py`, `cm15_maskpool.py`, `cm14_struct_test.py` (AlphaFold structure, 3d);
  - `cm17_pdb_index.py`, `cm18_interface.py`, `cm19_iface_test.py` (experimental interfaces, 3e);
  - `cm26_nonlinearity.py` (capture nonlinearity, co-abundance and co-capture, 3g);
  - `cm20_pilot_select.py`, `cm21_windows.py` (pilot set and co-folding jobs, 3f),
    `cm22_val_score.py` (validation gate against 9ef3), `cm23_pilot_test.py` (the four predictions),
    `cm24_pilot_figure.py`, `cm25_reliability.py` (test-retest and split-half ceilings);
  - `cm_launch.py` and `cm_summary.py` (sweeps);
  - `slurm/cm0*.sbatch`, `slurm/cm1[3689].sbatch`, `slurm/boltz_val.sbatch`,
    `slurm/boltz_pilot.sbatch`, `slurm/boltz_rep.sbatch`, `slurm/cm26.sbatch`.
- **Results:** `data/campaign/`:
  - `runs/` (every group: config, R², out-of-fold predictions, latents, saved models);
  - `ridge.json`, `cm07_baseline.json`, `cm09_stats.json`;
  - `attrib/` (associations, annotated features);
  - `cm12_complex.json`, `cm16_network.json`, `cm14_struct.json`, `cm19_iface.json`,
    `cm26_nonlinearity.json`;
  - `cm20_pilot.json`, `cm22_val.tsv`, `cm23_pilot.json`, `cm25_reliability.json`;
  - `pilot_proteins.tsv`, `boltz_jobs.tsv`, `pilot_windows.tsv` (4,787 co-folds),
    `pilot_scores.tsv`, `rep_pairs.tsv`;
  - `boltz/` (4,787 + 600 predicted structures and confidence files, 2.1 GB);
  - `struct.tsv`, `interface.tsv`, `iface_entry.tsv`, `pdb_manifest.tsv`;
  - `proteome_predictions.tsv`.
- **External data:** `data/external/`: `complexes/` (Complex Portal), `string/`, `sifts/`,
  `afdb_human/` (AlphaFold human proteome tar), `pdb_assemblies/cif/` (29,816 RCSB biological assemblies,
  11 GB, fetched by `code/fetch_pdb_assemblies.sh`). Boltz-2 weights live in `~/.boltz` (3.7 GB), and
  the co-folding environment is the `boltz` conda env (boltz 2.2.1, torch 2.14.1+cu130,
  cuequivariance kernels; CUDA 13 is required by the Blackwell RTX PRO 6000 cards).
- **Figures:** `reports/figures/campaign/`.
