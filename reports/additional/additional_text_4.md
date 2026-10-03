---
title: "Additional text 4: do the conclusions depend on how the runs are normalised?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

GA_33 runs were median-normalised, which removed a rise of about 0.9 log2 units in run medians from
35 to 43 °C (a loading or global-binding difference that the data cannot resolve). The salt workbook
was used as delivered. Do the conclusions survive other choices?

# What can and cannot change

Median normalisation, normalisation to the bait's own signal, and no normalisation all subtract a
single constant from each run. Across-protein comparisons are then unchanged, except for proteins
missing from some runs, whose replicate means draw on different runs. These three choices are
therefore a weak test, included for completeness. What could change the results is a difference in
**dynamic range** between runs: if, say, the 43 °C runs compressed the intensity scale, abundant
proteins would appear to fall and scarce ones to rise, which would mimic the strong dependence of the
heat response on abundance. Two normalisations remove such differences: **scale** (each run centred
and scaled to a common interquartile range) and **quantile** (each run mapped onto a common intensity
distribution).

# Results

**1. Dynamic range.** In GA_33 the 43 °C runs are slightly *wider*, not compressed (interquartile range
1.59 and 1.69 against 1.48 and 1.55 at 35 °C, for DNAJA1 and DNAJB11; standard deviation ratio 1.05).
In the salt workbook the runs had already been median-aligned upstream (run medians 12.51-12.63), and
the 75 and 150 mM runs are mildly narrower than the 0 mM runs (1.51 and 1.55 against 1.64).

**2. GA_33 under five normalisations.**

| Norm. | Target | r vs median | Reliab. | rho, abund. | ESMC R2 | Class effect |
|:--|:--|--:|--:|--:|--:|:--|
| median | heat 43 °C | 1 | 0.992 | -0.43 | 25.3 % | TM -0.52 |
| none | | 1.000 | 0.992 | -0.43 | 25.3 % | TM -0.52 |
| bait | | 1.000 | 0.991 | -0.43 | 25.3 % | TM -0.52 |
| scale | | 0.997 | 0.991 | -0.50 | 24.4 % | TM -0.53 |
| quantile | | 0.996 | 0.991 | -0.50 | 24.1 % | TM -0.53 |
| median | stau. 37 °C | 1 | 0.755 | -0.04 | 6.5 % | kinases -1.04 sd |
| scale | | 0.999 | 0.753 | -0.11 | 6.3 % | kinases -1.03 sd |
| quantile | | 0.998 | 0.752 | -0.09 | 6.3 % | kinases -1.03 sd |
| median | DNAJB11 pref. | 1 | 0.959 | +0.12 | 44.2 % | signal +0.35, nuc. -0.31 |
| scale | | 0.993 | 0.959 | +0.01 | 44.2 % | signal +0.36, nuc. -0.30 |
| quantile | | 0.992 | 0.957 | +0.00 | 43.5 % | signal +0.36, nuc. -0.30 |

*ESMC R2: ridge, final layer for heat and preference, layer 50 for staurosporine, folds grouped by
sequence cluster, one penalty (10^4) throughout. Heat 37 °C behaves like heat 43 °C (reliability
0.86-0.87, R2 12.0-12.5 %).*

**3. The salt titration under four normalisations.**

| Norm. | Target | r vs delivered | Reliab. | rho, abund. | ESMC R2 | TM effect |
|:--|:--|--:|--:|--:|--:|--:|
| as delivered | 150 mM | 1 | 0.992 | -0.44 | 18.2 % | +0.46 |
| median | | 1.000 | 0.992 | -0.44 | 18.2 % | +0.46 |
| scale | | 0.997 | 0.992 | -0.36 | 18.3 % | +0.45 |
| quantile | | 0.996 | 0.992 | -0.36 | 18.1 % | +0.45 |
| as delivered | 75 mM | 1 | 0.986 | -0.41 | 16.0 % | +0.32 |
| scale | | 0.988 | 0.986 | -0.25 | 15.7 % | +0.31 |
| quantile | | 0.987 | 0.986 | -0.24 | 15.6 % | +0.31 |

# Conclusions

- **Every sequence result is robust to normalisation.** The targets keep at least 97 % of their variance in common
  with the original (r >= 0.987), reliabilities by at most 0.008, sequence R2 by at most 1.2 points, and the class
  effects (transmembrane, kinases, signal peptide, nuclear) by at most 0.02.
- **The heat response's dependence on abundance is not a dynamic-range artefact.** Equalising the
  runs' dynamic range makes it stronger (-0.43 to -0.50), not weaker. It reflects a real change in the
  composition of the pull-downs between 35 and 43 °C: the 43 °C and 35 °C profiles correlate at about
  0.6 with nearly equal spread (standard deviation ratio 1.05), and in that situation the change between
  them is necessarily anti-correlated with the starting level.
- **Part of the salt response's dependence on abundance is technical.** The salt runs are mildly
  compressed relative to the 0 mM runs, and removing that weakens the dependence from -0.44 to -0.36
  (150 mM) and from -0.41 to -0.24 (75 mM). This changes nothing in the sequence results, which the
  main report had already shown to be independent of abundance.
- **The global 0.9 log2 rise from 35 to 43 °C** remains unresolvable from these data: loading and a
  real increase in total binding look identical in a pull-down. It affects no across-protein result.

# Critical evaluation

- **The constant-offset normalisations are not a real test** (explained above); the conclusions rest on
  the scale and quantile comparisons.
- **Intensity-dependent normalisation was not used.** Methods such as loess on intensity would remove
  any dependence of the response on abundance by construction, which would erase the very thing under
  question rather than test it. The dynamic-range comparison is the appropriate test.
- **One ridge penalty.** Fixing the penalty at the value earlier fits chose keeps the comparison across
  normalisations exact; R2 values are a few tenths of a point below fully tuned ones (e.g. 25.3 % against
  25.7 % for heat).
- **The salt workbook's upstream processing is unknown.** Its runs were already median-aligned; whether
  other corrections were applied cannot be determined from the file.

# Methods files

`code/ad04_normalisation.py` (GA_33) and `code/ad04b_salt_normalisation.py` (salt).
Results: `data/ga_data/ad04_results.json`, `data/pd_data/ad04b_results.json`.
