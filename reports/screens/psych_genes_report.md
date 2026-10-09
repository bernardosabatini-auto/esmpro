---
title: "Drug-sensitive HSPB1 partners and schizophrenia / bipolar risk genes"
date: "9 October 2026"
geometry: margin=2.2cm
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
---

# Summary

**Question.** Are the proteins that respond to the anticonvulsants (FS73) or to lithium (FS76) in
the HSPB1 pull-down also risk genes for schizophrenia or bipolar disorder?

**Data.** The risk genes come from the two large exome-sequencing studies:

- **SCHEMA** (schizophrenia): 24,248 cases.
- **BipEx** (bipolar disorder): 13,933 cases, 14,422 controls.

Both publish gene-level results through the Broad's public browsers. I downloaded them and linked
them to the screens' UniProt accessions through HGNC.

**Answer: no.** None of the 51 drug-sensitive proteins is an established or near-significant risk
gene. As a group, they carry no more genetic association than other proteins in the pull-down. The
reverse also holds: the 21 established risk genes that the pull-down detects do not respond to any
drug.

| Test | Result |
|---|---|
| Are drug hits risk genes? | No. The best are CA5B (schizophrenia p 0.018), IFRD2 (0.022), PGM3 (bipolar p 0.026) and GALE (0.045). About 5 such p < 0.05 genes are expected by chance among 51 hits; 4 were seen |
| Do hits carry more association as a group? | No, for any drug or for the Li-specific set. All p ≥ 0.07 against random sets of the same protein length (Fig. 1) |
| Does drug sensitivity rise with genetic association across all ~6,300-7,900 proteins? | No. Correlations −0.016 to +0.040, explaining at most 0.16% of the variance (Fig. 2) |
| Do established risk genes respond to a drug? | No. 19 of the 32 SCHEMA FDR-5% genes and 2 of the top BipEx genes are detected. The largest drug effect is TRIO falling with Li (t −3.0, q 0.49), which is Mg-like (Fig. 3, 4) |
| Anything after correction for the 40 tests run? | Nothing. The smallest adjusted q is 0.053 (topiramate vs schizophrenia, R² 0.16%) |

**Interpretation.** In these screens, the drugs and lithium act on HSPB1 partners that are mostly
metabolic enzymes and nucleotide-handling proteins. The proteins with strong rare-variant risk are
mainly chromatin regulators, ubiquitin ligases, and synaptic and neuronal proteins. Many of these are
present in the pull-down, but none responds.

This does not rule out a link. It means the link, if there is one, is not through the same proteins
at the level these data can resolve. Section 4 lists what this test cannot see.

# 1. Data and methods

**Risk genes.**

| | SCHEMA (schizophrenia) | BipEx (bipolar) |
|---|---|---|
| File | `SCHEMA_gene_results.tsv.bgz` | `BipEx_gene_results.tsv.bgz`, group "Bipolar Disorder" |
| Gene-level test | rare PTV and damaging-missense burden; case-control and de novo meta-analysis (P meta, Q meta) | Fisher tests of PTV and of damaging missense against gnomAD non-psychiatric controls; gene p = smaller of the two × 2 |
| Genes tested | 18,143 | 19,391 |
| Strong genes | 32 at FDR 5% (10 exome-wide) | 1 at p ≤ 1e-4 (AKAP11); 6 at p ≤ 0.001 |

BipEx has much less power than SCHEMA. Its gene p-values sit below the uniform line for most of their
range (Fig. 1B, grey curve). That pattern means a conservative test, not an absence of signal, but it
leaves little to correlate with.

**Drug sensitivity.** Each protein gets one value per drug: −log10 p of the drug's effect.

| Drug | Test behind the value |
|------------------|----------------------------------|
| Carbamazepine, lamotrigine, topiramate (FS73) | 2-df F test of both doses against control |
| Lithium (FS76) | dose trend at 0.25 mM Mg, the condition where the Li effects are |
| Mg^2+^ (FS76) | Mg effect, as a comparison (changes 6,129 proteins) |

The drug hits are the ones from the screen report:

- FS73: 21 hits, q ≤ 0.05;
- FS76: 30 Li hits;
- FS76: 11 Li-specific proteins (section 4.5 of the screen report).

**Length control.** Longer genes collect more rare variants, so their burden tests have more power and
smaller p-values. Every set test therefore compares against random sets drawn within the same
protein-length deciles. The correlations are partial Spearman, controlling for length and for the
protein's pull-down level.

**Multiple testing.** 40 tests: 5 drugs × 2 diseases × up to 4 tests each (correlation, hits,
top 1%, top 5%), plus the Li-specific set. All are BH-adjusted together.

# 2. Drug hits against genetic association

![Quantile-quantile plots of the schizophrenia (A) and bipolar (B) p-values of the drug hits. The grey curve is all proteins detected in either screen. The schizophrenia curve rises above the diagonal at the top: those are the true risk genes, and they are present in the pull-down. Every drug hit sits on or below the background curve, and the four labelled genes are within chance.](../figures/psych/qq_hits.png){width=100%}

| Hit set | n | Mean −log10 p, schizophrenia (random sets) | p | Mean −log10 p, bipolar (random sets) | p |
|---|---|---|---|---|---|
| Carbamazepine | 7 | 0.28 (0.42) | 0.77 | 0.31 (0.23) | 0.24 |
| Lamotrigine | 12 | 0.49 (0.42) | 0.28 | 0.18 (0.22) | 0.64 |
| Topiramate | 2 | 0.87 (0.37) | 0.07 | 0.16 (0.21) | 0.52 |
| Lithium | 29 | 0.43 (0.43) | 0.50 | 0.30 (0.24) | 0.18 |
| Li-specific | 11 | 0.36 (0.46) | 0.71 | 0.11 (0.26) | 0.93 |

Could the test have seen a real overlap? It could have seen a strong one. For the 29 Li hits, the
cut-off for "more than chance" is a mean −log10 p of 0.60, against 0.43 for random sets. A single
gene at p ≈ 10^−5^ among the hits, or two at p ≈ 10^−3^, would have crossed it. Every SCHEMA FDR-5%
gene has p below 10^−4^. So one established schizophrenia gene among the Li hits would have shown.

The topiramate p of 0.07 comes from one of only two hits: CA5B, at schizophrenia p 0.018 (q 0.82).
That is not evidence.

# 3. Across all proteins

![Every detected protein. x: genetic association (−log10 p); y: drug sensitivity (−log10 p). The blue line is the least-squares fit with its 95% band. Red: drug hits, and labelled risk genes. Across 6,300-7,900 proteins the lines are flat. The established risk genes on the right are not drug-sensitive, and the drug hits at the top are not risk genes. The Mg^2+^ column shows a strong, reproducible perturbation that is equally unrelated to genetic risk.](../figures/psych/scatter_all.png){width=100%}

| Drug | Schizophrenia: partial rho (p) | Bipolar: partial rho (p) |
|---|---|---|
| Carbamazepine | −0.016 (0.19) | +0.017 (0.19) |
| Lamotrigine | −0.009 (0.48) | −0.011 (0.36) |
| Topiramate | +0.040 (0.0013; q 0.053) | −0.007 (0.58) |
| Lithium | +0.007 (0.53) | −0.011 (0.34) |
| Mg^2+^ (comparison) | −0.005 (0.65) | +0.004 (0.70) |

The topiramate correlation is the only nominally significant one. It explains 0.16% of the variance
and does not survive correction. Topiramate also has no reproducible effect on the pull-down beyond
CA5B, so its per-protein p-values are mostly noise. A weak correlation of noise with genetics most
likely reflects a property shared by both measures that the length and level controls do not
capture. I do not count it.

The looser sets give a few nominal signals, none surviving correction:

| Set | p | What drives it |
|---|---|---|
| Top 1% most carbamazepine-sensitive, bipolar | 0.003 (q 0.06) | five genes at bipolar p 0.014-0.045 (SEMA4B, ARHGAP5, ANXA11, AMER1, GALE). The 5% set is not significant (0.09) |
| Top 5% most Li-sensitive, schizophrenia | 0.011 (q 0.15) | mostly TRIO; the rest are genes at p 10^−3^-10^−2^ |
| Top 5% most Li-sensitive, bipolar | 0.047 (q 0.47) | TICRR (p 0.0005); the rest weak |

A real link would show up for the strongest drug effects first and weaken in looser sets. The
carbamazepine signal does the opposite: it rests on five barely significant genes and vanishes at 5%.

# 4. The established risk genes in the pull-downs

| Set | Total | Detected in FS73 or FS76 |
|---|---|---|
| SCHEMA FDR 5% | 32 | 19: AKAP11, CUL1, DNM3, EIF2S3, FAM120A, H1-4, HERC1, KDM6B, NR3C2, PREP, RB1CC1, SETD1A, SLF2, SRRM2, STAG1, SV2A, TRIO, XPO7, ZMYM2 |
| BipEx p ≤ 0.001 | 6 | 2: DOP1A, TICRR (AKAP11 is counted in the SCHEMA row) |

Half of all mapped genes are detected (8,932 of 18,143), so 19 of 32 is about what is expected. The ones missing are
mostly neuron-only proteins absent from HEK293 lysate: GRIN2A, GRIA3, CACNA1G, SP4.

![Moderated t of every drug contrast for the risk genes detected. Numbers mark |t| ≥ 3. The only drug cells at |t| ≥ 3 are TRIO with Li (−3.0); TICRR with Li (+2.9) and SRRM2 with topiramate 10 µM (+2.9) fall just short. About one such cell is expected by chance among the 137 drug cells. The Mg^2+^ column is strong for many genes, but Mg changes 70% of all proteins, so that is the base rate.](../figures/psych/risk_genes_t.png){width=70%}

![Lithium dose curves (blue: 0.25 mM Mg; orange: 2.5 mM Mg) of the risk genes in FS76. * marks the SCHEMA exome-wide genes. TRIO drifts down with Li at low Mg, by about 0.08 log2 per 10-fold more Li (q 0.49); Mg lowers it more. This is the Mg-like pattern of most Li hits, and too weak to call. The rest are flat.](../figures/psych/risk_genes_li.png){width=100%}

Two genes are worth naming even though neither responds:

- **AKAP11** is detected in both screens and does not move (largest |t| under any drug is 1.4). It
  is the strongest gene in BipEx, an FDR gene in SCHEMA, and a binding partner of GSK3B, a classic
  lithium target.
- **SV2A** is the target of levetiracetam. It is detected in FS76 only, so it could not be tested
  with the FS73 drugs. Levetiracetam was not in the screen.

# 5. Critical evaluation: what this test cannot see

1. **Different kinds of genetic evidence.** Exome burden genes are a narrow slice of risk: rare,
   damaging variants in about 30 genes. Most heritability is common variation in GWAS loci (PGC),
   which point to hundreds of genes and to neuronal and synaptic biology. The GWAS genes were not
   asked about here. They are the natural next set, using the PGC prioritised genes or MAGMA
   gene-level statistics, with the same length-matched tests.
2. **The wrong cell type.** HEK293 lysate lacks most synaptic proteins, which is where the
   schizophrenia exome signal concentrates (GRIN2A, GRIA3, CACNA1G). A lysate from neurons or brain
   would let those genes be tested at all.
3. **The pull-down measures binding to HSPB1, not the drug's direct targets.** A drug that binds a
   risk-gene product without changing its HSPB1 association is invisible here.
4. **Weak bipolar data.** With one exome-wide gene, BipEx gives little to correlate with. The
   lithium comparison, the one with the strongest prior, is therefore the weakest test. The SCHEMA
   comparison is the informative one.
5. **Few hits.** 2 to 30 per drug. The tests can detect one strong risk gene among the hits
   (section 2), but not a diffuse enrichment of weak ones.
6. **Gene-to-protein mapping.** HGNC links each Ensembl gene to its reviewed UniProt entries. Where
   one accession maps to several genes, I kept the best-tested. This affects a handful of proteins
   and none of the risk genes.

# 6. Files

- Code: `code/fs05_psych_genes.py`.
- Downloaded data: `data/annot/exome/` (SCHEMA, BipEx) and `data/annot/hgnc_complete_set.txt.gz`.
- Results: `data/fs_data/psych/`:
  - `fs05.json` (all tests, including the 40-test family with q-values);
  - `fs05_hit_genetics.tsv` (every hit with its SCHEMA and BipEx statistics);
  - `fs05_risk_gene_t.tsv`;
  - `fs05_per_protein.tsv` (all proteins, all drugs, both diseases).
- Figures: `reports/figures/psych/`.
