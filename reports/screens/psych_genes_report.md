---
title: "Drug-sensitive HSPB1 partners and schizophrenia / bipolar genetics"
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

**Data.** There are two kinds of genetic evidence, and both were used.

| | Schizophrenia | Bipolar disorder |
|---|---|---|
| Rare coding variants (exome sequencing) | SCHEMA, 24,248 cases | BipEx, 13,933 cases |
| Common variants (GWAS) | PGC3, European: 53,386 cases, 77,258 controls | PGC 2024, European, no 23andMe: up to 59,287 cases, 781,022 controls |

The exome studies publish gene-level results. For the GWAS, I computed gene-level association from
the public summary statistics with MAGMA, the standard tool. Everything is linked to the screens'
UniProt accessions through HGNC.

**Answer.**

- **Anticonvulsants: no link** with either kind of evidence.
- **Lithium, rare variants: no link.**
- **Lithium, common variants: a weak lead that does not hold up yet.** The most lithium-sensitive
  proteins carry more bipolar GWAS association than random sets, and somewhat more schizophrenia
  association. The two controls from the same screen do not: lithium at 2.5 mM Mg^2+^, and the Mg^2+^
  effect. But the signal does not reappear when each half of the replicates is analysed on its own.
  The strongest individual cases are also not the lead genes of their GWAS loci.

| Test | Rare variants (Part A) | Common variants (Part B) |
|---|---|---|
| Are drug hits risk genes? | No. 4 hits at nominal p < 0.05, against 5 expected by chance | 4 Li hits are MAGMA-significant: BAG6, SKIC2, PPP4C (schizophrenia) and PC (bipolar). None is the lead gene of its locus, and two lie in the MHC. No hit is one of the 57 PGC3 prioritised genes present in the pull-down |
| Do hits carry more association as a group? | No (all p ≥ 0.07) | Anticonvulsants no. Li hits, bipolar: p 0.011, or 0.024 with the MHC excluded |
| Does drug sensitivity rise with association across all proteins? | No (rho −0.016 to +0.040) | No (rho −0.019 to +0.019) |
| The most Li-sensitive 2-10% of proteins | not tested | Enriched for bipolar (top 5%: p 0.003) and schizophrenia (p 0.005), MHC excluded. Controls flat. **Not seen in independent halves of the replicates** |
| Known risk genes in the pull-down respond to a drug? | No (19 SCHEMA genes detected) | No beyond the Li hits above. 20 drug effects at \|t\| ≥ 3 among 2,772, against about 15 expected |
| Best result after correcting for all tests | q 0.053 (40 tests) | q 0.024 (96 tests): Li top 5% vs bipolar GWAS |

**Interpretation.** The anticonvulsant hits have no genetic connection to either disorder.

For lithium there is a hint, from common variants only, that the proteins it moves in this assay
overlap bipolar risk. The hint has three things for it:

- it is specific to lithium at low Mg^2+^;
- it is spread across many loci rather than one;
- it is strongest for bipolar disorder, the disorder lithium treats.

And three things against it:

- it fails the split-half replication;
- its individual genes sit next to, not on, the GWAS peaks;
- it is one of 96 tests.

It is worth a pre-registered test in a new lithium screen, ideally in neuronal lysate. It is not a
finding yet.

# Part A. Rare-variant (exome) genes

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

# 5. Critical evaluation of Part A: what this test cannot see

1. **A narrow slice of risk.** Exome burden genes cover rare, damaging variants in about 30 genes.
   Most heritability is common variation in GWAS loci, tested in Part B.
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

# Part B. Common-variant (GWAS) genes

# 6. Data and methods

| | PGC3 schizophrenia | PGC bipolar 2024 |
|---|---|---|
| Source | Trubetskoy et al. 2022, figshare `scz2022`, European autosomes | O'Connell et al. 2025, figshare `bip2024`, European, no 23andMe |
| SNPs used by MAGMA | 7.55 million (98.6% matched to the reference) | 6.9 million |
| Genes tested | 18,447 | 18,355 |
| Significant genes (Bonferroni, p < 2.7 × 10^−6^) | 705 | 238 |
| Top genes | DPYD, CACNA1C and others: the published loci | the FADS/FEN1 cluster, CACNA1C: the published loci |

**Gene-level test.** MAGMA v1.10, SNP-wise mean model, 1000 Genomes European LD reference. A gene
gets the SNPs from 35 kb upstream to 10 kb downstream, as in PGC3. The run takes 6 minutes on one
node (`slurm/magma_pgc.sbatch`).

**PGC3 prioritised genes.** The 120 genes of PGC3 Extended Data Table 1. These are genes picked
within the loci by fine-mapping, SMR or rare-variant support; 104 have a UniProt entry.

**Checks that the genetics behaves as expected:**

- Schizophrenia and bipolar gene scores correlate (rho 0.30). This is expected: the two disorders
  share much of their common-variant risk.
- Schizophrenia GWAS and SCHEMA exome scores barely correlate (rho 0.010). This is also expected:
  the published overlap is a handful of genes.

**Controls specific to GWAS:**

- **Gene size.** A gene with more SNPs gets a smaller MAGMA p, so random sets are drawn within
  deciles of SNP count.
- **Shared signal between neighbouring genes (LD).** Neighbouring genes inherit each other's signal.
  - The extended MHC (chromosome 6, 25-34 Mb) is the worst case: long-range LD there lifts every
    gene. Every test is run with and without it.
  - Each hit near a strong signal is checked: is it the lead gene of its ±500 kb window?
- **Outliers.** A rank version of each set test stops one extreme gene from carrying a set.
- **Multiple testing.** 96 tests, BH-adjusted together.

# 7. Drug hits against GWAS association

![Quantile-quantile plots of the MAGMA p-values of the drug hits. The grey background curve, all proteins in either pull-down, rises steeply because common-variant risk is spread over thousands of genes. Hits must be compared with that curve, not with the diagonal. Most hits sit on it. The Li hits labelled above it are discussed below.](../figures/psych/gwas_qq_hits.png){width=100%}

Each cell gives the hit set's mean −log10 p, the mean for random sets of the same SNP counts, and
the p-value.

| Hit set | n | Schizophrenia | Same, MHC excluded | Bipolar | Same, MHC excluded |
|---|---|---|---|---|---|
| Carbamazepine | 7 | 1.31 vs 1.54, p 0.55 | p 0.48 | 1.17 vs 1.14, p 0.39 | p 0.37 |
| Lamotrigine | 11 | 0.98 vs 1.52, p 0.84 | p 0.81 | 1.70 vs 1.13, p 0.09 (rank p 0.03) | p 0.09 (rank p 0.03) |
| Topiramate | 1 | p 0.12 | p 0.11 | p 0.07 | p 0.07 |
| Lithium | 27 | 2.35 vs 1.47, **p 0.022** (rank p 0.16) | 1.71 vs 1.43, p 0.21 | 1.72 vs 1.09, **p 0.011** (rank p 0.023) | 1.62 vs 1.07, **p 0.024** (rank p 0.060) |
| Li-specific | 9 | 2.80 vs 1.36, **p 0.019** (rank p 0.19) | 1.24 vs 1.27, p 0.44 | 2.54 vs 1.02, **p 0.002** (rank p 0.009) | 2.50 vs 0.98, **p 0.005** (rank p 0.085) |

PDHA1 and SLC9A7 are X-linked and not in the autosomal GWAS files, so the Li-specific set has 9
testable proteins.

- **Schizophrenia.** The Li signal is the MHC: it disappears when the MHC is excluded.
- **Lamotrigine, bipolar.** The rank test reaches p 0.03, from DHFR and PTP4A1 near p 10^−4^. It
  does not survive correction, and I do not count it.
- **Bipolar.** The signal survives MHC exclusion but weakens in the rank version (p 0.06-0.09). That
  means a few strong genes carry it, mainly PC and GET4.

**Are the strongly associated hits the genes the GWAS points to?** Mostly not. Each row is a hit
with MAGMA p ≤ 10^−5^, ranked among all genes in its ±500 kb window.

| Hit | Disorder | Hit's p | Rank in window | Lead gene of the window (p) | Note |
|---|---|---|---|---|---|
| BAG6 (Li) | schizophrenia | 3.7 × 10^−14^ | 8 of 56 | MICB (4.9 × 10^−17^) | MHC class III |
| SKIC2 (Li) | schizophrenia | 6.1 × 10^−8^ | 51 of 62 | MICB | MHC class III |
| PPP4C (Li) | schizophrenia | 8.9 × 10^−10^ | 12 of 41 | HIRIP3 (6.2 × 10^−14^) | 16p11.2, a gene-dense region with long LD |
| PC (Li) | bipolar | 6.5 × 10^−9^ | 6 of 26 | RCE1 (3.0 × 10^−10^) | 11q13 |
| GET4 (Li) | bipolar | 3.4 × 10^−6^ | 2 of 13 | SUN1 (2.4 × 10^−6^) | 7p22 |
| PSME4 (carbamazepine) | schizophrenia | 6.9 × 10^−6^ | 2 of 8 | GPR75 (5.6 × 10^−6^) | |
| UHRF1 (Li) | schizophrenia | 1.6 × 10^−5^ | 1 of 16 | UHRF1 | lead, but not significant |

None of these hits is the lead gene of a significant locus. A MAGMA p for a gene near a strong peak
partly reflects its neighbours, so these hits are near risk loci, not shown to be the risk genes.
None of the 57 PGC3 prioritised genes in the pull-downs is a drug hit.

**An observation, not a finding.** Three lithium responders lie within 0.5 Mb of each other in MHC
class III: BAG6, SKIC2 and ATF6B (the last a Li-specific protein that is not among the 30 hits).
Only 1.1% of detected proteins lie in the extended MHC, so about 0.4 would be expected among about 35 Li
responders. But the three are unrelated in function, I noticed the cluster after the fact, and the
MHC's LD makes any genetic reading impossible.

# 8. All proteins: overall correlation, and the most lithium-sensitive tail

![Every detected protein. x: MAGMA −log10 p (dashed line: Bonferroni); y: drug sensitivity (−log10 p). Blue: least-squares fit and 95% band. Across 6,300-7,900 proteins the lines are flat for every drug. Red marks drug hits, and significant GWAS genes with some drug sensitivity. The Mg^2+^ column shows that a strong, reproducible perturbation is also unrelated to GWAS risk.](../figures/psych/gwas_scatter_all.png){width=100%}

| Drug | Schizophrenia: partial rho (p) | Bipolar: partial rho (p) |
|---|---|---|
| Carbamazepine | +0.012 (0.35) | −0.014 (0.27) |
| Lamotrigine | −0.008 (0.50) | −0.014 (0.25) |
| Topiramate | +0.003 (0.80) | +0.008 (0.54) |
| Lithium | +0.011 (0.35) | +0.009 (0.45) |
| Mg^2+^ (comparison) | +0.016 (0.16) | +0.012 (0.29) |

The partial correlations control for protein length, pull-down level and SNP count. Across the whole
proteome there is nothing. But a real effect confined to a few hundred proteins would not show in a
correlation over 7,000, so the tail was tested directly.

![GWAS association of the most drug-sensitive proteins, MHC excluded. y: how far the set's mean MAGMA −log10 p lies above random sets with the same SNP counts. Grey band: 95% of random sets for the lithium curve. Lithium at 0.25 mM Mg^2+^ (red) rises above the band for the top 2-10% in both disorders. The two controls stay flat: lithium at 2.5 mM Mg^2+^ (orange) has the same samples' noise and no lithium effect, and the Mg^2+^ effect (dark blue) is strong but not lithium. So do lamotrigine and carbamazepine.](../figures/psych/gwas_tail_curves.png){width=100%}

| Ranking (top 5%, MHC excluded) | Schizophrenia: excess (p) | Bipolar: excess (p) |
|---|---|---|
| **Lithium trend, 0.25 mM Mg^2+^** | **+0.22 (0.005)** | **+0.20 (0.003)** |
| Lithium trend, 2.5 mM Mg^2+^ (control) | −0.07 (0.80) | +0.01 (0.43) |
| Mg^2+^ effect (control) | +0.04 (0.32) | +0.03 (0.33) |
| Lamotrigine | −0.03 (0.61) | −0.01 (0.57) |
| Carbamazepine | −0.08 (0.82) | −0.02 (0.61) |

Several things argue that the lithium tail signal is not trivial:

- Matching the random sets on pull-down level as well as SNP count gives the same p (checked with
  the MHC included).
- The signal is spread out. In bipolar, the 25 proteins in the Li top 5% with MAGMA p ≤ 10^−4^ fall
  in 20 different loci, and the five strongest contribute only half of the excess.
- Of all the rankings tried, it is the one with a biological reason to be there.

**It fails the replication test.** If the signal reflects how proteins respond to lithium, a ranking
built from two replicates should find it, and so should a ranking from the other two.

![Each bar ranks proteins by the lithium trend tested on two replicates only, takes the top 5% (MHC excluded), and measures their GWAS excess. Pairs of bars that share no replicate are complements: BR 1,2 with 3,4; 1,3 with 2,4; 1,4 with 2,3. One half of one split is significant for bipolar (replicates 2 and 4, p 0.005); its complement is not (p 0.45). Schizophrenia shows nothing in any half.](../figures/psych/gwas_split_half.png){width=85%}

Two readings fit this result.

| Reading | What it says | Fit to the data |
|---|---|---|
| Too little power | The Li response outside the 30 hits is weak (proteome-wide reliability 0.14). With two replicates per half the ranking is mostly noise, so a real tail signal could be lost | Possible. The full-data excess is +0.20; the half-data mean is +0.05 for bipolar and +0.006 for schizophrenia. I did not estimate how much a real signal should shrink with half the data |
| A chance tail | The signal is a fluctuation in one full-data ranking | Fits the replication failure. It is the strongest of 96 tests (q 0.024); the next are q 0.06-0.10 |

The data cannot separate these readings. A signal that neither independent half finds on its own is
not established.

# 9. GWAS genes present in the pull-downs

Detection rates, against 49% for all genes:

- 57 of 104 PGC3 prioritised genes are detected;
- 368 of 681 significant schizophrenia genes;
- 122 of 222 significant bipolar genes.

![Moderated t for every drug contrast, for the PGC3 prioritised genes (P) and the MAGMA-significant genes (S, schizophrenia; B, bipolar) with \|t\| ≥ 3 for some drug. Numbers mark \|t\| ≥ 3. The strong drug cells are the lithium hits of section 7: PC, BAG6, PPP4C, SKIC2. Among 2,772 drug cells for GWAS genes, 20 reach \|t\| ≥ 3, against about 15 expected by chance. The Mg^2+^ column is strong for many genes because Mg^2+^ changes 70% of all proteins.](../figures/psych/gwas_genes_t.png){width=72%}

# 10. Critical evaluation of Part B

1. **Gene assignment by position.** MAGMA gives each gene the SNPs that lie near it. In gene-dense
   regions the true risk gene can be a neighbour, as the locus table shows for every strong hit.
   Fine-mapping (the PGC3 prioritised genes) avoids this, and no drug hit is among those genes.
2. **The tail result is one of many tests and does not replicate.** It is the reason this report
   says "lead". A pre-specified test in an independent lithium screen would settle it. That test is:
   the top 5% by Li trend at low Mg^2+^, MHC excluded, against bipolar MAGMA, with the 2.5 mM Mg^2+^
   arm as the control.
3. **The wrong cell type, again.** GWAS risk concentrates in neuronal genes. Half the significant
   GWAS genes are detected here, but HEK293 lysate is not where they act.
4. **European data only.** Both studies also have other-ancestry and multi-ancestry results. Those
   need matched LD references and were not run.
5. **The X chromosome is not tested.** PDHA1 and SLC9A7, two of the strongest Li-specific proteins,
   are X-linked.
6. **The pull-down measures binding to HSPB1, not drug targets.** This limitation applies to both
   parts.

# 11. Files

- Code:
  - `code/fs05_psych_genes.py` (exome);
  - `code/fs06_psych_gwas.py` (GWAS);
  - shared tests in `code/pdpipe/genetics.py`;
  - MAGMA job `slurm/magma_pgc.sbatch` (6 min on one node).
- Downloaded data:
  - `data/annot/exome/` (SCHEMA, BipEx);
  - `data/annot/gwas/` (PGC summary statistics, MAGMA v1.10, 1000 Genomes EUR reference, MAGMA gene results `magma_scz.genes.out`, `magma_bd.genes.out`);
  - `data/annot/hgnc_complete_set.txt.gz`.
- Results in `data/fs_data/psych/`:
  - `fs05.json`, `fs05_hit_genetics.tsv`, `fs05_risk_gene_t.tsv`, `fs05_per_protein.tsv`;
  - `fs06.json` (all tests, tail curves, split halves, locus checks, 96-test family with q-values);
  - `fs06_hit_gwas.tsv`, `fs06_gwas_gene_t.tsv`.
- Figures: `reports/figures/psych/`.
