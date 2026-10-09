---
title: "HSPB1 drug and lithium screens (FS73, FS76)"
date: "9 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Summary

Both screens are HA-HSPB1 pull-downs from HEK293 lysate at 37 °C, with the drug or salt added to the
lysate. Each was analysed with the same new pipeline (`code/pdpipe`, described in section 2). The
pipeline reproduces the lab's limma estimates in both workbooks (r 0.91-0.998 per contrast).

| | FS73: anticonvulsants | FS76: lithium |
|---|---|---|
| Broad change of the pull-down | none for any drug | none for Li; very large for Mg^2+^ |
| Proteins that change | 21 hits, all robust to dropping any replicate | 30 Li hits, almost all down |
| What the hits are | lamotrigine: DHFR (a known lamotrigine target), the nucleoside kinases DCK and TK2, GART, PP4 subunits, p38 and JNK kinases; topiramate: the carbonic anhydrase CA5B | Mg^2+^-dependent enzymes (3.3-fold enriched); the BAG6-GET4-UBL4A complex |
| Main conclusion | the hits look like the drug acting on its target protein, not on HSPB1. Lamotrigine and topiramate targets come off HSPB1, as if binding stabilises them; most carbamazepine hits come on, as if binding opens them | at low Mg^2+^, Li^+^ acts like extra Mg^2+^ on most hits; at 2.5 mM Mg^2+^ it does almost nothing. About 11 proteins change in ways Mg does not predict: a Na^+^/Li^+^ transporter (SLC9A7), pyruvate dehydrogenase E1, the SKI complex, and several ATP/GTP enzymes |

The screens differ in kind from the temperature and salt experiments (GA_20, GA_33). There, thousands
of proteins move together. Here the treatments leave the pull-down as a whole unchanged and move a few
specific proteins. Magnesium is the exception: raising it from 0.25 to 2.5 mM changes 6,129 proteins.

# 1. The data

| | FS73 | FS76 |
|---|---|---|
| Bait | HA-HSPB1, 2 $\mu$g, 1 mg lysate | HSPB1 |
| Arms | control; carbamazepine, lamotrigine, topiramate at 10 and 100 $\mu$M | LiCl 0, 0.3, 1, 3, 5, 10 mM, traded against NaCl (constant ionic strength), at Mg^2+^ 0.25 and 2.5 mM |
| Samples used | 36 (4-6 per arm; 7 outliers removed upstream) | 44 (3-4 per arm; 4 removed upstream) |
| Proteins | 6,611 (6,100 detected in every sample) | 8,855 (7,287 in every sample) |
| Input | per-sample normalised log2 intensities from the lab's workbook (median normalisation) | same |

There are no input (lysate) measurements for either screen, so a change in the pull-down cannot be
split into a change in binding and a change in how much of the protein is in solution.

# 2. The standard pipeline

Every screen now goes through the same steps, so results from different experiments come out in the
same form (`python -m pdpipe.run <screen>`, configured by a short YAML file per screen):

1. **Load** the workbook into one standard table (proteins by samples, plus a sample sheet).
2. **Quality control**: sample correlations, principal components, detection per sample.
3. **Model**: for each protein, one mean per condition plus a replicate (BR) term when replicates
   behave as batches. They do in both screens (BR explains 28% and 21% of within-arm variance, against
   17% and 9% by chance). Variances are shrunk across proteins as limma does. Missing values are left
   out, never filled in.
4. **Check against the lab's numbers.**
5. **Reproducibility**: each comparison is computed separately in two halves of the replicates, and
   the two halves are correlated across proteins. That gives the share of the comparison's spread
   that is reproducible. It is the ceiling for anything that tries to explain it. It is tested
   against shuffled labels.
6. **Hit checks**: every hit recomputed leaving out each replicate; the top 10 hits from one half
   looked up in the other half; dose agreement and drug-to-drug agreement measured on disjoint
   replicates, so two comparisons that share a control are not correlated by that shared noise.

One lesson from building it is now written into the code. Both screens carry strong sample-to-sample
patterns that are not explained by the design. The largest within-arm pattern holds 19% of the
within-arm variance in FS73 and 21% in FS76, and it tracks how much bait each sample captured (r up to
-0.83). Removing these patterns looked like it made every comparison reproducible, even drug against
drug. That was manufactured: a correction learned from the same samples it is applied to shrinks each
half's own noise but not the noise the halves share. When the correction is learned on one half and
applied to the other, it changes nothing. The results below are therefore uncorrected.

# 3. FS73: anticonvulsants

## 3.1 No drug changes the pull-down as a whole

![FS73. A: reproducible share of each drug-versus-control profile across all proteins; the black line is the 95th percentile with shuffled labels. B: the lamotrigine 100 $\mu$M profile in one half of the replicates against the other half, with the 12 hits in red. C: the 10 strongest hits found in one half, looked up in the other half (each dot is one split); the black line is their size in the half that found them.](../figures/FS73/overview.png){width=100%}

| Comparison | Hits (q <= 0.05) | Reproducible share | Shuffled-label p | Top 10 hits in the other half: mean change (same sign) |
|---|---|---|---|---|
| carbamazepine 10 | 0 | 0.07 | 0.44 | 0.05 (53%) |
| carbamazepine 100 | 7 | 0.02 | 0.56 | 0.26 (62%) |
| lamotrigine 10 | 1 | 0.17 | 0.25 | 0.05 (57%) |
| lamotrigine 100 | 12 | 0.27 | 0.12 | **0.53 (80%)** |
| topiramate 10 | 2 | 0.10 | 0.28 | 0.11 (63%) |
| topiramate 100 | 0 | -0.31 | 0.94 | -0.07 (44%) |

Across all 6,611 proteins, no drug profile is reproducible beyond what shuffled labels give (panel A).
That rules out a broad effect on HSPB1, such as a change in its activity or in the overall amount it
captures. Only lamotrigine at 100 $\mu$M has hits that come back strongly in held-out replicates
(panel C): the top 10 found in one half keep about half their size, and 80% keep their sign, in the
other half. Carbamazepine 100 replicates partly. With 4-6 replicates per arm the shuffled-label test
cannot go below p 0.03, so it can only reject a broad effect, not detect a narrow one.

## 3.2 The hits are robust and drug-specific

![Every FS73 hit, every sample. Each colour is one arm; the bar is the arm mean. The comparison in which the protein was called is in brackets.](../figures/FS73/hits.png){width=100%}

Every one of the 21 hits keeps its sign, and at least 70% of its size, when any single
replicate is left out. So none is carried by one sample. Each is specific to one drug.

| Drug | Hits | Pattern |
|---|---|---|
| lamotrigine 100 $\mu$M | DCK -1.54, PIP4K2C -1.10, PPP4R3A -1.02, TRERF1 -0.90, DHFR -0.77, PPP4R2 -0.76, PTP4A1 -0.50, TK2 -0.45, MAPK14 -0.39, ACOT9 -0.33, MAPK8 -0.30, GART -0.22 | all down; DCK already -0.45 at 10 $\mu$M; several others partly down at 10 $\mu$M |
| carbamazepine 100 $\mu$M | PTGR1 -0.97, M6PR -0.72; GALE +0.95, FUS +0.70, ADH5 +0.67, EWSR1 +0.49, PSME4 +0.27 | mixed; PTGR1, M6PR, ADH5 and GALE already shift at 10 $\mu$M |
| topiramate 10 $\mu$M | BASP1 -1.08, CA5B -0.64 | CA5B equally down at 100 $\mu$M (-0.59; not called there because that arm is noisier) |

Changes are log2, drug against control.

**Lamotrigine.** DHFR is a known lamotrigine target. Lamotrigine is a diaminotriazine, the chemical
class of antifolates, and it weakly inhibits DHFR. GART is a folate-dependent enzyme of purine
synthesis, and DCK and TK2 are nucleoside kinases. All are nucleotide or folate enzymes that could
bind a nucleobase-like ring. The other folate-binding proteins in the data do not move (TYMS +0.14,
MTHFD1 +0.02). Kinases as a class shift very slightly (median -0.03 vs -0.01, p 0.013), so the
effect is confined to a few of them.

**Topiramate.** Topiramate inhibits carbonic anhydrases. CA5B is one of only two carbonic anhydrases
detected, and it drops at both doses. The other, CA9, does not move.

**Carbamazepine.** Its hits go in both directions and have no obvious shared target. FUS and EWSR1
rise together, both FET-family RNA-binding proteins.

**Two ways a drug can change the pull-down.** Thermal proteome profiling sees both kinds of ligand.
Most stabilise their target, and some destabilise it.

- **Binding closes or stabilises the protein.** The drug fills a pocket, the protein becomes more
  compact, or the drug sits on the surface HSPB1 would bind. Less comes down.
- **Binding opens the protein.** The drug shifts a domain, exposes a hydrophobic patch, or loosens a
  subunit interface. More comes down.

Either way the pull-down reports that the drug acts on the protein. The direction says which
mechanism.

| Drug | Hits down / up | Top 50 down | Reading |
|----------|----------|-------|----------------------------------|
| lamotrigine 100 $\mu$M | 12 / 0 (binomial p 0.0005) | 48 of 50 | consistently stabilising or occluding; already visible at 10 $\mu$M for DCK, TRERF1, PTP4A1, PPP4R2/R3A, MAPK14 |
| topiramate 10 $\mu$M | 2 / 0 | 26 of 50 | CA5B down at both doses |
| carbamazepine 100 $\mu$M | 2 / 5 | 31 of 50 | mostly opening: ADH5 already up at 10 $\mu$M (+0.19, p 0.004), GALE nearly (+0.27, p 0.07); FUS, EWSR1 and PSME4 rise only at 100 $\mu$M |

So the opening mechanism probably accounts for most of the carbamazepine hits. ADH5 and GALE rise
with dose, which is what a binding event should do. FUS and EWSR1 rise together and only at the high
dose. They are FET-family proteins with long disordered low-complexity regions, so a small molecule
at 100 $\mu$M could change their conformation or their tendency to condense. Condensates or
aggregates that pellet with the beads would also raise their level without any HSPB1 binding, so the
FET pair is the least certain of these.

## 3.3 Dose and drug agreement, on disjoint replicates

![Each panel compares two comparisons, one measured on one half of the replicates and the other on the other half, so they share no samples. Hits are in red; the line is the regression with its 95% band. Top row: 10 against 100 $\mu$M of the same drug. Bottom row: drug against drug at 100 $\mu$M.](../figures/FS73/concordance.png){width=100%}

Across all proteins, 10 and 100 $\mu$M of the same drug agree only weakly (r 0.06-0.11, averaged over
all splits). Two different drugs do not agree at all (r 0.01-0.07). The scatter is dominated by
noise because so few proteins respond. The hits sit far off the cloud, and in the dose panels they
sit in the same direction at both doses. In the drug-against-drug panels they sit near zero for the
other drug.

# 4. FS76: lithium and magnesium

## 4.1 Magnesium changes the pull-down broadly and reproducibly

![A: Mg^2+^ 2.5 against 0.25 mM, averaged over all Li doses. B: the same profile in two disjoint halves of the replicates. C: the Mg effect against the protein's pull-down level, with binned means. D: by protein class.](../figures/FS76/mg_effect.png){width=100%}

Raising Mg^2+^ ten-fold changes 6,129 of 8,855 proteins at q <= 0.05: 3,503 up and 2,626 down. The
spread is 0.49 log2 (sd). The profile is almost perfectly reproducible between halves (reproducible
share 0.98, panel B). That is the largest single effect in any HSPB1 experiment so far, and it is the
first principal component of the whole FS76 data (64% of all variance).

The bait itself falls 0.35 log2. Ribosomal proteins fall (cytosolic -0.24, mitochondrial -0.20,
median), and integral membrane proteins rise (+0.19). These classes explain only 4.4% of the Mg
effect, and the pull-down level explains 7.9%, against a ceiling of 98%. So most of the Mg effect is
reproducible but unexplained. It is a good target for the sequence models. It does not resemble the
salt effect of GA_20 (section 5, r 0.11), so it is not a generic ionic-strength effect.

## 4.2 Lithium moves a small set of proteins, almost all down

![Every Li hit: each sample against Li concentration (log scale; 0 drawn at 0.09 mM), blue at 0.25 mM Mg^2+^, orange at 2.5 mM. [Mg2+] marks proteins annotated as Mg^2+^-dependent in UniProt.](../figures/FS76/li_dose_curves.png){width=100%}

The dose trend is the slope of the six arm means on log10 Li: the log2 change per ten-fold more
lithium. The lab's spline dose model is in the workbook. Our trend and per-dose tests agree with it on
the strong hits, and our per-dose estimates match its estimates (r 0.92-0.96).

| Test | Proteins at q <= 0.05 |
|---|---|
| Li dose trend at Mg 0.25 mM | 28 |
| Li dose trend at Mg 2.5 mM | 2 (GET4, PC) |
| any Li dose differs from none, Mg 0.25 / 2.5 | 22 / 5 |
| trend differs between the two Mg levels | 8 |
| Mg 2.5 against 0.25 | 6,129 |

Thirty proteins pass at least one Li test. Twenty-five of them go down. Twenty-eight of the 30 keep
their sign at low Mg when any one replicate is left out. The exceptions are INTU, carried by one
10 mM sample, and UHRF1.

Across all proteins the Li profile is not reproducible (reproducible share 0.14 at low Mg and -0.20
at high Mg; shuffled-label p 0.31 and 0.81). As in FS73, lithium acts on a few proteins and leaves
the rest of the pull-down alone.

The hits are coherent:

- **Mg^2+^-dependent enzymes.** FAHD2A, NT5C3B, PGM3, DUT, LAP3, ADK, PC, GDPD1 and HDHD5 are
  metal-dependent hydrolases, mutases or kinases. Phosphoglucomutase-family enzymes such as PGM3 are
  classic lithium targets.
- **The BAG6 complex.** BAG6, GET4 and UBL4A form one complex. All three fall steeply, as does GET3,
  the ATPase they hand tail-anchored proteins to. BAG6 is itself a holdase chaperone.
- **The textbook lithium targets do not move.** GSK3B, IMPA1, IMPA2, INPP1 and BPNT1/2 change by at
  most 0.15 log2 and are not significant. So the pull-down does not report inhibition as such.

## 4.3 The Li effect disappears at high Mg

![A: reproducible share of the Mg effect and of the Li dose trends across all proteins. B: Li trend at low against high Mg on disjoint replicates, all proteins. C: the 30 Li hits only; the dotted line is equal effect at both Mg levels.](../figures/FS76/li_reproducibility.png){width=100%}

All 30 hits respond less at 2.5 mM Mg^2+^ than at 0.25 mM. Their trends at high Mg are 0.17 times
their trends at low Mg [95% CI 0.10, 0.24]. The trends correlate (r 0.67), so the same proteins
respond, just less. This is what competition predicts. Li^+^ can occupy a Mg^2+^ site only while the
site is empty, and a ten-fold excess of Mg^2+^ fills it first.

## 4.4 Lithium acts on these proteins the way Mg^2+^ does

![A: share of proteins annotated as Mg^2+^-dependent (UniProt cofactor or keyword) among the N strongest Li responders, as a multiple of the share among all proteins. Dashed: Zn^2+^-binding proteins, a control class. B: Li trend at low Mg against the Mg effect, all proteins, disjoint replicates. C: the Li hits: change at 10 mM Li (low Mg) against the Mg effect, the two from different arms.](../figures/FS76/li_mg.png){width=100%}

| Test | Result |
|---|---|
| Mg^2+^-dependent proteins among the top 30 / 50 Li responders, low Mg | 27% / 24%, against 7.6% of all proteins (3.5x and 3.2x; p 0.001 and 0.0003) |
| same at high Mg | 17% / 14% (p 0.07, 0.08) |
| Zn^2+^-binding proteins (control) among the top 30 / 50 | 3% / 4%, against 11.5% (depleted) |
| Mg effect of the 30 Li hits | median -0.21 vs +0.05 for all proteins (p 0.005) |
| 10 mM Li effect against the Mg effect, Li hits | slope 0.72 [0.41, 1.03], r 0.67; 22 of 30 in the same direction |

Two of these results point the same way. The proteins lithium lowers at low Mg^2+^ are the ones that
raising Mg^2+^ also lowers. And 10 mM lithium reproduces about 70% of what a ten-fold rise in Mg^2+^
does to them (panel C, and the dose curves: the blue curves fall toward the orange baseline). Read
plainly: on these proteins Li^+^ fills the metal site that Mg^2+^ would fill, and a metal-loaded
enzyme binds HSPB1 less. This again fits "ligand-bound means less chaperone-bound". Here the ligand
is the metal ion.

Eight of the 30 hits do not follow this pattern, and two of them stand out. PDHA1 and PDHB, the two
subunits of the pyruvate dehydrogenase E1 enzyme, both rise with Li at low Mg^2+^ (+0.30 and +0.21 at
10 mM). Raising Mg^2+^ leaves them unchanged (+0.02, -0.02). E1 binds its thiamine diphosphate
cofactor through Mg^2+^. A Li^+^ in that site is not a substitute for Mg^2+^: it may leave the
cofactor loosely held and the enzyme more open, so more comes down. This is the opening mechanism,
with the metal ion as the ligand. Two partners of the SKI complex, SKIC2 and SKIC3, fall with Li but
rise with Mg^2+^, which is another sign that Li^+^ does not always act like Mg^2+^.

Across all proteins, Li does not resemble the Mg effect (panel B, r 0.03). Mg^2+^ acts on thousands of
proteins and through many routes, while Li^+^ mimics it only at a few vacant metal sites.

The Mg^2+^ annotation is incomplete. Neither GDPD1 nor HDHD5 counts as Mg^2+^-dependent here, though
their families use divalent metals. The enrichment is therefore probably understated.

## 4.5 Lithium effects that Mg^2+^ does not predict

The Mg^2+^ relation above explains most Li hits. To find the rest, each protein is held to the most
generous version of that explanation: 10 mM Li at 0.25 mM Mg^2+^ doing everything that 2.5 mM Mg^2+^
does. A Li effect counts as Li-specific only if two conditions hold:

- the Li effect itself is significant (q <= 0.05 over all proteins);
- it exceeds that full-mimicry prediction in its own direction (p <= 0.05).

![Li effect at 10 mM (low Mg) against the Mg effect, all proteins. The shaded wedge, between zero and the identity line, is what Mg mimicry can explain, from none to complete. Blue: Li hits inside it. Red: Li hits outside it.](../figures/FS76/li_specific_scatter.png){width=70%}

Of the 27 proteins with a significant 10 mM Li effect, 15 fall inside the wedge (Mg-like) and 12
fall outside. One of the 12, INTU, was detected in a single 10 mM sample and is dropped. The other 11
are graded with dose: the 3 and 5 mM arms, which are separate samples, move the same way (dose
correlation -0.77 to -1.00, ATF6B -0.60). Each survives leaving out any replicate.

![Dose curves of the Li-specific proteins (blue: 0.25 mM Mg^2+^, orange: 2.5 mM), and two Mg-like hits that were also flagged at high Mg.](../figures/FS76/li_specific_curves.png){width=100%}

| Kind | Protein | Li 10 mM (log2) | Mg effect | What it is |
|---|---|---|---|---|
| Mg does nothing | SLC9A7 | -1.03 | -0.01 | NHE7, a Na^+^(K^+^)/H^+^ exchanger: its transport site binds Na^+^ and Li^+^ |
| | PDHA1 | +0.29 | +0.06 | pyruvate dehydrogenase E1$\alpha$; Mg^2+^-thiamine diphosphate site (PDHB, its partner, also rises) |
| | HMGCS1 | -0.28 | +0.08 | HMG-CoA synthase |
| | GARS1 | -0.24 | -0.02 | glycyl-tRNA synthetase (ATP) |
| | DNM1L | -0.24 | -0.07 | DRP1, mitochondrial fission GTPase |
| | ATF6B | -0.73 | +0.11 | ER-stress transcription factor (weakest grading) |
| Opposite to Mg | SKIC2 | -0.22 | +0.28 | SKI complex RNA helicase (ATP) |
| | SKIC3 | -0.22 | +0.24 | SKI complex scaffold, SKIC2's partner |
| Beyond the Mg level | TSR1 | -0.69 | -0.21 | ribosome-assembly factor (GTPase-like fold) |
| | PC | -0.39 | -0.23 | pyruvate carboxylase (ATP, biotin; activated by K^+^) |
| | GET4 | -1.43 | -1.22 | BAG6 complex; mostly Mg-like, slightly beyond |

These proteins point to two ways lithium could act that Mg cannot mimic.

- **A monovalent-cation site.** In this design Li^+^ replaces Na^+^. SLC9A7 transports Na^+^ and
  Li^+^, and PC is activated by monovalent cations. Here the change could come from Li^+^ in the site
  or from losing Na^+^; the design cannot tell which.
- **Li^+^ alongside Mg^2+^ on a nucleotide.** Five of the 11 bind ATP or GTP (TSR1, DNM1L, GARS1,
  SKIC2, PC), against 15% of all proteins (p 0.017). Li^+^ is known to bind together with Mg^2+^ on the
  phosphate chain of ATP and GTP rather than replace it, and raising Mg^2+^ would not reproduce that.
  This is a lead, not a result. The set is small, and it is not clearly different from the Mg-like
  hits (3 of 15; p 0.22).

Two pairs of complex partners move together: SKIC2 with SKIC3, and PDHA1 with PDHB. A single noisy
protein would not do that, so the pairs make these calls more credible.

**The high-Mg arms are not fully consistent.** At 2.5 mM Mg^2+^, the 3 and 5 mM Li arms carry a faint
copy of the low-Mg Li profile among the top 50 responders. It is about one sixth of the low-Mg size
(r 0.44 and 0.62), and all four 5 mM samples show it. That would fit a small residual Li effect when
the Mg sites are mostly full. But the 10 mM arm at high Mg shows none of it (r -0.12), so many curves
dip at 5 mM and rebound at 10 mM. A real effect should not vanish at the highest dose. The Li effects
"at high Mg" (GET4, PC, TSR1, ALDH18A1, ADK) therefore lean on the 5 mM dip and are tentative.

# 5. Across screens

![HSPB1 control pull-downs on the 5,194 proteins quantified in all four (log2, centred). Each panel has the regression line, r and slope.](../figures/cross/control_levels.png){width=78%}

| Pair | r |
|---|---|
| FS76 Mg 0.25 vs FS76 Mg 2.5 (same experiment) | 0.91 |
| FS73 control vs FS76 (no Li) | 0.67-0.72 |
| GA_20 0 mM salt vs FS73 / FS76 | 0.53 / 0.65-0.67 |
| Mg effect (FS76) vs salt effect (GA_20, 75 or 150 mM) | 0.11 |

The HSPB1 pull-down is similar but far from identical between experiments. Across experiments the
control profiles share 28-52% of their variance; within FS76 the two Mg arms share 83%. Two things
follow. A comparison between screens should be made on effects, each against its own control, not on
raw levels. And lysis buffer, Mg^2+^ and salt are large enough variables that they should be recorded
for every screen. Mg^2+^ and NaCl change the pull-down in largely different ways.

# 6. Critical evaluation

- **Few replicates.** With 3-6 replicates per arm, a test with shuffled labels cannot reach p below
  0.03-0.12. So the proteome-wide reproducibility tests can only say "no broad effect". They cannot
  confirm narrow ones. The hits rest on the per-protein tests, on leaving out each replicate, on dose
  agreement, and on biology that was not used to call them. All four agree for lamotrigine and for
  lithium.
- **Several tests per screen.** The 30 Li hits are the union of four tests at q <= 0.05 each, so the
  false discovery rate of the union is somewhat above 5%. The weakest entries are UHRF1, DOLPP1 and
  INTU: none has a significant trend, and INTU and UHRF1 fail when one replicate is left out. They
  should be treated as unconfirmed.
- **Variance shrinkage is weak.** It borrows little across proteins here (prior df 2.3 and 1.8),
  because noise varies a lot between proteins. The q values are close to those of ordinary t-tests.
  Our hit counts are similar to the lab's (FS76 10 mM at low Mg: 27 vs 21, 19 shared), with
  differences at the margin from the replicate term.
- **Pull-down only.** Without input, "less in the pull-down" cannot be split into less binding and
  less in solution. Drug-induced precipitation of DCK or DHFR would look the same here. An input
  measurement of the lamotrigine and 10 mM Li arms would settle it.
- **A rise has two explanations as well.** More of a protein in the pull-down can mean the drug
  opened it, so HSPB1 binds it. It can also mean the drug made it aggregate or condense, so it
  pellets with the beads with or without HSPB1. The second is most plausible for FUS and EWSR1. A
  bead-only pull-down with and without carbamazepine separates the two.
- **The Mg effect could be partly a batch effect.** It is huge and perfectly reproducible, and the
  bait itself drops 0.35 log2. If the two Mg arms were prepared or run as separate batches, part of
  the effect could be processing. The data cannot tell; the run order can.
- **The nuisance patterns are real but not removable here.** They track how much bait each sample
  captured. A correction learned and applied on the same samples manufactured reproducibility. The
  cross-fitted correction changed nothing. Spiking a fixed amount of a foreign protein into each
  sample would give a correction that does not come from the data being corrected.

# 7. Questions for the lab and next steps

1. Were the two Mg^2+^ arms processed and injected interleaved, or as blocks? This decides how much of
   the 6,129-protein Mg effect is chemistry. Please also check the 2.5 mM Mg / 10 mM Li arm: it
   lacks the faint Li effect that the 3 and 5 mM arms at the same Mg show.
2. What is the base NaCl concentration in FS76? At 10 mM Li, how much of the NaCl was replaced? A
   control that removes the same NaCl without adding Li (choline or K^+^ in its place) would separate
   Li-specific effects (SLC9A7, PC) from effects of losing Na^+^.
3. Which vehicle was used for the drugs in FS73, and does the control arm contain it?
4. Is input (lysate) available for any arm? The lamotrigine 100 $\mu$M and Li 10 mM / Mg 0.25 mM arms
   would show whether a fall in the pull-down is a fall in binding.
5. To tell opening from stabilising directly, three experiments would help:
   - limited proteolysis (LiP-MS) or a thermal shift on the same lysate with drug: an opened protein
     is cut more and melts lower, a stabilised one the reverse;
   - a bead-only pull-down with carbamazepine 100 $\mu$M, which shows whether the risers need HSPB1;
   - for PDH, Li titrated against thiamine diphosphate, which should reverse the rise if Li acts by
     loosening the cofactor.
6. Next on our side:
   - a sequence model for the Mg effect, which is reproducible (ceiling 98%) and mostly unexplained
     by protein class (4.4%);
   - running GA_20, GA_33 and the GA_22/24 supernatants through the same pipeline, so every screen
     has the same QC, reproducibility and hit-check figures.

*Code: `code/pdpipe/` (pipeline), `code/fs00_convert.py` (workbook to table), `code/fs01_anticonvulsants.py`,
`code/fs02_lithium.py`, `code/fs03_cross.py`. Per-protein results: `data/fs_data/FS73/contrasts.tsv`,
`data/fs_data/FS76/fs02_tests.tsv`.*
