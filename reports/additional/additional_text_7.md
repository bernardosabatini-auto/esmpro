---
title: "Additional text 7: does complex membership explain the heat response?"
date: "3 October 2026"
geometry: margin=2.2cm
fontsize: 10pt
---

# Question

Measured thermal stability explains only about 1 % of the 43 vs 35 °C response (additional text 1).
One hypothesis for why: heat disassembles protein complexes, and the released subunits become DNAJ
clients whatever their own stability. It predicts that (1) at matched stability, complex subunits
gain more co-chaperone association than other proteins, (2) subunits of the same complex move
together, and (3) both effects are weaker at 37 °C, where little is perturbed.

# Data

Human complexes from the EBI Complex Portal (curated; 2,167 complexes with at least two protein
subunits after resolving sub-complexes; 3,478 proteins). CORUM, a larger catalogue, could not be
downloaded. 2,471 of the 8,217 GA_33 proteins with a heat response are subunits of at least one
complex. Meltome melting temperatures as in additional text 1.

# Results

**1. Complex subunits gain more at 43 °C, at matched stability.** Difference in the 43 vs 35 °C
response, subunits minus other proteins, adjusted for melting temperature (quadratic), level in the
35 °C pull-down (quadratic) and nuclear, cytoplasmic and membrane location; 95 % interval over
sequence clusters:

| | Raw | Adjusted |
|:--|--:|--:|
| Heat 43 °C | +0.11 (p 7e-11) | **+0.17 [+0.12, +0.22]** |
| Heat 37 °C | +0.03 (p 5e-8) | +0.04 [+0.02, +0.05] |

The effect is heat-dependent, as predicted. It is concentrated at intermediate stability:

| Melting temperature third | Subunits | Other proteins |
|:--|--:|--:|
| Least stable | +0.21 (812) | +0.30 (1,563) |
| Middle | **+0.45** (796) | +0.28 (1,585) |
| Most stable | +0.15 (671) | +0.12 (1,704) |

Subunits show the stability window of additional text 1 more sharply than other proteins: they gain
most when moderately stable and least when least stable. But membership does not account for the
window itself; the curvature in melting temperature is unchanged when membership is added to the model
(-0.063 against -0.067).

**2. Subunits move together, but at both temperatures.** The responses of a complex's detected
subunits are more alike than those of random sets of the same size (804 complexes with at least three
detected subunits):

| Share of the response variance shared within a complex | 43 °C | 37 °C |
|:--|--:|--:|
| Against random proteins | 37 % | 36 % |
| Against random subunits of other complexes | 27 % | 29 % |
| Same, after removing level, Tm and location | 24 % / **14 %** | 36 % / **31 %** |

*p <= 0.002 throughout.* Coherence is as strong at 37 °C as at 43 °C, so it is not a heat effect. Its
most likely source is that pull-downs capture stable complexes as units: when the co-chaperone holds
one subunit, its partners come along. Once level and location are removed, coherence relative to other
complexes' subunits is lower at 43 °C (14 %) than at 37 °C (31 %), which is what partial dissociation
of complexes by heat would produce, but this comparison alone does not establish it.

**3. Which complexes move.** Mean 43 vs 35 °C response of the detected subunits:

| Gain most | | Lose most | |
|:--|--:|:--|--:|
| eIF3 translation initiation factor (13) | +2.71 | Retention and splicing complex (3) | -1.73 |
| RNase H2 (3) | +2.68 | CDK-activating kinase (3) | -1.57 |
| COPII vesicle coat (8) | +2.53 | Chromosomal passenger complex (4) | -1.13 |
| COP9 signalosome (8) | +2.49 | Nucleosome (3) | -0.97 |
| GINS (3) | +2.45 | | |

Cytoplasmic machinery for translation initiation, vesicle traffic and protein degradation gains most;
chromatin-bound nuclear complexes lose most, in line with the class effects found earlier.

# Interpretation

Complex membership matters in the direction the hypothesis predicts. Subunits gain more co-chaperone
association with heat at matched stability, abundance and location, and do so most when moderately
stable. But the effect is small. A +0.17 shift carried by 30 % of proteins accounts for well under 1 %
of the variance of a response whose spread is 1.1 log2 units, comparable to melting temperature. Neither
stability nor complex membership explains much of the heat response; protein class and the sequence
model explain far more.

# Critical evaluation

- **Catalogue coverage.** Complex Portal is curated but incomplete; many real complexes are missing, so
  some "non-members" are subunits. That dilutes the membership effect rather than creating it.
- **Disassembly is inferred, not observed.** Pull-downs do not show whether a complex is intact. The
  lower coherence at 43 °C is suggestive only.
- **Coherence has a technical explanation.** Co-purification of intact complexes produces coherence at any
  temperature, which fits the 37 °C result.
- **The input lysate** would show whether gaining subunits are released from complexes (constant total
  level) or newly abundant.

# Methods files

`code/ad07_complexes.py`; results in `data/ga_data/ad07_results.json`.
