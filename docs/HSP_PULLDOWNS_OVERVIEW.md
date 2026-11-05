---
title: "HSP pull-downs: dataset, goals and methods"
date: 5 November 2026
---

# The dataset

Six affinity pull-down experiments of human chaperones out of cell lysate, quantified by DIA mass
spectrometry. Together they give 33 analysis targets over a union of 11,570 proteins. Every manipulation is
in vitro: the bait is added to lysate, the perturbation is applied in the tube, and the beads are pulled.

| Run | Bait | What was varied | Fractions measured | Proteins | Targets |
| --- | --- | --- | --- | ---: | ---: |
| GA_20 | HSPB1 | KCl 75, 150 mM | pull-down | 7,176 | 3 |
| GA_22 | HSPB1 | KCl 60, 120 mM; staurosporine 0–10 µM | pull-down + post-IP supernatant | 8,950 | 9 |
| GA_24 | HSPB1 | KCl 60, 120 mM; NAD / NADP | pull-down + post-IP supernatant | 9,001 | 9 |
| GA_33 | DNAJA1, DNAJB11 | 37 and 43 °C vs 35 °C; ± 10 µM staurosporine | pull-down + pre-IP lysate | 8,836 | 9 |
| FS_73 | HSPB1 | anticonvulsant screen | pull-down | 6,611 | 1 |
| FS_76 | HSPB1 | Mg²⁺ 2.5 vs 0.25 mM | pull-down | 8,017 | 2 |

A "target" is one quantity a model is asked to predict: either an absolute level (how much of a protein the
beads carried) or a contrast (how that level moved under a perturbation). GA_22 and GA_24 contribute three
fractions each — the pull-down, the unbound supernatant, and the bound:free partition that divides one by
the other — which is why they carry nine targets apiece.

Replication is five biological replicates per condition with about three injections each. The drug arms
exist but sit outside the campaign, which used the drug-free series only.

**No experiment includes a bait-free IP.** Nothing in the dataset passes over the beads without the bait, so
bead carry-over and genuine binding are not separated anywhere. That gap shapes several of the results below.

# The goals

The starting question was whether a protein's amino acid sequence predicts how much of it a chaperone pulls
down, and whether it predicts how that changes when the tube is perturbed.

The appeal is that sequence is known for every protein. If a model trained on measured proteins generalises
to unmeasured ones, it turns a pull-down into a proteome-wide prediction, and the features it relies on say
something about what the chaperone recognises.

The campaign was built so that a negative answer would be as informative as a positive one, which meant
committing to two things up front:

1. **Hold out by sequence similarity, not at random.** Homologues clustered at 30 % identity go into the
   same held-out block, so a model cannot score well by recognising a close relative it already saw.
2. **Report accuracy against a ceiling, not in absolute terms.** Every target gets a split-half reliability
   estimate, and results are quoted as R² divided by that ceiling — the share of the *reproducible* variance
   explained. A target measured at reliability 0.65 cannot be predicted better than that, and saying so
   prevents a noisy measurement from reading as a failed model.

That framing turned the second, larger question into the one worth asking: **what does sequence miss, and is
the missing part structured or is it noise?** Most of the work ended up here, because the answer turned out
to be that the residual is strongly structured — and the structure is not what we first assumed.

# Methods we tried

Five families of method, run in roughly this order. Each was chosen to answer what the previous one left
open.

**1. Sequence models.** Protein sequences were embedded with ESMC-6B, reading two internal layers (50 and
80) plus a sparse autoencoder over them. Two predictors were fitted on those features: ridge regression, and
a multi-task MLP trained on all 33 targets at once so that targets could share structure. Folds come from
MMseqs2 clustering at 30 % identity, five held-out blocks, with the ridge penalty chosen on an inner split
held out the same way.

**2. A named-property baseline.** The same targets predicted from interpretable quantities instead: measured
abundance, sequence length, charge per residue, pI, hydrophobicity, predicted disorder, melting temperature
from a public meltome, and UniProt keywords. This exists to ask whether the embedding is doing anything a
short list of properties could not.

**3. The complex-mate correlation.** For each protein, take its residual — what the sequence model missed —
and correlate it against the mean residual of the other subunits of its Complex Portal complex, within the
same held-out block. This asks whether the misses are independent across proteins or shared within
assemblies. We came to call the phenomenon *the complex effect*.

**4. Three structural tiers.** Having found a large complex effect, the natural hypothesis was that it is an
interface property — something about which surface is exposed or occluded. Three independent tests:
AlphaFold monomer surface features (electrostatics, salt bridges); burial at native interfaces measured on
29,816 experimental PDB assemblies; and Boltz-2 co-folding of candidate chaperone–client pairs.

**5. A four-part decomposition.** When the structural tiers all returned nothing, we stopped asking what the
complex effect *is made of* structurally and asked where in the protocol it arises:

- **the measurement-stage gradient** — the same correlation computed on the lysate, the unbound supernatant,
  the pull-down and the partition. A stage the sample never reached cannot have created the effect;
- **a flexible abundance control** — abundance projected out as 155 decile indicators and their interactions
  rather than one linear term, so a nonlinear capture curve cannot hide inside it;
- **abundance-discordant partners** — the correlation recomputed using only complex-mates drawn from a
  *different* abundance quintile. If co-abundance mediates the effect, this collapses it;
- **co-capture asymmetry** — whether a protein's own measured binding strength, taken from an independent
  run, predicts how much it shares with its complex-mates.

Throughout, uncertainty comes from a cluster bootstrap over the same 30 % identity groups, so error bars are
not inflated by homologues counted twice.

# What came of them

Sequence predicts about a quarter of what is reproducibly there, structure adds nothing, and the large
shared residual turned out to be mostly a property of the lysate rather than of the capture.

| Method | Result | Reading |
| --- | --- | --- |
| Ridge on ESMC-6B | mean R²/ceiling **0.245** | sequence carries real but partial signal |
| Multi-task MLP | **0.273** | sharing across targets adds 0.028 — small and real |
| Named properties alone | abundance is the single strongest predictor of a baseline pull-down | much of "binding" is how much protein was there |
| Complex-mate correlation | mean r = **0.336** | subunits of one complex miss the prediction together, in the same direction |
| AlphaFold surface features | **0.000** added | not surface electrostatics |
| Native-interface burial (29,816 assemblies) | no residual explained | not steric competition for an interface |
| Boltz-2 co-folding | nothing added; half the score was run-to-run noise | predicted interfaces do not carry it |

The decomposition is where the interpretation changed. The complex effect is **largest in the supernatant**
(0.607 at KCl 60), falls in the pull-down (0.401), and falls again in the partition (0.305) — and the
supernatant never passes over the beads. Capture therefore cannot have created it. Replacing the linear
abundance control with decile indicators moved the mean only 0.336 → 0.304, but restricting partners to a
*different* abundance quintile moved it to **0.214**, with the positive control (`GA33_input`, which is
lysate abundance itself) collapsing to −0.03.

So roughly a third of the complex effect is co-abundance, most of the remainder is lysate-level
co-behaviour, and only about 0.20–0.23 is left that could be chaperone biology. This also explains, after
the fact, why all three structural tiers were null: they were looking for an interface property, which could
only ever have been the smallest of the four components.

One earlier conclusion was withdrawn on this evidence. An earlier section had claimed the complex effect "is
not abundance" on the strength of a linear projection; that control was too weak for a partner-mean
statistic, and the claim was corrected rather than defended.

# Picking it up

**First, a warning about the repository.** It holds two unrelated projects that share a directory and
nothing else. The root `CLAUDE.md`, every `code/gate*.py`, and most of `reports/` belong to ESM-ProteinAE, a
protein structure predictor. None of it applies to the pull-downs. Read `CLAUDE.md` expecting orientation
and you will be thoroughly misled.

| What | Where |
| --- | --- |
| Campaign report, 31 pages | `reports/campaign/hsp_mlp_campaign_report.md` (+ `.pdf`, `.html`) |
| GA_33 report | `reports/protein_response_report.md` |
| Supplementary texts | `reports/additional/additional_text_1–9.md` |
| Campaign analysis scripts | `code/cm01`–`cm26` (34 files) |
| GA_33 analysis scripts | `code/ad01`–`ad15` (24 files) |
| Reusable pipeline | `code/pdpipe/study.py`, `code/pdpipe/levels.py` |
| Pipeline documentation | `docs/PIPELINE.md` |
| Preprocessed data tables | `data/export/`, described by `docs/DATA_EXPORT.md` |
| Study configs | `code/pdpipe/configs/studies/` — `TEMPLATE.yaml`, `ga22_24.yaml` |
| Campaign data | `data/campaign/` — `targets.npz`, `targets_meta.json`, `folds.npz`, `oof.npz` |
| Job scripts | `slurm/study.sbatch`, `slurm/cm*.sbatch` |

To run the pipeline on a new lysate:

```
python -m pdpipe.study --study <name> --check     # what your data can and cannot support
python -m pdpipe.study --study <name> --dry-run   # the plan, nothing executed
sbatch slurm/study.sbatch <name>                  # the full run, on the cluster
```

Two cluster facts that would otherwise cost a day each:

- Submit to account `kempner_bsabatini_lab`, partition `kempner_rtx`. The CPU account `bsabatini_lab` has
  MaxSubmit 0, so even wholly CPU-bound analysis has to hold a GPU it will not use.
- The `dendritic_env` torch predates the Blackwell cards. `torch.cuda.is_available()` returns True and then
  every kernel fails with "no kernel image is available". The pipeline probes with a real kernel and falls
  back to CPU, saying so.

Start with the campaign report's *How to read this report* section. It defines the six terms — target, fold,
ceiling, residual, complex-mate, complex effect — that everything else is written in.

# Where things stand

The campaign is closed and written up as a 31-page report
(`reports/campaign/hsp_mlp_campaign_report.md`). All analysis is reproducible on new data through
`pdpipe.study`, which reruns every level and gates each one on which fractions exist — a study with only a
pull-down runs what a pull-down supports and says plainly what the missing fractions would have bought.

Three things are settled enough not to revisit: structure adds nothing over sequence; the complex effect is
a lysate property, not a capture artefact; and measured abundance is the strongest single predictor of a
baseline pull-down.

What is open divides into questions for the bench and one measurement design issue:

- **The missing control is a bait-free IP.** It is the one fraction no experiment has, and it would do two
  things at once: separate genuine binding from bead carry-over, and settle the capture question directly
  instead of by the indirect staging argument. The pipeline already handles it the moment it exists.
- **The perturbation steps are too small to fit curves.** The noise floor is ~0.17 log2 units per protein
  and constant; signal is set entirely by step size. A 2 °C step gives signal-to-noise 1.3 and is wasted MS
  time, while 8 °C gives 5.8. The salt titration has not saturated at 120–150 mM, so no release midpoint can
  be fitted from the three points we have.
- **Open questions for the lab:** whether salt is present during binding or only in the wash (this changes
  the interpretation entirely), the units on NAD(P), whether equal volume or equal protein was injected, and
  two supernatant blocks that failed.

The largest single opportunity is temperature. The 43 °C data is measured beautifully (reliability 0.97,
signal-to-noise 5.8), but 43 °C is still the foot of the melting curve for most proteins. Extending to
40–60 °C in 4 °C steps, with a heat pulse followed by capture at a common temperature and a bait titration
to control for competition, would yield a per-protein stability estimate — and one that reports client
recognition rather than aggregation, which is a different and potentially earlier event than classical
thermal profiling measures.
