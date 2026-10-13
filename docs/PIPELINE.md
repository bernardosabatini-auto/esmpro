# Analysing a new lysate

This reproduces every level of the HSP campaign on new data. It is driven by one study config and gates
each level on what was actually measured, so a study with only a pull-down runs what a pull-down supports
and says plainly what the missing fractions would have bought.

```bash
cd code
python -m pdpipe.study --study mynew --check      # what your data can and cannot support
python -m pdpipe.study --study mynew --dry-run    # the plan; nothing is executed
python -m pdpipe.study --study mynew              # run every level the data support
python -m pdpipe.study --study mynew --levels complex,decompose --resume
python -m pdpipe.study --list                     # the levels, their costs and prerequisites
```

On the cluster, where the CPU account cannot submit at all:

```bash
sbatch slurm/study.sbatch mynew                                     # full run: 15 cores is right
sbatch -c 2 --mem 16G slurm/study.sbatch mynew --levels targets,complex   # subset: ask for less
```

The script's 15 cores are sized for `predict`, the one level with real CPU parallelism — the ridge
eigendecomposes a 5,120 x 5,120 Gram per target per fold and a full run measures 87 % CPU efficiency.
Every other level is serial and a subset run measures 11 %, which is what the cluster's Job Defense Shield
reports on. Command-line flags beat the script's `#SBATCH` directives, so trim them when you run a subset.
GA_22 peaked at 1.6 GB of memory; the 48 GB default is headroom for a larger study, not a measurement.

---

## 1. Describe the experiment

Two layers of config, because they answer different questions.

**Per fraction** (`code/pdpipe/configs/<name>.yaml`) — *how do I parse this MS report?* One file per MS
run: the source path, a regex that pulls the design fields out of the sample names, which fields define a
condition, and the contrasts of interest. Copy `ga22.yaml` (DIA-NN) or `fs73.yaml` (proteoDA workbook).
The sample-name regex is the only fiddly part; everything else follows from it.

**Per study** (`code/pdpipe/configs/studies/<name>.yaml`) — *which fractions belong together, and what do
I want from them?* Copy `TEMPLATE.yaml`. `ga22_24.yaml` is a worked example.

```yaml
name: MYSTUDY
bait: P04792
fractions:
  ip:    myrun_ip      # REQUIRED
  sup:   myrun_sup     # optional
  input: myrun_input   # optional
  mock:  myrun_mock    # optional
reference: "0|0"
contrasts: [KCl60_stau0]
conditions_include: ['\|0$']
options: {layers: [50, 80], mlp: false, structure: false}
```

---

## 2. What each fraction buys

This is the part worth reading before the experiment rather than after.

| Fraction | What it is | What it enables |
|---|---|---|
| `ip` | the pull-down | required; baselines and perturbation contrasts |
| `sup` | post-IP supernatant | the **bound:free partition**, and the measurement-stage gradient of report §3g |
| `input` | pre-IP lysate | **enrichment**, and abundance measured in *this* lysate rather than borrowed from PaxDb |
| `mock` | **bait-free IP** (beads only, or untagged control) | **specific binding** (IP − mock), and a *direct* test of the capture hypothesis |

**`sup` is the cheapest useful addition** — it is the same tube, already in hand after the IP. With it you
get the partition, which divides out how much protein was present, and you get a second measurement stage,
which is the minimum for the §3g comparison that showed the complex effect is mostly a lysate property.

**`input` matters more than it looks.** Report §4 found measured abundance to be the strongest single
predictor of a baseline pull-down. Without a pre-IP lysate the pipeline falls back to the supernatant
(same tube, but depleted exactly for the best binders) and then to PaxDb (other cells, other preparation).
The `--check` output names which one it used, and the report records it.

**`mock` is the fraction the campaign never had, and it is the one worth adding.** Two distinct things:

1. **Specific binding.** `spec = ip − mock` separates genuine binding from bead carry-over. Neither
   partition nor enrichment can do this: both divide out *amount*, not *stickiness*. A protein abundant in
   the lysate and sticky to agarose looks like a strong binder in every other readout.
2. **A direct capture control.** Report §3g had to argue indirectly — complex-level structure is *higher*
   in the supernatant than in the pull-down, so capture cannot have created it. A bait-free IP settles it
   outright: it passes over the same beads with no bait, so any complex effect it shows was made by
   capture itself. If `mock` is present the pipeline runs that comparison and prints it.

---

## 3. The levels

| Level | Does | Needs | Cost |
|---|---|---|---|
| `load` | parse each fraction, QC, replicate structure | — | minutes |
| `targets` | baselines, contrasts, derived fractions, split-half ceilings | `load` | minutes |
| `embed` | ESMC-6B states + SAE for proteins not already cached | — | ~1 min / 1,600 proteins, GPU |
| `folds` | MMseqs2 30 % clustering into 5 held-out blocks | — | minutes |
| `predict` | out-of-fold ridge on the embedding | targets, embed, folds | minutes |
| `properties` | abundance/length baseline, and what sequence adds | targets, folds | minutes |
| `complex` | the complex effect (§3c), permutation null, bootstrap | targets, predict | ~10 min |
| `decompose` | what the complex effect is made of (§3g) | complex, **≥2 fractions** | ~5 min |
| `structure` | AlphaFold / experimental interfaces / co-folding (§3d–3f) | predict | **hours of GPU** |
| `report` | assemble markdown, render PDF | targets | minutes |

Prerequisites are pulled in automatically: `--levels complex` also runs `load`, `targets`, `embed`,
`folds` and `predict` unless `--resume` finds them already done.

**`structure` is off by default and should stay off.** All three tiers returned nothing across 33 targets,
and §3g then explained why — they were looking for an interface property, which could only ever have
accounted for the smallest of the four components the complex effect is made of. Turn it on to test a new
hypothesis, not as routine analysis.

**Runtime of `predict`.** The ridge eigendecomposes a 5,120 x 5,120 Gram matrix twice per target per
fold (once to choose alpha on a cluster-held-out inner split, once to fit). On GPU that is seconds; on CPU
it is roughly 5-6 minutes per target. `dendritic_env`'s torch predates the Blackwell cards (sm_120), so it
falls back to CPU automatically and says so. For a faster run set `options.layers: [80]`, which is one
layer instead of two: 2,560 features, about eight times quicker, and worth about 0.01 of ceiling less.

**`predict` runs ridge, not the multi-task MLP.** The campaign's MLP beat ridge by +0.028 of ceiling, which
is real but small, and it is a ~10 GPU-h sweep. Set `options.mlp: true` for the instructions to run it.

---

## 4. Outputs

Everything lands in `data/study/<NAME>/`:

```
plan.json          capabilities and the level plan, as run
ip/ sup/ ...       each fraction in standard form (matrix, samples, proteins, meta)
qc.json  qc_*.png  per-fraction QC
targets.npz        the target matrix; targets_meta.json carries each target's ceiling
folds.npz          cluster assignment and the five blocks
oof.npz            out-of-fold predictions
predict.json       R2 and R2/ceiling per target
complex.json       the complex effect per target, with null and bootstrap CI
decompose.json     stage gradient, abundance controls
report.md/.pdf
```

Campaign reference values are printed beside the new ones, so a drift is visible immediately: ridge mean
R²/ceiling 0.245, multi-task MLP 0.273, mean complex effect 0.336, and the abundance series
0.336 → 0.326 (linear) → 0.304 (flexible) → 0.214 (abundance-discordant partners).

---

## 5. Reading the complex effect

Defined fully in the campaign report's "How to read this report". In short: take a protein's **residual**
(measured minus what sequence predicted), average the residuals of the other subunits of its complex, and
correlate the two across proteins. It is ~0.34 in the campaign, meaning subunits of one complex miss the
prediction together and in the same direction.

`decompose` then asks what that is made of:

- **the stage gradient** — the same correlation at each measurement stage. Falling from `input`/`sup`
  through `ip` means the effect is a lysate property; rising means capture created it;
- **a flexible abundance control** — decile indicators rather than a linear term, so a nonlinear capture
  curve cannot hide in it;
- **abundance-discordant partners** — the correlation recomputed using only complex-mates from a different
  abundance quintile. A large drop means co-abundance. In the campaign this was the test that mattered: a
  linear projection said "not abundance" and was wrong by about a third.

**The abundance covariate must not be a copy of the target.** Without an `input` fraction the pipeline takes
abundance from the supernatant, and then for a `sup_*` target — or a `part_*` target, since `part = ip - sup`
— the residual and the covariate carry the same measurement noise. Projecting it out strips shared noise from
both sides and the control reads "co-abundance" when it is not. Those targets are rerun against PaxDb, which
is an independent measurement, and the log marks them `[vs PaxDb: ...]`; with no PaxDb they are excluded from
the mean and the log says so. The campaign was never exposed to this — its `abundance_here` is `GA33_input`,
a separate pre-IP lysate — but a study with only `ip` and `sup` is.

---

## 6. Troubleshooting

**"conditions_include removed every condition"** — the condition key is the fraction config's `condition:`
fields joined by `|`, in order. GA_22 is `KCl|stau` so its reference is `"0|0"`; GA_24 is
`cof|conc|KCl` so its reference is `"CTRL|0|0"`. The error prints the conditions actually present.

**MMseqs2 missing** — set `MMSEQS_BIN`. Without it `folds` falls back to one cluster per protein and warns:
homologs can then straddle blocks and every accuracy number becomes optimistic. Do not report numbers from
a fallback run.

**New proteins not in the embedding cache** — `embed` writes `missing.fasta` and prints the `cm01_embed.py`
command. Run it, then re-run the level.

**A level is declared but not implemented** — the driver says so by name and continues, rather than
failing silently.
