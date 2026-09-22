# Challenge: fold-validation negative (2026-09-16)

CONCLUSION UNDER REVIEW: "Three independent pose-invariant metrics agree the heads emit a generic compact blob, not a fold. Option C, properly tested, does not produce folds."

VERDICT: **premature** (the headline "no folds" is very likely true; the evidence chain
actually used to reach it is broken in the same way as the two previous overturned results)

## OBJECTIONS (ranked)

### 1. [class: signature] Two of the three "independent" metrics are sitting on their own floors. The TM column is censored, not measured.

**Objection:** TM-score was not measured for 80-90% of the proteins in the three
failing arms; those entries were imputed as 0.0. And 3Di identity at 0.146-0.167 is
exactly the background identity between *unrelated real proteins*. Neither metric has
any resolution in this regime, so they cannot corroborate anything.

**Evidence:**
- `notes/gate6_fold_validation_results.json`, field `tm_found`: true_z 100/100,
  mean_z **20/100**, MSE head **10/100**, FAPE head **15/100**. Everything else is
  `tms.get(name, 0.0)` (gate6_fold_validation.py:118). So "TM = 0.039" is
  `15 real values + 85 zeros`, and "TM = 0.014" is `10 real values + 90 zeros`.
- Cause: `foldseek easy-search` was run WITHOUT `--exhaustive-search 1` /
  `--prefilter-mode 2`. `-e inf --max-seqs 2000` relaxes the E-value cut but leaves
  the 3Di k-mer/ungapped **prefilter** on. The prefilter scores on 3Di. The bad arms
  are at 3Di background, so they never reach TMalign. The metric is a *detector*
  whose gate is the other metric — TM and 3Di here are not independent; TM is
  downstream of 3Di.
- Unrelated same-length protein pairs score TM ~0.2-0.3, essentially never 0.0
  (Xu & Zhang 2010). A mean of 0.012 over 100 length-matched pairs is not a
  physically attainable value; it is an artifact signature (mass piled exactly at
  0.0, i.e. 80-90% of the distribution at one characteristic value), not a noise
  distribution.
- 3Di floor, computed just now from the 30 stored ground-truth 3Di strings in
  `notes/gate6_3di_eval_results.json` (435 unrelated GT-vs-GT pairs, same ungapped
  `compare_3di`): **mean 0.142, median 0.131, p90 0.230**. Composition-only chance
  identity (sum p_i^2) = 0.108. So MSE head 0.146, FAPE head 0.165, mean_z 0.167 are
  ALL inside the unrelated-protein background. This also answers the "mean_z beats
  the trained head on 3Di" puzzle: it is not an artifact and not a paradox, it is
  three draws from the same background distribution whose per-pair p90 is 0.23.

**Direction of bias:** both errors push the *same* way — they make the trained head
look worse and flatter than it is. The imputed zeros suppress the FAPE head's TM;
the 3Di floor hides any sub-fold improvement. So the claim "training moved FAPE while
TM and 3Di stayed put" is exactly the claim these two artifacts would manufacture.
Note that on the numbers actually present, TM went 0.012 (mean_z) -> 0.014 (MSE) ->
**0.039** (FAPE head), i.e. a 2.8x move in the same direction as the FAPE improvement.
The table does not show decoupling; it shows a censored metric moving the same way.

**What survives:** `frac TM > 0.5 = 0.00` is almost certainly still correct — a pair
at TM>0.5 would have had high 3Di similarity and would have passed the prefilter.
"No folds" survives. "FAPE is a weak proxy for fold" does not; that claim rests
entirely on the decoupling, and the decoupling is unmeasured.

**Also note** the pseudo-backbone construction is fine as a descriptor input
(positive control 3Di 0.968 / TM 0.997 proves it), but it is the *input to the
prefilter*, so a degenerate blob's 3Di string is what causes the censoring. The
construction is not corrupting the descriptor; it is faithfully reporting that the
blob has no 3Di signal, and Foldseek is then silently declining to align it.

**If true:** the sentence "three independent pose-invariant metrics agree" must be
withdrawn; you have one metric (FAPE) plus two floors. The subsidiary conclusion
"my FAPE is a weak proxy for fold" is unsupported and should not be used to justify
abandoning the loss.

**Resolving test:** rerun `gate6_fold_validation.py` with
`--exhaustive-search 1` (or `--prefilter-mode 2`) so all 100 pairs are TMaligned,
and add two floors: (a) TM and 3Di for 100 *shuffled* GT-vs-GT pairs, (b) TM for the
`random_z` arm. Everything else in the script is unchanged. **~1.5 h** (mostly the
GPU decode of 4 arms x 100 proteins x 20 ODE steps, already known to fit).

---

### 2. [class: power] The training run moved the model through ~2% of the metric's dynamic range, never early-stopped, and used the exact config a previous review already called underpowered.

**Objection:** "Option C, properly tested" overstates what was tested. The loss was
fixed; the run was not.

**Evidence** (`notes/gate6_fape_fixed_results.json`, `notes/gate6_corrected_eval_results.json`):
- Anchors on the corrected FAPE: true_z 0.042, random_z 0.992. Full range 0.950.
- Gap closed from random toward true: warm-start/MSE head 3.9%, mean_z 3.1%,
  epoch-1 FAPE head 5.3%, epoch-10 FAPE head 6.8%. **Ten epochs bought 1.5
  percentage points of a 100-point range.**
- `best_epoch: 10` of 10, patience 4 never triggered, val monotone decreasing at
  every epoch, train loss also still falling (1.764 -> 1.690, no overfit —
  train >> val here because of the 10% unclamped samples). This is a run stopped by
  the epoch counter, not by evidence.
- 10k train / 8.2M params, and MEMORY.md records that a prior challenger already
  flagged precisely "10k data, 3 steps, MSE warm-start" as underpowered, and that an
  80k / 10-step run was started in response. The corrected run reverted to the
  config that was already judged insufficient.
- **Train/eval ODE mismatch:** trained with `ode_steps: 3`, evaluated with
  `ODE_STEPS = 20`. The head is optimising z against the 3-step Euler map; it is
  scored on the 20-step map. Untested, and it can only hurt the head (true_z is
  a fixed point of both; a learned z need not be).

**Direction of bias:** every one of these cuts toward a false negative. None of
them makes the task artificially easy.

**If true:** "Option C does not produce folds" becomes "Option C, run for 10 epochs
on 10k proteins from a frame-committed warm start with a train/eval decoder
mismatch, was still improving monotonically when we stopped it." That is not a
project-level negative. Note this is also the cheapest possible conclusion to
reach — abandoning Option C costs nothing, converging it costs ~14 h/epoch.

**Resolving test, two tiers:**
(a) Cheap sanity, do first: re-score the existing FAPE head at **3** ODE steps to
size the mismatch, and score the epoch-1 vs epoch-10 checkpoints on TM (exhaustive)
to see whether the 1.5-point FAPE move is any structural move at all. **~1 h.**
(b) Credible negative: 80k train, 10 ODE steps, random init, run to genuine early
stop (patience triggered), then fold-validate. If TM stays below the shuffled-GT
floor after that, the negative is real. **~3-5 days wall clock**, unattended.

---

### 3. [class: attribution / implication] The diagnosis and the conclusion contradict each other, and the instrument that would separate them was written and never run.

**Objection:** "My hand-rolled FAPE is a weak proxy for fold" and "Option C does not
produce folds" cannot both be load-bearing. Option C *is* "train on FAPE". If the
training signal is a weak proxy for fold, then what failed is the loss, and the
hypothesis (structural loss through the frozen decoder recovers folds) is untested.
If instead the loss is a good proxy, then FAPE 0.921 is meaningful and the
"weak proxy" clause should go. The log asserts both.

Competing mechanisms for "FAPE moved, RMSD/TM did not", none discriminated:
1. FAPE is fold-blind (the stated cause).
2. `clamp_distance = 20.0` with `error / 10.0` saturates: mean_z (20 A Kabsch) scores
   0.963 and random_z (75 A Kabsch) scores 0.992. A 55 A difference in global error
   is worth 0.03 of loss. Above ~10 A the loss is nearly flat, so it is **not** fold-
   blind — it is fold-*saturated*, which is a different problem with a different fix
   (schedule the clamp / use unclamped early), and which also means gradient toward
   global arrangement is largely zeroed during training.
3. Ca-pseudo-frames built from 3 consecutive Ca are short-baseline and noisy, so the
   per-frame signal is locally dominated even before clamping.
4. The metrics it was compared against are censored (objection 1), so "did not move"
   was never established.

**The instrument already exists and was not run:** `gate6_fold_calibration.py`
(written 06:50 today, 5 min after the validation script) sweeps
sigma in {0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.20} on-manifold and reports
TM / 3Di / FAPE for each. `notes/gate6_fold_calibration_results.json` **does not
exist**. That curve is the only thing that converts a FAPE delta into a TM delta,
and it is what tells you whether FAPE 0.92 is 10% or 99% of the way to hopeless.
The project has now drawn a project-level conclusion about a metric's validity while
the written-but-unrun calibration of that metric sits on disk.

**If true:** the recommended remedy does not follow from the diagnosed cause. If the
cause is clamp saturation, the fix is a clamp schedule, not abandoning Option C.

**Resolving test:** run `gate6_fold_calibration.py` as written (add
`--exhaustive-search 1` first, per objection 1), and add two extra arms at
clamp = 50 A / no clamp to see whether the FAPE ordering of the 4 model arms
survives declamping. **~2 h.**

---

## STRONGEST OPPOSING CASE (i.e. the negative is right)

The positive control is airtight: true_z decodes to TM 0.997, 3Di 0.968, 0.17 A
Kabsch, so the decoder, the pseudo-backbone construction, the PDB writer, the
Foldseek path and the 8-dim channel are all verified sufficient end to end. Nothing
downstream of z is broken. The head's Kabsch RMSD is 15.01 A against a mean_z
blob at 20.17 A and true_z at 0.17 A — Kabsch RMSD is uncensored, needs no
prefilter, and has full dynamic range, and on it the FAPE head has closed roughly
26% of the distance from mean_z to true_z in *superposed* terms while remaining
88x worse than the control and 3.5x worse than ESMFold's 4.33 A. `frac TM > 0.5 = 0`
is robust to the censoring, because a same-fold pair would have passed the
prefilter. Three separate training regimes (MSE-on-z at 3k/20k/100k, deep heads to
10L, and now corrected FAPE) have all landed in the same 14-21 A band, which is the
signature of a model collapsing to the conditional mean rather than of any single
measurement bug. And on scale grounds this is expected: ESMFold freezes the PLM but
trains a folding trunk one to two orders of magnitude larger than 8.2M params, on
PDB plus millions of distilled AFDB structures, with a learned combination over all
PLM layers rather than the last layer alone. An 8.2M head on 10k structures reading
only layer 33 is far below any published configuration that reaches non-trivial
TM-score, so the negative is the expected result and the burden is on the
positive claim.

## CHECKS PERFORMED

All five classes considered explicitly. **Signature:** found — TM mass piled at
exactly 0.0 for 80-90% of entries (imputation artifact, not noise), and 3Di sitting
on a measured 0.142 unrelated-protein background; objection 1. **Power:** found —
1.5 points of a 100-point range, never early-stopped, 10k/3-step/warm-start config
previously judged insufficient; objection 2. **Implication:** found — the
FAPE-vs-TM calibration sweep was coded and never run, and `random_z` FAPE 0.992 vs
`mean_z` 0.963 (a 55 A Kabsch difference worth 0.03 of loss) was reported three
gates ago and never used to bound the metric's dynamic range; objections 1 and 3.
**Attribution:** found — "FAPE is fold-blind" not discriminated against clamp
saturation, short-baseline Ca frames, or censored comparators; objection 3.
**Direction:** checked — every named limitation (warm start, 3-vs-20 ODE steps,
10k data, prefilter censoring, 3Di background) cuts toward a false negative; none
cuts toward a false positive. Additionally checked and found nothing: the 8-dim
channel (true_z at 0.17 A settles it), the pseudo-backbone construction (positive
control settles it, and it is not compressing the low end — it is faithfully
reporting no signal), and TM length-sensitivity across 32-256 residues (a real
statistical concern, and the mean over a length-heterogeneous set is the wrong
statistic, but it is moot while 85% of the values are imputed zeros; report median
and length-stratified TM after the exhaustive rerun).
