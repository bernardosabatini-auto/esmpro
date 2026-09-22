# Challenge: latent calibration curve / project-level negative

CONCLUSION UNDER REVIEW: "Frozen ESM-2 + frozen ProteinAE + a trained head is not a viable path to structure prediction."

VERDICT: **unsupported**

---

## OBJECTIONS (max 3, ranked)

### 1. [class: attribution] The "FAPE" loss is not rotation-invariant. It algebraically reduces to a clamped, un-superposed per-residue coordinate error, so the entire rationale for Option C (escape the PCA-frame problem) was never actually implemented, and every number in the decisive experiment is contaminated by global pose error.

**Objection.** In `gate6_fape_train.py:114-164`, both the predicted and the true point clouds are expressed in the *same* frame `(rot_true, trans_true)`. Therefore

    error_ij = || R_i^T (pred_j - t_i) - R_i^T (true_j - t_i) || = || pred_j - true_j ||

The frame index `i` cancels exactly. The loss is `mean_j clamp(||pred_j - true_j||, 10)/10` — clamped mean absolute coordinate deviation with no superposition, counted N times per residue. Real FAPE places predicted points in *predicted* frames and true points in *true* frames; that difference is the whole source of its alignment-freedom. Verified numerically with the project's own function:

| input | loss |
|---|---|
| identical structures | 0.00001 |
| **perfect structure, rotated 90 deg** | **0.851** |
| **perfect structure, translated 5 A** | **0.861** |
| perfect structure + 1 A local noise | 0.089 |
| trained MSE head | 0.939 |
| trained FAPE head | 0.907 |
| mean-z baseline | 0.965 |
| random-z | 0.993 |

A structurally *perfect* prediction in the wrong pose scores 0.85 — worse than sigma=0.80 isotropic latent noise (0.781, ~11 A RMSD by D1) and only 0.06 away from the trained heads. All models in this project live inside the narrow band 0.85-0.99 that this metric reserves for pose error alone. Two direct consequences:

- The stated reason for choosing Option C over MSE ("FAPE is rotation-invariant, bypasses the PCA canonicalization problem", memory + phase1 log) is false for this implementation. The FAPE head was trained against a loss that still fully penalizes the PCA-frame commitment previously diagnosed as the MSE head's root cause. **The hypothesis Option C was built to test has not been tested.**
- The secondary claim is also unsupported in both directions: "MSE head sits above the isotropic curve -> coherent global frame error" is the correct reading of the evidence, but the follow-on "FAPE training removed the structured frame error, what remains is generic ignorance" cannot be established with a metric that is maximally sensitive to exactly that error and provides no rotation-invariant channel. The FAPE head has never been scored with any pose-invariant structural metric. The 3Di evaluation (14-21% identity vs 83% for reconstruction, `notes/gate6_3di_eval_results.json`, 2026-09-13) predates both FAPE checkpoints and covers only the MSE head.

This also disposes of scrutiny item 5 (Ca pseudo-frames): the frames are vacuous, so their quality is irrelevant — which is itself the symptom.

**Evidence:** `gate6_fape_train.py:114-164`; `notes/gate6_latent_calibration_results.json`; `notes/gate6_normz_rescore_results.json`; `notes/gate6_3di_eval_results.json` (mtime 09-13 21:59, both `best_fape_*.pt` mtimes 09-14 and 09-15).

**If true:** The project-level negative collapses to "a head trained against a non-invariant clamped coordinate loss, through a decoder whose output pose depends on a freshly sampled noise vector each call (`_sample_initial_noise`, no fixed seed), plateaus at the pose-error floor of that metric." That is a statement about the loss, not about frozen ESM-2, not about 8-dim latents, and not about the frozen decoder. Note also that the headline claim over-reaches independently: ESMFold trains only its folding trunk on a *frozen* ESM-2, so "frozen PLM + trained trunk" is demonstrably viable at scale; what is untested here is doing it with an 8M-param head and 80k structures.

**Resolving test:** Decode 200 val proteins with `best_fape_full.pt`, `best_deep10L.pt`, mean-z and true-z at 20 ODE steps; report (a) Kabsch-superposed Ca RMSD, (b) TM-score, (c) 3Di identity, (d) a corrected FAPE that builds frames from the *predicted* structure for predicted points. If superposed RMSD / TM show the FAPE head is materially better than mean-z (e.g. TM > 0.3) the negative is dead; if TM ~ 0.17 and 3Di ~ 15% it is confirmed on a metric that cannot be blamed. (~2-3 hours; reuses `gate6_3di_eval.py` and `gate6_latent_calibration.py`.)

---

### 2. [class: power] 100% clamped loss zeroes the gradient for roughly 80% of residues throughout training — the precise pathology OpenFold documented for early-phase AF2 training. Both FAPE runs were trained in that regime.

**Objection.** `fape_loss` is always called with the default `clamp_distance=10.0` (lines 294, 336); there is no unclamped fraction and no clamp annealing. From the reported value alone: FAPE 0.907 means mean clamped deviation 9.07 A out of a 10 A ceiling, so if the unclamped residues average 5-7 A, only ~19-31% of residues carry any gradient at all; the rest contribute exactly zero. At true_z (0.053) essentially 100% carry gradient. AlphaFold2 deliberately leaves 10% of batches unclamped, and OpenFold reports that batch-level clamping "is potentially problematic during the volatile early phase of training, when FAPE values can be extremely large and frequent clamping zeroes gradients for most of the residues in each crop", and that fixing it sped convergence ~30%. This run started at 0.914 — the volatile early phase, permanently. The observed signature fits: train loss falls monotonically 0.914 -> 0.854 over 17 epochs (local geometry, the unclamped near-diagonal residues) while val is flat at 0.908-0.910 — learning confined to the channel that still has gradient.

**Evidence:** `gate6_fape_train.py:114,152,294,336`; `notes/gate6_fape_results.json` (17-epoch history, best epoch 9).

**If true:** The plateau at ~0.907 is a property of an optimizer that cannot see global structure, not of the information content of the embeddings. It compounds objection 1: the ~20% of residues that do carry gradient are the ones already near-correct, i.e. the loss teaches local geometry and pose, never fold.

**Resolving test:** (a) Instrument one validation pass to log the fraction of residues at the clamp for each arm (true_z, mean-z, both heads) — settles the magnitude immediately (~0.5 h). (b) Retrain 5 epochs on the 20k subset with a corrected, rotation-invariant FAPE and sample-level clamping at 90% (OpenFold's fix) plus clamp annealing 30 A -> 10 A. If val drops below 0.85 the negative is void. (~6-8 h.)

---

### 3. [class: implication] The "6.4% of the gap closed, 74% needed, 11x to go" arithmetic is computed on a metric that is saturated in exactly the regime where every model sits, and the curve has no mean-z control point on its own axes.

**Objection.** Because of the 10 A clamp, the metric's derivative with respect to structural error goes to zero above ~0.8: the interval 0.85 (perfect structure, wrong pose) to 0.993 (random) spans everything from "correct fold" to "noise". Treating displacement along that interval as linear in structural quality is not defensible in either direction — it can equally understate real progress or dress up nothing as progress. The curve itself shows the compression: sigma 0.8 -> 1.2 -> 2.0 moves z-RMSE 0.62 -> 0.82 -> 1.04 and FAPE only 0.781 -> 0.890 -> 0.953, while D1 says the corresponding RMSD goes from ~11 A to ~20 A to ~44 A. A 4x change in Angstroms is 0.17 in this metric. Separately, `gate6_latent_calibration.py:80-105` never evaluates mean-z (or a variance-matched control) through the same (z-RMSE, cos, FAPE) pipeline, so "the MSE head is above the curve, the FAPE head is on it" has no baseline: a predictor collapsed toward the conditional mean is low-variance and anisotropic and would also sit off an isotropic-noise curve. The "on the curve -> generic ignorance" reading is one of at least three (variance collapse; global pose offset; genuine isotropic ignorance) and was not discriminated.

**Evidence:** `notes/gate6_latent_calibration_results.json` (curve + heads, no mean-z row); `notes/d1_results.json` (sigma 0.7 -> 11.5 A, 1.0 -> 20.0 A, 2.0 -> 44.0 A).

**If true:** The "11x further improvement, not a tuning gap" figure has no defined units and should not be the basis for a go/no-go. Re-expressed in Angstroms via D1, the same data says "current heads produce ~20 A structures", which is a statement the project already had from 3Di and did not need this experiment to reach.

**Resolving test:** Add three rows to `gate6_latent_calibration.py`: mean-z, per-dimension-variance-matched random z, and z perturbed with residue-correlated (low-frequency) noise at matched RMSE; and report superposed RMSD alongside FAPE for every row so the axis is in Angstroms. (~1-2 h.)

---

## Not objections (checked, direction stated)

- **Isotropic vs correlated noise (item 1).** The worry that the curve *overstates* the needed latent accuracy is backwards on the current evidence: both heads score *worse* than isotropic noise of equal z-RMSE, so as a requirement the curve is if anything optimistic. Direction of bias: favors the negative. (But objection 1 supplies a non-ignorance explanation for the excess.)
- **z-RMSE axis meaningless for FAPE-trained heads (item 2).** Correct that the axis is not the training objective, and the inversion (worse z-RMSE, better FAPE) is expected — it is evidence the FAPE head found a different solution, not evidence the axis is broken. It does not by itself invalidate the curve as a *decoder sensitivity* measurement, which is what it is.
- **Convergence of the FAPE run (item 6).** This one cuts *for* the negative and the log did not use it: `fape_10L` ran 17 epochs to convergence (best epoch 9, then val rises while train keeps falling) and reached 0.9080; the 80k / 10-ODE-step run reached 0.9069 in 3 epochs. 8x data and 3.3x ODE steps bought 0.001. The previous review's "underpowered" objection has been answered empirically — within this loss.
- **3 vs 20 ODE steps (item 4).** true_z at 0.053 bounds the decoder's contribution at 3 steps to ~0.5 A, so few-step decoding is not the bottleneck near the manifold. Folded into objection 1's resolving test anyway since pose jitter from the random initial noise is step-count dependent.

---

## STRONGEST OPPOSING CASE

The negative may well be right, and here is the case that survives all three objections: the project's one rotation-invariant structural measurement, the 3Di evaluation, showed the MSE head at 9-29% 3Di identity against 80-95% for straight reconstruction — genuinely random structures, on a metric immune to every complaint above. The MSE track was pushed to 8.2M params on ~80k samples and moved sigma only 0.637 -> 0.621 -> 0.603 against a 0.678 baseline, with a 10-layer head barely beating a 2-layer one — the flattening of a capacity-and-data curve, not a tuning gap. D1 requires sigma < 0.20 for 2 A structures and the best head is at 0.60, a 3x gap that three separate scaling attempts failed to dent. Under that reading, the FAPE run's metric bug changes only the sharpness of the last data point, not the trend, and the honest summary is "an 8M-param head on frozen final-layer ESM-2 embeddings and 80k structures is not a competitive folding model" — which was already established and is unsurprising. The distinction that matters is that this weaker claim is about *scale*, and the conclusion under review states something much stronger and architectural.

---

## CHECKS PERFORMED

All five classes considered. Signature: examined (objection 1, the clamp band 0.85-0.99, and the train-falls/val-flat shape in objection 2). Power: examined — and the convergence evidence cuts *for* the negative, stated above. Implication: examined (objection 3; D1's Angstrom axis never joined to the FAPE axis). Attribution: examined (objection 1; three competing readings of "on the curve" in objection 3). Direction of bias: examined — isotropic-noise caveat and the easy-split caveat both cut toward the negative and are stated as such; the metric bug and the clamped gradient both cut against it.
