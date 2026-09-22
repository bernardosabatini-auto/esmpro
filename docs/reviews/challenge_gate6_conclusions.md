# Challenge: Gate 6 Conclusions and Root-Cause Attribution

CONCLUSION UNDER REVIEW: "The fundamental bottleneck is PCA canonicalization, not the regression architecture" (status block, phase1_log.md line 635)

VERDICT: premature

---

OBJECTIONS (max 3, ranked):

## 1. [class: attribution] The root-cause claim rests on uncorrected D2 numbers; the sign-check follow-up invalidated the headline evidence but the conclusion was never re-examined.

**Objection:** The status block (line 628) and MEMORY.md both cite "66% of TM>0.8 pairs have >90 deg frame divergence" as the primary evidence that PCA canonicalization is the fundamental bottleneck. The D2 sign-check follow-up (line 452, timestamped one hour AFTER the status block) showed this was "largely a sign-convention artifact" -- the corrected number is 11% above 90 degrees with a corrected mean angle of 55 degrees, not 110.7 degrees. Critically, the frame-angle-vs-latent-distance correlation (rho=0.373), which was cited as being STRONGER than TM-vs-latent (rho=0.169), was computed only on the uncorrected angles. After sign correction halved the mean angle, this correlation was never recomputed. If the sign-corrected rho drops below 0.169, the central claim -- that frame noise is the dominant source of latent variation -- falls apart, and the failure would require a different explanation (e.g., the latent space simply is not metrically organized for regression, as Gate 4's rho=0.169 already suggested).

**Evidence:** D2 raw mean angle 110.7 deg (line 410); D2 sign-corrected mean 55.0 deg (line 463); rho=0.373 computed on raw angles only (line 426-427); sign-check explicitly states "PARTIAL_SIGN" verdict (d2_sign_check_results.json); status block timestamp 10:00 UTC (line 623) vs sign-check timestamp 11:00 UTC (line 454); status block never updated.

**If true:** The root cause shifts from "PCA canonicalization makes z unpredictable" to "ProteinAE's latent space is not metrically smooth enough for regression regardless of frame" -- a fundamentally different bottleneck that points to different next steps (the equivariant-AE recommendation might be unnecessary; Option C might also fail).

**Resolving test:** Recompute the frame-angle-vs-latent-distance Spearman rho using sign-corrected frame angles on the same 309 TM>0.8 pairs. If the corrected rho remains above 0.169 (the TM-vs-latent correlation), the frame-contamination story survives in weakened form. If it drops below 0.10, the attribution to canonicalization is unsupported. Estimated cost: ~1 hour (rerun D2 analysis script with the existing 4-sign-combo correction applied to the angle computation, no new data needed).

---

## 2. [class: power] Gate 6 trained a 13.3M-parameter model on 2,401 samples, overfit by epoch 5, and no regularization sweep or smaller architecture was attempted -- the negative result cannot distinguish "unlearnbable target" from "insufficient data / wrong model capacity."

**Objection:** The model has 5,539 parameters per training sample (13.3M / 2,401) and 5.2 parameters per scalar target value (13.3M / 2.55M residue-level values). It overfit aggressively: by epoch 5 (the best checkpoint), train MSE was already 0.361 vs val 0.409 (12% gap); by epoch 10, val exceeded the mean-z baseline (ratio 1.001). The plan (Gate 6 item 2) says "sweep minimally: learning rate, backbone depth, whether to fine-tune the top few ESM layers." None of these sweeps were done -- a single configuration was trained once. The 10.8% improvement over mean-z is the kind of marginal signal that can be destroyed by overfitting in an overparameterized regime. A linear probe (10,248 parameters), a 2-layer model at d_model=128 (~1.3M params), or simple L2 regularization on the output could perform identically or better, and would separate the question of data quantity from model capacity. The plan's reduced dataset (3k from 20k) was explicitly a feasibility compromise; the log treats the negative as conclusive without acknowledging that the experiment was designed to be quick, not definitive.

**Evidence:** 13.3M params (gate6_results.json); train 2,401 samples (line 516); best epoch 5, val/train ratio 1.134 at best epoch (history in gate6_results.json); epoch 10 ratio 1.001 (val exceeds baseline); no hyperparameter sweep mentioned in the log; plan specifies sweeps at Gate 6 item 2; 7/20 test targets where mean-z BEATS the model (gate6_decode_results.json per-target analysis).

**If true:** The 18.7 A RMSD is a property of this specific overparameterized, under-regularized, under-sampled experiment -- not a property of the ESM-to-z mapping in general. A properly regularized model on 20k samples (as the plan specified) could close more of the gap. This does not mean the approach would succeed, but it means this experiment does not prove it cannot.

**Resolving test:** Train a linear probe (1280 -> 8, no transformer, ~10k params) and a small model (d_model=128, 2 layers, ~1.3M params) on the same data with the same splits. If the linear probe matches the transformer's 10.8% improvement, the signal is linear and the transformer added nothing -- the dataset is too small for a complex model. If a smaller model does significantly better on val (less overfitting), the 13.3M model was simply the wrong capacity choice. Estimated cost: ~2 hours (both models train in under 2 minutes each on this data; the evaluation pipeline already exists).

---

## 3. [class: direction] The MMseqs2 30%-identity split created an easier evaluation than the plan's Foldseek structural clustering; the Gate 6 failure on this easier split is more damning than acknowledged, not less.

**Objection:** The Gate 5 review correctly notes the substitution of MMseqs2 for Foldseek but labels it "more conservative" and "stricter." This is true only for sequence leakage. For the actual task (sequence -> structure), a 30%-identity split with 98.8% singletons is effectively random, which means structural homologs (proteins below 30% sequence identity that share the same fold) can appear in both train and test. This makes the regression task EASIER, because the model can see structurally similar targets during training for some test proteins. The plan specified Foldseek structural clustering precisely to prevent this fold leakage. A proper structural split would yield worse test performance than the 18.7 A already observed. The log's D2 follow-up (line 487) corrected this: "Gate 5's note about MMseqs2 should state that 98.8% singletons mean the val/test split is easier than a Foldseek-clustered split. The Gate 6 failure happened on the easier split." But the status block does not incorporate this observation -- the failure is treated as caused by canonicalization rather than as a structural lower bound on what this approach can achieve. The easier split means the negative is MORE credible as a negative, not less.

**Evidence:** 18,767 clusters from 18,888 proteins = 98.8% singletons (line 499); plan specifies Foldseek clustering (Gate 5 item 1, item 6); Gate 5 review calls it "stricter" (gate5_review.md line 6); D2 follow-up correction at line 487; status block does not reference the split-difficulty issue.

**If true:** Even if the canonicalization problem were fully fixed (the equivariant-AE path), the regression might still fail on a properly clustered split. The easier split gives the model every advantage and it still lands at 18.7 A. This strengthens the negative and weakens the claim that canonicalization is the sole bottleneck.

**Resolving test:** No additional experiment needed -- this is purely a question of correct interpretation. The status block should acknowledge that the failure occurred on a split that is easier than what the plan specified, which makes the negative more robust, not less. If desired, the existing data can be re-split by Foldseek 3Di clustering (already available from Gate 4's analysis of 19,094 structures with 13,125 clusters) and Gate 6 rerun to confirm the result is the same or worse. Estimated cost: ~2 hours.

---

STRONGEST OPPOSING CASE:

The sign-check showed that the raw D2 numbers exaggerated the problem by roughly 2x, but a corrected mean frame angle of 55 degrees among TM>0.8 pairs is still large enough to be a substantial source of noise. D1 shows the decoder requires sigma < 0.10 for 1 A RMSD, and even the corrected canonicalization noise is well above this threshold. A 13.3M-parameter transformer on 2,401 samples is certainly overparameterized, but the fact that the best-epoch model only improved 10.8% over mean-z -- while the train set was already at 78% of baseline -- suggests that the signal-to-noise ratio in the z targets is genuinely poor, not just that the model was too big. The easiest split and the overparameterized model both cut in directions that should have HELPED the regression, and it still failed massively (18.7 A vs 0.22 A ceiling). If you fix the signs, shrink the model, enlarge the dataset, and use a proper structural split, you might go from 18.7 A to perhaps 12-15 A -- still catastrophically far from the 2 A threshold that D1 says the decoder needs. The conclusion is probably directionally correct even if the specific evidence trail is sloppy: the gap between what the decoder needs (sigma < 0.10) and what per-residue regression from sequence can provide is likely too large for this approach.

---

CHECKS PERFORMED:

1. **Distributional signature:** Checked the per-target model-vs-mean-z RMSD ratios. Model beats mean-z on only 13/20 targets; 7/20 targets mean-z is better. The median ratio (0.969) is much closer to 1.0 than the mean (0.914), indicating a few targets with large improvements pull the average while the bulk barely improves. This is consistent with the model learning a slight shrinkage toward mean-z rather than protein-specific structure. Checked, consistent with the 10.8% improvement being thin but real.

2. **Underpowered negative:** The 13.3M-param model on 2,401 samples with overfitting at epoch 5 and no hyperparameter sweep is the core of Objection 2. The plan specified 20k samples and sweeps; neither was done. The negative is underpowered. Flagged as Objection 2.

3. **Unexamined implication:** D1 establishes that sigma < 0.10 is needed for 1 A RMSD. The model's equivalent sigma is 0.625. Even a 10x improvement (from a larger dataset, better model, fixed canonicalization) would give sigma ~0.06, which D1 maps to ~0.7 A RMSD -- this would actually be competitive with ESMFold. But the question is whether a 10x MSE reduction is plausible. The current model closes 10.8% of the gap; closing 90% is a different proposition entirely. This implication was not computed in the log but is the key feasibility number. Checked; noted but not elevated to a separate objection because it depends on the resolution of Objections 1 and 2.

4. **Single-cause attribution:** The root cause is attributed solely to PCA canonicalization. Alternative explanations consistent with the evidence: (a) ProteinAE's latent space is not metrically smooth for regression (Gate 4 rho=0.169 already showed this); (b) the dataset is too small and the model too large; (c) the sequence-based split is too easy but 2,401 samples is too few. The sign-check partially undermined the canonicalization attribution. Flagged as Objection 1.

5. **Direction of bias:** The easier split (MMseqs2 vs Foldseek) makes the task easier, so the failure is more credible. The overparameterized model overfitting makes the MSE number less reliable (could improve with proper regularization), so the negative is less credible on that axis. These cut in opposite directions. The split issue is Objection 3; the model capacity issue is Objection 2.

