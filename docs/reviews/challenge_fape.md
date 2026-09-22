# Challenge: FAPE Training Experiment

CONCLUSION UNDER REVIEW: "The head cannot learn a generalizable sequence-to-structure mapping. Frozen ESM embeddings don't contain enough information to predict 3D structure through this decoder."

VERDICT: unsupported

---

OBJECTIONS (max 3, ranked):

## 1. [class: power] The experiment is underpowered across at least three axes simultaneously: 10k training samples (vs 80k for the MSE head that was warm-started from), 3 ODE steps (vs 20 at inference), and a warm-start from an objective-mismatched checkpoint -- any one of these could explain the result.

**Objection:** The FAPE experiment used 10k training samples, but the MSE-trained checkpoint it warm-starts from (best_deep10L.pt) was trained on ~80k samples (dataset_100k.h5 train split: 79,653). This is an 8x reduction in training data. The MSE experiments in this very project showed clear improvement scaling from 3k to 20k (sigma 0.637 to 0.621), and the earlier challenge review explicitly identified 2.4k samples as insufficient. The training curve confirms the expected consequence: train FAPE drops steadily from 0.914 to 0.854 over 17 epochs while val FAPE barely moves (0.915 to 0.908, a reduction of 0.007 in absolute terms). This is textbook overfitting on insufficient data. Meanwhile, the 3-step Euler ODE has dt=0.333 per step -- the literature on neural ODE training establishes that large step sizes corrupt gradient quality, and the ProteinAE decoder was designed for 20 steps (dt=0.05). The gradients flowing back through 3 large Euler steps are a 6.7x coarser approximation of the true gradient than what the decoder was designed for. Finally, the warm-start from an MSE checkpoint trained to predict z in a specific PCA-canonicalized frame creates an initialization that has learned to commit to a specific rotation. The FAPE loss is rotation-invariant and needs the model to find z values that decode correctly regardless of frame -- a fundamentally different basin of the loss landscape. The MSE initialization may actively trap the optimizer near a bad local minimum. Each of these three problems (data, gradient quality, initialization) independently could explain the negative result; together they make it impossible to attribute the failure to "frozen ESM embeddings lack information."

**Evidence:** gate6_fape_train.py line 374: n_steps=3; line 396: 10k subset from 100k dataset; line 417: warm-start from best_deep10L.pt; dataset_100k.h5 train split has 79,653 samples; gate6_fape_results.json: train loss 0.914->0.854 (steady decrease), val loss 0.915->0.908 (stalled); gate6_deep_head_results.json: deep10L trained on full train split (79,653 samples).

**If true:** The failure is a property of this specific misconfigured experiment, not a property of frozen ESM embeddings. The correct conclusion would be at most "this particular FAPE training configuration did not work" -- not that the approach is fundamentally impossible.

**Resolving test:** Run FAPE training from random initialization (not MSE warm-start) on the full ~80k training set with 10 ODE steps, lr sweep over {1e-5, 5e-5, 1e-4}, for at least 30 epochs. If FAPE drops below 0.7 (halfway between mean-z at 0.965 and the current 0.908), the experiment was underpowered. If it remains above 0.9 across all configurations, the negative becomes credible. (~12 hours: 3 lr settings x ~4 hours each, parallelizable.)

---

## 2. [class: attribution] The conclusion jumps from "this head architecture failed with FAPE loss" to "frozen ESM embeddings lack information," but the MSE experiments in this same project proved the opposite -- the head CAN extract structural signal from ESM embeddings (sigma=0.603 vs mean-z 0.678, closing 20% of the gap at 20k). The FAPE failure is more parsimoniously explained by the training setup.

**Objection:** The project's own MSE experiments demonstrated that frozen ESM-2 embeddings DO contain structure-predictive information. The 10-layer d256 head on ~80k samples achieved sigma=0.603 (test MSE=0.364) vs mean-z baseline sigma=0.678 (test MSE=0.459) -- a 20.7% reduction in test MSE, highly significant. D4 established that sigma tracks RMSD at rho=0.681, confirming that better z prediction translates to better structure. The conclusion attributes the FAPE failure to "frozen ESM embeddings don't contain enough information," but the MSE result falsifies this specific claim within the same project. The actual failure is that the FAPE training loop could not exploit the information that the MSE training loop could. The most likely explanations are: (a) gradient quality through 3 ODE steps is too poor to learn, (b) the FAPE loss landscape through the frozen decoder has pathological curvature, or (c) the experiment was too small. None of these implicate the ESM embeddings.

**Evidence:** gate6_deep_head_results.json: deep10L test_sigma=0.6033, ratio=0.7929 (20.7% below baseline); phase1_log.md D4 stratification: sigma tracks RMSD at rho=0.681; gate6_fape_results.json: model barely improves over mean-z despite using a checkpoint that already outperforms mean-z on MSE.

**If true:** The conclusion should be revised from "ESM embeddings lack information" to "FAPE training through a 3-step frozen ODE decoder did not converge, likely due to gradient quality or training scale issues." The recommended next step would be different: instead of abandoning frozen ESM or declaring the problem unsolvable, one would fix the training setup (more ODE steps with gradient checkpointing, or adjoint method, or more data, or random init).

**Resolving test:** Decode the MSE-trained head's z predictions through the same 3-step decoder and compute FAPE. If the MSE-trained head gives FAPE substantially below 0.908 without any FAPE training, it proves that the MSE objective already found better z values than the FAPE training loop could -- confirming the failure is in the FAPE optimization, not the embeddings. (~1 hour: just a forward pass evaluation with existing checkpoints.)

---

## 3. [class: signature] The training curve (train drops 0.914->0.854, val flat at 0.908) is the signature of overfitting on small data, not of a fundamental information bottleneck. An information bottleneck would show BOTH train and val stalling near mean-z.

**Objection:** If frozen ESM embeddings truly lacked the information to predict structure, the training loss itself would not decrease -- the model could not even memorize the training set, because the inputs would be uninformative. But train FAPE drops steadily from 0.914 to 0.854 over 17 epochs with no sign of plateauing. This 0.060 reduction in train FAPE represents learning -- the head IS extracting information from ESM embeddings to produce z values that decode to better-than-mean-z structures on the training set. The problem is that this learning does not generalize, which at 10k samples with an 8.2M parameter model (820 parameters per sample) is the expected behavior of any overparameterized model, regardless of whether the underlying task is solvable. A true information bottleneck would manifest as both train AND val losses plateauing near the mean-z baseline (0.965), because the model would find nothing to learn even on the training set. The observed pattern -- train improving, val flat -- is diagnostic of "solvable task, insufficient data."

**Evidence:** gate6_fape_results.json: epoch 1 train=0.914 -> epoch 17 train=0.854 (monotonic decrease over all 17 epochs); val=0.915 -> 0.908 (best epoch 9, then stalls); 8,227,592 parameters / 10,000 samples = 823 params per sample.

**If true:** The conclusion must be weakened from "fundamentally impossible" to "requires more data." The recommended next step would be to scale up training data, not to abandon the approach.

**Resolving test:** Train the same architecture from random init on the full ~80k samples. If train FAPE drops below 0.80 AND val drops below 0.85, the task is learnable with more data. If train drops but val remains above 0.90, then generalization is the issue and one should try smaller architectures or regularization. (~4 hours.)

---

STRONGEST OPPOSING CASE:

Even granting all three objections, the project has already established that predicting z IS the folding problem (D4: sigma tracks RMSD at rho=0.681, "predicting z correctly IS predicting structure correctly"). The MSE experiments show that a frozen 650M-param ESM-2 plus an 8M-param head on 80k samples closes only 20% of the gap to true-z (sigma 0.603 vs ceiling 0.0 and baseline 0.678). D1 shows the decoder needs sigma < 0.20 for useful structures (2 A RMSD). At the current rate of improvement (0.637 at 3k -> 0.621 at 20k -> 0.603 at 80k), reaching sigma=0.20 would require either orders of magnitude more data or a fundamentally different model capacity -- which is exactly what ESMFold provides with 3B parameters and end-to-end training. The FAPE experiment may be badly configured, but the MSE experiments already established the core difficulty: a small head on frozen embeddings makes slow progress toward a target (sigma < 0.20) that is extremely far from the current sigma=0.60. The FAPE experiment did not need to succeed for the broader conclusion to hold. The strongest version of the negative is not "ESM embeddings lack information" (which the MSE results disprove) but rather "the information in ESM embeddings is insufficient for this task at this model scale" -- and that conclusion IS supported by the MSE experiments even if the FAPE experiment is discarded entirely.

---

CHECKS PERFORMED:

1. **Distributional signature:** The training curve shape (monotonically decreasing train, flat val) is diagnostic of overfitting on small data, not information bottleneck. An information bottleneck would show flat train loss. Checked; this is Objection 3.

2. **Underpowered negative:** 10k samples, 3 ODE steps, MSE-mismatched warm-start, and a single hyperparameter configuration. The experiment is underpowered on multiple axes. The MSE experiment that provides the warm-start weights used 80k samples. Checked; this is the core of Objection 1.

3. **Unexamined implication:** The true-z FAPE ceiling (0.053) was reported but it is unclear whether it was computed at 3 or 20 ODE steps. If computed at 20 steps, the comparison is unfair because the model trains at 3 steps where decoder output quality is much lower. The MSE-trained head's FAPE was never computed, which would directly show whether the MSE objective already found better z values. Checked; incorporated into Objection 2's resolving test.

4. **Single-cause attribution:** The conclusion attributes failure to "frozen ESM embeddings lack information," but the MSE experiments in this same project demonstrated they DO contain structural information (sigma=0.603, 20% below mean-z). The FAPE failure has at least three alternative explanations (data size, gradient quality, initialization mismatch). Checked; this is Objection 2.

5. **Direction of bias:** The warm-start from MSE cuts in an ambiguous direction. It gives a better starting point (epoch 1 at 0.914 vs mean-z 0.965) but may trap the optimizer in the MSE solution's basin, preventing exploration of FAPE-optimal solutions. The 10k subset makes the task harder (less data), but the easier split (from earlier analysis showing no structural clustering) makes it easier. On balance, the reduced data size is the dominant confounder and it makes the negative less credible, not more.
