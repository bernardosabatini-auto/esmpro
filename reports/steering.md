# steering — navigating and conditioning the generative model

Answering two questions: if the data latent cannot be walked in, what is the generative capacity worth, and can generation be steered without retraining?

**Jobs** `code/gate28_latent_explore.py` (49065270), `code/gate30_steering.py` (49090811), `code/gate29_designability.py` (49090568). Models: 840M for the traversal, 459M pair 128x8 for steering (our best on real structures), inverse folding by the 174M co-design head, independent refolding by ESMFold2-Fast.

## 1. The data latent is not walkable, the prior is (gate 28, 840M, 64 proteins)
| latent source | Ca-Ca A | in 3.7-3.9 | clash | Rg/expected | helix | strand | TM to source |
|---|---|---|---|---|---|---|---|
| real structure | 3.86 | 0.82 | 0.000 | 1.35 | 0.43 | 0.24 | 1.00 |
| **unconditional sample** | **3.85** | **0.78** | **0.000** | **1.35** | 0.36 | 0.21 | **0.26** |
| real + noise 0.2 | 4.59 | 0.05 | 0.005 | 1.28 | 0.11 | 0.17 | 0.72 |
| real + noise 0.4 | 6.86 | 0.02 | 0.008 | 1.30 | 0.04 | 0.24 | 0.51 |
| slerp between two reals, 0.5 | 2.73 | 0.06 | 0.032 | 1.05 | 0.21 | 0.03 | 0.30 |

Given only a length and the CFG null token, the model produces chains with correct bond geometry, no clashes, real compactness and near-real secondary structure, at TM 0.26 to the protein whose length it borrowed: **new folds, not copies**. Interpolating or perturbing a finished latent instead collapses the chain. Note also that the noise-0.2 row keeps TM 0.72 while its bond lengths are 4.59 +- 1.82 A: **TM is blind to local geometry**, which is worth remembering for every other number in this project.

## 2. Navigate the noise, not the output (gate 30)
| mode | Ca-Ca | in-range | clash | helix | TM to real |
|---|---|---|---|---|---|
| prior sample | 3.84 | 0.84 | 0.000 | 0.42 | 0.29 |
| noise slerp 0.25 / 0.5 / 0.75 | 3.84 / 3.84 / 3.84 | 0.86 / 0.85 / 0.85 | 0.000 | 0.39 / 0.38 / 0.35 | 0.27 |

Interpolating the ODE's **initial noise** and integrating gives a continuous family in which every member is as well-formed as a direct sample: Ca-Ca 3.84 and zero clashes throughout, against 2.73 A and 0.032 clashes for the same interpolation done on the output latent. The space is navigable; you have to go through the flow.

## 3. Gradient guidance through the decoder works, within a window
No retraining. At each step form the one-step estimate `x1 = x_t + (1-t)v`, decode it with gradient (the ProteinAE decoder is differentiable, as `struct_loss.md` already used), and push `x1` down the gradient of an objective on the coordinates. Gradient flows only through the decoder, not the trunk.

**Objective: radius of gyration, target 0.80x the unguided value (16.4 -> 13.1 A).**
| scale | Ca-Ca | in-range | clash | Rg achieved | helix |
|---|---|---|---|---|---|
| 0 | 3.84 | 0.84 | 0.000 | 16.4 | 0.42 |
| **2** | **3.84** | **0.87** | **0.001** | **14.3** | 0.38 |
| 10 | 11.96 | 0.06 | 0.139 | 16.3 | 0.07 |
| 40 | 18.51 | 0.01 | 0.083 | 21.0 | 0.03 |

At scale 2 the objective moves 64 % of the way to target **with bond geometry and clash rate unchanged**. Past that the guidance destroys the protein and stops achieving the objective at all (Rg at scale 40 goes the wrong way). There is a usable window and it is narrow.

**Objective: bring two residues within 8 A.** This failed at first and the reason was my parameterisation, not the model. The radius-of-gyration gradient is spread over every residue; a single-pair contact concentrates its gradient on 2 of ~150, so the same nominal scale applies a per-residue force two orders of magnitude larger and tears the chain apart (Ca-Ca 7.53 A already at scale 0.5). Normalising the guidance gradient to unit RMS per sample (`--guide-norm`) makes a scale mean the same displacement whatever the objective:

| scale (normalised) | Ca-Ca | in-range | clash | contact distance | helix |
|---|---|---|---|---|---|
| 0 | 3.84 | 0.87 | 0.000 | 24.3 A | 0.42 |
| 0.02 | 3.84 | 0.85 | 0.000 | 22.4 | 0.42 |
| 0.05 | 3.84 | 0.86 | 0.000 | 19.6 | 0.40 |
| 0.1 | 3.84 | 0.86 | 0.000 | 14.7 | 0.36 |
| **0.2** | **3.84** | **0.87** | **0.000** | **8.2** | 0.29 |

**The constraint is satisfied -- 24.3 A down to 8.2 A against an 8 A target -- with bond geometry and clash rate untouched.** Helix content falls from 0.42 to 0.29, which is what forming a long-range contact should do to the fold. Under the same normalisation the Rg objective reaches 14.9 A at scale 0.2, also with perfect geometry. So both a global and a hard pairwise design constraint are steerable at sampling time, with no retraining.

**Motif inpainting** (hold 30 % of residues on the real latent's own noise-to-data path): Ca-Ca 3.95, clash 0.003, and TM to the source protein rises from 0.29 to 0.36. Geometry survives and the motif is partly respected, but 30 % of residues fixed buying 0.07 TM means the scaffold is not tightly following the motif yet.

## 4. Designability: the measurement is blocked by our inverse-folder (gate 29)
Loop: structure -> sequence (co-design head, temperature 1) -> ESMFold2-Fast -> TM back.

| structures | n | scTM | above 0.5 |
|---|---|---|---|
| **real experimental** | 52 | **0.376** | 0.17 |
| unconditional samples | 52 | 0.313 | 0.02 |
| slerp 0.5 | 60 | 0.264 | 0.00 |
| real + noise 0.2 | 52 | 0.249 | 0.00 |

**Genuine experimental structures only score 0.376 through this pipeline.** The control establishes that the ceiling is our inverse-folding head (0.18 native recovery), not the structures. Unconditional samples reach 0.313, which is 83 % of that ceiling and clearly above both degenerate modes, but with a control this weak the test cannot distinguish "somewhat less designable" from measurement noise. **The honest verdict is inconclusive, and the fix is a real inverse-folding model** (ProteinMPNN, ~1.7M parameters, reachable from the cluster) rather than our prototype head.

## Reading
1. The generative capacity is real and it is reachable: unconditional samples are well-formed and novel, and noise-space interpolation gives smooth families of valid structures.
2. Conditioning at sampling time works for both a global geometric objective and a hard pairwise contact, with geometry fully preserved, once the guidance gradient is normalised so a scale means the same displacement for every objective. Un-normalised, objectives whose gradient is concentrated on few residues destroy the structure; that was a bug in the knob, not a limit of the method.
3. Whether any of this yields *designable* proteins is not yet measurable with our own inverse-folder. That is the next thing to fix, and it is cheap.
