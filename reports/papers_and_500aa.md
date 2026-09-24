# SimpleFold and SimpleDesign: what they change for us, and the path to 500-residue proteins

Read 2026-09-24 from `papers/simpleFold.pdf` (Apple, arXiv 2509.18480) and `papers/simpleDesign.pdf` (TMLR 08/2026). Two measurements were run in response (below).

## 1. What the papers do

**SimpleFold** — folding as flow matching in Cartesian all-atom space with a plain DiT (adaLN, QK-norm, SwiGLU, RoPE), *no pair representation, no triangle updates, no MSA*. Frozen ESM2-3B (all layers) conditions the trunk. Trained on 2M structures (PDB + SwissProt + AFESM cluster representatives at pLDDT > 0.8) at 100M-1.1B, and 8.6M for the 3B model. Recipe details that matter: (i) an LDDT loss on the one-step estimate x_hat in addition to the flow loss, which they say is *required* for local accuracy; (ii) timesteps resampled toward t = 1 (logit-normal m 0.8, s 1.7); (iii) targets rigidly aligned to x_hat before the loss; SO(3) augmentation instead of equivariance; (iv) **repeated batching**: each GPU carries B_c copies of the same protein with different t and noise; (v) pre-train at crop 256, fine-tune at 512 with half the batch; (vi) 200-500 SDE steps at inference (50 steps collapses CAMEO TM from 0.83 to 0.65); (vii) a pLDDT head trained afterwards on the frozen model's residue tokens. Results: CASP14 TM 0.611 (100M) to 0.720 (3B) vs ESMFold 0.701; CAMEO22 0.80-0.84 vs ESMFold 0.85. Clear scaling with parameters and data.

**SimpleDesign** — joint sequence + structure generation in one transformer: masked-token cross-entropy for the sequence and a Cartesian Ca flow loss for the structure, sampled with independent time axes so the same model does folding (t_seq = 1), inverse folding (t_str = 1) and co-design in between. Tokenizer-free, ESM2-650M initialised, trained on AFESM 1.8M cluster representatives (32-512 residues, pLDDT > 85) then SwissProt. Competitive co-designability with much simpler machinery than tokenised multimodal PLMs.

## 2. Where we stand relative to them
- Our model is a **latent** flow: 8 numbers per residue, a frozen decoder does the geometry. That is why 25 Euler steps suffice where SimpleFold needs 200-500, and why our 459M model folds a 256-residue protein in well under a second on one GPU. This is the efficiency edge to keep.
- We do use a pair track (theirs is the "unnecessary" component). Our own ablations agree with their thesis more than not: at the plateau the pair track buys ~0.01 TM; the conditioner (ESMC-6B) bought 0.17 and data 0.03-0.05. SimpleFold's scaling curves say the same thing: capacity and data, not inductive bias.
- Their 100M model on 2M structures reaches CASP14 0.611 on *full-length* targets up to 1000 residues; our 459M on 473k reaches 0.703 on CASP15/16 domains <= 256 residues. Not comparable sets, but the data ratio (4-18x more structures) is the obvious gap. AFESM cluster representatives are a ready-made source (1.9M at pLDDT > 0.8, 5M clusters total); our Phase B plan to build 2.3M from AFDB should use it instead.

## 3. Measured today because of the papers
**a. Does the frozen autoencoder work beyond 256 residues?** (`code/gate17_long_roundtrip.py`, job 48287014). 234 experimental X-ray chains of 257-640 residues, encoded and decoded with the shipped ProteinAE:

| length | n | TM | TM > 0.9 | RMSD | decoder time |
|---|---|---|---|---|---|
| 257-320 | 51 | 0.999 | 100 % | 0.16 A | 23 ms |
| 321-384 | 52 | 0.999 | 100 % | 0.18 A | 30 ms |
| 385-448 | 41 | 0.999 | 100 % | 0.19 A | 39 ms |
| 449-512 | 43 | 0.999 | 100 % | 0.22 A | 48 ms |
| 513-640 | 38 | 0.999 | 100 % | 0.22 A | 61 ms |

Latent norm exactly on the manifold (2.828). **The 8-dim latent and the frozen decoder are not the obstacle to 500-residue proteins**; nothing in the autoencoder has to be retrained.

**b. Repeated batching** (`REPEAT_COPIES` in `gate10_pair_flow.py`, profile `slurm/step_profile_repeat.sbatch`): see the addendum at the end once the profile returns. The point: the pair track depends only on the sequence, so R noisy copies of a protein can share one pair computation, and the pair track is half of our step time.

## 4. Ideas worth taking, ranked by expected value per GPU-hour
1. **Data: AFESM cluster representatives, 32-512 residues, pLDDT > 80.** Both papers train on exactly this. Replaces the hand-built 2.3M AFDB plan; the build pipeline (`gate8`) already does download + canonicalise + encode at ~30 structures/s per GPU.
2. **Repeated batching** for the pair model (efficiency; needs a quality A/B because fewer unique proteins per step).
3. **Structural auxiliary loss on the decoded one-step estimate.** We already decode x_hat for recycling; an LDDT-style distance loss through the *differentiable* decoder gives the flow a structure-aware gradient the latent MSE lacks. CLAUDE.md's warning is about a latent-MSE term, not a geometric one. Cost ~+40 % per step; test on the 174M/80k twin.
4. **Timestep resampling toward t = 1** (one-line change, cheap A/B at 80k). In latent space the late-t regime decides fine detail just as in coordinate space.
5. **Joint sequence-structure (SimpleDesign-style) on the latent.** Add a masked-sequence head and a sequence time axis: the same latent flow then does folding, inverse folding and co-design, and the frozen decoder turns every sample into a backbone. This is the "generative capacity of the latent" the user asked for, and it is a capability ESMFold does not have. A Phase C item, after data scale.
6. Not worth it for us: 200-500 sampling steps (our decoder removes the need), SO(3) augmentation (our latents are canonicalised), all-layer ESM mixing (tested, null).

## 5. Extending to 500 residues: plan
Prerequisites now known: autoencoder OK to 640 (above); the pair track and attention are O(L^2) memory, the triangle update O(L^3) compute.

1. **Data (CPU/GPU build, ~1 day):** AFESM/AFDB cluster representatives 257-512 residues (target 300-500k) plus PDB chains 257-512 (the gate15 selector already lists 157k candidate chains at <= 2.5 A before clustering). Store ragged; encode with the frozen encoder; embed with ESMC-6B. Host-RAM budget: ESMC at 2560-d fp16 is 5 KB/residue, so 400k proteins x 350 residues = 700 GB per node cache — too much. Either a PCA-512 projection of the ESMC embedding (needs a quality check at 80k first) or fp8 storage, or online ESMC (roughly doubles step time).
2. **Model changes (small):** replace the 256-entry absolute position table with RoPE or extend it (the relative attention bias already exists, clamp 32); pair memory law B x L^2 means the budget drops 4x at L = 512 (about 27 proteins per H200 step at budget 108); the fused triangle update helps here most. Spatial + contiguous cropping at 512 for anything longer (SimpleFold, AF3 style).
3. **Training schedule (SimpleFold's):** keep pre-training at <= 256 (what we have), then a **fine-tuning stage at <= 512** from the best checkpoint with half the batch and a low learning rate — exactly the `--warm-start` path built today. Expected cost: 512-residue steps are ~4x the pair compute per protein, so a 10-epoch fine-tune on 300k long proteins is on the order of 2-3 days on 8 H200s; budget accordingly.
4. **Evaluation:** CASP15/16 domains 257-512 are already downloaded and excluded only by length (68 domains); CAMEO-style full chains for the external comparison; ESMFold2-Fast on the same inputs.
5. **Decision gate:** run the 174M model at 512 on the 80k+long data on RTX first to see whether the pair track is still worth its L^3 at 512; if not, the SimpleFold result says a pair-free DiT with more data may be the better 512-residue model, and it is 2.5x cheaper per step.
