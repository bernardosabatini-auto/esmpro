# sample_selector — can we pick the best of 8 samples without the answer?

**Job** 48136941, kempner_rtx, 2026-09-24, 26 min. Code `code/gate14_sample_selector.py`, job `slurm/sample_selector.sbatch`. Checkpoint `best_pf_459M_esmc_afdb.pt` (pair64 + ESMC-6B + 473k), w = 2, 50 Euler steps, K = 8. Log `logs/sample_sel_48136941.out`.

## Question
Best-of-8 has been worth +0.03-0.06 TM on every set, but the oracle needs the true structure. PLAN.md A4 asked how much of that gap can be recovered without it. Before training a confidence head, this measures three signals that need no training and no answer:

- **agree** — consensus: mean Kabsch-superposed TM-like score of a sample against the other seven (the medoid).
- **cycle** — re-encode the decoded backbone with the frozen ProteinAE encoder and compare with the sampled latent (on-manifold check).
- **dspread** — decode the same latent twice (the decoder is a 3-step flow from noise) and score the two decodes against each other (decoder confidence).
- **combo** — mean within-protein rank of the three.

Ground-truth TM per sample by Foldseek TM-align (exhaustive), coverage 100 % on all three sets.

## Result
TM of the chosen sample; "gap rec." = fraction of (oracle − mean-of-8) recovered; Spearman = mean within-protein rank correlation between the signal and the true TM.

| set | n | first | mean-of-8 | oracle | agree | cycle | dspread | combo |
|---|---|---|---|---|---|---|---|---|
| held-out (offset 1000) | 100 | 0.759 | 0.758 | 0.795 | **0.768** (27 %, ρ 0.29) | 0.760 (7 %) | 0.751 (−17 %) | 0.761 (10 %) |
| CASP15/16 domains | 80 | 0.698 | 0.700 | 0.743 | **0.706** (15 %, ρ 0.15) | 0.692 (−17 %) | 0.693 (−14 %) | 0.700 (1 %) |
| no-neighbour < 0.6 | 266 | 0.573 | 0.570 | 0.631 | **0.583** (20 %, ρ 0.20) | 0.565 (−10 %) | 0.569 (−3 %) | 0.573 (5 %) |

Correct-fold fraction moves the same way: on the no-neighbour set consensus lifts TM > 0.5 from 0.64 to 0.67 (oracle 0.79).

## Reading
1. **Consensus works, weakly:** +0.01 TM everywhere, one fifth to one quarter of the oracle gap. It is free at inference (Kabsch on 28 pairs) and should be the default when we already sample K.
2. **Re-encoding and decoder spread carry no information** about sample quality (ρ ≈ 0, sometimes negative). The decoder reproduces any on-manifold latent faithfully, good or bad, so "is it on the manifold" does not separate right folds from wrong ones; the samples are all valid proteins, just not always the right one. This also means the cycle signal is not a useful auxiliary loss.
3. **The oracle gap itself is only 0.04-0.06.** Even a perfect selector is worth less than the conditioner swap (+0.17 on CASP) or the data scale-up. A learned confidence head (the PLAN.md A4 proposal) would at best add another ~0.02 over consensus and costs a labelled training set of sampled structures; deprioritised behind Phase B.
4. **Sample-to-sample variation is mostly ambiguity, not noise.** Within a protein the samples' TMs vary (else no gap), yet nothing about the samples themselves predicts which is right. The model puts mass on several folds and only the sequence can arbitrate — the remedy is a better posterior (data, conditioner, recycling), not a picker.

## What it changed
- `gate14_sample_selector.py` gives any checkpoint's oracle gap and consensus gain on any set in one job.
- PLAN.md A4 is answered: use consensus (+0.01) when sampling K, skip the confidence head for now.
- No effect on the H200 main line, which is at epoch 30 of 60 (selection-set TM 0.732).
