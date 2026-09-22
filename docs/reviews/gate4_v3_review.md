VERDICT: pass-with-issues

FINDINGS:
- [severity: major] Primary latent-TM correlation (rho=0.231) is reported on an unbalanced set; the stratified latent correlation is much lower (0.093) and is not prominently reported
  Evidence: The diagnostic JSON (gate4_diagnostic_results.json) reports the stratified rho_tm_vs_aligned_z = 0.093 (the only stratified latent metric). The stratified rho for the pooled latent cosine (the metric that gives the headline 0.231) is NOT computed or reported in either the JSON or the log. The overall rho=0.231 is computed over 546,860 pairs of which 96% fall in TM 0.2-0.5. Per-bin analysis confirms near-zero latent correlation in TM 0.3-0.7 (rho ranging from -0.03 to +0.04). The overall rho is driven almost entirely by the contrast between different-fold pairs (TM<0.3) and same-fold pairs (TM>0.5), not by local smoothness within similar structures.
  Why it matters: The plan asks whether "proteins with similar folds have similar latents." The per-bin analysis shows they largely do not, except at TM > 0.9 (n=67). The overall rho=0.231 is not a measure of local smoothness -- it measures gross fold discrimination. Citing rho=0.231 as the "primary measurement" overstates the latent space's structural informativeness for downstream regression. For the regression task, what matters is whether nearby sequences map to nearby latents within a fold family, which is the regime where per-bin rho is near zero.
  Required action: Report the stratified pooled-latent-cosine rho alongside the overall value. The stratified number is the one that should inform the go/no-go decision.

- [severity: minor] max(qtmscore, ttmscore) is described as "the default TM-align convention" but this is not quite accurate
  Evidence: gate4_smoothness_v3.py line 13 and phase1_log.md line 184-185. Standard TM-align (Zhang lab) normalizes by the second (target) structure length, not the shorter. max(qtm, ttm) = normalize by shorter protein, which is a valid symmetrization convention used in some benchmarks, but it is not the TM-align default. This choice is more lenient than normalizing by the longer protein, inflating TM-scores for pairs of different lengths.
  Why it matters: The choice itself is defensible (and all values are now properly in (0,1]), but the stated justification is inaccurate. More importantly, the literature comparison (Jakubec & Hoksza 2024) reports for both TMmin and TMmax; the diagnostic's max(qtm,ttm) matches TMmin, which is the more lenient normalization.
  Required action: Correct the description. No re-run needed since the convention is valid and consistently applied.

- [severity: minor] TM > 0.9 per-bin rho=0.913 is based on only 67 pairs and has wide confidence intervals
  Evidence: gate4_diagnostic per-bin table. With n=67, the approximate 95% CI for Spearman rho is [0.67, 1.00]. While the point estimate is impressive, it could reflect as few as ~10 within-cluster pairs dominating the bin. This number is cited in the log as evidence of "local smoothness" but cannot support strong conclusions.
  Why it matters: The conclusion "the latent space is locally very smooth near identical folds" depends heavily on this single small-n bin. The adjacent bin (TM 0.8-0.9, n=276) has rho=-0.111, which is the opposite direction and borderline significant (z=-1.8).
  Required action: Acknowledge the small sample size and the contradictory TM 0.8-0.9 bin when interpreting this result.

- [severity: minor] ESM-2 mean-pooled sanity check (stratified rho=0.179) remains 2.5x below literature values (|rho|~0.46 for ESM-1b mean-pooled Euclidean on PISCES)
  Evidence: Jakubec & Hoksza (Bioinformatics 2024, btad786) report rho=-0.46 (|rho|=0.46) for Average Distance (mean-pooled ESM-1b) vs TMmin on PISCES benchmarks. The diagnostic's stratified mean-pooled ESM-2 L2 gives rho=0.179. The aligned ESM-2 L2 gives 0.386, which is closer to the literature's EBAplain (0.52-0.64 range). The sanity check passes the 0.1 threshold but the mean-pooled metric remains substantially below literature even after stratification.
  Why it matters: The 0.1 threshold was set lower than the original review implied (~0.3-0.5 expected). However, the aligned metric (0.386) is in a reasonable range, and the dataset composition (AFDB-derived, not curated PISCES pairs) likely explains some of the gap. The sanity check resolution is adequate but should note the remaining discrepancy.
  Required action: Note in the log that the mean-pooled ESM correlation remains below literature values and that the sanity check passes primarily on the aligned metric.

SPOT-CHECK:
1. Pair counts: v3 reports 62,250 from 250 structures. 250*249 = 62,250. Confirmed. Diagnostic reports 546,860 from 740 structures. 740*739 = 546,860. Confirmed.
2. TM distribution bin sums: v3 bins sum to 62,250 (23949+35641+2491+157+12). Confirmed. Diagnostic bins sum to 546,860 (5253+207880+257307+58829+12072+3792+1384+276+67). Confirmed.
3. Stratified sample size: diagnostic uses 5 bins with min_per_bin = 1727 (the count in [0.7, 1.01) = 1384+276+67 = 1727). Total = 5*1727 = 8,635. Matches reported value. Confirmed.
4. v3 TM range [0.101, 1.000]: consistent with plot showing no values > 1.0, confirming the alntmscore fix is effective.
5. Correlation consistency: v3 rho_pooled_z = 0.237, diagnostic rho_pooled_z = 0.231. The 0.006 difference is expected from different structure sets drawn from the same pool. Consistent.
6. Sign convention: code computes spearmanr(TM, -distance), so positive rho means higher TM correlates with lower distance (more similar). Convention is correct.

CHECKS PERFORMED:
1. Impossible/implausible numbers: All TM-scores now in (0,1]. Correlation values are plausible. The ESM stratified mean-pooled rho (0.179) is below literature but not impossible given dataset differences. Checked, no blocking issues.
2. Substituted/approximate methods: v1's hand-rolled TM-score was correctly invalidated. v2's alntmscore was correctly replaced with max(qtmscore, ttmscore). v3 and diagnostic both use Foldseek with --exact-tmscore 1. Checked, fixes are adequate.
3. Mismatched evaluation sets: v3 (250 structs) and diagnostic (740 structs) are drawn from the same pool with different sampling. The gate is self-contained; no mismatch with other gates. Checked, nothing found.
4. Reference-set contamination: All structures are AFDB predictions. For a smoothness test this is appropriate (testing in-distribution behavior). Checked, no concern for this gate.
5. Training-set leakage: Not applicable -- Gate 4 tests latent space geometry, not predictive accuracy. Checked, nothing found.
6. Conclusions depending on exclusions: All valid pairs are included (no inf filtering needed since both runs report 100% coverage). The only selection is the structure sampling strategy, which is explicitly described. The per-bin analysis reveals the overall rho is driven by between-bin contrast, not local smoothness -- this IS effectively an exclusion concern (the headline number is dominated by the TM<0.3 bulk). Flagged under major finding.
7. Statistics hygiene: Means, medians, and ranges are reported for TM-scores. Per-bin correlations are reported with sample sizes. The n=67 bin has a point estimate reported without CI. p-values are unreliable due to directed-pair doubling. Checked, minor issues noted.
8. Tests that cannot fail: The gate correctly failed against the 0.5 threshold. The sanity check passed at 0.1 but would have failed at literature-expected levels (~0.4) for mean-pooled metrics. The test is not vacuous. Checked, nothing found.
9. Identifiers and data integrity: Pair counts match exactly. Bin sums match totals. TM range is valid. No truncation or dropped records evident. Checked, nothing found.
