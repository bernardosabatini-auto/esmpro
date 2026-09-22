VERDICT: pass

FINDINGS:
- [severity: minor] Clustering method substituted: plan specifies Foldseek cluster-based splitting, implementation uses MMseqs2 at 30% sequence identity.
  Evidence: gate5_build_dataset.py lines 156-169 use `/home/guest/bin/mmseqs cluster --min-seq-id 0.3`. Plan (Gate 5, item 1) says "Sample across Foldseek clusters" and (item 6) "hold out by Foldseek cluster or CATH superfamily."
  Why it matters: MMseqs2 at 30% sequence identity produces more, smaller clusters (98.8% singletons), making the split effectively near-random. Proteins below 30% sequence identity can share the same fold, so remote structural homologs can appear in both train and test. This makes the evaluation EASIER (weaker split) for the sequence-to-structure regression task, not stricter — the opposite of what "more conservative" implies. (The label "more conservative" is correct only for preventing sequence leakage, which is not the relevant concern for a structural prediction task.) A Foldseek structural split would be harder and would likely produce worse test performance.
  Required action: Acknowledge in the dataset manifest that the split is sequence-based (MMseqs2 30% identity), not structure-based, and that remote structural homologs may leak across splits. This matters for Gate 7 interpretation.

- [severity: minor] Validation code claims cluster disjointness but only checks member disjointness.
  Evidence: gate5_build_dataset.py step5_validate (lines 379-388) checks that no protein name appears in multiple splits (`all_members & members`), but does not check that no cluster ID appears in multiple splits. The log states "Split disjointness: verified OK (no cluster in multiple splits)."
  Why it matters: Cluster-level disjointness is guaranteed by construction in step3_split (all members of a cluster are assigned to the same split), so the claim is correct. But the validation code does not actually verify it -- it could not detect a bug in step3_split that assigned members of the same cluster to different splits. Independent verification confirms cluster disjointness holds (0 clusters appear in multiple splits).
  Required action: None blocking. The claim is correct.

- [severity: minor] 15 structures silently dropped during FASTA extraction (not logged).
  Evidence: 18,903 PDB files on disk but only 18,888 appear in the MMseqs2 cluster TSV. 15 structures (e.g., AF-Q3V541-F1-model_v6, AF-A0A5K4F7N6-F1-model_v6) failed sequence extraction and were silently excluded. The step2_cluster function catches exceptions with a bare `except: pass` (line 147).
  Why it matters: 15/18,903 (0.08%) is negligible and does not affect downstream results. However, the plan (Gate 5, item 2) explicitly requires "log every permanent failure. Do not silently drop them."
  Required action: None blocking. The attrition is trivial but the silent drop violates the plan's logging requirement.

SPOT-CHECK:
Independently recomputed split counts by replaying the clustering + splitting logic:
- Read cluster TSV (18,767 clusters, 18,888 members)
- Sorted cluster IDs, shuffled with seed 42, split 80/10/10
- Matched the 3,000 HDF5 entries against cluster assignments
- Result: train=2401, val=309, test=290 -- exact match with metadata.json and log

Additionally verified:
- HDF5 file size: 993,325,689 bytes = 0.99 GB (matches reported value)
- All 3,000 entries have consistent shapes: z=(N,8) float32, esm2_emb=(N,1280) float16, ca_coords=(N,3) float32
- All z vectors have L2 norm = 2.8284 (= sqrt(8)) exactly, confirming LayerNorm constraint
- Zero NaN values in z or ESM-2 embeddings across all 3,000 entries
- Length range [33, 256], mean=133.0, median=123 -- matches log
- 0 clusters appear in multiple splits (verified independently, not just from validation code)
- Gate 6's 20 evaluation targets are the first 20 alphabetical entries from the test split -- consistent

CHECKS PERFORMED:
1. Impossible/implausible numbers: checked, nothing found. Counts, sizes, and distributions are internally consistent and plausible.
2. Substituted methods: checked, MMseqs2 used instead of Foldseek for clustering (minor, documented above). No hand-rolled metrics.
3. Mismatched evaluation sets: checked, Gate 6 evaluation targets are from this dataset's test split (20/20 matched).
4. Reference-set contamination: checked, all structures are from AFDB (AlphaFold predictions by design per plan). Not a Gate 5 issue; the plan accounts for this in Gate 1.
5. Training-set leakage: checked, ProteinAE trained on AFDB so these structures are in-distribution. ESM-2 trained on UniRef sequences. For dataset building (encoding) this is by design. Evaluation interpretation (Gate 7) must note this.
6. Conclusions that depend on exclusions: checked, 0 failures reported, 15 structures silently dropped pre-clustering (0.08%, irrelevant to final 3000). No data excluded from the reported 3000.
7. Statistics hygiene: checked, length distribution reported with mean, median, range, and per-bin percentages. Split counts reported individually. Nothing found.
8. Tests that cannot fail: checked, the pass threshold (75% of target) is lenient but not vacuous -- encoding failures could bring it below. The actual result (100%) is clear.
9. Identifiers and data integrity: checked, all 3000 entries have valid AF-*-F1-model_v6 identifiers, zero NaN values, zero shape mismatches, zero truncated records.
