# leakage_audit — how much of the held-out score is neighbours? (2026-09-23)

**Jobs** 47938503 / 47941200 (Foldseek audit, `code/gate9_leakage_audit.py`),
47944460-63 (subset scoring, `code/gate7_score.py --names-file`).

## Question
The 473k-protein model scores 0.660 TM held-out against 0.609 for the 80k
model. The new training proteins were taken from the same AFDB index as the
validation set without structural clustering against it. How much of the
score, and of the gain, comes from training proteins that are structurally
close to the validation proteins?

## Method
For each of the 2,000 seed-42 validation proteins (the gate set is the first
100, the held-out slice positions 1000-1099), Foldseek TM-align search
(`-s 9.5`, prefilter on, `--exact-tmscore 1`) against all 473,184 training
structures as Ca pseudo-backbones, recording the best hit separately in the
original 80k train split and in the 393k AFDB additions. TM normalised by the
validation query's length ("qtm") is the leakage-relevant number; the max of
both normalisations (first pass) is an upper bound. Raw alignments:
`notes/gate9_val_vs_train_aln.tsv.gz`; per-protein results:
`notes/gate9_leakage_audit.json`.

## Nearest training structure (held-out slice, n = 100, qtm)
| threshold | original 80k train | with AFDB additions |
|---|---|---|
| nearest TM > 0.5 | 83% | 96% |
| nearest TM > 0.6 | 58% | 85% |
| nearest TM > 0.7 | 35% | 72% |
| nearest TM > 0.9 | 5% | 13% |
| mean nearest TM | 0.644 | 0.757 |

The other 1,800 validation proteins and the gate set give the same picture
within 2-3 points. The 100k split was clustered by MMseqs2 sequence identity
(30%), not by structure, so even the original train split contains close
structural relatives of most validation proteins.

## Scores on validation proteins WITHOUT a close training neighbour
Both best checkpoints, w=2, 25 Euler steps, K=8, coverage 100% in every row.
Subsets exclude the gate (selection) proteins.

| subset | n | 80k model TM / TM>0.5 / RMSD / best-of-8 | 473k model TM / TM>0.5 / RMSD / best-of-8 |
|---|---|---|---|
| all held-out (reference) | 100 | 0.609 / 71% / 10.06 / 0.668 | 0.660 / 74% / 9.01 / 0.716 |
| nearest train TM < 0.6 | 266 | 0.439 / 29% / 16.38 / 0.496 | 0.474 / 41% / 15.34 / 0.535 |
| nearest train TM < 0.5 | 93 | 0.396 / 14% / 18.52 / 0.453 | 0.430 / 24% / 17.25 / 0.487 |

## Reading
- **Most of the absolute score is neighbours.** On proteins with no training
  structure above TM 0.6, both models fall from ~0.63 to ~0.45 TM, and the
  correct-fold rate from ~72% to 29-41%. The models mainly reproduce folds
  they have seen.
- **The data gain is real but smaller than it looked**: +0.035 TM and +12
  points of correct folds on the no-neighbour subsets, versus +0.05 and +3
  points on the plain held-out slice. Both are within about one standard
  error on these subset sizes, so "real" is a fair reading, "large" is not.
- Novel-fold proteins are also intrinsically harder (odd folds, lower pLDDT),
  so part of the drop is difficulty, not leakage. ESMFold would drop on them
  too, though far less: the gap to it is the generalisation gap.

## Consequences
1. Every headline number in `reports/` is a "seen-fold" number. Model
   comparisons from here on report the no-neighbour subsets as well
   (`notes/val_noneighbour_lt0.5.txt`, `..._lt0.6.txt`).
2. Any further data step must structurally cluster the training set against
   validation and test first.
3. The breakthrough lever is generalisation, i.e. pair reasoning inside the
   generative model (Gate 10), not more data of the same kind.
