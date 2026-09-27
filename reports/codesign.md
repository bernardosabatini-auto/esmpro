# codesign — joint sequence + structure generation (gate 20 / gate 24)

**What** A SimpleDesign-style co-design model: one trunk predicts the latent flow velocity *and* the masked residues of the sequence. `code/gate20_codesign.py` adds a sequence head to the pair-flow network, embeds masked sequences with ESMC (mask token id 32) and trains a cross-entropy term weighted by an independently sampled sequence-corruption time `t_seq`, with probability `p_fold` of a pure folding step. `code/gate24_codesign_eval.py` evaluates three modes on 100 held-out validation proteins (offset 1000, 25 structure steps, 10 unmasking rounds, guidance 2).

**Run** `cd_174M_p64x6_esmc` (job 48782843, 4 RTX, 175.5M, pair 64x6, ESMC conditioning, the 80k short set), paused at epoch 20 to free the node; resumable from `last_cd_174M_p64x6_esmc.ckpt`. Evaluation job 48842723.

## Does the joint objective cost folding accuracy?
Validation TM every two epochs, against the folding-only twin trained on identical data:

| epoch | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| co-design | 0.288 | 0.408 | 0.557 | 0.618 | 0.632 | 0.644 | 0.653 | 0.660 | 0.668 | 0.672 |
| folding only | 0.357 | 0.558 | 0.643 | 0.664 | 0.677 | 0.690 | 0.694 | 0.696 | 0.705 | 0.705 |

The sequence head costs 0.033 TM at epoch 20, down from 0.069 at epoch 6: the tax is real but shrinking, and the joint model is not diverging from the folding solution.

## The three modes at epoch 20
| quantity | epoch ~12 | epoch 20 |
|---|---|---|
| fold TM (t_seq = 1, sanity) | 0.645 | **0.699**, coverage 100 % |
| inverse-fold native recovery | 0.14 | **0.230** |
| inverse-fold scTM (own model refolds the design) | 0.26 | 0.366 |
| co-design scTM | 0.31 | 0.350, 8 % > 0.5 |
| co-design identity to the native sequence | 0.11 | 0.105 |
| co-design structure TM to the native structure | — | 0.284 |

## The sequences are degenerate, and that is the headline
Composition of the 100 designs against their natives:

| | Shannon entropy | most common residue's share |
|---|---|---|
| co-design sequences | 0.16 bits | 0.98 |
| inverse-fold sequences | 2.35 bits | 0.41 |
| native sequences | 3.96 bits | 0.13 |

The co-design pathway has collapsed outright: a typical design is poly-leucine with a few glycines. Its scTM of 0.350 is therefore an artifact, not a designability measurement, since the model folds a poly-leucine sequence into a helix that partly resembles the helical structure it co-generated. Do not quote it.

The inverse-folding pathway is degenerate but not collapsed, and its recovery is genuinely above the baselines that its own composition bias would produce:

| baseline | recovery |
|---|---|
| uniform over 20 residues | 0.050 |
| sampling from the validation composition | 0.060 |
| always predict L (the modal residue, 9.9 % of positions) | 0.097 |
| **model** | **0.230** |

Per-native-residue recall shows where the signal is: glycine 0.61, leucine 0.43, alanine 0.24, arginine 0.19, glutamate 0.16, valine 0.08. The model recovers turn and helix-core classes and little else, which is what a weak but structurally conditioned sequence model looks like.

## Why it collapses: confirmed as a decoding artifact (job 48845318)
`unmask_round` filled the most confident masked positions with the **argmax** residue. For a head whose marginal is dominated by leucine that is self-reinforcing: each round commits more leucines, ESMC re-embeds a low-complexity sequence, and the next round is more confident still. Re-running the identical checkpoint with the residue **sampled** at temperature 1 (confidence still orders the positions; `--temp`, now the recommended setting) settles it:

| | argmax | temperature 1 | native |
|---|---|---|---|
| co-design sequence entropy | 0.16 bits | **3.95 bits** | 3.96 bits |
| co-design most common residue | 0.98 | **0.14** | 0.13 |
| inverse-fold sequence entropy | 2.35 bits | **3.89 bits** | 3.96 bits |
| inverse-fold native recovery | 0.230 | 0.179 | — |
| inverse-fold scTM | 0.366 | **0.378** | — |
| co-design scTM | 0.350 | 0.321 | — |
| co-design identity to native | 0.105 | 0.069 | — |

Sampling restores native-like composition exactly, and self-consistency does not suffer: the inverse-folding scTM is slightly **better** sampled than argmax (0.378 vs 0.366). So the head's per-position distributions were fine and only the decoder was pathological. Recovery falls from 0.230 to 0.179 because argmax maximises recovery by construction, but 0.179 is still 1.8x the always-modal-residue baseline of 0.097 and 3x the composition-sampling baseline of 0.060. Fold TM is unchanged (0.691 vs 0.699), as it must be, since folding does not use the sequence head.

The honest designability number is therefore **co-design scTM 0.321 with 5 % above 0.5**, from compositionally realistic sequences. That is weak, and it is the number to improve.

## Reading
1. Joint sequence and structure generation runs in this architecture and costs 0.033 TM of folding accuracy at epoch 20, shrinking with training.
2. The inverse-folding pathway has real signal: 0.179 recovery sampled (1.8x the always-modal-residue baseline, 3x composition sampling), 0.230 with argmax, from sequences whose composition matches natives.
3. The co-design pathway looked collapsed only because of argmax unmasking; sampled at temperature 1 it produces native-like composition with the same self-consistency, and its designability is scTM 0.321 with 5 % above 0.5. Use `--temp 1`, never argmax.
4. Every self-consistency number here refolds the design with the **same** model, which is the weakest possible check. The designs are written to `notes/cd_174M_p64x6_esmc_designs.fasta` for an independent ESMFold2 fold, which is the number that would actually count.
5. This is a 175M prototype stopped at epoch 20 on the 80k set. It establishes the mechanism, not a competitive design result.
