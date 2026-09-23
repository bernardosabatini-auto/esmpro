# afdb_build — Gate 8 training-set scale-up (2026-09-22)

**Jobs** 47844170 (shard 0/2) and 47844171 (shard 1/2), kempner_rtx, one GPU +
15 cores each, 75 and 73 minutes. `code/gate8_build_afdb.py`,
`slurm/build_afdb.sbatch`.

## What was built
The Genie2 AFDB index (`afdbreps_l-256_plddt_80`: AFDB cluster
representatives, length <= 256, mean pLDDT >= 80; 588,571 entries) minus the
99,832 entries already in `dataset_100k.h5` leaves 488,739 candidates. For
each: download the v6 model from EBI, read N/CA/C/O directly from the PDB
(a direct parser, validated identical to ProteinAE's `ProteinProcessor` on the
backbone atoms), PCA-canonicalise with the shipped `canonicalize.py`, encode
with the frozen ProteinAE encoder in batches of 64 on the GPU, and store the
8-dim latent, canonical CA and N/CA/C coordinates, sequence, mean pLDDT and
AFDB version. No ESM embeddings are stored: training uses
`gate7_latent_flow.py --esm online`, which runs frozen ESM-2 per batch.

| | shard 0 | shard 1 |
|---|---|---|
| candidates | 244,370 | 244,369 |
| written | 196,675 | 196,856 |
| not served by EBI (404 on v6 and v4) | 47,542 | 47,331 |
| outside 32-256 residues | 153 | 182 |
| wall time | 4509 s | 4407 s |
| file size | 4.10 GB | 4.11 GB |

**Total: 393,531 new training proteins**, 4.9x the original train split;
with the 79,653 of the 100k set the training set is 473,184 proteins.
Validation and test are unchanged (the 100k splits), so every number stays
comparable.

## Checks
- On 13 entries that overlap the 100k set, re-encoded latents match the
  stored ones to 0.06% relative error and CA coordinates exactly; sequences
  match. The direct parser matches the slow parser to 1e-3 in latent space.
- Sampled records: lengths 32-256, mean pLDDT ~85, latents finite with
  per-residue L2 = 2.83 (on the LayerNorm manifold).
- 19.4% of the index is no longer served by EBI. The original 100k build saw
  a similar 15% failure rate, so this is the index aging, not a fault.

## Caveats
- The Genie2 index is AFDB cluster representatives, so redundancy within
  the new set is limited, but no fresh structural clustering was run against
  the 100k validation/test splits. Homologs of validation proteins may be in
  the new training data. The 100k set's own split was MMseqs2-clustered, and
  section 9b of CLAUDE.md notes the split is already lenient. A Foldseek
  clustering of the full 573k set against val/test is the right follow-up
  before any published claim.
- The PDB files were not kept (~60 GB); the builder is resumable and re-runs
  from the index in ~75 minutes per shard.

## Consumer
`slurm/launch_afdb_big.sh`: 459M latent flow, online ESM-2 with learnable
all-layer mix, 2 H200 nodes (8 GPUs), train = 100k train split + both shards.
