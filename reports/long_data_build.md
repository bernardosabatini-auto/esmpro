# long_data_build — training and test data for 257-512-residue proteins

**Built 2026-09-24** (user request: "Download and prepare the data to extend to 500 AA"). Jobs 48294266/67/79/80 (AFDB shards, one RTX each, ~30 min: 14 min download+encode at 48 proteins/s, 17 min ESMC), 48296051 (PDB + CASP, 12 min). Code: `gate8_build_afdb.py`, `gate15_select_pdb.py`, `gate15_build_pdb.py`, `gate13_build_casp.py`, `gate11_precompute_esmc.py` (all now take the length window from `BUILD_MAX_LEN`), `gate18_make_long_val.py`. Job scripts `slurm/build_long_afdb.sbatch`, `slurm/build_long_pdb_casp.sbatch`.

## Prerequisite (papers_and_500aa.md)
The frozen ProteinAE round-trips 257-640-residue experimental chains at TM 0.999 / 0.16-0.22 A; the latent stays on its manifold. Nothing in the autoencoder needed retraining.

## Sources and selection
| set | source | selection | final count |
|---|---|---|---|
| **AFDB long train** | AFDB v6 Foldseek cluster representatives, metadata `2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv.gz` (2.60M reps) | 257-512 residues (510k), mean pLDDT >= 70 (161k; only 77k have >= 80, so 70 was used and `plddt_mean` is stored per protein for filtering at training time) | 160,800 built, 0 download failures; **159,799 train** after hold-out and guard |
| **AFDB long val** | same | 1,000 drawn (seed 42) from shard 3 | `dataset_long_val{,_esmc}.h5` (val split; train split empty) |
| **PDB long train** | `pdb_seqres`, X-ray/EM <= 3 A, SEQRES 257-512, standard residues (46.7k unique sequences) | CASP<=512 sequence guard (115 removed), MMseqs2 50 % identity clusters -> 16,431 best-resolution chains; built with >= 80 % of SEQRES observed | 13,865 built; **13,133** after guards |
| **CASP <= 512 test** | CASP15 TS-domains + CASP16 monomer domains | length 32-512, phase duplicates removed | **109 domains** (80 <= 256 + 29 long); `dataset_casp512{,_esmc}.h5`, names `notes/casp_domains_le512.txt` |

Length statistics (AFDB long): min 257, median 334, max 512, mean 347; median pLDDT 78.6, 44 % >= 80. Embeddings: ESMC-6B last layer, fp16, ragged; 4 x 72.7 GB (AFDB) + PDB long; host-RAM cache for the full mixed set is ~90 GB per rank on the H200 nodes (1.4 TB requested), so no compression was needed.

## Leakage guards applied (all sequence-level unless stated)
- PDB long vs **validation** (100k val + long val): 367 chains > 50 % identity removed (26 were > 90 %).
- PDB long vs **CASP <= 512**, structural (Foldseek TM-align): 365 chains with TM > 0.7 to a CASP domain removed. (Sequence guard at selection: 115.)
- AFDB long vs **CASP <= 512**: 54/109 domains have a hit, 6 above 50 % identity, **1 above 90 %** — that protein removed. As with the short set, most CASP domains have a training *fold* neighbour (fold-space coverage), not a sequence neighbour.
- Long val vs long train: cluster representatives are by construction distinct Foldseek clusters; no extra filter.

## Controls
- Round trip (true latent -> decoder) on the long PDB chains: TM 0.999 / 0.18 A (200/200); on CASP <= 512: TM 0.997 / 0.20 A (109/109). Coverage 100 %.
- Every AFDB long latent has per-residue L2 = 2.828 (on the manifold).
- End-to-end smoke test of the 512 window (`long_smoke`, 2 RTX, DDP, repeated batching R = 2, long validation split): batches up to 448 residues, evaluation on long proteins with 100 % coverage, no errors.

## Training-side changes made for the long stage
`ESM_PROAE_MAX_LEN=512` sets the window (RamSplit caches, dataset padding, position table, scorers); the residue budget stays bs x 256^2 so a 512-residue batch holds bs/4 proteins (floor fixed from bs to 1); `--val-h5` takes the selection split from the long hold-out; `--warm-start` from a 256-window checkpoint extends the position table (learned rows kept, new rows = mean row). Profile on real 257-512 batches (one RTX, 459M): budget 36 -> 6.5k residues/step in 1.5 s at 38 GB; R = 4 at budget 12 -> 7.6k residues/step in 1.1 s at 37 GB. Launcher: `slurm/launch_long_finetune.sh` (8 H200, warm start, R copies, long validation).

## What is not done
- No spatial/contiguous cropping: proteins are whole (<= 512); anything longer is excluded, so the model still cannot fold > 512.
- pLDDT 70-80 AFDB proteins (52 % of the long set) are lower-confidence targets; the first long run uses them all, and a >= 80 subset is one flag away if the long-val curve says they hurt.
- CASP domains longer than 512 (39 of the 204 units) remain excluded.
