# ESM-ProteinAE data: deleted 2026-11-20

The ESM-ProteinAE **data** was deleted on 2026-11-20 at the user's instruction. The code, the
`ProteinAE_v1` decoder package, the run reports and `CLAUDE.md` were all kept, so this records what
the scripts will no longer find. `data/` is gitignored, so nothing here was ever in the repository
and none of it is recoverable from git history.

Recovered: **~1.43 TB**.

Deliberately kept: `data/esmc6b` (ESMC-6B weights, 24 GB) is shared with the HSP pull-down half
(`cm01_embed`, `pd03_embed`, `pd12_sae_features` default to it) and is a public downloadable model.
Also kept: `annot`, `campaign`, `export`, `external`, `fs_data`, `ga_data`, `pd_data`, `sae`,
`esmc6b_sae`, `study` — all pull-down.

## What went

| Path | Size | Files | Reproducible from |
|---|---:|---:|---|
| `data/phase1_dataset/` | 1.4 TB | 234 | `structures.tar.gz` was inside it, so **no** — see below |
| `data/esmfold2_fast/` | 25 GB | 14 | re-download |
| `data/esm2/` | 2.5 GB | 5 | re-download (ESM-2 650M, public) |
| `data/pdb/` | 99 MB | 12 | `gate15_build_pdb.py` |
| `data/afdb_clusters/` | 53 MB | 3 | `gate8_build_afdb.py` |
| `data/casp/` | 7.2 MB | 2 | `gate13_build_casp.py` |
| `data/ensembles/` | 195 KB | 3 | `gate22_build_ensembles.py` |
| `data/afdb_long_index.txt` | 2.6 MB | 1 | `gate18_make_long_val.py` |
| `data/afdb_short_extra_index.txt` | 10 MB | 1 | `gate8_build_afdb.py` |
| `data/genie2_afdb_index.txt` | 15 MB | 1 | `gate8_build_afdb.py` |

## The one that cannot be rebuilt

`CLAUDE.md` section 9b said the dataset was reproducible from the bundle because
`data/phase1_dataset/structures.tar.gz` (2.36 GB, 99,903 AFDB PDB files) shipped with it. That
archive lived **inside** the deleted directory, so that claim no longer holds. Rebuilding
`dataset_100k.h5` now means re-fetching the AFDB structures from source before
`gate5_build_dataset.py` and `gate6_100k.py` can run.

Checksums of the two checkpoints, from `checksums.md5`, in case a copy is found elsewhere:

```
c63bfc300060b036383e35b4c6834fe3  data/phase1_dataset/smallscale_final_40k.pt
84d69c6de2c7c29de7fb61defee170bf  ProteinAE_v1/checkpoints/ae_r1_d8_v1.ckpt
```

## Full listing as it stood at deletion

```
--- data/phase1_dataset ---
  59866882  phase1_dataset/afdb_long_seqs.fasta
   2032937  phase1_dataset/all_val_seqs.fasta
 797012758  phase1_dataset/backbone_100k.h5
 702077829  phase1_dataset/best_cd_174M_p64x6_esmc.pt
       610  phase1_dataset/best_cd_174M_p64x6_esmc.pt.meta.json
 204466065  phase1_dataset/best_h200_16L512.pt
       228  phase1_dataset/best_h200_16L512.pt.meta.json
 684505073  phase1_dataset/best_h200_24L768.pt
       237  phase1_dataset/best_h200_24L768.pt.meta.json
  32957083  phase1_dataset/best_h200_bb_bond.pt
       336  phase1_dataset/best_h200_bb_bond.pt.meta.json
  32956433  phase1_dataset/best_h200_bb.pt
       347  phase1_dataset/best_h200_bb.pt.meta.json
  32957343  phase1_dataset/best_h200_cabb_bond.pt
       337  phase1_dataset/best_h200_cabb_bond.pt.meta.json
  32957083  phase1_dataset/best_h200_ca_bond.pt
       336  phase1_dataset/best_h200_ca_bond.pt.meta.json
  32957473  phase1_dataset/best_h200_full_bs160.pt
       238  phase1_dataset/best_h200_full_bs160.pt.meta.json
  32956693  phase1_dataset/best_h200_full.pt
       229  phase1_dataset/best_h200_full.pt.meta.json
 698539387  phase1_dataset/best_lf_174M_esmc.pt
       474  phase1_dataset/best_lf_174M_esmc.pt.meta.json
 694596789  phase1_dataset/best_lf_174M_mix.pt
      1066  phase1_dataset/best_lf_174M_mix.pt.meta.json
 694595997  phase1_dataset/best_lf_174M.pt
       171  phase1_dataset/best_lf_174M.pt.meta.json
1836824891  phase1_dataset/best_lf_459M_afdb.pt
      1055  phase1_dataset/best_lf_459M_afdb.pt.meta.json
1836826035  phase1_dataset/best_lf_459M_afdb_rtx.pt
      1075  phase1_dataset/best_lf_459M_afdb_rtx.pt.meta.json
1836824891  phase1_dataset/best_lf_459M_ddp6.pt
       182  phase1_dataset/best_lf_459M_ddp6.pt.meta.json
1836824891  phase1_dataset/best_lf_459M_h200.pt
       182  phase1_dataset/best_lf_459M_h200.pt.meta.json
1843137695  phase1_dataset/best_lf_459M_nopair_long512_r4b.pt
       597  phase1_dataset/best_lf_459M_nopair_long512_r4b.pt.meta.json
1843137409  phase1_dataset/best_lf_459M_nopair_long512_r4.pt
       597  phase1_dataset/best_lf_459M_nopair_long512_r4.pt.meta.json
1836823461  phase1_dataset/best_lf_459M.pt
       171  phase1_dataset/best_lf_459M.pt.meta.json
 702007007  phase1_dataset/best_pf_174M_cond_L80.pt
       595  phase1_dataset/best_pf_174M_cond_L80.pt.meta.json
 702010153  phase1_dataset/best_pf_174M_cond_lastlayer.pt
       591  phase1_dataset/best_pf_174M_cond_lastlayer.pt.meta.json
 702008438  phase1_dataset/best_pf_174M_deposited.pt
       684  phase1_dataset/best_pf_174M_deposited.pt.meta.json
 702006321  phase1_dataset/best_pf_174M_distil.pt
       697  phase1_dataset/best_pf_174M_distil.pt.meta.json
 702008438  phase1_dataset/best_pf_174M_distil_x2.pt
       687  phase1_dataset/best_pf_174M_distil_x2.pt.meta.json
 706769449  phase1_dataset/best_pf_174M_p128x8.pt
       432  phase1_dataset/best_pf_174M_p128x8.pt.meta.json
 702039757  phase1_dataset/best_pf_174M_p64x6_esmc.pt
       535  phase1_dataset/best_pf_174M_p64x6_esmc.pt.meta.json
 702011525  phase1_dataset/best_pf_174M_p64x6_esmc_r4_lddt.pt
       584  phase1_dataset/best_pf_174M_p64x6_esmc_r4_lddt.pt.meta.json
 702009810  phase1_dataset/best_pf_174M_p64x6_esmc_r4.pt
       547  phase1_dataset/best_pf_174M_p64x6_esmc_r4.pt.meta.json
 702011868  phase1_dataset/best_pf_174M_p64x6_esmc_r4_tlate.pt
       581  phase1_dataset/best_pf_174M_p64x6_esmc_r4_tlate.pt.meta.json
 702011868  phase1_dataset/best_pf_174M_p64x6_esmc_r8_tlate.pt
       591  phase1_dataset/best_pf_174M_p64x6_esmc_r8_tlate.pt.meta.json
 697428474  phase1_dataset/best_pf_174M_p64x6.pt
       431  phase1_dataset/best_pf_174M_p64x6.pt.meta.json
 702015437  phase1_dataset/best_pf_174M_p64x6_rec_esmc.pt
       603  phase1_dataset/best_pf_174M_p64x6_rec_esmc.pt.meta.json
 702008781  phase1_dataset/best_pf_174M_twoteacher.pt
       687  phase1_dataset/best_pf_174M_twoteacher.pt.meta.json
1855631402  phase1_dataset/best_pf_459M_distil_x2.pt
       689  phase1_dataset/best_pf_459M_distil_x2.pt.meta.json
1845626510  phase1_dataset/best_pf_459M_esmc_afdb.pt
       526  phase1_dataset/best_pf_459M_esmc_afdb.pt.meta.json
1856680928  phase1_dataset/best_pf_459M_gen_pdrop50.pt
       679  phase1_dataset/best_pf_459M_gen_pdrop50.pt.meta.json
1855673595  phase1_dataset/best_pf_459M_p128x8_ctlft.pt
       536  phase1_dataset/best_pf_459M_p128x8_ctlft.pt.meta.json
1855676007  phase1_dataset/best_pf_459M_p128x8_esmc_afdb.pt
       537  phase1_dataset/best_pf_459M_p128x8_esmc_afdb.pt.meta.json
1856685678  phase1_dataset/best_pf_459M_p128x8_long512_cycle2.pt
       680  phase1_dataset/best_pf_459M_p128x8_long512_cycle2.pt.meta.json
1856682353  phase1_dataset/best_pf_459M_p128x8_long512.pt
       681  phase1_dataset/best_pf_459M_p128x8_long512.pt.meta.json
1856686153  phase1_dataset/best_pf_459M_p128x8_long512_scratch.pt
       691  phase1_dataset/best_pf_459M_p128x8_long512_scratch.pt.meta.json
1855673595  phase1_dataset/best_pf_459M_p128x8_pdbft.pt
       526  phase1_dataset/best_pf_459M_p128x8_pdbft.pt.meta.json
1855646531  phase1_dataset/best_pf_459M_p128x8_rec_esmc_afdb.pt
       595  phase1_dataset/best_pf_459M_p128x8_rec_esmc_afdb.pt.meta.json
1855631877  phase1_dataset/best_pf_459M_twoteacher.pt
       699  phase1_dataset/best_pf_459M_twoteacher.pt.meta.json
3361199469  phase1_dataset/best_pf_840M_longfrac40.pt
       680  phase1_dataset/best_pf_840M_longfrac40.pt.meta.json
3361204140  phase1_dataset/best_pf_840M_p128x8_long512_1p3M.pt
       690  phase1_dataset/best_pf_840M_p128x8_long512_1p3M.pt.meta.json
3361205178  phase1_dataset/best_pf_840M_p128x8_long512_anneal.pt
       690  phase1_dataset/best_pf_840M_p128x8_long512_anneal.pt.meta.json
     24192  phase1_dataset/casp512_seqs.fasta
     13714  phase1_dataset/casp_seqs.fasta
3461148544  phase1_dataset/dataset_100k_esmc_ctl_0.h5
3423434339  phase1_dataset/dataset_100k_esmc_ctl_1.h5
3420341844  phase1_dataset/dataset_100k_esmc_ctl_2.h5
3464069231  phase1_dataset/dataset_100k_esmc_ctl_3.h5
3446865360  phase1_dataset/dataset_100k_esmc_ctl_4.h5
3460280739  phase1_dataset/dataset_100k_esmc_distil_0.h5
3422554995  phase1_dataset/dataset_100k_esmc_distil_1.h5
3419466827  phase1_dataset/dataset_100k_esmc_distil_2.h5
3463194239  phase1_dataset/dataset_100k_esmc_distil_3.h5
3445986359  phase1_dataset/dataset_100k_esmc_distil_4.h5
61466473088  phase1_dataset/dataset_100k_esmc.h5
61475635252  phase1_dataset/dataset_100k_esmc_L80.h5
32719901096  phase1_dataset/dataset_100k.h5
72691890256  phase1_dataset/dataset_afdb_long_0_esmc.h5
1480750982  phase1_dataset/dataset_afdb_long_0.h5
72601697072  phase1_dataset/dataset_afdb_long_1_esmc.h5
1479434438  phase1_dataset/dataset_afdb_long_1.h5
72704923728  phase1_dataset/dataset_afdb_long_2_esmc.h5
1480896665  phase1_dataset/dataset_afdb_long_2.h5
72642744048  phase1_dataset/dataset_afdb_long_3_esmc.h5
1480020682  phase1_dataset/dataset_afdb_long_3.h5
93873638496  phase1_dataset/dataset_afdb_short2_0_esmc.h5
3063188602  phase1_dataset/dataset_afdb_short2_0.h5
93776142668  phase1_dataset/dataset_afdb_short2_1_esmc.h5
3061703694  phase1_dataset/dataset_afdb_short2_1.h5
93868645084  phase1_dataset/dataset_afdb_short2_2_esmc.h5
3063194619  phase1_dataset/dataset_afdb_short2_2.h5
93680637548  phase1_dataset/dataset_afdb_short2_3_esmc.h5
3060471921  phase1_dataset/dataset_afdb_short2_3.h5
135038212624  phase1_dataset/dataset_afdb_train_0_esmc.h5
4098460896  phase1_dataset/dataset_afdb_train_0.h5
135399192208  phase1_dataset/dataset_afdb_train_1_esmc.h5
4105630357  phase1_dataset/dataset_afdb_train_1.h5
  90967408  phase1_dataset/dataset_apo_esmc.h5
   1798572  phase1_dataset/dataset_apo.h5
 216269320  phase1_dataset/dataset_cameo22_esmc.h5
   3096456  phase1_dataset/dataset_cameo22.h5
 115382272  phase1_dataset/dataset_casp512_esmc.h5
   2078980  phase1_dataset/dataset_casp512.h5
  32443052  phase1_dataset/dataset_casp_esm2.h5
  63649452  phase1_dataset/dataset_casp_esmc.h5
   1200320  phase1_dataset/dataset_casp.h5
  85108436  phase1_dataset/dataset_codnas_esmc.h5
   1624532  phase1_dataset/dataset_codnas.h5
 605261772  phase1_dataset/dataset_exp_val_esmc.h5
  15687445  phase1_dataset/dataset_exp_val.h5
1814506192  phase1_dataset/dataset_long_val_esmc.h5
  36705179  phase1_dataset/dataset_long_val.h5
24435384360  phase1_dataset/dataset_pdb_long_esmc.h5
 503786852  phase1_dataset/dataset_pdb_long.h5
13947290880  phase1_dataset/dataset_pdb_train_esmc.h5
 404776173  phase1_dataset/dataset_pdb_train.h5
2808379889  phase1_dataset/last_cd_174M_p64x6_esmc.ckpt
 613432875  phase1_dataset/last_h200_16L512.ckpt
2053563477  phase1_dataset/last_h200_24L768.ckpt
  98896867  phase1_dataset/last_h200_bb_bond.ckpt
  98894027  phase1_dataset/last_h200_bb.ckpt
  98898707  phase1_dataset/last_h200_cabb_bond.ckpt
  98897507  phase1_dataset/last_h200_ca_bond.ckpt
  98897163  phase1_dataset/last_h200_full_bs160.ckpt
  98894779  phase1_dataset/last_h200_full.ckpt
2778448971  phase1_dataset/last_lf_174M.ckpt
2794213619  phase1_dataset/last_lf_174M_esmc.ckpt
2778445979  phase1_dataset/last_lf_174M_mix.ckpt
7347398625  phase1_dataset/last_lf_459M_afdb.ckpt
7347379057  phase1_dataset/last_lf_459M_afdb_rtx.ckpt
7347358393  phase1_dataset/last_lf_459M.ckpt
7347366265  phase1_dataset/last_lf_459M_ddp6.ckpt
7347371961  phase1_dataset/last_lf_459M_h200.ckpt
7372638213  phase1_dataset/last_lf_459M_nopair_long512_r4b.ckpt
7372626757  phase1_dataset/last_lf_459M_nopair_long512_r4.ckpt
 937451243  phase1_dataset/last_lf_base.ckpt
  13879129  phase1_dataset/last_lf_ddp_smoke.ckpt
   4379627  phase1_dataset/last_lf_online_smoke.ckpt
   4376321  phase1_dataset/last_lf_smoke.ckpt
2808109863  phase1_dataset/last_pf_174M_cond_L80.ckpt
2808131669  phase1_dataset/last_pf_174M_cond_lastlayer.ckpt
2808117380  phase1_dataset/last_pf_174M_deposited.ckpt
2808118125  phase1_dataset/last_pf_174M_distil.ckpt
2808120900  phase1_dataset/last_pf_174M_distil_x2.ckpt
2827175405  phase1_dataset/last_pf_174M_p128x8.ckpt
2789802480  phase1_dataset/last_pf_174M_p64x6.ckpt
2808245089  phase1_dataset/last_pf_174M_p64x6_esmc.ckpt
2808148152  phase1_dataset/last_pf_174M_p64x6_esmc_r4.ckpt
2808146249  phase1_dataset/last_pf_174M_p64x6_esmc_r4_lddt.ckpt
2808206310  phase1_dataset/last_pf_174M_p64x6_esmc_r4_tlate.ckpt
2808137958  phase1_dataset/last_pf_174M_p64x6_esmc_r8_tlate.ckpt
2808148741  phase1_dataset/last_pf_174M_p64x6_rec_esmc.ckpt
2808126049  phase1_dataset/last_pf_174M_twoteacher.ckpt
7422641516  phase1_dataset/last_pf_459M_distil_x2.ckpt
7382618420  phase1_dataset/last_pf_459M_esmc_afdb.ckpt
7426809934  phase1_dataset/last_pf_459M_gen_pdrop50.ckpt
7422824383  phase1_dataset/last_pf_459M_p128x8_ctlft.ckpt
7422830339  phase1_dataset/last_pf_459M_p128x8_esmc_afdb.ckpt
7426836897  phase1_dataset/last_pf_459M_p128x8_long512.ckpt
7426830264  phase1_dataset/last_pf_459M_p128x8_long512_cycle2.ckpt
7426892649  phase1_dataset/last_pf_459M_p128x8_long512_scratch.ckpt
7422824383  phase1_dataset/last_pf_459M_p128x8_pdbft.ckpt
7422699123  phase1_dataset/last_pf_459M_p128x8_rec_esmc_afdb.ckpt
7422643869  phase1_dataset/last_pf_459M_twoteacher.ckpt
13444899617  phase1_dataset/last_pf_840M_longfrac40.ckpt
13444938902  phase1_dataset/last_pf_840M_p128x8_long512_1p3M.ckpt
13444932912  phase1_dataset/last_pf_840M_p128x8_long512_anneal.ckpt
    376069  phase1_dataset/long_val_seqs.fasta
  23851329  phase1_dataset/old/best_codesign_smoke.pt
       531  phase1_dataset/old/best_codesign_smoke.pt.meta.json
  23827353  phase1_dataset/old/best_disteval_smoke.pt
       589  phase1_dataset/old/best_disteval_smoke.pt.meta.json
1843136551  phase1_dataset/old/best_lf_459M_nopair_long512.pt
       597  phase1_dataset/old/best_lf_459M_nopair_long512.pt.meta.json
  24089005  phase1_dataset/old/best_long_smoke.pt
       676  phase1_dataset/old/best_long_smoke.pt.meta.json
 702015437  phase1_dataset/old/best_pf_174M_p64x6_rec_esmc.pt
       603  phase1_dataset/old/best_pf_174M_p64x6_rec_esmc.pt.meta.json
1855688323  phase1_dataset/old/best_pf_459M_p128x8_rec_esmc_afdb.pt
       572  phase1_dataset/old/best_pf_459M_p128x8_rec_esmc_afdb.pt.meta.json
  23840156  phase1_dataset/old/best_rec_smoke.pt
       569  phase1_dataset/old/best_rec_smoke.pt.meta.json
  23827107  phase1_dataset/old/best_struct_smoke.pt
       582  phase1_dataset/old/best_struct_smoke.pt.meta.json
  95434806  phase1_dataset/old/last_codesign_smoke.ckpt
  95338442  phase1_dataset/old/last_disteval_smoke.ckpt
7372619141  phase1_dataset/old/last_lf_459M_nopair_long512.ckpt
  96384646  phase1_dataset/old/last_long_smoke.ckpt
2808121669  phase1_dataset/old/last_pf_174M_p64x6_rec_esmc.ckpt
7422874611  phase1_dataset/old/last_pf_459M_p128x8_rec_esmc_afdb.ckpt
  95395609  phase1_dataset/old/last_rec_smoke.ckpt
  95337320  phase1_dataset/old/last_struct_smoke.ckpt
   4814976  phase1_dataset/pdb_long_seqs.fasta
  89134597  phase1_dataset/short2_seqs.fasta
  32957083  phase1_dataset/smallscale_final_40k.pt
       112  phase1_dataset/smallscale_final_40k.pt.meta.json
2533581089  phase1_dataset/structures.tar.gz
  76249582  phase1_dataset/train_seqs.fasta
   1656868  phase1_dataset/val_seqs.fasta
--- data/esmfold2_fast ---
      8214  esmfold2_fast/config.json
       354  esmfold2_fast/filelist.txt
      1580  esmfold2_fast/.gitattributes
4983974660  esmfold2_fast/model-00001-of-00006.safetensors
4997408792  esmfold2_fast/model-00002-of-00006.safetensors
4979069192  esmfold2_fast/model-00003-of-00006.safetensors
4934556472  esmfold2_fast/model-00004-of-00006.safetensors
4997460576  esmfold2_fast/model-00005-of-00006.safetensors
1242793352  esmfold2_fast/model-00006-of-00006.safetensors
      2916  esmfold2_fast/modeling_esmfold2_remote.py
    191458  esmfold2_fast/model.safetensors.index.json
     10752  esmfold2_fast/README.md
       327  esmfold2_fast/tokenizer_config.json
      2879  esmfold2_fast/tokenizer.json
--- data/esm2 ---
       724  esm2/esm2_t33_650M_UR50D/config.json
2609506392  esm2/esm2_t33_650M_UR50D/model.safetensors
       125  esm2/esm2_t33_650M_UR50D/special_tokens_map.json
        95  esm2/esm2_t33_650M_UR50D/tokenizer_config.json
        93  esm2/esm2_t33_650M_UR50D/vocab.txt
--- data/pdb ---
       175  pdb/casp_struct_homologs_gt0.7.txt
  22545420  pdb/clusters-by-entity-30.txt
     32461  pdb/exp_val_chains.tsv
      2639  pdb/long_casp_struct_homologs_gt0.7.txt
      2569  pdb/long_val_homologs_gt0.5.txt
   5329373  pdb/pdb_entry_type.txt
  67128902  pdb/pdb_seqres.txt.gz
   3050586  pdb/resolu.idx
   3770288  pdb/selected_chains.fasta
    468850  pdb/selected_chains_long.tsv
    661797  pdb/selected_chains.tsv
      2949  pdb/val_homologs_gt0.5.txt
--- data/afdb_clusters ---
   4650712  afdb_clusters/afdb_long_257_512_plddt70.tsv
        26  afdb_clusters/afdb_long_casp_gt0.9.txt
  49929202  afdb_clusters/repmeta_v6.tsv.gz
--- data/casp ---
   1657982  casp/casp15.targets.TS-domains.public_12.20.2022.tar.gz
   5870506  casp/casp16.targets_monomer_trimmed2domains.tgz
--- data/ensembles ---
     47324  ensembles/apo.csv
    107206  ensembles/cameo2022.csv
     44361  ensembles/codnas.csv
--- loose index files ---
   2678505  afdb_long_index.txt
  10440473  afdb_short_extra_index.txt
  15030781  genie2_afdb_index.txt
```
