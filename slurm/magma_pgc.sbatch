#!/bin/bash
#SBATCH -J magma_pgc
#SBATCH -A kempner_bsabatini_lab
#SBATCH -p kempner_rtx
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 0-02:00
#SBATCH -o logs/magma_pgc_%j.log
#SBATCH -e logs/magma_pgc_%j.log
# MAGMA v1.10 gene-level GWAS tests (SNP-wise mean model) for PGC3 schizophrenia (EUR) and PGC bipolar
# 2024 (EUR, no 23andMe), 1000 Genomes EUR LD reference, NCBI build 37 genes, window 35 kb up / 10 kb down
# (as in PGC3). CPU only; on kempner_rtx because the CPU account cannot submit (MaxSubmit 0). N per SNP = 2 x the half-effective N the files report. 16 chromosome batches run in parallel.
set -eu
G=/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/data/annot/gwas
cd $G; M=$G/magma/magma
[ -f g1000_eur/g1000_eur.bim ] || { unzip -o -q g1000_eur.zip -d g1000_eur; }
B=$(ls g1000_eur/*.bim | head -1); B=${B%.bim}
[ -f eur_35_10.genes.annot ] || $M --annotate window=35,10 --snp-loc $B.bim --gene-loc geneloc/NCBI37.3.gene.loc --out eur_35_10
zcat PGC3_SCZ_eur.tsv.gz | grep -v '^##' | awk -F'\t' 'NR==1{print "SNP\tP\tN";next}{print $2"\t"$11"\t"2*$14}' > scz.pval
zcat bip2024_eur_no23andMe.gz | awk 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; print "SNP\tP\tN";next}{print $c["SNP"]"\t"$c["P"]"\t"2*$c["Neff_half"]}' > bd.pval
wc -l scz.pval bd.pval
for t in scz bd; do
  for k in $(seq 1 16); do $M --batch $k 16 --bfile $B --pval $t.pval ncol=N --gene-annot eur_35_10.genes.annot --out magma_$t > /dev/null & done
  wait
  $M --merge magma_$t --out magma_$t
  wc -l magma_$t.genes.out
done
echo MAGMA_DONE
