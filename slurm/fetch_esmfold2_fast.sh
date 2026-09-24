#!/bin/bash
cd /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/data/esmfold2_fast
TOK=$(cat ~/.cache/huggingface/token)
for f in $(cat filelist.txt); do
  case $f in images/*) continue;; esac
  curl -sL -H "Authorization: Bearer $TOK" -o $f "https://huggingface.co/biohub/ESMFold2-Fast/resolve/main/$f" && echo "got $f $(du -h $f | cut -f1)"
done
echo DOWNLOAD_DONE
