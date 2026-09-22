#!/bin/bash
cd /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/data/esm2/esm2_t33_650M_UR50D
for f in config.json vocab.txt tokenizer_config.json special_tokens_map.json model.safetensors; do
  curl -sL -o $f "https://huggingface.co/facebook/esm2_t33_650M_UR50D/resolve/main/$f" && echo "got $f $(du -h $f | cut -f1)"
done
echo DOWNLOAD_DONE
