#!/bin/bash
cd /n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae/data/esmc6b
TOK=$(cat ~/.cache/huggingface/token)
for f in config.json special_tokens_map.json tokenizer.json modeling_esmc_remote.py model.safetensors.index.json model-00001-of-00006.safetensors model-00002-of-00006.safetensors model-00003-of-00006.safetensors model-00004-of-00006.safetensors model-00005-of-00006.safetensors model-00006-of-00006.safetensors; do
  curl -sL -H "Authorization: Bearer $TOK" -o $f "https://huggingface.co/biohub/ESMC-6B/resolve/main/$f" && echo "got $f $(du -h $f | cut -f1)"
done
echo DOWNLOAD_DONE
