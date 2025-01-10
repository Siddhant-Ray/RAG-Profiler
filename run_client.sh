#!/bin/bash

# Set the environment variables
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

export HF_HOME=/tmp/tf_cache
# export TIKTOKENS_CACHE_DIR=./llangchain_cache
export TOKENIZERS_PARALLELISM=false

# Use the arguments from bash directly using $@

LD_LIBRARY_PATH=/dataheart/siddhantray/lmcache_docs/rag_profiler/venv_rag_pf/lib/python3.12/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH
python profile_test.py $@
