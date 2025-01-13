#!/bin/bash

# Set the environment variables
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

export HF_HOME=/tmp/tf_cache
# export TIKTOKENS_CACHE_DIR=./llangchain_cache
export TOKENIZERS_PARALLELISM=false
export LLAMA_INDEX_CACHE_DIR="/tmp/sid_llama/llama_index_cache"

# Use the arguments from bash directly using $@

LD_LIBRARY_PATH=/dataheart/siddhantray/lmcache_docs/rag_profiler/venv_test_req/lib/python3.12/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH

python local_server.py $@