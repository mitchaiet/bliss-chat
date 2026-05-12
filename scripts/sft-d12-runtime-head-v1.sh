#!/bin/bash
# Exact-runtime Bliss SFT on v4 data, freezing the transformer and training
# only lm_head.weight.
set -euo pipefail

BLISS_REPO_DIR="${BLISS_REPO_DIR:-$HOME/bliss-sft-v1}"
export BLISS_MODEL_TAG="${BLISS_MODEL_TAG:-d12_runtime_head_v1}"
export BLISS_SFT_JSONL="${BLISS_SFT_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v4_train.jsonl}"
export BLISS_VAL_JSONL="${BLISS_VAL_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v4_val.jsonl}"
export BLISS_TRAIN_SCOPE=lm_head
export BLISS_GRAD_CLIP="${BLISS_GRAD_CLIP:-0.10}"
export BLISS_ABORT_LOSS="${BLISS_ABORT_LOSS:-8.0}"
export BLISS_FIRST_TOKEN_WEIGHT="${BLISS_FIRST_TOKEN_WEIGHT:-1.0}"
export BLISS_END_TOKEN_WEIGHT="${BLISS_END_TOKEN_WEIGHT:-2.0}"
export BLISS_EMBEDDING_LR=0.0
export BLISS_MATRIX_LR=0.0
export BLISS_UNEMBEDDING_LR="${BLISS_UNEMBEDDING_LR:-0.00005}"
export BLISS_NUM_ITERATIONS="${BLISS_NUM_ITERATIONS:-800}"

exec "$BLISS_REPO_DIR/scripts/sft-d12-runtime-v1.sh"
