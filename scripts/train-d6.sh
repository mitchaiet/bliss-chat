#!/bin/bash
# Train a tiny d6 nanochat model on a CUDA box (~1-2 min on CUDA workstation).
# Run this on the training machine (Ubuntu / training host) inside the nanochat checkout .
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
source .venv/bin/activate
WANDB_RUN="${WANDB_RUN:-dummy}"

# 1) Pretraining data + tokenizer (only needed once)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000

# 2) Pretrain d6 (~30M params) for 5K steps
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=500 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --num-iterations=5000 \
    --run="$WANDB_RUN"
