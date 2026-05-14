#!/bin/bash
# Train a d12 nanochat model (~110M params) — 50 min on CUDA workstation.
# Run this on the training machine inside the nanochat checkout .
set -euo pipefail
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate

# Need ~30 shards for proper Chinchilla scaling at 110M params
python -m nanochat.dataset -n 30

python -m scripts.base_train \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=32 \
    --total-batch-size=65536 \
    --eval-every=1000 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=1000 \
    --num-iterations=10000 \
    --run="${WANDB_RUN:-dummy}"
