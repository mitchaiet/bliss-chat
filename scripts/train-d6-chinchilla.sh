#!/bin/bash
# Train a Chinchilla-optimal d6 nanochat model (target ratio 20 vs the
# default 12). Should produce a much better-trained tiny model — same
# 30M params, but ~5x more tokens. Wall clock ~10-15 min on RTX 6000.
#
# Run on the training box (~/nanochat).
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
source "$HOME/nanochat/.venv/bin/activate"
WANDB_RUN="${WANDB_RUN:-dummy}"

# nanochat computes num_iterations from --target-param-data-ratio when
# --num-iterations is not set. Ratio 20 = Chinchilla-optimal.
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --target-param-data-ratio=20 \
    --eval-every=2000 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=2000 \
    --run="$WANDB_RUN"
