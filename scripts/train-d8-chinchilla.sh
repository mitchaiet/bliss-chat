#!/bin/bash
# Train d8 (~57M params) Chinchilla-optimal — middle ground between
# the small d6 (30M, fast on P4) and d12 (110M, slow but coherent).
#
# Run on the training host. Should take ~25-35 min depending on
# auto-computed iteration count.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
source "$HOME/nanochat/.venv/bin/activate"
WANDB_RUN="${WANDB_RUN:-dummy}"

python -m scripts.base_train \
    --depth=8 \
    --max-seq-len=1024 \
    --device-batch-size=32 \
    --total-batch-size=32768 \
    --target-param-data-ratio=20 \
    --eval-every=2000 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=2000 \
    --run="$WANDB_RUN"
