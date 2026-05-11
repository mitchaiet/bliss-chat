#!/bin/bash
# d12 trained Chinchilla-optimal (ratio 20 vs the original run's ratio 12).
# Same architecture, same XP int8 file size — just sees ~3.4x more training
# tokens. Drop-in replacement for MODEL.NCB.
#
# Wall clock estimate: ~80 minutes on RTX 6000 Blackwell.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
source "$HOME/nanochat/.venv/bin/activate"
WANDB_RUN="${WANDB_RUN:-dummy}"

# IMPORTANT: don't pass --num-iterations; nanochat auto-derives it from
# --target-param-data-ratio so we hit ratio 20 exactly.
python -m scripts.base_train \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=32 \
    --total-batch-size=65536 \
    --target-param-data-ratio=20 \
    --eval-every=2000 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=2000 \
    --run="$WANDB_RUN"
