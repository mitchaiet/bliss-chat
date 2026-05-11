#!/bin/bash
# SFT attempt with the Muon -> AdamW swap (in-loop, via the patched
# chat_sft.py). Cold-start the optimizer state so we don't drag any
# Muon momentum from the pretrain checkpoint.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source "$HOME/nanochat/.venv/bin/activate"

python -m scripts.chat_sft \
    --model-tag=d12_patched \
    --max-seq-len=1024 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --embedding-lr=0.00005 \
    --unembedding-lr=0.00005 \
    --matrix-lr=0.00005 \
    --init-lr-frac=0.1 \
    --warmup-ratio=0.05 \
    --load-optimizer=0 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --chatcore-every=-1 \
    --num-iterations=500 \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1 \
    --run=dummy
