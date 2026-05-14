#!/bin/bash
# SFT attempt #N: force COMPUTE_DTYPE=torch.float32 to test hypothesis 3
# from 10-known-issues.md (bf16 forward overflow on long SmolTalk samples).
#
# Run on the training host.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE=float32
source "$HOME/nanochat/.venv/bin/activate"

# Use the patched-special-token d12 base (already prepared earlier).
# Conservative LR + warmup as in the prior attempt.
python -m scripts.chat_sft \
    --model-tag=d12_patched \
    --max-seq-len=1024 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --embedding-lr=0.03 \
    --unembedding-lr=0.0008 \
    --matrix-lr=0.002 \
    --init-lr-frac=0.05 \
    --warmup-ratio=0.1 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=-1 \
    --num-iterations=500 \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1 \
    --run=dummy
