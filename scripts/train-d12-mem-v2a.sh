#!/bin/bash
# Track A: continued pretraining of bliss-d12-curated-c20-v1 on the v2
# memory/multi-turn curated mixture. Resumes from step 33600 with the
# tail-end (decayed) learning rate and a higher curated-mix probability.
#
# Produces base_checkpoints/bliss-d12-mem-c20-v2a/model_037600.pt in
# roughly 20-25 minutes on the RTX PRO 6000.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export BLISS_CURATED_PARQUET_DIR="${BLISS_CURATED_PARQUET_DIR:-$HOME/bliss-chat-data/bliss_pretrain_curated_v2}"
export BLISS_CURATED_PROB="${BLISS_CURATED_PROB:-0.35}"
source "$HOME/nanochat/.venv/bin/activate"

SRC_TAG="bliss-d12-curated-c20-v1"
MODEL_TAG="${MODEL_TAG:-bliss-d12-mem-c20-v2a}"
RESUME_STEP=33600
EXTRA_STEPS="${EXTRA_STEPS:-4000}"
TOTAL_STEPS=$((RESUME_STEP + EXTRA_STEPS))

SRC_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$SRC_TAG"
DST_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$DST_DIR"
for f in model_033600.pt optim_033600_rank0.pt meta_033600.json; do
  [ -f "$DST_DIR/$f" ] || cp "$SRC_DIR/$f" "$DST_DIR/$f"
done

python -m scripts.base_train \
  --depth=12 \
  --max-seq-len=1024 \
  --device-batch-size=32 \
  --total-batch-size=65536 \
  --num-iterations="$TOTAL_STEPS" \
  --target-param-data-ratio=-1 \
  --resume-from-step="$RESUME_STEP" \
  --eval-every=1000 \
  --eval-tokens=524288 \
  --core-metric-every=-1 \
  --sample-every=2000 \
  --save-every=2000 \
  --model-tag="$MODEL_TAG" \
  --run="${WANDB_RUN:-dummy}"
