#!/bin/bash
# Bliss-native curated d12 base training: mix plain Q:/A: Bliss docs into
# nanochat base pretraining from step 1. No chat-special-token SFT.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export BLISS_CURATED_PARQUET_DIR="${BLISS_CURATED_PARQUET_DIR:-$HOME/bliss-chat-data/bliss_pretrain_curated_v1}"
export BLISS_CURATED_PROB="${BLISS_CURATED_PROB:-0.12}"
mkdir -p "$NANOCHAT_BASE_DIR"
source "$HOME/nanochat/.venv/bin/activate"
WANDB_RUN="${WANDB_RUN:-bliss-d12-curated-c20-v1}"
MODEL_TAG="${MODEL_TAG:-bliss-d12-curated-c20-v1}"

# Keep same architecture/perf envelope as current XP d12. For full run, leave
# num-iterations unset and target-param-data-ratio=20. For smoke tests, set
# NUM_ITERATIONS=20 in the environment.
ARGS=(
  --depth=12
  --max-seq-len=1024
  --device-batch-size=32
  --total-batch-size=65536
  --eval-every=2000
  --eval-tokens=524288
  --core-metric-every=-1
  --sample-every=2000
  --model-tag="$MODEL_TAG"
  --run="$WANDB_RUN"
)
if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  ARGS+=(--num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1)
else
  ARGS+=(--target-param-data-ratio=20)
fi

python -m scripts.base_train "${ARGS[@]}"
