#!/bin/bash
# Track B: full d12 Chinchilla-ratio-20 retrain from scratch with the v2
# memory/multi-turn curated mixture (multi-turn recall, Notes-line
# selective recall, Context: grounding, reasoning) mixed into base
# pretraining. Same recipe that produced bliss-d12-curated-c20-v1, richer
# mixture. ~3h on the RTX PRO 6000.
set -euo pipefail
cd "$HOME/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export BLISS_CURATED_PARQUET_DIR="${BLISS_CURATED_PARQUET_DIR:-$HOME/bliss-chat-data/bliss_pretrain_curated_v2}"
export BLISS_CURATED_PROB="${BLISS_CURATED_PROB:-0.15}"
source "$HOME/nanochat/.venv/bin/activate"

MODEL_TAG="${MODEL_TAG:-bliss-d12-mem-c20-v2b}"

ARGS=(
  --depth=12
  --max-seq-len=1024
  --device-batch-size=32
  --total-batch-size=65536
  --eval-every=2000
  --eval-tokens=524288
  --core-metric-every=-1
  --sample-every=4000
  --save-every=4000
  --model-tag="$MODEL_TAG"
  --run="${WANDB_RUN:-dummy}"
)
if [[ -n "${NUM_ITERATIONS:-}" ]]; then
  ARGS+=(--num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1)
else
  ARGS+=(--target-param-data-ratio=20)
fi

python -m scripts.base_train "${ARGS[@]}"
