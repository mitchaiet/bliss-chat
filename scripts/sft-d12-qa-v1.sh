#!/bin/bash
# Bliss d12 Q/A SFT. Renders examples as:
#   Q: <prompt>
#   A: <answer>\n
set -euo pipefail

BLISS_REPO_DIR="${BLISS_REPO_DIR:-$HOME/bliss-sft-v1}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/nanochat}"
BLISS_MODEL_TAG="${BLISS_MODEL_TAG:-d12_qa_v1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE=float32
export BLISS_PLAIN_CHAT=1
export BLISS_PLAIN_STYLE="${BLISS_PLAIN_STYLE:-qa}"
export BLISS_SFT_JSONL="${BLISS_SFT_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v3_train.jsonl}"
export BLISS_VAL_JSONL="${BLISS_VAL_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v3_val.jsonl}"
export BLISS_GRAD_CLIP="${BLISS_GRAD_CLIP:-0.50}"
export BLISS_ABORT_LOSS="${BLISS_ABORT_LOSS:-12.0}"
export BLISS_FIRST_TOKEN_WEIGHT="${BLISS_FIRST_TOKEN_WEIGHT:-8.0}"
export BLISS_END_TOKEN_WEIGHT="${BLISS_END_TOKEN_WEIGHT:-12.0}"
BLISS_TOTAL_BATCH_SIZE="${BLISS_TOTAL_BATCH_SIZE:-4096}"
BLISS_EMBEDDING_LR="${BLISS_EMBEDDING_LR:-0.00010}"
BLISS_UNEMBEDDING_LR="${BLISS_UNEMBEDDING_LR:-0.00020}"
BLISS_MATRIX_LR="${BLISS_MATRIX_LR:-0.00005}"
BLISS_NUM_ITERATIONS="${BLISS_NUM_ITERATIONS:-1600}"

cd "$NANOCHAT_DIR"
source "$NANOCHAT_DIR/.venv/bin/activate"
export PYTHONPATH="$NANOCHAT_DIR:${PYTHONPATH:-}"

python "$BLISS_REPO_DIR/tools/patch_nanochat_bliss_sft.py" --nanochat "$NANOCHAT_DIR"
python "$BLISS_REPO_DIR/tools/check_bliss_sft_data.py" --plain --plain-style "$BLISS_PLAIN_STYLE" --nanochat "$NANOCHAT_DIR" --jsonl "$BLISS_SFT_JSONL" --max-tokens 512
python "$BLISS_REPO_DIR/tools/check_bliss_sft_data.py" --plain --plain-style "$BLISS_PLAIN_STYLE" --nanochat "$NANOCHAT_DIR" --jsonl "$BLISS_VAL_JSONL" --max-tokens 512

if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/$BLISS_MODEL_TAG/model_033600.pt" ]; then
  mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/$BLISS_MODEL_TAG"
  cp "$NANOCHAT_BASE_DIR/base_checkpoints/d12/model_033600.pt" "$NANOCHAT_BASE_DIR/base_checkpoints/$BLISS_MODEL_TAG/model_033600.pt"
  cp "$NANOCHAT_BASE_DIR/base_checkpoints/d12/meta_033600.json" "$NANOCHAT_BASE_DIR/base_checkpoints/$BLISS_MODEL_TAG/meta_033600.json"
fi

python -m scripts.chat_sft \
  --model-tag="$BLISS_MODEL_TAG" \
  --model-step=33600 \
  --max-seq-len=256 \
  --device-batch-size=4 \
  --total-batch-size="$BLISS_TOTAL_BATCH_SIZE" \
  --embedding-lr="$BLISS_EMBEDDING_LR" \
  --unembedding-lr="$BLISS_UNEMBEDDING_LR" \
  --matrix-lr="$BLISS_MATRIX_LR" \
  --init-lr-frac=0.05 \
  --warmup-ratio=0.08 \
  --warmdown-ratio=0.25 \
  --load-optimizer=0 \
  --eval-every=50 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --num-iterations="$BLISS_NUM_ITERATIONS" \
  --mmlu-epochs=0 \
  --gsm8k-epochs=0 \
  --run=dummy
