#!/bin/bash
# Bliss d12 SFT v1 canary. Run on the GPU workstation.
set -euo pipefail

BLISS_REPO_DIR="${BLISS_REPO_DIR:-$HOME/bliss-sft-v1}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE=float32
export BLISS_SFT_JSONL="${BLISS_SFT_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v1_train.jsonl}"
export BLISS_VAL_JSONL="${BLISS_VAL_JSONL:-$BLISS_REPO_DIR/data/bliss_sft_v1_val.jsonl}"
export BLISS_GRAD_CLIP="${BLISS_GRAD_CLIP:-0.25}"
export BLISS_ABORT_LOSS="${BLISS_ABORT_LOSS:-4.0}"

cd "$NANOCHAT_DIR"
source "$NANOCHAT_DIR/.venv/bin/activate"
export PYTHONPATH="$NANOCHAT_DIR:${PYTHONPATH:-}"

python "$BLISS_REPO_DIR/tools/patch_nanochat_bliss_sft.py" --nanochat "$NANOCHAT_DIR"
python "$BLISS_REPO_DIR/tools/check_bliss_sft_data.py" --nanochat "$NANOCHAT_DIR" --jsonl "$BLISS_SFT_JSONL" --max-tokens 512
python "$BLISS_REPO_DIR/tools/check_bliss_sft_data.py" --nanochat "$NANOCHAT_DIR" --jsonl "$BLISS_VAL_JSONL" --max-tokens 512

if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d12_c20_special_v1/model_033600.pt" ]; then
  python "$BLISS_REPO_DIR/tools/repair_special_rows.py" \
    --src "$NANOCHAT_BASE_DIR/base_checkpoints/d12" \
    --out "$NANOCHAT_BASE_DIR/base_checkpoints/d12_c20_special_v1" \
    --tokenizer "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" \
    --step 33600 \
    --bos-blend 0.25
fi

python -m scripts.chat_sft \
  --model-tag=d12_c20_special_v1 \
  --model-step=33600 \
  --max-seq-len=256 \
  --device-batch-size=4 \
  --total-batch-size=8192 \
  --embedding-lr=0.0 \
  --unembedding-lr=0.00005 \
  --matrix-lr=0.00005 \
  --init-lr-frac=0.1 \
  --warmup-ratio=0.1 \
  --load-optimizer=0 \
  --eval-every=10 \
  --eval-tokens=131072 \
  --chatcore-every=-1 \
  --num-iterations=400 \
  --mmlu-epochs=0 \
  --gsm8k-epochs=0 \
  --run=dummy
