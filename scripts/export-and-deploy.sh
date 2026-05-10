#!/bin/bash
# Export a nanochat checkpoint to NCB1 + NCT1 binaries and stage for deploy.
# Usage: export-and-deploy.sh <model_tag> [step]   e.g. d6, d12
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$HERE/tools"
DEPLOY="$HERE/build/deploy"
TAG="${1:-d6}"
STEP="${2:-}"

CKPT_DIR="$HOME/.cache/nanochat/base_checkpoints/$TAG"
TOK_PKL="$HOME/.cache/nanochat/tokenizer/tokenizer.pkl"

mkdir -p "$DEPLOY"

# 1) Tokenizer
echo "[export] tokenizer -> $DEPLOY/TOKENIZER.NCT"
python "$TOOLS/export_tokenizer.py" --src "$TOK_PKL" --out "$DEPLOY/TOKENIZER.NCT"

# 2) Model (int8 quantized)
echo "[export] model $TAG -> $DEPLOY/MODEL.NCB"
ARGS="--src $CKPT_DIR --out $DEPLOY/MODEL.NCB --int8"
if [ -n "$STEP" ]; then ARGS="$ARGS --step $STEP"; fi
python "$TOOLS/export_ncb.py" $ARGS

ls -lh "$DEPLOY/"
