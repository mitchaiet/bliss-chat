#!/bin/bash
# Build the native (Linux/macOS) nc_run for benchmarking and CI-style
# checks. This is the binary bench/run_bench.py and the eval scripts
# drive; the XP release binaries come from scripts/build-xp.sh instead.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
CC="${CC:-gcc}"
"$CC" -O2 -std=gnu99 -o build/nc_run_native src/nc_run.c src/nc_tokenizer.c -lm
echo "built build/nc_run_native"
