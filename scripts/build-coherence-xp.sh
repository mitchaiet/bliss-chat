#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$HERE/build/slm"
CC=i686-w64-mingw32-gcc
command -v "$CC" >/dev/null
bash "$HERE/scripts/build-slm.sh"
bash "$HERE/scripts/build-coherence-gui-xp.sh"
for binary in "$BUILD/XPCHAT.EXE" "$BUILD/SLM_RUN_SSE2.EXE"; do
    imports="$(i686-w64-mingw32-objdump -p "$binary")"
    if printf '%s\n' "$imports" | grep -Eiq 'DLL Name:.*(api-ms-win|ucrt|vcruntime|libgcc|libwinpthread)'; then
        printf 'ERROR: XP-incompatible runtime imports in %s\n' "$binary" >&2
        exit 1
    fi
done
printf 'Built XP GUI and SSE2 backend in %s\n' "$BUILD"
