#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$HERE/build/slm"
CC=i686-w64-mingw32-gcc
command -v "$CC" >/dev/null
bash "$HERE/scripts/build-slm.sh"
i686-w64-mingw32-windres "$HERE/src/resource.rc" -O coff -o "$BUILD/resource.o" \
    --include-dir="$HERE/assets" --include-dir="$HERE/src"
"$CC" -O2 -std=gnu99 -mcrtdll=msvcrt-os -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 \
    -march=pentium-m -mtune=pentium-m -msse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-avx \
    -static -static-libgcc -mwindows -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
    "$HERE/src/xpchat.c" "$BUILD/resource.o" -o "$BUILD/XPCHAT.EXE" \
    -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid -lm
for binary in "$BUILD/XPCHAT.EXE" "$BUILD/SLM_RUN_SSE2.EXE"; do
    imports="$(i686-w64-mingw32-objdump -p "$binary")"
    if printf '%s\n' "$imports" | grep -Eiq 'DLL Name:.*(api-ms-win|ucrt|vcruntime|libgcc|libwinpthread)'; then
        printf 'ERROR: XP-incompatible runtime imports in %s\n' "$binary" >&2
        exit 1
    fi
done
printf 'Built XP GUI and SSE2 backend in %s\n' "$BUILD"
