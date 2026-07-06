#!/bin/bash
# build-xp.sh equivalent for the MSYS2 MINGW32 environment on Windows.
# Same flags and outputs as scripts/build-xp.sh (Linux cross host); only
# the windres binary name differs (unprefixed in MSYS2).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/src"
BUILD="$HERE/build"
mkdir -p "$BUILD"
export PATH=/mingw32/bin:$PATH

CC=i686-w64-mingw32-gcc
CRT_FLAG=""
if echo 'int main(void){return 0;}' | $CC -x c - -mcrtdll=msvcrt-os -o /tmp/nc_crt_probe.exe >/dev/null 2>&1; then
  CRT_FLAG="-mcrtdll=msvcrt-os"
fi
rm -f /tmp/nc_crt_probe.exe
CFLAGS_BASE="$CRT_FLAG -O2 \
  -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -static -static-libgcc \
  -Wno-implicit-function-declaration"
CFLAGS_SSE2="$CFLAGS_BASE -msse -msse2 -mtune=pentium-m -march=pentium-m"
CFLAGS_SSE3="$CFLAGS_BASE -msse -msse2 -msse3 -mtune=pentium4 -march=pentium4"

echo "[build] NC_RUN_SSE2.EXE"
$CC $CFLAGS_SSE2 -std=c99 "$SRC/nc_run.c" "$SRC/nc_tokenizer.c" -o "$BUILD/NC_RUN_SSE2.EXE" -lm
echo "[build] NC_RUN_SSE3.EXE"
$CC $CFLAGS_SSE3 -std=c99 "$SRC/nc_run.c" "$SRC/nc_tokenizer.c" -o "$BUILD/NC_RUN_SSE3.EXE" -lm
cp "$BUILD/NC_RUN_SSE2.EXE" "$BUILD/NC_RUN.EXE"

echo "[build] resource.o"
windres "$SRC/resource.rc" -O coff -o "$BUILD/resource.o" --include-dir="$HERE/assets"

echo "[build] XPCHAT.EXE"
$CC $CFLAGS_BASE -mwindows "$SRC/xpchat.c" "$BUILD/resource.o" -o "$BUILD/XPCHAT.EXE" \
  -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid

ls -la "$BUILD"/NC_RUN.EXE "$BUILD"/NC_RUN_SSE2.EXE "$BUILD"/NC_RUN_SSE3.EXE "$BUILD"/XPCHAT.EXE
echo "=== DLL imports ==="
objdump -p "$BUILD/XPCHAT.EXE" | grep "DLL Name" || true
objdump -p "$BUILD/NC_RUN_SSE2.EXE" | grep "DLL Name" || true
