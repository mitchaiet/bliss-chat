#!/bin/bash
# Cross-compile NC_RUN.EXE and XPCHAT.EXE for Windows XP (i686, no AVX, msvcrt only).
# Requires: i686-w64-mingw32-gcc (Linux mingw-w64 cross toolchain).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/src"
BUILD="$HERE/build"
mkdir -p "$BUILD"

CC=i686-w64-mingw32-gcc
# Some MinGW toolchains support -mcrtdll=msvcrt-os;
# Ubuntu's win32 runtime package does not. Default MinGW i686 already links
# msvcrt.dll, so keep the flag only when the compiler accepts it.
CRT_FLAG=""
if echo 'int main(void){return 0;}' | $CC -x c - -mcrtdll=msvcrt-os -o /tmp/nc_crt_probe.exe >/dev/null 2>&1; then
  CRT_FLAG="-mcrtdll=msvcrt-os"
fi
rm -f /tmp/nc_crt_probe.exe
CFLAGS_BASE="$CRT_FLAG -O2 \
  -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -static -static-libgcc \
  -Wno-implicit-function-declaration"

# Ship two backend binaries in the portable installer:
# - SSE2: safest path for Pentium M / Pentium 4-era XP laptops (no SSE3).
# - SSE3: keeps the previous optimized build for newer Pentium 4 Prescott/Core-era CPUs.
# XPCHAT.EXE detects CPU features at runtime and launches the best available backend.
CFLAGS_SSE2="$CFLAGS_BASE -msse -msse2 -mtune=pentium-m -march=pentium-m"
CFLAGS_SSE3="$CFLAGS_BASE -msse -msse2 -msse3 -mtune=pentium4 -march=pentium4"

echo "[build] NC_RUN_SSE2.EXE (Pentium M/Pentium 4 compatible inference engine)"
$CC $CFLAGS_SSE2 -std=c99 \
   "$SRC/nc_run.c" "$SRC/nc_tokenizer.c" \
   -o "$BUILD/NC_RUN_SSE2.EXE" -lm

echo "[build] NC_RUN_SSE3.EXE (optimized inference engine)"
$CC $CFLAGS_SSE3 -std=c99 \
   "$SRC/nc_run.c" "$SRC/nc_tokenizer.c" \
   -o "$BUILD/NC_RUN_SSE3.EXE" -lm

# Backward-compatible filename for existing tooling/tests. Use the safest backend.
cp "$BUILD/NC_RUN_SSE2.EXE" "$BUILD/NC_RUN.EXE"

echo "[build] resource.o (icon + version info)"
i686-w64-mingw32-windres "$SRC/resource.rc" -O coff -o "$BUILD/resource.o" \
  --include-dir="$HERE/assets"

echo "[build] XPCHAT.EXE (Win32 GUI)"
$CC $CFLAGS_BASE -mwindows \
   "$SRC/xpchat.c" "$BUILD/resource.o" \
   -o "$BUILD/XPCHAT.EXE" \
   -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid

ls -lh "$BUILD/"NC_RUN.EXE "$BUILD/"NC_RUN_SSE2.EXE "$BUILD/"NC_RUN_SSE3.EXE "$BUILD/"XPCHAT.EXE

echo
echo "[build] Verify XP-era DLL imports (should be standard XP system DLLs):"
i686-w64-mingw32-objdump -p "$BUILD/NC_RUN_SSE2.EXE"  | grep "DLL Name" || true
i686-w64-mingw32-objdump -p "$BUILD/NC_RUN_SSE3.EXE"  | grep "DLL Name" || true
i686-w64-mingw32-objdump -p "$BUILD/XPCHAT.EXE"  | grep "DLL Name" || true
