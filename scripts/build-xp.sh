#!/bin/bash
# Cross-compile NC_RUN.EXE and XPCHAT.EXE for Windows XP (i686, no AVX, msvcrt only).
# Requires: i686-w64-mingw32-gcc (Linux mingw-w64 cross toolchain).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/src"
BUILD="$HERE/build"
mkdir -p "$BUILD"

CC=i686-w64-mingw32-gcc
# Some MinGW toolchains (e.g. Homebrew) support -mcrtdll=msvcrt-os;
# Ubuntu's win32 runtime package does not. Default MinGW i686 already links
# msvcrt.dll, so keep the flag only when the compiler accepts it.
CRT_FLAG=""
if echo 'int main(void){return 0;}' | $CC -x c - -mcrtdll=msvcrt-os -o /tmp/nc_crt_probe.exe >/dev/null 2>&1; then
  CRT_FLAG="-mcrtdll=msvcrt-os"
fi
rm -f /tmp/nc_crt_probe.exe
CFLAGS_COMMON="$CRT_FLAG -O2 -msse -msse2 -msse3 -mtune=pentium4 -march=pentium4 \
  -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -static -static-libgcc \
  -Wno-implicit-function-declaration"

echo "[build] NC_RUN.EXE (inference engine)"
$CC $CFLAGS_COMMON -std=c99 \
   "$SRC/nc_run.c" "$SRC/nc_tokenizer.c" \
   -o "$BUILD/NC_RUN.EXE" -lm

echo "[build] resource.o (icon + version info)"
i686-w64-mingw32-windres "$SRC/resource.rc" -O coff -o "$BUILD/resource.o" \
  --include-dir="$HERE/assets"

echo "[build] XPCHAT.EXE (Win32 GUI)"
$CC $CFLAGS_COMMON -mwindows \
   "$SRC/xpchat.c" "$BUILD/resource.o" \
   -o "$BUILD/XPCHAT.EXE" \
   -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid

ls -lh "$BUILD/"NC_RUN.EXE "$BUILD/"XPCHAT.EXE

echo
echo "[build] Verify XP-era DLL imports (should be standard XP system DLLs):"
i686-w64-mingw32-objdump -p "$BUILD/NC_RUN.EXE"  | grep "DLL Name" || true
i686-w64-mingw32-objdump -p "$BUILD/XPCHAT.EXE"  | grep "DLL Name" || true
