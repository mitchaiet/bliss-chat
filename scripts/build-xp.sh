#!/bin/bash
# Cross-compile NC_RUN.EXE and XPCHAT.EXE for Windows XP (i686, no AVX, msvcrt only).
# Requires: i686-w64-mingw32-gcc (Homebrew mingw-w64).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/src"
BUILD="$HERE/build"
mkdir -p "$BUILD"

CC=i686-w64-mingw32-gcc
CFLAGS_COMMON="-mcrtdll=msvcrt-os -O2 -msse -msse2 -msse3 -mtune=pentium4 -march=pentium4 \
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
   -lcomctl32 -lcomdlg32 -ladvapi32

ls -lh "$BUILD/"NC_RUN.EXE "$BUILD/"XPCHAT.EXE

echo
echo "[build] Verify XP-only DLL imports (must be only KERNEL32/USER32/GDI32/COMCTL32/MSVCRT/ADVAPI32):"
i686-w64-mingw32-objdump -p "$BUILD/NC_RUN.EXE"  | grep "DLL Name" || true
i686-w64-mingw32-objdump -p "$BUILD/XPCHAT.EXE"  | grep "DLL Name" || true
