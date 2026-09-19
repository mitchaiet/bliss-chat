#!/usr/bin/env bash
# Build only the native GUI; preserve the frozen inference runtime and weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$HERE/build/slm"
mkdir -p "$BUILD"
i686-w64-mingw32-windres "$HERE/src/resource.rc" -O coff -o "$BUILD/resource.o" \
  --include-dir="$HERE/assets" --include-dir="$HERE/src"
# APP_VERSION is what the About box reports. Keep it equal to the installer's
# RELEASE_VERSION so the GUI and the package can never claim different versions.
APP_VERSION="${APP_VERSION:-$(sed -n 's/.*!define RELEASE_VERSION "\(.*\)".*/\1/p' "$HERE/installer/coherence-portable.nsi" | head -1)}"
: "${APP_VERSION:?could not determine release version}"
printf 'Building GUI for version %s\n' "$APP_VERSION"
i686-w64-mingw32-gcc -O2 -std=gnu99 -mcrtdll=msvcrt-os \
  -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -DAPP_VERSION="\"$APP_VERSION\"" \
  -march=pentium-m -mtune=pentium-m -msse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-avx \
  -static -static-libgcc -mwindows -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
  "$HERE/src/xpchat.c" "$BUILD/resource.o" -o "$BUILD/XPCHAT.EXE" \
  -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid -lm
python3 "$HERE/tools/audit_xp_binary.py" "$BUILD/XPCHAT.EXE" --out "$BUILD/gui-xp-audit.json"
