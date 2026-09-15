#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/build/slm"
${CC:-cc} -O3 -std=c99 -Wall -Wextra -Werror -fno-fast-math \
  "$HERE/src/slm_run.c" "$HERE/src/slm_tokenizer.c" -lm -o "$HERE/build/slm/slm_run"
if command -v i686-w64-mingw32-gcc >/dev/null 2>&1; then
  probe="$(mktemp /tmp/slm-crt-probe.XXXXXX)"
  crt_flag=()
  if printf 'int main(void){return 0;}\n' | i686-w64-mingw32-gcc -x c - -mcrtdll=msvcrt-os -o "$probe" >/dev/null 2>&1; then
    crt_flag=(-mcrtdll=msvcrt-os)
  fi
  rm -f "$probe"
  i686-w64-mingw32-gcc -O3 -std=c99 -Wall -Wextra -Werror -fno-fast-math \
    "${crt_flag[@]}" \
    -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -march=pentium-m -mtune=pentium-m \
    -msse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-avx -mfpmath=sse \
    -static -static-libgcc -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
    "$HERE/src/slm_run.c" "$HERE/src/slm_tokenizer.c" -lm -o "$HERE/build/slm/SLM_RUN_SSE2.EXE"
  imports="$(i686-w64-mingw32-objdump -p "$HERE/build/slm/SLM_RUN_SSE2.EXE")"
  if printf '%s\n' "$imports" | grep -Eiq 'DLL Name:.*(api-ms-win|ucrt|vcruntime|libgcc|libwinpthread)'; then
    printf 'ERROR: backend imports a runtime unavailable on Windows XP.\n' >&2
    exit 1
  fi
fi
