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
  # Clang generates a faster 32-bit SSE2 loop for Q6X4. Use MinGW only
  # for its XP headers/libraries and legacy CRT link. GCC remains a fallback.
  win_flags=(-O3 -std=c99 -Wall -Wextra -Werror -fno-fast-math
    -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -march=pentium4
    -msse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-avx -mfpmath=sse)
  win_compiler="${SLM_WIN32_COMPILER:-auto}"
  mingw_root="${MINGW_SYSROOT:-$(i686-w64-mingw32-gcc -print-sysroot)}"
  mingw_headers=""
  for include in "$mingw_root/i686-w64-mingw32/include" "$mingw_root/include"; do
    if [[ -f "$include/windows.h" ]]; then mingw_headers="$include"; break; fi
  done
  if [[ "$win_compiler" != gcc ]] && command -v clang >/dev/null 2>&1 && [[ -n "$mingw_headers" ]]; then
    for source in slm_run slm_tokenizer; do
      clang --target=i686-w64-windows-gnu --sysroot="$mingw_root" -isystem "$mingw_headers" \
        "${win_flags[@]}" -c "$HERE/src/$source.c" -o "$HERE/build/slm/$source.win32.o"
    done
    i686-w64-mingw32-gcc "${crt_flag[@]}" -static -static-libgcc \
      -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
      "$HERE/build/slm/slm_run.win32.o" "$HERE/build/slm/slm_tokenizer.win32.o" \
      -lm -o "$HERE/build/slm/SLM_RUN_SSE2.EXE"
    printf 'Win32 compiler: Clang, Pentium 4 SSE2; legacy MinGW CRT link\n'
  elif [[ "$win_compiler" == auto || "$win_compiler" == gcc ]]; then
    i686-w64-mingw32-gcc "${win_flags[@]}" -mtune=pentium4 "${crt_flag[@]}" \
      -static -static-libgcc -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
      "$HERE/src/slm_run.c" "$HERE/src/slm_tokenizer.c" -lm -o "$HERE/build/slm/SLM_RUN_SSE2.EXE"
    printf 'Win32 compiler: GCC, Pentium 4 SSE2\n'
  else
    printf 'Requested Clang cross compiler needs clang and MINGW_SYSROOT with XP headers.\n' >&2
    exit 1
  fi
  imports="$(i686-w64-mingw32-objdump -p "$HERE/build/slm/SLM_RUN_SSE2.EXE")"
  if printf '%s\n' "$imports" | grep -Eiq 'DLL Name:.*(api-ms-win|ucrt|vcruntime|libgcc|libwinpthread)'; then
    printf 'ERROR: backend imports a runtime unavailable on Windows XP.\n' >&2
    exit 1
  fi
fi
