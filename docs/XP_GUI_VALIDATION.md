# Windows XP GUI candidate validation

Candidate: `2.0.0-candidate.20260915` (Windows numeric version `2,0,0,0`).
The GUI uses UTF-8 on the backend pipe and UTF-16 for chat controls, clipboard,
search, and SAPI text. New installations start with a 512-token context and
temperature 0; valid saved sampling preferences still apply. About and Model
Info display the model identity reported by the backend. Legacy deployment filenames remain
`NC_RUN.EXE`, `MODEL.NCB`, and `TOKENIZER.NCT`.

## Reproduce the portable helper checks

From the repository root:

```sh
python3 tests/test_xpchat_transport.py -v
```

The runner extracts the current implementations from `src/xpchat.c`, inserts
them into the two `tests/xpchat_*_harness.c.in` harnesses, and compiles temporary
executables with this command shape:

```sh
clang -std=c99 -Wall -Wextra -Werror -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer /tmp/GENERATED_HARNESS.c -o /tmp/GENERATED_HARNESS
```

`XPCHAT_TEST_CC` can select another host compiler;
`XPCHAT_TEST_SANITIZERS=0` disables sanitizers when unavailable. The default
sanitized run must report:

```text
GUI_CONTROL_EOT_QUEUE 519 checks passed
UTF8_GUI_BOUNDARY_CHECKS 79 passed
```

The Unicode scenarios include ASCII, accented Latin, Japanese, a supplementary
emoji, smart punctuation, combining characters, and CRLF. Every possible split
between two input chunks and one-byte input chunks are checked. Posted valid
text must contain complete UTF-8 codepoints; known UTF-16 values and exact
round trips are checked. Truncated and overlong input must produce visible
replacement characters in the conversion helpers. The harness substitutes a
16-bit `WCHAR` and its string-length function on non-Windows hosts.

The queue scenarios cover control/chat acknowledgment order, full capacity,
overflow rejection, ring wraparound, draining, and reuse. Win32 Interlocked
functions are sequential stubs in this harness: these checks establish FIFO
logic, not synchronization or actual message-loop behavior.

## Saved-settings compatibility

```sh
python3 tests/test_xpchat_settings.py -v
```

The additional four tests use the current settings loader and model-name helper
with a fake registry. They check supported sampling values, invalid registry
types/sizes, absent values, NaN/infinity, un-terminated or malformed UTF-16
strings, and model-banner recognition. They also check the relevant source
wiring for context, pre-save/pre-send normalization, and dynamic identity.
The C harness runs under the same ASan/UBSan defaults as the transport tests.

The GUI always launches with `-c 512`. It does not read a saved context setting,
so an old `Context=1024` cannot increase its KV allocation or request a context
larger than the candidate's 512-token header. Sampling is loaded only from
`HKCU\Software\bliss-chat\SettingsCoherentV2`; the prior sampling key is left
untouched and is not imported. Valid values in the new key are preserved:
temperature 0–5, top-p above 0 and at most 1, and maximum output 1–512 tokens.
Invalid persisted values fall back to defaults, including 128 output tokens.
`MaxTok=512` limits generation; it does not allocate a 512-token output buffer or
increase the context. The native runtime further caps generation to available
context after the prompt. Oversized system prompts remain subject to the
backend's token-aware check, which retains the previous prompt on rejection.

The GUI recognizes legacy model banners, Qwen banners, and explicit
`INFO MODEL <identity>` messages. Optional `INFO SOURCE <provenance>` updates the
source field separately; without one, the field says the backend did not report
it. Status notices do not replace model identity. About and Model Info use
Unicode text and the reported fields, so the GUI no longer hard-codes a model
family or parameter count.

## Reproduce the XP GUI cross-build

From the repository root, with the macOS MinGW tools used for this candidate:

```sh
mkdir -p build/slm
/usr/local/bin/i686-w64-mingw32-windres src/resource.rc -O coff \
  -o /private/tmp/bliss-gui-resource.o --include-dir=assets
/usr/local/bin/i686-w64-mingw32-gcc -mcrtdll=msvcrt-os -O2 -std=c99 \
  -D_WIN32_WINNT=0x0501 -DWINVER=0x0501 -march=pentium-m \
  -msse2 -mno-sse3 -mno-avx -static -static-libgcc -mwindows \
  -Wl,--major-subsystem-version,5,--minor-subsystem-version,1 \
  src/xpchat.c /private/tmp/bliss-gui-resource.o \
  -o build/slm/XPCHAT.EXE \
  -lcomctl32 -lcomdlg32 -ladvapi32 -lole32 -loleaut32 -luuid
/usr/local/bin/i686-w64-mingw32-objdump -p build/slm/XPCHAT.EXE
```

`-mcrtdll=msvcrt-os` is required with this compiler: its default CRT otherwise
introduces UCRT imports unavailable on the target XP installation. The candidate
was verified as PE32 with subsystem 5.1 and imports limited to ADVAPI32,
COMCTL32, COMDLG32, GDI32, KERNEL32, msvcrt, ole32, OLEAUT32, SHELL32, and USER32.

## Evidence limits

The portable checks passed with AddressSanitizer and UndefinedBehaviorSanitizer,
and the GUI cross-build succeeded. They do not establish a successful run on
Windows XP or the Pentium M target, font coverage, actual clipboard/SAPI behavior,
thread scheduling, UI usability, or the combined package's memory/performance.
Filesystem path APIs remain ANSI; arbitrary Unicode filenames are not supported
by this change. The native backend has separate numerical and IPC tests.
