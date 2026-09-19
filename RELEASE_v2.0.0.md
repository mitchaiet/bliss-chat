# Bliss Chat 2.0.0 — native XP, fully local

Version 2.0.0 finalizes the RC line. The conversational model is the one frozen
for RC1 and RC2; what changed since RC2 is the inference engine underneath it,
which now uses every processor in the machine and reads its weights in a layout
built for the Pentium 4. There is still no internet lookup and no remote helper.

## Changes since RC2

- **Lossless Q6X4 model layout.** The six-bit weights and their FP32 group
  scales are preserved exactly, then rearranged so four output rows are
  interleaved per column group. `pmaddwd` produces four outputs with no
  horizontal reduction, and Windows maps the prepared file directly instead of
  unpacking a copy into memory. The file grows from 274.87 MiB to 359.37 MiB.
  See [Pentium 4 speed build](docs/Q6X4_PERFORMANCE.md).
- **All processors used.** Linear layers, attention heads, activations and
  convolution channels are split across a worker pool. Rows are divided into
  contiguous blocks and each one is computed by the same code as before, so the
  thread count never changes an answer. Control it with `-j`, `--threads` or
  `SLM_THREADS`; the default is one thread per processor.
- **System prompt cached.** The key/value and convolution state for the system
  prefix is snapshotted, restored when a new chat starts, and can persist to
  disk keyed by the model and executable hashes. Starting a fresh chat no longer
  recomputes the same 56 tokens.
- **Clang-built SSE2 backend** linked against the legacy Microsoft C runtime,
  with a build-time check that rejects any import Windows XP does not provide.

## Measured speed

Decoding, in an ARM64 QEMU virtual machine with Pentium 4-compatible processors
and 512 MiB of memory. These are not measurements on physical hardware.

| Threads | Decode | Prefill |
|---|---:|---:|
| 1 | 1.07 tok/s | 1.27 tok/s |
| 2 | 2.10 tok/s | 2.53 tok/s |
| 4 | 3.7 tok/s | 4.5 tok/s |

For comparison, the RC2 runtime on one processor decoded at about 0.19 tok/s.
Working set is 373 MiB, which leaves roughly 35 MiB free in a 512 MiB guest.

## Download and use

Run the portable EXE, or extract the whole folder ZIP and launch `XPCHAT.EXE`.
Chats, memories and imported documents live in `%APPDATA%\bliss-chat`.

The runtime and the model must be replaced together. Version-3 model files are
rejected by older executables, and this executable still reads the packed
version-1 and version-2 files, so keep a matched pair when rolling back.

## Validation and limits

Answers are unchanged by every optimization in this release. The 48-case
held-out suite, 56 turns in total, produced byte-identical replies from the
single-threaded build and from a ten-thread build, and the prefix cache tests
report bit-exact continuations.

Known limits carried forward from the RC acceptance review:

- Arithmetic and logic remain unreliable. This is a 350M-parameter model.
- It sometimes elaborates where a one-word answer was requested.
- Physical Windows XP hardware remains untested; all results above come from
  emulation, which favors this layout over the packed one. On a real Pentium 4,
  where memory bandwidth rather than instruction cost is the limit, the smaller
  packed file may perform better.
- The executables are unsigned.
