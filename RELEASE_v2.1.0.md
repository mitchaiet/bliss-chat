# Bliss Chat 2.1.0 — faster prompts

The model is unchanged from 2.0.0 and from the RC line before it. This release
changes how a prompt is read. It is still fully local, with no internet lookup.

## What changed

Until now, reading a prompt cost one full pass over the model for every token
in it. A single token turns each weight matrix into a matrix-vector product, so
all 359 MiB had to stream past the processor 30 times for a 30-word question
and hundreds of times for an imported document.

Prefill now happens in batches. Several tokens go through each layer together,
which turns the same work into a matrix-matrix product: the weights are fetched
from memory once and each block is reused from cache for the rest of the batch.
This applies to your messages, to imported documents, to saved notes, and to
the replay that happens when a long conversation drops its oldest turn.

Generation is untouched. A reply is produced one token at a time by nature, so
it still runs through the original path at the same speed.

## Measured

Prefilling a 383-token prompt, same executable, on a real processor:

| Threads | One at a time | Batched | Gain |
|---|---:|---:|---:|
| 1 | 9.74 s | 5.98 s | 1.63x |
| 4 | 3.84 s | 2.67 s | 1.44x |

In the Windows XP virtual machine the same change is worth only about 1.09x.
That is a limit of emulation, not of the code: QEMU charges the same price for
a cache hit as for a main-memory read, and avoiding those reads is exactly what
batching does. A physical Pentium 4 has a much wider gap between cache and
memory than the machine used for the table above, so the gain there should be
larger. Nobody has measured it on real hardware yet.

`SLM_BATCH` sets the batch size and defaults to eight, which costs about
840 KiB. Setting it to 1 restores the old behavior.

## Validation

Batching must never change an answer, and it does not. `tests/test_slm_batch.py`
prefills three prompts of different lengths at batch sizes 1, 2, 3, 8 and 16 and
requires the resulting logits to be byte-identical. That passes on both the
ARM64 and x86-64 SSE2 builds. The 48-case held-out suite, 56 turns in total,
returns all 48 answers unchanged, and the prefix-cache and kernel oracle tests
still pass.

## Limits

Unchanged from 2.0.0:

- Arithmetic and logic remain unreliable. This is a 350M-parameter model.
- It sometimes elaborates where a one-word answer was requested.
- Physical Windows XP hardware remains untested.
- The executable is unsigned.
