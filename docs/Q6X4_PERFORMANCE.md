# Pentium 4 lossless Q6 layout

The version-3 Q6X4 file preserves all learned six-bit weight values, every
FP32 group scale, and all other tensors. Four output rows are interleaved
on disk so SSE2 `pmaddwd` produces four outputs without horizontal reductions.
The runtime can share activation loads across adjacent four-row blocks.
Activation values retain the existing Q16 scaling; FP32 group accumulation
keeps its original order. `--float-activations` retains the reference path.
The packed version-1 and version-2 formats remain supported.

The model grows from 288,226,560 bytes (274.87 MiB) to 376,831,232 bytes
(359.37 MiB). XP maps the prepared file directly; it does not allocate an
unpacking copy. Additional aligned activation scratch space is about 36 KiB
for this model. The tokenizer and 512-token context are unchanged.

## File format

The 256-byte SLMODEL1 header has version 3 at byte 8 and storage width 8 at
byte 12. This identifies losslessly expanded Q6, not newly quantized Q8.
Architecture 1, group size 64 and row counts divisible by four are required.
Dimensions, hybrid layer tags and tensor order follow version 2.

For a matrix with `G = columns / 64`, weight `(row, column)` is stored at:

```
((row / 4) * G + column / 64) * 256
  + ((column % 64) / 2) * 8 + (row % 4) * 2 + column % 2
```

The byte stores the signed six-bit value plus 32. The matrix's scale array
follows its weight array, at `((row / 4) * G + group) * 4 + row % 4`.
All remaining tensors are copied verbatim. Old executables reject version 3;
ship the model and new runtime together. Keep the old pair for rollback.

## Reproduce

Run conversion once on the packaging host (Python and NumPy required):

```sh
python3 tools/expand_q6.py packed/MODEL.NCB fast/MODEL.NCB --report conversion.json
bash scripts/build-slm.sh
cc -O3 -fno-fast-math -msse2 tests/test_slm_q6x4.c -lm -o build/test_q6x4
build/test_q6x4
python3 tests/test_expand_q6.py
```

The converter refuses existing outputs, verifies every packed weight bit
round-trips, shuffles scales as bytes, and rejects truncated or trailing data.
No Python, GPU, server, or network connection is needed on the XP machine.

## Measurements

Windows XP VM results measure ARM64 QEMU 11.1 TCG (Apple M1 Pro host) with
Pentium 4-compatible virtual CPUs and 512 MiB RAM. They are not measurements
on a physical Pentium 4; TCG executes every SSE2 instruction through a helper
call, so a real 3 GHz Pentium 4 should be several times faster per core.

`bench/slm_speed_xp.c` (built with `-DSLM_PROFILE`) on the installed
Q6X4 model, warm second pass, 2026-09-17. Host wall-clock time agreed with
the guest's QueryPerformanceCounter and tick clocks within a second when the
host was otherwise idle.

| Threads | vCPUs | Decode, 32 tokens | Prefill, 64 tokens | Linear share |
|---|---|---:|---:|---:|
| 1 | 4 | 1.07 tok/s | 1.27 tok/s | 87.1% (SwiGLU 11.5%) |
| 2 | 4 | 2.10 tok/s | 2.53 tok/s | 95.5% |
| 4 | 4 | 3.7 tok/s | 4.5 tok/s | 97.3% |
| 5 | 6 | 3.2 tok/s | 3.9 tok/s | 97.7% |
| 6 | 6 | 3.7 tok/s | 4.4 tok/s | 98.0% |
| 6, SLM_SPIN=2000 | 6 | 2.9 tok/s | 3.5 tok/s | 97.7% |

Beyond four threads the host, not the guest, is the limit: the 10-core M1 Pro
was also running other work, and TCG's per-instruction helpers plus MTTCG
memory barriers leave little headroom. Eight virtual CPUs bug-checked
Windows XP under sustained load and the spinning pool degraded badly there,
so the VM runs with four.

## Batched prefill

A single token turns each weight matrix into a matrix-vector product, so
reading a prompt costs one pass over the whole model per token. Feeding
several tokens at once turns it into a matrix-matrix product: the weights are
fetched from memory once and each 256-byte block is then reused from L1 for
the rest of the batch. `SLM_BATCH` sets the size, default eight, costing about
840 KiB of scratch.

Measured on the x86_64 SSE2 build on a real processor, prefilling a 383-token
prompt with the same executable:

| Threads | One at a time | Batched | Gain |
|---|---:|---:|---:|
| 1 | 9.74 s | 5.98 s | 1.63x |
| 4 | 3.84 s | 2.67 s | 1.44x |

In the QEMU virtual machine the same change is worth only about 1.09x, and
that understates it rather than contradicting it. Emulation charges the same
price for a cache hit as for a main-memory read, which is precisely the cost
batching removes, so the emulator hides the benefit. A physical Pentium 4 has
a far wider gap between L1 and DRAM than the host used for the table above,
so the gain there should be larger than 1.63x. That remains unmeasured.

Decoding is unaffected: it generates one token at a time by nature and still
runs through the single-token path.

Reference points on one vCPU: the packed Q6 runtime with Q16 activations
decoded at about 0.40 tok/s, and the original RC2 runtime at about 0.19 tok/s.
Working set was 373 MiB in every run, leaving about 35 MiB free in the guest.
The first pass after a process start pays a 70 to 90 second page-in of the
mapped model from the emulated IDE disk; the GUI shows this as "Loading
model...".

Answers were unchanged in every configuration: the 48-case held-out suite
(56 turns) produced byte-identical replies with 1 and 10 threads on the
x86_64 SSE2 host build, and the prefix-cache tests report bit-exact logit
continuations.

The VM boots Windows XP Professional SP3 with the ACPI multiprocessor HAL
(`/HAL=halmacpi.dll /KERNEL=ntkrnlmp.exe` in boot.ini) and
`-smp N,sockets=1,cores=N` in QEMU. Keep QEMU's `tb-size` at 128 on ARM64
hosts; larger translation caches abort in `do_patch_instruction`.
