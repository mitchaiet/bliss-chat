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

Validation results will be recorded with the selected executable's hashes.
Windows XP VM results measure ARM64 QEMU TCG with one Pentium 4-compatible
CPU and 512 MiB RAM. They are not measurements on a physical Pentium 4.
