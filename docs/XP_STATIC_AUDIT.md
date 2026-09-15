# XP executable audit

Run after the final build:

```sh
python3 tools/audit_xp_binary.py build/slm/SLM_RUN_SSE2.EXE \
  build/slm/XPCHAT.EXE --out build/slm/xp-static-audit.json
```

The script records exact executable hashes, PE32/subsystem5.1, DLL imports,
and disassembled instructions. It rejects modern CRT imports and known
post-SSE2 instructions. This is a static check, not physical XP execution
or an exhaustive API-availability proof.

The prebuilt MinGW numeric-conversion helpers contain `F3 0F BC` byte
sequences disassembled as `tzcnt`. Intel documents that this encoding
executes `bsf` on processors without BMI1. Both return the same count for
nonzero input; their zero-input and flag behavior differ. This is therefore
recorded separately rather than treated as an illegal opcode.

In the current binaries the sites belong to MinGW's `__trailz_D2A`,
`__d2b_D2A` and `__strtodg` helpers. Inspected uses have explicit nonzero
tests, an inserted nonzero significand bit, or the numeric helper's
nonzero-significand invariant; subsequent arithmetic/tests replace flags.
No LZCNT, SSE3, SSSE3, SSE4 or AVX instructions were found in the inspected
builds. Recheck the generated report whenever the toolchain changes.

Sources: [Intel instruction-set reference, TZCNT](https://cdrdv2-public.intel.com/835757/325383-sdm-vol-2abcd.pdf),
[GCC bit-operation builtins](https://gcc.gnu.org/onlinedocs/gcc/Bit-Operation-Builtins.html).
