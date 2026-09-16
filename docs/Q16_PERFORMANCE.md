# Q6 runtime speed update — 2026-09-16

The model weights, tokenizer, prompt template and context limit are unchanged.
Q6 matrix products now use groupwise signed 16-bit activation values and SSE2
integer multiply/add instructions. FP32 is retained for scales, attention,
normalization, convolution, residuals and accumulation between groups.
`--float-activations` selects the original RC1/RC2 arithmetic path.

This is an inference optimization, not another training run. It introduces
small rounding differences; it does not make the small model's existing
reasoning errors disappear.

## Measured speed

The fixed-token benchmark separates input processing from generation and
excludes tokenization, GUI work and initial model setup. Warm passes compare
the same tokens, weights and context positions in one process at a time.

| Test | RC2 FP32 activations | Q16 activations | Generation speedup |
|---|---:|---:|---:|
| Modern T2 CPU, one thread, SSE2-only executable | 9.159 tok/s | 20.831 tok/s | 2.27x |
| Windows XP VM, final kernel, warm pass | 0.156 tok/s | 0.401 tok/s | 2.58x |

Linux used 16 input tokens followed by 16 fixed decode tokens, with the first
of three passes discarded. XP used 3 input tokens and 4 decode tokens over
two passes. The VM has one Pentium 4-compatible CPU and 512 MiB RAM, running
under native ARM64 QEMU TCG on a Mac. **These are not measurements of a
physical Pentium 4 or Pentium M.** An earlier prototype reached 0.465 tok/s; the table reports the final
installed build. Raw values and installation verification are retained in the
accompanying validation record.

Input processing improved 2.26x in the final Linux build. Whole-chat time
also includes input processing, turn-ending tokens and, when needed, history
replay. The RC2 UI divides total elapsed turn time by generated tokens, so
its displayed rate is lower than the isolated generation rate above.
Text still streams immediately as each visible token is generated.

In a real GUI check, the same nine-token reply to `hi` completed in **46.5
seconds**, down from **110.3 seconds** (2.37x faster end to end). Text was
visible while generation was still active, and the UI returned to Ready.
This is one observed comparison, not a repeated latency benchmark.

## Response and compatibility checks

- The final binary reproduced **all 56 responses and token counts** across
  the existing 48-case regression suite. Mechanical passes stayed at 24/48.
  This is a regression comparison on an existing suite, not fresh evidence
  of general model quality or a guarantee of identical answers to every prompt.
- 16,384 independent Q6/Q16 integer-dot and activation-error checks passed
  in scalar and SSE2 modes, including extreme values and unaligned buffers.
- Existing Q6 FP32, Q4/Q8 FP32 and Q4/Q8 integer kernel tests passed.
- Independent Hugging Face forward checks at 1, 12 and 47 tokens passed;
  all next-token choices matched. Maximum absolute logit difference was
  0.00556 and maximum mean absolute difference was 0.00085.
- A complete 512-token context passed under a 384 MiB Linux virtual-memory
  limit, with peak RSS 301,632 KiB. All nine IPC, saved-note, recovery,
  replay and context-rollover checks also passed under that limit.
- The Windows executable is PE32, subsystem 5.1, imports only KERNEL32 and
  legacy MSVCRT, and passed the post-SSE2 instruction audit. Actual guest
  execution complements this static check; physical laptop validation remains.

The extra activation scratch space is twice the widest matrix-input dimension
in bytes. No model-sized unpacking buffer or extra model copy is allocated.

Reports are in [performance-20260916](performance-20260916/).
The frozen selection and original evaluation reports remain unchanged and
describe the original FP32-activation runtime.

## Build and reproduce

```sh
bash scripts/build-slm.sh
cc -O3 -fno-fast-math -msse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-avx \
  bench/slm_speed.c src/slm_tokenizer.c -lm -o build/slm/bench_speed
build/slm/bench_speed MODEL.NCB --float-activations
build/slm/bench_speed MODEL.NCB
cc -O3 -msse2 tests/test_slm_q16.c -lm -o build/slm/test_q16
build/slm/test_q16
```

Run SSE2 benchmark commands on an x86 host. The unchanged model SHA256 is
`ac4fe748bfa8f61af3b55f13e5435c299ca4c5abad71cd24c233f6ff24a33953`.
The optimized Windows runtime SHA256 is
`71a570f3bf7b6cd46705caffcbcfe88c04aa982bf6ab84210cf998b0b384bd4c`.

## Install or roll back

Close Bliss, back up the three `NC_RUN*.EXE` files, and replace them with the
optimized runtime under the same names. Keep `XPCHAT.EXE`, model, tokenizer,
settings, chat history and documents. Restore the saved executable files to
roll back. The standalone speed patch contains no model weights.

The previously observed RC2 GUI crash when closing the window is outside this
runtime update and remains unresolved.
