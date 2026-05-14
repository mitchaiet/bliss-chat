# xp-llm: project overview

A complete chat-LLM stack that runs natively on Windows XP / Pentium 4 (2004
hardware). The project covers everything from PyTorch training on a
modern CUDA GPU to a hand-written C inference engine that loads a custom
binary format and runs in ~1500 LOC on a 21-year-old machine.

## The problem

A modern small chat model (e.g. SmolLM2-135M, Qwen2.5-0.5B) is theoretically
small enough to fit on memory-constrained Windows XP-era hardware, but in practice does
not run usably:

- **`llama.cpp` cross-compiled for XP** drops to a scalar codepath because
  the P4 lacks AVX/AVX2 (its fast vec-dot kernels are gated on those).
  Measured throughput on the Windows XP-era Pentium 4 machine with SmolLM2-135M Q4_0:
  **~25 sec/token**, with quantization artifacts producing partly-garbage
  output (mode collapse on certain prompts).
- Modern Windows binaries link against UCRT (`api-ms-win-crt-*.dll`) which
  doesn't exist on XP.
- Modern Linux toolchains for cross-compile produce binaries that depend on
  Vista+ APIs (SRWLock, condition variables, PrefetchVirtualMemory).

## The approach

Drop the off-the-shelf path. Train a model **sized for the hardware** and
write the inference engine to match the model **exactly**, with no
abstraction tax.

1. **Training**: Karpathy's [nanochat](https://github.com/karpathy/nanochat)
   on Linux + an CUDA training workstation. Model sizes:
   - `d6` (~30M params, 1.6 min) — demo tier, used for first end-to-end
     bring-up.
   - `d12` (~110M params, 54 min) — coherent base LM, current target.
2. **Architecture**: nanochat's specific transformer variant. Llama-arch base
   (RMSNorm, RoPE, GQA-capable) plus several non-standard components:
   - ReLU² MLP (instead of SwiGLU)
   - QK-norm
   - Per-layer **value embeddings** with per-head sigmoid gates
   - Per-layer learnable residual scalars (`resid_lambdas`, `x0_lambdas`)
   - "Smear": previous-token embedding mixed into current via gate
   - "Backout": mid-layer residual subtracted before final norm
   - Sliding-window attention with per-layer L/S pattern
   - Logit softcap (`15 * tanh(logits/15)`)
3. **Export**: Custom `NCB1` binary format. Single file, 256-byte header
   followed by tensors in fixed order. Per-row symmetric int8 quantization
   for matrix weights and embedding tables; fp32 for scalars and rotary
   tables. Result: a 110M-param model lands at **~280 MB** (down from 793 MB
   fp32). Custom `NCT1` tokenizer format from the tiktoken pickle.
4. **Inference**: `nc_run.c` + `nc_tokenizer.c`, ~1500 LOC of C99. Single
   threaded, scalar math, mmap-style heap load of the entire model file.
   No SIMD intrinsics — on the P4 with only SSE2/SSE3, vectorization
   doesn't pay enough to justify the complexity, and `ggml`-style fallbacks
   were measured to be slower than scalar for these tiny models.
5. **GUI**: `xpchat.c`, native Win32 with RichEdit and a COMCTL32 progress
   bar. Spawns the inference binary once, talks to it over anonymous pipes
   with a tiny sentinel-line protocol.
6. **Cross-compile**: build host → XP via `i686-w64-mingw32-gcc`,
   statically linked against legacy `msvcrt.dll` (with several patches to
   work around Vista+ assumptions in libstdc++).

## Result

| Setup | Model | Speed on XP Pentium 4 |
|---|---|---|
| `llama.cpp` (XP-patched) + SmolLM2-135M Q8 | 135M params, vanilla llama-arch | ~25 sec/token |
| **`nc_run` + nanochat-d6 int8** | 30M params, custom arch | **~3 tok/s** |
| `nc_run` + nanochat-d12 int8 | 110M params, custom arch | ~0.5–1 tok/s expected |

The custom path is **~100× faster** than off-the-shelf for d6, with fluent
English output. d12 is more coherent but slower.

## What's in this folder

This `/context` folder is the deep-dive technical writeup of the project,
intended as the canonical reference for anyone (including future-us)
trying to understand why the code looks the way it does.

- `00-overview.md` — this file
- `01-system-architecture.md` — how the pieces fit together
- `02-inference-engine.md` — `nc_run.c` forward pass, every step
- `03-tokenizer.md` — BPE + ASCII pre-tokenizer, NCT1 format
- `04-binary-formats.md` — NCB1/NCT1 byte-level layout
- `05-quantization.md` — int8 per-row symmetric quant
- `06-cross-compile.md` — XP toolchain, every flag explained
- `07-gui-and-protocol.md` — XPCHAT.EXE Win32 design, sentinel protocol
- `08-nanochat-architecture.md` — what nanochat is, why we picked it,
  what's non-standard about its transformer
- `09-training-history.md` — what we trained, configs, results, lessons
- `10-known-issues.md` — SFT NaN, multi-turn, etc.
