# Bliss Chat XP architecture

Current public release: **v1.2.0**, distributed as one self-contained portable
Windows XP `.exe` that embeds `XPCHAT.EXE`, `NC_RUN.EXE`, `MODEL.NCB`,
`TOKENIZER.NCT`, `MODEL_VERSION.txt`, and `release-manifest.json`.

A complete, end-to-end mapping from a fresh PyTorch checkpoint to live
inference on Windows XP-era hardware. Each layer of the system, what it does, and why
it's shaped the way it is.

## Hardware target

Target class: Windows XP-era 32-bit machines with Pentium 4-class CPUs, SSE2, limited RAM, and no modern GPU compute path.

Public docs intentionally avoid private lab-machine identifiers, network details, and exact personal hardware specs.

## Process & wire layout at runtime

```
+--------------------------------------------------------------------------+
| bliss-chat-xp-v1.2.0-...-portable.exe (NSIS self-extracting wrapper)     |
|  - visible extraction/progress UI                                         |
|  - temp payload cleaned up when the GUI exits                             |
|                                                                          |
|  +-------------------------------+    stdin pipe    +------------------+ |
|  | XPCHAT.EXE  (Win32 GUI)       | ---------------> | NC_RUN.EXE       | |
|  | - RichEdit transcript         |                  | (inference)      | |
|  | - live progress/status        |    stdout pipe   |                  | |
|  | - Speak last reply via SAPI   | <--------------- | sentinels+tokens | |
|  +-------------------------------+                  +------------------+ |
|                                                         | read from temp   |
|                                                         v                  |
|                                          MODEL.NCB (279 MB) + TOKENIZER.NCT|
+--------------------------------------------------------------------------+
```

`XPCHAT.EXE` spawns `NC_RUN.EXE` once at startup with anonymous pipes. The
two processes communicate using a tiny line-based protocol on the inference
binary's stdout:

- `\x01READY\n` — model loaded, ready for input.
- `\x01INFO <text>\n` — model description (drives the GUI's "Model:" label).
- `\x01EOT\n` — end of one assistant turn.
- `\x01ERR <text>\n` — error.

Anything else on stdout is response text, streamed token-by-token to the GUI
(which appends it to the transcript on the UI thread).

The GUI sends one user message per line on stdin. EOF on stdin makes the
backend exit cleanly. v1.2.0 restores the prefixed KV snapshot before each user
turn so a bad answer does not contaminate the next prompt.

The GUI's **Speak last reply** button uses XP's built-in `SAPI.SpVoice` COM
automation asynchronously, so the release does not bundle a separate audio or
TTS engine.

## Inference engine: `src/nc_run.c`

Pure C99, ~1500 LOC including the tokenizer. No external libraries beyond
libc + WinSock (for older deploy variants). Single-threaded scalar math —
no SIMD intrinsics — because the Pentium 4 has only SSE2/SSE3 and `ggml`'s
SSE3-only path is slower than scalar for these tiny models anyway.

### Architecture details (must match nanochat's `gpt.py` exactly)

1. **Token embedding** lookup (per-row int8 dequant on the fly).
2. **Norm** after embedding (RMSNorm, no learnable scale).
3. **Smear** — mix previous token's pre-smear normed embedding into current via
   sigmoid-gated scalar. Cheap bigram trick. Kept off (zero) at first token.
4. **Save x0** — initial post-norm, post-smear embedding for residual lambdas.
5. **For each of N transformer blocks:**
   a. `x = resid_lambdas[i]*x + x0_lambdas[i]*x0` — per-layer residual mix.
   b. **Attention**:
      - Norm input
      - Q/K/V linear projections (int8 quantized)
      - For VE layers: lookup `value_embeds[i][token_id]`, gate by `3*sigmoid(ve_gate · x[:12])` per kv-head, add to V
      - RoPE on Q and K (precomputed cos/sin tables, base 100000)
      - QK norm (RMSNorm per head)
      - Q*1.2, K*1.2 (sharper attention)
      - Sliding window attention with per-layer window mask. Single token per step writes to KV cache; attention reads the visible window.
      - Output projection (int8) + residual.
   c. **MLP**:
      - Norm input
      - c_fc projection (int8) → 4D dim
      - **ReLU squared** activation (`max(0,x)²`) — NOT SwiGLU
      - c_proj projection (int8) → D dim
      - Residual.
   d. **Backout snapshot** at midpoint layer (`n_layer // 2`).
6. **Subtract backout**: `x -= backout_lambda * x_mid` to remove low-level features.
7. Final RMSNorm.
8. **LM head** (int8 quantized) → `pad_vocab_size` logits.
9. **Logit softcap**: `15 * tanh(logits / 15)`.
10. Sample with temperature + top-p.

### Memory layout

The `.ncb` file is read into a single heap allocation. All tensor pointers
are simple offsets into that buffer; no separate per-tensor allocations
during inference. KV cache is a separate allocation: `n_layer × seq_len × kv_dim × 4`
bytes for K and again for V (`~10 MB` for d6 at seq_len=512).

Working buffers per forward pass: ~50 KB scratch for activations.

### Quantization

Symmetric per-row int8 with one fp32 scale per row. Encode (in
`tools/export_ncb.py`):

```
abs_max = max(|row|)
scale   = abs_max / 127
q_row   = round(row / scale).clip(-127, 127).astype(int8)
```

Decode (in `nc_run.c::matmul_int8`): the dot product `Σ wq[c] * x[c]` is
accumulated as `int×float→double`, then multiplied by `scale[r]` once per
row at the end. No 16-bit intermediate; not vectorized.

This is enough quality for the model and small enough to fit on XP.
For the embedding tables (`wte`, `value_embeds`) we dequantize on lookup
into a row-sized stack buffer, then use the resulting fp32 row directly.

## Tokenizer: `src/nc_tokenizer.c`

Loads the `.nct` file, builds a hash table from byte-sequence → token id
(open addressing, linear probing, FNV-1a hash). Stays in memory.

**Encode**:
1. Apply the regex-style pre-tokenizer (ASCII approximation of GPT-4 split).
2. For each chunk:
   a. Initialize: each byte becomes its single-byte token.
   b. Greedily merge: scan adjacent pairs, find the one whose concatenated
      bytes have the lowest token id in the vocab. Apply that merge. Repeat
      until no more merges are found.
   c. Emit the resulting token ids.

**Decode**: just look up `id → bytes` and concatenate. Special tokens are
suppressed (their stored bytes are the human-readable name, not output text).

The ASCII pre-tokenizer covers:
- contractions (`'s`, `'t`, `'d`, `'m`, `'ll`, `'ve`, `'re`)
- letter runs with optional leading non-letter
- 1–2 digit numbers
- punctuation runs (with optional leading space)
- whitespace runs and newlines

For non-ASCII input, the regex falls through to the whitespace/punctuation
rules and bytes still encode through the byte-fallback tokens — fine for
English chat, suboptimal for other languages.

## Cross-compile (`scripts/build-xp.sh`)

Compiler: `i686-w64-mingw32-gcc` (Linux mingw-w64 cross toolchain).
Critical flags:

| Flag | Why |
|---|---|
| `-mcrtdll=msvcrt-os` | Link against legacy `msvcrt.dll` (XP has it). Default UCRT (`api-ms-win-crt-*`) doesn't exist on XP. |
| `-D_WIN32_WINNT=0x0501` | Mark XP as target so headers gate Vista+ APIs out. |
| `-march=pentium4 -msse3` | Generate code that runs on the P4. |
| `-static -static-libgcc` | No DLL deps for libgcc. |
| `-O2` | Aggressive optimization, but **not** `-O3 -ffast-math` — we hit precision issues on the model output. |

Verify the produced `.exe` only depends on XP-era system DLLs. For v1.2.0,
`XPCHAT.EXE` also imports `ole32.dll` and `OLEAUT32.dll` for SAPI/Microsoft Sam:

```
i686-w64-mingw32-objdump -p NC_RUN.EXE | grep "DLL Name"
```

## Training dashboard: `server/nc_dashboard.py`

Single Python file, stdlib only. HTTP server on `:8899`. Two endpoints:

- `GET /` — static HTML+JS, polls `/api/status` every 3 seconds.
- `GET /api/status` — JSON: GPU stats from `nvidia-smi`, parsed log files
  from `~/nanochat-logs/`, list of running processes.

Runs as a systemd user service (`nc-dashboard.service`) with linger
enabled so it survives SSH disconnect (remote SSH otherwise reaps
detached processes).

## Why custom-build instead of `llama.cpp`?

Initial approach was to cross-compile `llama.cpp` for XP and run a
chat-tuned llama-arch model. This works but:

- **No fast SIMD path**: ggml's optimized vec_dot kernels are gated on
  AVX/AVX2/SSE4.1. The P4 only has up to SSE3, so we fell through to
  scalar reference code.
- **CPU_REPACK fallback**: ggml warns at load time that it can't use its
  preferred buffer type and falls back to a slower path.
- **Result**: SmolLM2-135M Q8 ran at ~25 sec/token on the P4 with quant
  artifacts producing partly-broken output.

The nanochat path:
- Custom-train a small model sized to the hardware (30M for d6, 110M for d12)
- Hand-write inference matching the architecture exactly — no version churn,
  no abstraction tax, no AVX-only fast paths to fall off of.
- Result: **~3 tok/s** on the same P4 for the d6 model, ~100× faster.

## Known issues

- **SFT diverges to NaN** on small models. Loss bounces 1.7 → 3.0 → NaN
  by step 3 regardless of LR or dataset. Likely cause: random-init
  embeddings for the chat-special tokens (`<|user_start|>`, `<|assistant_start|>`,
  …) cause early activation overflow in the bf16 forward path. Fix is to
  initialize those embeddings from related token embeddings before SFT;
  not yet implemented.
- **Multi-turn**: each user message resets the KV cache (`state_reset`).
  Conversation memory across turns is not yet implemented.
- **Stop button** in the GUI is greyed out — would need to either kill
  and respawn the backend, or add a `\x01STOP\n` request sentinel that
  the inference loop polls for.
