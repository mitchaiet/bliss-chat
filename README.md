# xp-llm

A chat LLM that runs natively on Windows XP, end-to-end.

Train a custom transformer on a modern GPU, export to a tiny binary format,
run inference on a 21-year-old Pentium 4 via a hand-written C engine and a
native Win32 GUI.

## What this is

| Layer | Implementation |
|---|---|
| Training | [Karpathy `nanochat`](https://github.com/karpathy/nanochat) on Linux+CUDA |
| Model architecture | nanochat (RMSNorm, RoPE, QK-norm, ReLU² MLP, value embeddings, residual lambdas, smear, backout, softcap) |
| Checkpoint export | Custom `NCB1` binary format (int8 per-row quantized matrices, fp32 scalars and rotary tables) |
| Tokenizer | tiktoken BPE → custom `NCT1` binary format |
| Inference engine | `nc_run.c` — ~1500 LOC of C99, single-threaded, scalar math, mmap-based weight load |
| GUI | `xpchat.c` — Win32 (RichEdit + COMCTL32 progress bar), Win32 anonymous pipes for IPC |
| Wire protocol | Sentinel-line protocol on stdin/stdout: `\x01READY\n`, `\x01INFO ...\n`, `\x01EOT\n`, `\x01ERR ...\n` |
| Cross-compile | Mac → Win32 via `i686-w64-mingw32-gcc`, statically linked against legacy `msvcrt.dll` |
| Dashboard | `nc_dashboard.py` — single-file http.server + JS frontend, parses training logs |

Target hardware: **Dell Dimension 4700**, Pentium 4 @ 3 GHz, 512 MB DDR-400,
no AVX/SSE 4.x, integrated Intel 82915G graphics, IDE drive, Windows XP SP3.

## Repository layout

```
xp-llm/
├── README.md             this file
├── src/                  C source code
│   ├── nc_run.c          inference engine (compiled to NC_RUN.EXE)
│   ├── nc_tokenizer.c    BPE + ASCII pre-tokenizer
│   ├── xpchat.c          Win32 GUI (compiled to XPCHAT.EXE)
│   └── resource.rc       icon + version info
├── tools/                Python tooling, runs on the training machine
│   ├── export_ncb.py     PyTorch checkpoint → NCB1 (with optional --int8)
│   └── export_tokenizer.py  tiktoken pickle → NCT1
├── server/
│   └── nc_dashboard.py   training progress dashboard (stdlib only)
├── installer/
│   └── *.nsi             NSIS installer scripts
├── assets/
│   └── xp_tiny_llm.ico   app icon
├── scripts/              build + deploy automation
│   ├── build-xp.sh                cross-compile NC_RUN + XPCHAT for XP
│   ├── train-d6.sh                d6 ratio-12 (legacy short run)
│   ├── train-d6-chinchilla.sh     d6 ratio-20 (Chinchilla-optimal, ~9 min)
│   ├── train-d8-chinchilla.sh     d8 ratio-20 (untested middle ground)
│   ├── train-d12.sh               d12 ratio-12 (~50 min)
│   ├── sft-fp32.sh                SFT NaN-fix attempt (fp32 forward — falsified)
│   ├── export-and-deploy.sh       ckpt → NCB+NCT in build/deploy/
│   └── push-to-xp.sh              copy binaries+model to XP over telnet+HTTP
└── docs/
    └── ARCHITECTURE.md   detailed architecture writeup
```

## File formats

### `NCB1` (model)

Single-file binary, 256-byte header followed by tensors in fixed order.
Header fields: vocab size, layer count, head count, embed dim, head dim,
sequence length, value-embedding layer mask, sliding-window pattern mask,
rotary base, softcap. See `tools/export_ncb.py` and `nc_run.c::ncb_header_t`.

Two dtypes:
- `dtype=0` (fp32) — float32 row-major, no quantization.
- `dtype=1` (int8) — per-row symmetric quantized matrices: `[int8 weights row-major][fp32 row scales]`. Embeddings (`wte`, `value_embeds`) and the LM head all use this scheme. Per-layer scalars and rotary tables stay fp32.

A 30M-param d6 model lands at **75 MB int8** vs **295 MB fp32**.

### `NCT1` (tokenizer)

Header + sequence of `(id u32, nbytes u32, bytes[])` records, regular tokens then specials,
EOF marker. The C side builds a hash table at load time for byte-sequence → id lookup.

The pre-tokenization regex is an ASCII approximation of tiktoken's GPT-4 split
pattern (`\p{L}` → `[A-Za-z]`, `\p{N}` → `[0-9]`); good enough for English chat,
non-ASCII still encodes via byte-level fallback tokens.

## Quick start

### One-time setup (training machine, Linux+CUDA)

```bash
# Install nanochat, train a small model
git clone https://github.com/karpathy/nanochat.git ~/nanochat
cd ~/nanochat
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv sync --extra gpu
source .venv/bin/activate

# Pretrain (use one of these):
bash /path/to/xp-llm/scripts/train-d6.sh    # 1.6 min, base demo (rambly)
bash /path/to/xp-llm/scripts/train-d12.sh   # 50 min, coherent base
```

### Build for XP (Mac with Homebrew mingw-w64)

```bash
brew install mingw-w64
cd xp-llm
bash scripts/build-xp.sh
# produces build/NC_RUN.EXE and build/XPCHAT.EXE
```

### Export model + tokenizer (training machine)

```bash
bash xp-llm/scripts/export-and-deploy.sh d12   # writes build/deploy/{MODEL.NCB,TOKENIZER.NCT}
```

### Push to the XP machine

```bash
bash xp-llm/scripts/push-to-xp.sh
```

(One-time prerequisites on the XP side: telnet enabled, `get.vbs` and `XPGET.EXE`
copied into `C:\xp-llm\`. See `docs/ARCHITECTURE.md`.)

### Run

Double-click the **XP Tiny LLM** desktop shortcut. Window opens, model loads
in a few seconds, status flips to `Ready`, dynamic label shows the actual
model description ("Model: nanochat-d12 110M (fp32 int8)"). Type, hit Enter.

## Performance on the Dell Dimension 4700

| Setup | Model | Speed |
|---|---|---|
| llama.cpp (XP-patched) + SmolLM2-135M Q8 | 135M params, vanilla llama-arch | ~0.04 tok/s |
| `nc_run.c` scalar + nanochat-d12 int8     | 110M, hand-written engine | 0.88 tok/s |
| **`nc_run.c` SSE2 + nanochat-d12 int8**   | 110M, +matmul+attention SIMD | **4.68 tok/s** |
| **`nc_run.c` SSE2 + d6-chinchilla int8**  | 30M, ratio-20 trained        | **~27 tok/s** |

The SSE2 path is **5.33× faster** than scalar on the same Pentium 4
(`linear()` average per call dropped 15075 µs → 2765 µs, attention
QK·V from 20.6 → 7.8 ms/forward). Compared to off-the-shelf llama.cpp
on the same hardware, the custom pipeline is roughly **100× faster**.

See `context/11-profiling.md` for full numbers.

## Slash commands

Type these as your message and the backend handles them, no model
roundtrip:

| Command | Effect |
|---|---|
| `/reset` | Drop the running KV cache, start a fresh conversation |
| `/info`  | Show model size, dtype, current turn index, KV occupancy |
| `/help`  | List the available slash commands |

The backend also auto-resets when the KV cache is within 64 tokens of
`sequence_len`, with a `[INFO] context full, conversation reset`
notice.

## Model variants

| File | Params | Size | Tokens/s on P4 | Quality |
|---|---|---|---|---|
| `MODEL.NCB`     | d12, 110M | 280 MB | 4.68 | multi-sentence English, recognizable facts |
| `MODEL_D6.NCB`  | d6, 30M   | 75 MB  | ~27   | shorter, factual answers, ramblier on free-form |

Both ship to `C:\xp-llm\`. To switch, swap which file `MODEL.NCB`
points to (or rename) — `NC_RUN.EXE` always loads the file named
`MODEL.NCB`.

## Status

- ✅ Pipeline working end-to-end (d6 + d12, both shipped)
- ✅ Custom binary formats verified across Mac/Linux/XP
- ✅ Cross-compile clean (KERNEL32 + MSVCRT only — no UCRT, no Vista APIs)
- ✅ GUI integration with dynamic model label and live progress bar
- ✅ SSE2 SIMD: matmul (5.45×) + attention helpers (2.6× softmax block)
- ✅ Multi-turn KV cache with `/reset` and auto-overflow
- ✅ d6 Chinchilla-trained model (val_bpb 1.165 → 1.075)
- ✅ Live training dashboard at `http://localhost:8899/`
- ❌ SFT chat-tuning — still diverges to NaN; bf16-overflow falsified.
     Next experiment: gradient clipping. See `context/10-known-issues.md`.
