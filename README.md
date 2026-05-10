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
│   ├── build-xp.sh       cross-compile NC_RUN + XPCHAT for XP
│   ├── train-d6.sh       train a tiny d6 model (~30M params, ~2 min on RTX 6000)
│   ├── train-d12.sh      train a real d12 model (~110M params, ~50 min on RTX 6000)
│   ├── export-and-deploy.sh  ckpt → NCB+NCT in build/deploy/
│   └── push-to-xp.sh     copy binaries+model to the live XP box over telnet+HTTP
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
| llama.cpp (XP-patched) + SmolLM2-135M Q8 | 135M params, vanilla llama-arch | ~25 sec/token |
| llama.cpp (XP-patched) + SmolLM2-135M Q4 | 135M params | ~25 sec/token (no SIMD speedup on P4 SSE3) |
| **`nc_run.c` + nanochat-d6 int8**         | 30M params, custom arch       | **~3 tok/s** |
| `nc_run.c` + nanochat-d12 int8 (planned)  | 110M params                   | ~0.5–1 tok/s expected |

The custom path beats off-the-shelf llama.cpp by **~100×** on this hardware
because of (a) tiny-model targeting and (b) zero AVX-gated codepath bloat.

## Status

- ✅ Pipeline working end-to-end with d6 base model
- ✅ Custom binary formats verified across Mac/Linux/XP
- ✅ Cross-compile clean (KERNEL32 + MSVCRT only — no UCRT, no Vista APIs)
- ✅ GUI integration with dynamic model label and live progress bar
- ✅ Live training dashboard at `http://localhost:8899/`
- ⏳ d12 training in progress
- ❌ SFT chat-tuning — diverges to NaN on small models, unresolved (likely needs
     custom embedding init for the chat-special tokens; out of scope for v1)
