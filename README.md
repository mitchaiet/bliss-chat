# bliss-chat

A chat LLM that runs natively on Windows XP, end-to-end.

Train a custom transformer on a modern GPU, export to a tiny binary format,
run inference on Windows XP-era hardware via a hand-written C engine and a
native Win32 GUI.

**No internet. No emulation. No cloud.** A real tiny LLM, packaged as one
self-contained Windows XP `.exe`, generating short local responses on a
Windows XP-era machine.

## Download

Drop the .exe on a Windows XP machine and double-click. It self-extracts
to a temp dir, launches the chat GUI, and cleans up on exit. Nothing else
to install.

- **[bliss-chat-xp-v1.3.0-memory-portable.exe][full]** — single-file portable EXE, d12 mem c20 v2 int8 model, persistent memory, knowledge-folder RAG, automatic CPU backend selection.

The file is also browsable on the [v1.3.0 release page](https://github.com/mitchaiet/bliss-chat/releases/tag/v1.3.0). SHA-256 in `RELEASE_v1.3.0.md`.

[full]: https://github.com/mitchaiet/bliss-chat/releases/download/v1.3.0/bliss-chat-xp-v1.3.0-memory-portable.exe

## Current release: v1.3.0 — memory

Bliss now remembers. The v1.3.0 model was retrained on a curated mixture
that teaches multi-turn recall, selective persistent-note recall,
`Context:`-grounded answering, and small-step reasoning — the bench
"correct" score went from 57% to **77%**, multi-turn memory from 27/40 to
**37/40**, with zero regression in general language modeling.

- One self-contained portable `.exe`; no separate model/tokenizer files.
- **Persistent memory**: `/remember`, `/memories`, `/forget`, a per-message
  Remember button, and a Tools menu — notes survive restarts and are used
  selectively by the model.
- **Restored chats really resume**: reopening a saved chat replays its recent
  turns into model context.
- **Knowledge folder RAG v2**: term-overlap retrieval with match-centered
  snippets and a Sources footer, in the trained `Context:` format.
- **Real top-p (nucleus) sampling**; automatic SSE2/SSE3 backend selection.
- **~72 MB less RAM** — the engine no longer keeps a second KV-cache copy.
- Multi-turn KV thread memory with `/reset` and smarter context rollover
  (the bounded summary now keeps both questions and answers).
- LM Studio-style runtime controls: `/template`, `/defaults`, `/preset`.
- XP-native Microsoft Sam text-to-speech via **Speak last reply**.
- Visible NSIS extraction/progress window for the large bundled model payload.

Known limitations:

- The model is tiny by modern standards and is not a general-purpose assistant.
- Arithmetic beyond small numbers, and multi-hop reasoning, remain weak.
- Answers are intentionally short; long-form composition is out of scope.

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
| Cross-compile | Linux → Win32 via `i686-w64-mingw32-gcc`, statically linked against legacy `msvcrt.dll` |
| Dashboard | `nc_dashboard.py` — single-file http.server + JS frontend, parses training logs |

Target hardware: **Windows XP-era Pentium 4-class machines** with SSE2 and limited RAM.

## How the model was trained (plain-English walkthrough)

This project has two halves: first, teach a small language model on a modern GPU; second, shrink and package that model so an old Windows XP computer can run it locally.

### 1. Start with a tiny transformer shape

A language model is a big stack of number tables. During training, those numbers are adjusted until the model gets good at guessing the next piece of text.

For Bliss Chat, the shipped model uses Karpathy's `nanochat` transformer code with a small `d12` layout:

- `d12` means 12 transformer layers.
- The model is around 110 million useful parameters before export.
- This is tiny compared with modern cloud assistants, but large enough to produce simple coherent English.
- It is small enough to fit into an XP-era app after compression and int8 export.

The goal was not to make a ChatGPT competitor. The goal was to prove a complete end-to-end local XP language-model pipeline.

### 2. Train it to predict text

Training uses a large pile of ordinary text. The model sees text broken into small pieces called tokens. At each step it tries to predict the next token.

Example idea:

```text
The capital of France is ___
```

At first the model guesses badly. The training program measures the error, then slightly changes the model's numbers so the next guess is better. Repeating that billions of times is what “training” means here.

The shipped Bliss model was trained with:

- Architecture: `d12`
- Batch size: 65,536 tokens per training step
- Training steps: 33,600
- Total training: about 2.2 billion tokens
- Training recipe: Chinchilla-style ratio 20, meaning the small model gets a lot of text for its size instead of being under-trained
- Final validation score: about 0.818 bits-per-byte on held-out text

Validation text is text the model does not train on. If the validation score improves, it means the model is learning patterns that generalize instead of only memorizing the training examples.

### 3. Test sample answers during training

During training, the system periodically asks the model simple prompts and saves the answers. This gives a human-readable sanity check alongside the numeric validation score.

Earlier small runs could produce English but rambled or repeated themselves. The d12 Chinchilla run gave more recognizable short facts, such as simple science and geography answers. It still makes mistakes, but it crossed the line from “toy gibberish” to “small local demo assistant.”

### 4. Try chat-tuning, then choose the safer base model

After base training, several chat-tuning experiments were attempted. Chat-tuning means showing the model examples shaped like conversations so it learns to answer as an assistant.

Those experiments were not shipped because they made behavior worse. Some runs became unstable; others learned the format but started echoing prompts, looping, or losing factual behavior.

So the release uses the stronger base model plus runtime guardrails instead of a damaged chat-tuned checkpoint. The app wraps user prompts in a simple question/answer format and asks for one short factual sentence.

### 5. Export the PyTorch checkpoint into a tiny XP-friendly file

The training checkpoint is a PyTorch file meant for a modern Linux/Python environment. Windows XP cannot use that directly.

The exporter converts it into `MODEL.NCB`, a custom binary format made for this project:

- Tensor names and Python objects are removed.
- Weights are written in the exact order the C inference engine expects.
- Large matrix rows are quantized from 32-bit floats to 8-bit integers.
- Small scale values are kept so the C code can approximately reconstruct the original numbers while running.

This makes the model much smaller while keeping greedy-output quality effectively the same in spot checks.

### 6. Export the tokenizer too

The tokenizer is the rulebook that turns text into token IDs and token IDs back into text.

The project exports that into `TOKENIZER.NCT`, another custom binary file. The XP C program loads it at startup, builds a small lookup table, and uses it for all chat input/output.

Without the tokenizer, the model would only see raw characters incorrectly. The model and tokenizer must match.

### 7. Write a C inference engine for Windows XP

The runtime engine is `src/nc_run.c`. It loads `MODEL.NCB` and `TOKENIZER.NCT`, then performs the transformer math directly in C.

That matters because Windows XP cannot rely on modern Python, PyTorch, CUDA, or cloud APIs. The release runs with ordinary XP-era system DLLs.

The engine also includes CPU-specific builds:

- `NC_RUN_SSE2.EXE` for older Pentium M / Pentium 4-class machines
- `NC_RUN_SSE3.EXE` for newer XP-era CPUs

The GUI detects the CPU at startup and launches the best backend automatically.

### 8. Package everything into one portable EXE

The final release bundles these files into one self-extracting Windows executable:

- `XPCHAT.EXE` — the Win32 chat interface
- `NC_RUN_SSE2.EXE` — safe backend for SSE2-only CPUs
- `NC_RUN_SSE3.EXE` — faster backend for SSE3 CPUs
- `MODEL.NCB` — the trained Bliss model
- `TOKENIZER.NCT` — the matching tokenizer
- release metadata and version info

When you double-click the portable `.exe`, it extracts to a temporary folder, starts the GUI, runs the model fully offline, and cleans up when you exit.

### 9. What the result can and cannot do

Bliss Chat can answer short, simple prompts locally on Windows XP hardware. It is best treated as a technical demo of a complete training-to-XP deployment pipeline.

It cannot match modern assistants. It can be terse, wrong, repetitive, or confused. The important achievement is that the whole stack — training, export, quantization, inference, GUI, packaging, and XP execution — works end to end without internet or emulation.


## Repository layout

```
bliss-chat/
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
git clone https://github.com/karpathy/nanochat.git /path/to/nanochat
cd /path/to/nanochat
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv sync --extra gpu
source .venv/bin/activate

# Pretrain (use one of these):
bash /path/to/bliss-chat/scripts/train-d6.sh    # 1.6 min, base demo (rambly)
bash /path/to/bliss-chat/scripts/train-d12.sh   # 50 min, coherent base
```

### Build for XP

Requires `i686-w64-mingw32-gcc` (Linux mingw-w64 cross toolchain) and
`makensis` (NSIS) if you want the portable installer.

```bash
sudo apt install gcc-mingw-w64-i686 nsis      # Debian/Ubuntu
# or your distro's equivalent

cd bliss-chat
bash scripts/build-xp.sh
# produces build/NC_RUN.EXE, build/NC_RUN_SSE2.EXE, build/NC_RUN_SSE3.EXE, and build/XPCHAT.EXE
```

To rebuild the portable EXE shipped on the Releases page:

```bash
makensis -DMODEL=build/deploy/MODEL.NCB \
         -DOUTFILE=dist/bliss-chat-xp-v1.2.1-auto-cpu-portable.exe \
         installer/portable.nsi
```

### Export model + tokenizer (training machine)

```bash
bash bliss-chat/scripts/export-and-deploy.sh d12   # writes build/deploy/{MODEL.NCB,TOKENIZER.NCT}
```

### Push to the XP machine

```bash
bash bliss-chat/scripts/push-to-xp.sh
```

(One-time prerequisites on the XP side: telnet enabled, `get.vbs` and `XPGET.EXE`
copied into `C:\xp-llm\`. See `docs/ARCHITECTURE.md`.)

### Run

Double-click the portable **Bliss Chat XP** `.exe`. A small extraction/progress
window appears while the bundled model unpacks, then the Win32 chat window opens.
Status flips to `Ready`, and the dynamic label shows the actual model description
(`Bliss d12 293M (int8)`). Type, hit Enter. After a reply completes, use
**Speak last reply** to read it aloud with XP's built-in Microsoft Sam/SAPI voice.

## Performance on Windows XP-era hardware

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
| `/info`  | Show model size, dtype, current turn index, KV occupancy, note count |
| `/help`  | List the available slash commands |
| `/remember <fact>` | Store a short persistent note (survives restarts) |
| `/memories` | List stored notes with their numbers |
| `/forget <n>` | Remove note n |

Persistent notes live in `%APPDATA%\bliss-chat\MEMORY.TXT` (plain text,
one note per line) and are injected into the model's system prefix as a
`Notes:` line, so Bliss can use them in any conversation. New notes take
effect at the next `/reset`, context rollover, or app start.

The backend also auto-resets when the KV cache is within 64 tokens of
`sequence_len`, rebuilding from a bounded thread summary that now keeps
both the questions and the answers.

## Model variants

| File | Params | Size | Tokens/s on P4 | Quality |
|---|---|---|---|---|
| `MODEL.NCB`     | d12 curated c20 | 279 MB | ~4.7 | short coherent answers, recognizable facts |
| `MODEL_D6.NCB`  | d6, 30M   | 75 MB  | ~27   | shorter, factual answers, ramblier on free-form |

The portable v1.2.1 release embeds `MODEL.NCB` and `TOKENIZER.NCT` inside the single `.exe`. The older d6 path remains useful for experiments but is not the primary release asset.

## Status

Shipping:

- Pipeline working end-to-end with a single-file XP portable EXE
- Custom binary formats verified across Linux and XP
- Cross-compile clean with XP-era system DLLs only; SAPI TTS uses `ole32.dll`/`OLEAUT32.dll`
- GUI integration with dynamic model label, live progress bar, and **Speak last reply**
- SSE2 SIMD: matmul (5.45×) + attention helpers (2.6× softmax block)
- Clean per-turn KV reset, `/reset`, and auto-overflow
- d12 curated c20 model release (`Bliss d12 293M (int8)`)
- Live training dashboard at `http://localhost:8899/`
- Local browser web chat harness for pre-packaging testing

Open:

- Improve answer quality beyond terse/demo responses while keeping XP-era hardware viable.
- SFT chat-tuning still diverges to NaN; bf16-overflow falsified. Next experiment: gradient clipping. See `context/10-known-issues.md`.
