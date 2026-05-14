# Bliss Chat XP v1.2.0

Single-file portable Windows XP milestone/demo release for the
`bliss-d12-curated-c20-v1` model.

## Download

Use the release asset:

- `bliss-chat-xp-v1.2.0-bliss-d12-curated-c20-v1-portable.exe`

Drop it on a Windows XP machine and double-click. The EXE self-extracts its bundled runtime/model/tokenizer to a temporary directory, launches the native Win32 chat UI, and cleans up when the app exits. No installer, no Python, no internet, and no separate model files are required.

This release is meant to show the complete offline XP pipeline working on real
legacy hardware. It is not a modern assistant replacement: expect short, simple
answers, with best results on terse factual prompts.

## What changed since v1.1.0

- Packages the release as one self-contained portable `.exe` asset.
- Uses the d12 curated c20 int8 model, reported as `Bliss d12 293M (int8)`.
- Bakes in the coherent test defaults: `ctx=256`, `temp=0.0`, `top_p=0.95`, and `max_tokens=128`.
- Adds prompt-assist for fragile tiny-model prompts such as guitar facts and simple compliments.
- Starts each user turn from a clean prefixed KV state to avoid prior bad turns contaminating new answers.
- Adds XP-native Microsoft Sam text-to-speech via SAPI: use `Speak last reply` after an assistant answer.
- Shows NSIS extraction/progress instead of silently unpacking the large model payload.
- Embeds the Bliss Chat icon and refreshed version metadata.
- Includes the local browser web chat harness for pre-packaging testing during development.

## Known limitations

- The model is tiny by modern standards and can be overly terse or generic.
- Coherence is improved by deterministic decoding, prompt shaping, and per-turn
  cache reset; answer quality is still the next major milestone.
- The “cool guitar fact” local smoke test currently returns a coherent but plain
  answer: `The guitar is a very old instrument.`

## Verification

Built with an `i686-w64-mingw32-gcc` cross-compiler and `makensis`.

- Artifact: `dist/bliss-chat-xp-v1.2.0-bliss-d12-curated-c20-v1-portable.exe`
- Size: `260,740,046` bytes
- SHA256: `90620ecb6a0de2ecc91f1dfcb21dd4e7dd0b55240d93e7c19d4a8a690ee3d901`
- Tests: `python3 -m pytest tests/test_nc_run_guardrails.py tests/test_xpchat_tts.py` → 7 passed
- DLL import check: XP-era system DLLs only, including `ole32.dll` and `OLEAUT32.dll` for SAPI.

## Model

- Model: `bliss-d12-curated-c20-v1`
- Architecture: d12, 12 layers, 768 embedding, 6 heads
- Export: int8 NCB
- Training steps: 33,600
- Final validation bpb: 0.818168
