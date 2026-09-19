# Bliss Chat

### [⬇ Download Bliss Chat 2.1.0 for Windows XP](https://github.com/mitchaiet/bliss-chat/releases/latest/download/bliss-chat-xp-v2.1.0-portable.exe)

One file, 248 MB, model included. Copy it to the XP machine and double-click it.
Nothing else to install, and nothing is ever sent online.

SHA-256 `a776dee18a0fa032a5c93b17bb9d5a215301a84613d429ec67704beae8e6cbba`.
See the [release notes](RELEASE_v2.1.0.md) for what changed and the
[build receipt](docs/releases/v2.1.0/release-receipt.json) for per-file hashes.
Physical XP validation is still pending.

The current source includes a [Pentium 4 speed build](docs/Q6X4_PERFORMANCE.md)
with a lossless model layout and a Clang-compiled SSE2 runtime. The runtime
splits every linear layer, attention head and activation across all
processors (`-j N`, `--threads N` or `SLM_THREADS=N`; default one thread per
processor), and reads a prompt in batches so the weights stream past the
processor once per batch rather than once per token (`SLM_BATCH`, default
eight). Both keep output bit-identical. The original packed model and the
[earlier Q16 improvement](docs/Q16_PERFORMANCE.md) remain supported.

A small language model with a native Windows XP chat interface. This branch
adds a new compact pretrained foundation, controlled chat adaptation and an
SSE2 C inference engine for the repository's target: **Dell Inspiron 8600,
Pentium M 1.40 GHz, 512 MB RAM, Windows XP**.

The candidate is designed for offline, short-form conversation. It uses a
512-token context and defaults to greedy, bounded replies. Long chats retain
recent context; saved notes can be retrieved separately. The GUI supports
saved conversations, Unicode chat text, notes and the existing knowledge folder.

## Candidate and evidence

Use the accompanying candidate ZIP, model card and evaluation report for the
selected weights and measured results. Extract all files into a new folder,
for example `C:\Bliss2`, and run `XPCHAT.EXE`. Keep the original installation
in its own folder. The model files must stay beside the executables.

The candidate has development-host inference, numerical, memory, IPC and
executable checks. **Physical XP startup, combined OS/application memory use,
response speed and GUI usability still need verification on the laptop.**
Small models can produce incorrect or unsupported answers even when their
language is fluent; the evaluation report records those limitations.

The old 100-question benchmark contains 44 questions also present in repository
training data. Its historical score is not an independent measure of
improvement. This work uses a separate development suite and a fresh final
acceptance suite, with both objective checks and review of answer meaning.

## Build and training

- [Native XP local documents and memories](docs/XP_LOCAL_WORKSPACE.md)

- [Side-by-side browser chat](apps/bliss-compare/README.md)
- [Fresh acceptance suite and review](bench/coherence/FINAL_ACCEPTANCE_REVIEW.md)

- [Training inputs, commands and checks](docs/COHERENCE_REPRODUCE.md)
- [Native engine and file format](docs/SLM_NATIVE.md)
- [LFM tokenizer and parity](docs/LFM_TOKENIZER.md)
- [Calibration data recipe](docs/CALIBRATION_DATA.md)
- [Training integration audit](docs/CALIBRATED_TRAINING_AUDIT.md)
- [XP executable audit](docs/XP_STATIC_AUDIT.md)
- [GUI validation](docs/XP_GUI_VALIDATION.md)

```sh
bash scripts/build-coherence-xp.sh
```

Training and export run on a modern Linux/CUDA workstation. XP inference
requires no Python, account or network connection. The model's publisher
license and modification notices accompany the package; the LFM Open License
is distinct from the Apache-2.0 data license.

## Previous release

The original nanochat implementation, training tools and release material are
preserved. See the [v1 documentation](docs/LEGACY_README_v1.md) and the
[v1.3.0 release](https://github.com/mitchaiet/bliss-chat/releases/tag/v1.3.0).
