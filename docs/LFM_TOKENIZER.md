# Native LFM2.5 tokenizer extension

`src/slm_tokenizer.c` now reads both the existing Smol `SLMTOK1` version 1 and
LFM version 2. The public C functions are unchanged. `slm_encode` never inserts
BOS: runtime chat framing must insert the single LFM BOS itself.

Version 2 keeps the binary layout and magic, with header version set to 2. It
selects the publisher's GPT4-style ordered split expression, including number
groups of one to three Unicode numeric characters, case-insensitive English
contractions, and the specified newline/whitespace alternatives. Unicode
letter/number/space ranges come from the installed Hugging Face tokenizers regex
implementation during export, avoiding a separate Python Unicode database.

Each token's flags field has these independent bits:

| Bit | Meaning |
| --- | --- |
| 1 | Skip when decoding generated text. |
| 2 | Match the literal spelling as an added token. |

Thus special added tokens use 3, ordinary added tokens use 2, and unused padded
entries use 1 with zero byte length. `slm_token_bytes` exposes only the skip bit
to the runtime. Ordinary added tokens such as `python`, `Mathias`, and the think
markers match regardless of the `specials` argument; special tokens match only
when enabled. The added-token scan uses a dedicated list rather than scanning
the entire vocabulary at every byte. Unsupported added-token strip/word options
are rejected by the exporter. No external regex or Unicode DLL is needed on XP.

## Export and test

The separate exporter supports the integration API:

```python
from export_lfm_tokenizer import export_tokenizer
metadata = export_tokenizer(folder, output, model_vocab=65536)
```

With the pinned publisher files, 64,402 tokenizer IDs become 65,536 model
entries: 1,134 empty, skip-only padded IDs. All model logits remain available;
padding is not disguised as an unrelated token. The export is 1,692,655 bytes.

From the repository root:

```sh
python3 tools/export_lfm_tokenizer.py MODEL_DIRECTORY TOKENIZER.SLT --model-vocab 65536
python3 tests/test_lfm_tokenizer.py --lfm-dir MODEL_DIRECTORY \
  --smol-dir SMOL_METADATA_DIRECTORY --smol-tokenizer EXISTING_SMOL_TOKENIZER.SLT
```

The test builds only `slm_tokenizer.c` and its persistent C driver. ASan/UBSan
are enabled by default; `--no-sanitize` is available where unavailable, and `CC`
selects the host compiler. No model weights or chat-quality cases are read.

The saved macOS report `LFM_TOKENIZER_PARITY.json` records:

- 1,770 exact LFM token-ID comparisons and 1,770 decoded-text comparisons,
  covering 885 fixtures with special parsing off and on. These include all 509
  added-token spellings and the existing 169 tokenizer fixtures.
- 3,348 LFM token metadata checks, including all 1,134 padded entries.
- 338 Smol version-1 token-ID comparisons and 338 decode comparisons against
  the existing binary tokenizer, preserving its 49,152-entry vocabulary.
- Six capacity/invalid-ID checks per format and rejection of nine corrupt files.

The tokenizer also cross-compiles as a 32-bit Pentium M/SSE2 object with MinGW,
`-mcrtdll=msvcrt-os`, and warnings treated as errors. These tests establish
tokenizer parity and selected bounds checks; full pretrained-model numerical
parity, response quality, and actual XP execution are separate validations.

The same sanitized suite passed on the Linux training host with tokenizers
0.22.2; local macOS used 0.21.4. Both exports were byte-identical, SHA256
`2e9a8349d5242ff0070c060ac9e5d15a05739727f1cd4941d631b23bb881bb2d`.
See `LFM_TOKENIZER_PARITY_LINUX.json` for the second report. The uploaded native
source hash is `c0e5db7aa8f741ad9f3fb65f30c3567837220e0ac46edec8a72d245baf421dcd`.
