# Small-model native Windows XP backend

## Architecture and memory

`src/slm_run.c` runs bias-free, tied-embedding Llama and LFM2 models exported
by `tools/export_slm.py`. The original Llama integration uses
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
or its merged fine-tunes. The 135M model uses the same architecture and tokenizer.
The model publisher distributes both under Apache-2.0.

The 360M configuration has 32 layers, width960, FFN2560,15 query heads and5
KV heads, head width64, learned RMSNorm (epsilon1e-5), split-half RoPE
(theta100000), SwiGLU and49,152 tokens. The configuration is read from the
exported header; dimensions and file bounds are checked before mapping tensors.
See the [publisher's config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json).

The default export is group64 signed Q4, with one FP32 scale per group.
Consecutive weights occupy the low and high nibbles of a byte after adding8.
Norm weights remain FP32. Embeddings and the output head share one mapped
matrix. Only the current embedding row is dequantized. There is no full FP32
embedding copy and no duplicate KV cache.

Exact 360M export size:203,739,136 bytes (194.30MiB). Context512 uses
41,943,040 bytes (40MiB) of FP32 KV cache. Tokenizer data is about1.26MiB;
its runtime lookup tables, GUI, code and scratch arrays add memory beyond
those figures. These sizes are not a measured Windows XP working set.

## Building

```sh
bash scripts/build-slm.sh
```

This produces `build/slm/slm_run` for the build host and, when MinGW is
installed, `build/slm/SLM_RUN_SSE2.EXE`. The XP binary is32-bit PE with
subsystem5.1 and targets Pentium M SSE2. The build probes
`-mcrtdll=msvcrt-os`; an import check rejects UCRT and external compiler
runtimes. The verified local compiler required this flag: `-static` alone
still produced modern `api-ms-win-crt-*` imports.

## Exporting

```sh
python tools/export_slm.py --model /path/to/hf/snapshot \
  --out build/slm/model --bits 4 --context 512
build/slm/slm_run build/slm/model/MODEL.SLM \
  build/slm/model/TOKENIZER.SLT --prompt 'What is gravity?'
```

The exporter runs on a modern machine and needs PyTorch and safetensors.
The XP runtime needs neither. `--bits 8` provides a larger weight reference;
`--bits 32` exists for numerical verification and does not fit the512MB target.
`--scale-method mse` optionally minimizes reconstruction error with a fixed
scale search; the default remains maximum-absolute scaling. Neither method
uses evaluation questions or calibration activations.

The binary header is256 bytes: magic `SLMODEL1`, version, bits, dimension,
FFN width, layer/query/KV counts, head width, vocabulary, context, group size,
BOS/EOS, a reserved field, RoPE theta and RMSNorm epsilon. All numbers are
little-endian. Ordered tensors are embedding, then each layer's two norms,
Q/K/V/O, gate/up/down, then final norm. Tensor matrices store packed weights
followed by FP32 group scales. FP32 matrices and norms have no scales.

### Exploratory LFM2.5 support

The same exporter and runtime also support the bias-free, tied-embedding
[LFM2.5-350M configuration](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/config.json).
This implementation is a candidate until its pretrained and native answer
quality passes evaluation. Its model license is
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE),
not the SmolLM2 Apache-2.0 license.

SLM model version2 retains the first72 header bytes, with architecture1 at
offset60. Offset72 stores convolution width3, offset76 the ChatML start ID,
and bytes80 onward contain one layer type each:0 attention,1 convolution.
Smol remains version1. Per LFM layer, tensors are operator_norm, ffn_norm;
then either Q/K head norms and Q/K/V/out matrices, or the FP32 depthwise
convolution kernel followed by in/out matrices; then feed_forward w1,w3,w2.
The final tensor is embedding_norm. FFN width is read from the actual
weights because the source's6656 setting expands to an actual width4608.

The hybrid has six attention layers and ten convolution layers. Attention
applies learned RMSNorm to each Q/K head before RoPE. Convolution projects
to B,C,x, stores B*x in a three-sample history and returns C*conv(B*x) before
the output projection. Kernel index0 uses the oldest sample. Every history
reset zeroes the convolution state before replaying retained tokens. KV
storage is allocated only for attention layers and convolution state only
for convolution layers.

For the official350M configuration, group64 Q4 weights occupy approximately
190.37MiB; context512 adds12MiB KV and0.117MiB convolution state. Tokenizer,
scratch arrays and application memory are additional. These are calculated
sizes, not a measured XP working set. The runtime accepts groups16,32 and64;64 remains the export default.
Group32 weights occupy211.50MiB, adding21.125MiB versus group64.

### Q6 for LFM

`--bits 6 --group 64` exports LFM matrices with six bits per weight. Each
64-weight group occupies 48 bytes: 32 bytes containing consecutive low
nibbles, followed by 16 bytes containing four high-two-bit fields each.
Stored values are signed integers offset by 32. Export uses the symmetric
range −31 through 31; the decoder also handles −32. Matrix scales remain
separate FP32 values, one per group. Norms and convolution kernels remain
FP32. Q6 is accepted only for LFM with group size 64.

The 350M Q6 model file is **288,226,560 bytes (274.87 MiB)**. Context 512
adds 12 MiB of KV cache and 0.117 MiB of convolution state, plus tokenizer
and runtime scratch space. These calculated sizes do not establish an XP
working set. The optimized Q6 runtime quantizes each activation group to
signed 16-bit values and uses SSE2 integer dot products. It adds only twice
the widest matrix-input dimension in bytes of scratch space. Weights and
the model format are unchanged. `--float-activations` selects the original
RC1/RC2 FP32 matrix path. Q4 and Q8 retain their existing activation defaults.
See [the speed and response comparison](Q16_PERFORMANCE.md).

Converted model headers carry a modification notice referring to
`MODEL_CARD.md` and `MODEL-LICENSE.txt`. It begins at byte 128 for the
350M configuration, outside the layer-type table. Redistributed packages
must retain the publisher's license and provide the named provenance files.

LFM chat prepends BOS once, then uses ChatML start6/end7. It stops generation
on EOS7; other skip-marked tokens are omitted from output while maintaining
their effect on model state. Its independent tokenizer version2 and export
live in `tools/export_lfm_tokenizer.py`; Smol's tokenizer version1 remains
supported. Numeric-token forward tests can bypass tokenizer comparisons
using `--skip-tokenizer` while the tokenizer is validated separately.

## Tokenizer correctness

The tokenizer implements the publisher's actual pipeline: isolate every
Unicode numeric character, apply the GPT2 byte-level splitting expression,
then use explicitly ranked BPE pair merges. The exporter converts GPT2's
byte-to-Unicode vocabulary into byte strings and exports Unicode letter,
number and whitespace ranges. No modern Unicode or regex DLL runs on XP.
The exported metadata records the Unicode database version.

SmolLM2 does not include every possible single-byte vocabulary entry and
has no unknown token for this BPE path. The native tokenizer follows the HF
behavior for these absent entries. Special tokens are explicit; ordinary
user text cannot inject ChatML role delimiters through the chat API.

Sources: [tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/tokenizer_config.json),
[numeric splitting](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/digits.rs),
[byte-level expression](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs).

## GUI protocol and memory

The runner accepts existing positional model/tokenizer paths and `-c`, `-t`,
`-p`, `-n`, `-s SEED`, and `-m MEMORY.TXT`. Files can be named `MODEL.NCB` and
`TOKENIZER.NCT` for existing packaging; their contents must be the new
formats. Installing a new model with an old backend, or the reverse, fails
header validation. Keep the old release artifacts separately.

The stdout protocol retains READY, INFO, ERR and EOT sentinels. STOP is
polled between generation tokens. Existing `/reset`, `/temp`, `/topp`,
`/maxtok`, `/seed`, `/system`, `/defaults`, `/preset`, `/info`, `/help`,
`/template`, and `/replay user<TAB>assistant` commands are accepted.
The template is fixed ChatML; `/template` reports it rather than changing
the architecture's prompt format.

Persistent notes retain the existing twelve-note, one-line-per-note
`MEMORY.TXT` format. `/remember`, `/memories`, `/forget n`, `/memreload` and
`/memfile path` are supported. Writes use a temporary sibling file and
replace the destination only on successful completion; failed writes roll
back the in-memory edit. Up to three notes with query word overlap are
included in the current user turn. This is lexical retrieval and may miss
synonyms. Notes do not change the system prompt's training wording.

History caches complete ChatML turns. When needed, the oldest complete
turns are removed and retained turns are prefetched again. This bounds RAM
but adds a visible computation cost on context rollover. Oversized user or
system input returns an error; a rejected system prompt preserves the
previous system and history. `/replay` reconstructs prior saved chats.

Default decoding is greedy, with a96-token answer limit. The default system
message is identical to `tools/prepare_coherence_data.py`.

## Verification and limits

```sh
cc -O3 -std=c99 -Wall -Wextra -Werror \
  tests/test_slm_kernels.c src/slm_tokenizer.c -lm -o build/slm/test_kernels
build/slm/test_kernels
python tests/test_slm_parity.py --hf /path/to/hf/snapshot \
  --export build/slm/model --binary /absolute/path/to/build/slm/slm_run \
  --float-activations --report build/slm/parity.json
python tests/test_slm_ipc.py --export build/slm/model \
  --binary /absolute/path/to/build/slm/slm_run --report build/slm/ipc.json
```

Verified on the Linux workstation:

-169 tokenizer fixtures matched HF tokenizers exactly, including randomized
  text, Unicode, numeric splitting, whitespace and special tokens.
-4,096 Q4 and4,096 Q8 signed integer dot-product cases matched independent
  scalar arithmetic using the SSE2 implementation.
-FP32 model forward passes at1,13 and49 input tokens matched HF logits with
  maximum absolute error below0.00005 and identical next-token argmax.
-Q4 weights with FP32 activations matched HF at the same positions with
  maximum absolute error below0.000032 and identical next-token argmax.
-The trained Q4 model passed saved notes, restart/reload/forget, recoverable
  oversized system/user input, replay/reset, persistent-note answering,
  EOT emission and context rollover checks.

The Q4/Q8 fast default uses Q8 activations. Its exact integer kernels passed, but
sequential native versus HF implementations accumulated activation-rounding
differences: mean logit differences around0.12–0.14 on the longer probes,
cosine similarity above0.999 and the same next-token argmax. This is not
strict full-model numerical parity. `--float-activations` is a slower
weight-only reference option that removes this discrepancy. It now uses
SSE2 unpacking and float dot products; compiling with
`-DSLM_FLOAT_DOT_SCALAR` retains the scalar float path for timing comparisons.
Quantized answer quality
must still be measured on held-out conversations; a successful kernel or
export check does not establish better answers.

The XP binary cross-compiles without warnings, imports only KERNEL32.dll
and msvcrt.dll, and passed a disassembly check for unintended SSE3/SSSE3/AVX
instructions. Actual startup, peak working set, responsiveness and speed
on the Dell Pentium M remain hardware checks.

Exploratory LFM arithmetic was additionally checked with deterministic random
weights from `tests/make_lfm_fixture.py`: three layers (conv, attention,
conv), nontrivial learned head norms, and1/12/47-token probes. FP32,
Q4 with FP32 activations, and Q4 with Q8 activations agreed with the matching
HF reference, with maximum absolute error below0.000002. The new SSE2
float path passed the same Q4 forward checks. `tests/test_slm_cache.c`
confirmed43 positions after reset are bit-identical to a fresh state.
`tests/test_lfm_chat.py` checked three exact plain-chat templates, including
an empty system message and a previous conversation turn. Tiny random
fixtures establish arithmetic behavior, not pretrained answer quality.

HF transformers4.57's slow LFM convolution cache has an edge case when
prefilling only one token: the next absolute position is written into the
middle cache slot. Quantized incremental parity therefore prefills at least
three tokens before decoding; full-sequence FP32 parity remains independent
of this cache path. Ordinary chat prompts already exceed three tokens.

The full unmodified LFM2.5-350M checkpoint subsequently passed strict FP32
(max absolute logit difference0.000087) and Q4/SSE2-FP32-activation
(max0.000036) checks on1/12/47-token probes; all next-token choices matched
HF. Its Q4 file is199,621,888 bytes. Fast Q8 activations had mean logit
differences0.10–0.17 and did not meet strict parity tolerance; this mode's
answer quality must be compared separately. A fixed eight-token Smol360M
CPU prefill measured median0.250s for Q8 activations,0.314s for SSE2 float
activations and1.395s for scalar float activations on the Linux workstation.
These are host timings, not Pentium M performance.

Group16/32/64 integer kernels passed12,288 Q4 plus12,288 Q8 independent
scalar-oracle checks on Linux SSE2. Full LFM group32 Q4/F32-activation
forward parity passed the same three probes (maximum logit difference
0.000112, all next-token choices identical). Group16 has kernel coverage;
no claim is made here about its full-model quality or XP working set.

Q6 passed 8,192 independent packing and floating-point dot-product cases
under ASan/UBSan on Linux SSE2 and the macOS scalar path. These include
every possible code in every group position and unaligned packed weights.
The deterministic LFM fixture passed whole-forward parity with maximum
absolute logit difference below 0.000001. Thirteen malformed Q6 files,
including truncated planes/scales and invalid format fields, were rejected
cleanly under ASan/UBSan.

The first-trained full LFM Q6 model passed all three 1/12/47-token forward
probes against HF loaded with the exact exported weights and scales:
maximum absolute logit difference 0.000060, with identical next-token
argmax values. The calibrated Q6 model passed the same checks with maximum
difference 0.000096. These checks validate export and runtime arithmetic;
answer quality is evaluated separately on the unchanged diagnostic suite.

`tests/test_slm_memory.py` fills all 512 context positions under a Linux
384 MiB virtual-memory limit and checks that all final logits are finite.
Both full Q6 candidates passed this test: first-trained peak RSS was
301,632 KiB (294.6 MiB), and calibrated peak RSS was 301,440 KiB (294.4 MiB).
These runs used the exact candidate files and the same immutable Linux
binary as their diagnostic answer evaluations.
`tests/test_slm_ipc.py --limit-mib 384` applies the same limit to protocol,
saved-note, reset, replay and rollover checks. Both Q6 candidates passed
all nine checks, including saved notes across restart, recoverable oversized
input, note retrieval and history rollover. These tests provide
Linux process evidence only; physical XP startup, working set and speed
still require the target laptop.
