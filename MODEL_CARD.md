# Bliss 2 — LFM2.5-350M Q6

Version: **2.0.0-candidate.20260915**. A compact, adapted language model for
short offline conversations on the repository's Windows XP target.

## Model and operation

The foundation is [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M),
pinned at `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
It has 354,483,968 parameters, with six attention layers and ten short-convolution
layers. It is an adapted LFM model; GPT-6 Astra assisted engineering, data design
and evaluation, and its weights are not part of this package.

The native model file uses group 64 six-bit weights and FP32 scales/norms.
The optimized backend uses groupwise signed 16-bit activation values for Q6
matrix products, with FP32 scales and accumulation between groups. Attention,
normalization, convolution and residual values remain FP32. The original
RC1/RC2 FP32 activation path remains available with `--float-activations`.
This runtime update does not change the model weights. See
[performance validation](docs/Q16_PERFORMANCE.md). Weights occupy
288,226,560 bytes (274.87 MiB); the tokenizer occupies 1,692,655 bytes.
The default context is 512 tokens and replies are capped at 128 tokens, with greedy
decoding. Longer conversations retain recent turns, and saved notes are
retrieved separately. Chat and note data stay on the local machine.

Extract the whole ZIP to a new folder, such as `C:\Bliss2`, and run
`XPCHAT.EXE`. Keep the model, tokenizer and executables together. The three
backend filenames contain the same SSE2 executable for compatibility with
the GUI's existing selection logic. Python, a GPU and internet access are
not required for inference.

## Training and selection

Training ran on the T2's RTX PRO6000 Blackwell GPU in a separate run directory.
The selected run is `lfm25-anchored`, checkpoint 413: rank 16 LoRA on Q, V and
output projections, learning rate 2e-5, batch 32 with accumulation 2, one epoch,
and KL weight 1.0 against the unchanged foundation on public retention examples.
Assistant-only loss excludes user/system text and padding. The run completed
413 updates in 190.6 seconds, and validation loss fell from 1.40393 to 1.10700.

The run contains 26,430 training dialogues, 5,747,284 rendered tokens and
2,472,527 supervised assistant tokens. Validation contains 1,460 dialogues.
Approximately 75% of rendered training tokens come from
[Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; the rest come from
independent procedural recall, note-grounding, unknown-information and simple
reasoning examples. Benchmark answers were not used for training. Upstream
pretraining exposure to public material cannot be excluded.

SmolLM2 and Qwen alternatives, several compression modes, and a broader
calibrated LFM adaptation were evaluated. The latter tied the selected model's
HF semantic score, but scored lower for coherence after native Q6 conversion.
It remains preserved on the T2. Selection was recorded before the fresh final
48-case suite was run; no weights or selection decisions were changed using
that suite's answers.

## Evaluation

These model-quality and original latency results describe the frozen RC1/RC2
package. The runtime-only update is documented separately above.

The 42-case development suite improved from 15 to 25 mechanical passes and from
4.500 to 6.524/8 on a structured review of relevance, correctness, coherence
and instruction following. These are small development-set comparisons,
not general capability percentages. Keyword checks sometimes accept wrong
conclusions and reject valid abstentions; the semantic review is essential.

On the fresh 48-case acceptance suite, the frozen package improved as follows:

| Measure | Original Bliss | Bliss 2 Q6 |
|---|---:|---:|
| Fully correct final replies, reviewed | 7/48 | 26/48 |
| Combined semantic score, out of 8 | 3.458 | 6.562 |
| Coherence dimension, out of 2 | 1.583 | 1.979 |
| Mechanical checks passed | 3/48 | 24/48 |

The candidate won 37 paired semantic comparisons, lost four and tied seven.
"Fully correct" counts final-answer content correctness, not perfect formatting
or compliance in every earlier acknowledgment. This compares complete backends,
including their different prompt templates, stopping rules and note handling.
Model labels were randomized per case, and scores were recorded before the
mapping was revealed. The AI reviewer had prior familiarity with the models;
there was no independent human rater. This small authored suite supports a
substantial improvement in conversation, recall and note use, with uncertainty
handled better on these cases. It does not establish broad reasoning ability.
Both models answered only one of eight arithmetic/logic cases fully correctly;
the candidate's arithmetic correctness mean was lower. Only two of eight
strict instruction cases were fully correct for the candidate.

Serial native CPU evaluation on the modern T2 host measured median reply
latency of 0.692 seconds for the original and 4.575 seconds for the candidate,
about 6.6 times slower. Answers differ in content and length. These are observed
user-turn times, not a matched-token speed benchmark or a Pentium M prediction.

The old 100-question repository benchmark contains 44 questions also found in
repository training data. Its historical score is not an independent
generalization measure. The supplied evaluation report and evidence archive
preserve prompts, responses, hashes, settings, failed experiments and review
limitations.

## Hardware and runtime validation

Working target: Dell Inspiron 8600, Pentium M 1.40 GHz with SSE2, 512 MB RAM and
Windows XP. The native backend is a 32-bit PE executable with subsystem 5.1
and legacy MSVCRT imports. Static instruction checks find no post-SSE2
instructions requiring a newer CPU; documented legacy BSF/TZCNT encodings
are handled separately in the audit.

On Linux, the selected model completed a full 512-token forward pass under a
384 MiB virtual-memory limit, peaking at 301,632 KiB (294.6 MiB) resident memory.
All nine protocol, persistence, recovery, replay and context-rollover checks
also passed under that limit. This excludes the GUI and operating system.

The actual Windows backend loaded in isolated Wine 9 configured for XP.
Three forward comparisons matched Linux's next-token choices, with maximum
logit difference 2.72e-5. Message framing and Unicode saved-note persistence
matched. Tokenizer, packed-math and full-model parity have separate tests.
Extracted GUI Unicode, queue and settings helpers passed their checks.

**Physical XP startup, combined OS/application memory pressure, sustained
response speed and GUI usability have not been verified on the laptop.**
The candidate is ready for that hardware check; Linux and Wine measurements
do not establish actual Pentium M performance.

## Known limitations

- Arithmetic, ordering, precise editing and following several constraints can fail.
- Fluent text can still contain invented facts, wrong referents or contradictions.
- Missing-information handling remains uneven; notes do not guarantee grounding.
- Practical advice can be unsound. Longer reasoning and long-form answers are weak.
- The 512-token context limits how much conversation or retrieved text can be used.
- GUI filesystem paths retain legacy ANSI limitations, despite Unicode chat transport.

## Provenance and license

The model is modified through adaptation, merging, quantization and native
serialization. Its header carries a modification notice. The original
publisher checkpoint remains unchanged on the T2. The native tokenizer
preserves publisher vocabulary and tokenization behavior in a new file format.

The model uses the **LFM Open License v 1.0**, reproduced in `MODEL-LICENSE.txt`.
It includes a commercial-use revenue condition; it is not an Apache-2.0 model.
See [publisher terms](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/LICENSE)
and `NOTICE.txt`. Public adaptation data is from the Apache-2.0 Smol-SmolTalk
dataset. Keep the model license and notices with redistributed packages.

Model SHA256:
`ac4fe748bfa8f61af3b55f13e5435c299ca4c5abad71cd24c233f6ff24a33953`

Tokenizer SHA256:
`2e9a8349d5242ff0070c060ac9e5d15a05739727f1cd4941d631b23bb881bb2d`

Windows backend SHA256:
`5d67e94ee4f261f956d6c132c551d05d0fbb96e436c7d44e447d009264634e4d`

`release-manifest.json` records every packaged file and its checksum.
