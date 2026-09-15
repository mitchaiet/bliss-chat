# Bliss Chat 2: training and evaluation

**Built and trained a substantially more coherent candidate on the T2 GPU.**
The selected model is **Bliss 2 LFM2.5-350M Q6**, version
`2.0.0-candidate.20260915`. It is the strongest combined coherence candidate
tested in this work. It fits the measured backend memory budget for the
repository's XP target, and its actual Windows backend passed Wine checks.
Physical laptop startup, speed and complete application memory use remain
unverified. Arithmetic and precise instruction following remain weak.

## Fresh acceptance results

Selection, model and executable hashes were frozen at
`2026-09-15T08:37:28.104813+00:00`, before the fresh 48-case suite ran. No weights
or selection decisions changed after seeing its results. Each model used its
native engine, a 512-token context, greedy decoding, seed 42 and a 128-token
reply cap. The test supplied actual generated conversation history and isolated
saved notes. It did not supply gold answers or use the GUI's local calculator.

| Measure | Original Bliss | Selected Bliss 2 |
|---|---:|---:|
| Fully correct final replies, reviewed | 7/48 | 26/48 |
| Semantic total, out of 8 | 3.458 | 6.562 |
| Coherence, out of 2 | 1.583 | 1.979 |
| Mechanical checks passed | 3/48 | 24/48 |
| Protocol failures | 0 | 0 |

Bliss 2 won **37** paired semantic comparisons, lost **4** and tied **7**.
The rubric gives 0–2 points each for relevance, correctness/grounding,
coherence and instruction following. Fluency does not guarantee correctness:
the selected model's arithmetic correctness mean was lower, despite more
relevant and better-formed answers.

"Fully correct" means final-answer content received correctness/grounding 2/2;
it does not require perfect formatting or compliance in earlier acknowledgments.
This compares complete deployment backends. The original uses a Q/A template,
punctuation/newline stopping and repetition penalty 1.12, while the candidate
uses ChatML and its own stopping and note handling. The experiment does not
isolate the causal effect of training or compression.

### Fully correct final replies by category

| Category, eight cases each | Original | Bliss 2 |
|---|---:|---:|
| Conversation meaning | 2 | 6 |
| Conversation memory | 0 | 5 |
| Saved-note grounding | 3 | 6 |
| Missing information / unknowns | 1 | 6 |
| Strict instructions | 0 | 2 |
| Arithmetic and logic | 1 | 1 |

The suite was authored separately from the training-data work. Model names
were randomized independently for each case; all semantic scores were written
and validated before the mapping was revealed. This was AI review with prior
familiarity with candidate behavior, not an independent human panel. Mechanical
checks are imperfect, and this small, authored suite is not a general capability
percentage. The evidence archive contains every prompt, answer, score, masking
map and run manifest. The final review also documents individual successes
and regressions.

The old repository's 100-question benchmark contains 44 questions present in
repository training files. Its previous score cannot establish independent
generalization. Our development suite had no normalized exact question matches
in the ten checked repository SFT files. Upstream pretraining overlap and
paraphrases cannot be ruled out.

## Hardware budget and speed

The repository target is a **Dell Inspiron 8600, Pentium M 1.40 GHz, 512 MB
RAM and Windows XP**, with an FX Go5200 GPU. Inference runs on the CPU; the old
GPU is not needed. The T2 workstation provided the modern training GPU.

The selected model has 354,483,968 parameters. Its native weight file is
288,226,560 bytes (274.87 MiB); the tokenizer is 1,692,655 bytes. Six-bit weights
in groups of 64 retain substantially more quality than the tested four-bit
exports. Q6 automatically uses FP32 activations, while packing weights compactly.
The 512-token key/value cache uses 12 MiB; convolution state is about 0.117 MiB.

| Check | Result and scope |
|---|---|
| Full 512-token context under a 384 MiB virtual-memory limit | Passed; peak Linux backend RSS 301,632 KiB, or 294.6 MiB; all final logits finite |
| Protocol, saved notes, recovery, replay and rollover | Nine checks passed under the same memory limit |
| Final 48-case backend peak RSS | Original 298,000 KiB; selected 293,780 KiB |
| Windows executable audit | PE32, subsystem 5.1, legacy MSVCRT and SSE2; exact final hashes audited |
| Actual Windows backend under Wine 9 configured for XP | Three next-token comparisons matched Linux; maximum logit difference 2.72e-5; control frames and Unicode note persistence matched |
| Actual XP laptop / complete GUI | Not available for validation |

The static audit separately documents MinGW numeric helpers whose TZCNT encoding
executes as BSF on pre-BMI processors, and the reviewed nonzero-input/flag
conditions. This audit is not an exhaustive Windows API availability proof.
The Wine report's `float_activations: false` records an omitted command-line
flag; Q6 still uses FP32 activations automatically.

Final timing ran serially on the same modern Linux host after other training
and validation jobs ended:

| Observed native CPU timing | Original | Bliss 2 |
|---|---:|---:|
| Median whole user-turn response | 0.692 s | 4.575 s |
| 95th-percentile response | 1.035 s | 7.280 s |
| Median first visible text | 0.568 s | 3.667 s |
| Median process startup, measured separately | 0.507 s | 4.959 s |

The new model was about **6.6 times slower** by median user-turn time. Replies
differ in length and content, so this is not a matched-token throughput test.
Modern-host timing does not predict Pentium M latency. Physical XP startup,
OS-plus-GUI memory pressure and sustained speed are still required before
calling this a validated laptop release.

## Design and training decisions

The original nanochat artifact has about 286 million actual parameters, including
roughly 151 million value-embedding parameters. Its architecture and available
pretraining were a poor starting point for the requested coherence improvement.
We compared compact pretrained foundations and retained the best observed
balance of native coherence and memory use.

The selected foundation is
[LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M), revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f`. Its hybrid architecture combines
ten short-convolution layers with six attention layers. We implemented native
C inference and publisher-compatible tokenization, including state reset and
replay, rather than requiring a modern Python runtime on XP.

### Development comparison

These 42 development cases informed selection. Their scores must not be
confused with the fresh 48-case acceptance results above. HF results use the
uncompressed Hugging Face implementation; native results measure actual
compressed model behavior. Retrieval placement and runtime settings differ,
so HF-to-native differences are not solely quantization effects.

| Candidate | Mechanical passes /42 | Semantic mean /8 |
|---|---:|---:|
| Original Bliss NCB | 15 | 4.500 |
| SmolLM2-360M foundation, HF | 21 | 5.905 |
| SmolLM2 first adaptation, HF | 26 | 6.238 |
| SmolLM2 anchored adaptation, HF | 25 | 6.095 |
| Qwen2.5-0.5B foundation, local FP32 | 21 | 5.833 |
| LFM2.5-350M foundation, HF | 23 | 6.119 |
| LFM first adaptation, HF | 26 | 6.500 |
| LFM broader calibrated adaptation, HF | 23 | 6.500 |
| LFM first adaptation, Q4 group64 / Q8 activations | 21 | 5.881 |
| LFM first adaptation, Q4 group32 / FP32 activations | 22 | 6.214 |
| **LFM first adaptation, Q6 group64 / FP32 activations** | **25** | **6.524** |
| LFM broader calibrated adaptation, Q6 | 23 | 6.357 |

Four adaptation runs completed: two SmolLM2 runs and two LFM runs. The broader
LFM experiment used independently authored corrective data, a larger adapter
and three epochs. It improved some missing-information cases but regressed on
conversation and reasoning examples after native export. We preserved that
experiment and selected the first LFM adaptation. No candidate won every
category, and this search does not prove a globally optimal model.

### Selected training recipe

- GPU: RTX PRO 6000 Blackwell Workstation Edition, about 96 GB VRAM, on mitch-xpt.
- Training run: `/home/mitch/bliss-runs/20260915-next/lfm25-anchored`.
- Rank-16 LoRA, alpha 32, dropout 0.05; Q, V and output projections.
- Learning rate 2e-5; cosine schedule, 6% warmup; gradient clipping 0.5.
- Batch 32 with accumulation 2; one epoch; seed 20260915.
- Assistant-only supervised loss; public retention rows use KL weight 1.0
  against detached predictions of the unchanged foundation.
- 26,430 training dialogues; 5,747,284 rendered tokens; 2,472,527 supervised
  assistant tokens. Validation: 1,460 dialogues.
- 413 updates in 190.6 seconds; selected checkpoint 413. Validation loss
  improved from 1.40393 to 1.10700.

Approximately 75% of rendered training tokens came from
[Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; the rest were procedural
examples for context recall, notes, missing information and simple reasoning.
Evaluation questions and answers were not training inputs. Masking and
retention audits checked the actual prepared rows and tokenizer boundaries.

The broader comparison used 20,564 unique dialogues, with explicit repeats of
2,048 independent calibration rows giving 26,708 rows per epoch. It trained
1,254 steps in 862.7 seconds and selected checkpoint 300 by validation loss.
That model is preserved but is not the packaged model.

Training used Python 3.10, PyTorch 2.9.1+cu128, Transformers 4.57.6 and PEFT
0.17.1. Exact executed source, data hashes, manifests and loss histories are
in the evidence archive; reproduction commands are in the source documentation.
GPT-6 Astra ultra assisted the engineering and evaluation. Its weights are
not part of the local model.

## Runtime and application changes

The new native engine includes packed Q6 matrix kernels, attention and
convolution state, a matching tokenizer, and bounded conversation replay.
Numerical checks include 8,192 Q6 kernel cases, retained Q4/Q8 regressions,
malformed-format rejection and full-model parity against dequantized exported
weights. The selected model's maximum parity error was 5.991e-5 across three
prefixes, with all next-token choices matching. These checks establish engine
fidelity, not reasoning quality or equivalence to uncompressed weights.

The GUI now transports UTF-8 chat text through Unicode controls, waits for
control acknowledgments correctly and validates settings. It defaults to a
512-token context, deterministic replies and at most 128 generated tokens.
Existing saved conversations, notes, knowledge-folder retrieval, calculator,
local time and Windows speech integration remain in the source. Filesystem
paths retain ANSI limitations. Extracted helper checks and existing source
regressions passed; the complete GUI was not exercised on physical XP.

## Artifacts, preservation and installation

- `BlissChat2-XP.zip`: model, tokenizer, XP GUI, SSE2 backend, licenses and
  checksummed release manifest. Extract into a new folder such as `C:\Bliss2`
  and run `XPCHAT.EXE`.
- `BlissChat2-Source.zip`: native source, GUI changes, training/export/build
  tools and reproduction documentation, without large foundation weights.
- `BlissChat2-Source.patch`: complete changes from source commit
  `3c21d79c74a75f3404faefaf9a0d0491bcd223c5`.
- `BlissChat2-Evidence.zip`: raw evaluations, reviews, training records,
  calibration audit and runtime verification.

The prior GPU model server on port 8001 was stopped as requested. Its model
files were preserved and it was not restarted. Training and selected weights
remain under `/home/mitch/bliss-runs/20260915-next`; the GPU is no longer
running training. The original `/home/mitch/bliss-chat` checkout retains its
original three untracked benchmark CSVs and no tracked changes. New source is
in the isolated `train/xp-coherence-20260915` worktree. Nothing was published or
pushed to GitHub; a source archive and patch preserve the uncommitted work.

The model uses the publisher's **LFM Open License v1.0**, including its
commercial-use revenue condition; it is not Apache-2.0. The exact model license
and modification notices accompany the package. Public adaptation data uses
Apache-2.0. See
[pinned model terms](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/LICENSE).

Selected model SHA256:
`ac4fe748bfa8f61af3b55f13e5435c299ca4c5abad71cd24c233f6ff24a33953`.

Final suite SHA256:
`9ef8940b4cacbcecd5d679e0eba90bfb5548eca488b7a37ae8fdcd63a9b1ca41`.

## Remaining limits

This is a much more coherent conversation candidate, with measured gains in
recall and note use. It can still invent facts, mishandle referents, give poor
practical advice and fail simple arithmetic or exact output constraints. The
short context restricts longer conversations. The largest outstanding product
question is real Pentium M responsiveness and complete memory use on the XP
laptop. Keep the old installation while checking this candidate on that hardware.
