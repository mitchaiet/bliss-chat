# Bliss next: independent evaluation v1

This directory is evaluation-only. Never include it in a training glob, distillation
prompt set, retrieval index, or model prompt. The runner feeds only each case's
`turns` and optional fictional `notes` to the model. `checks`, `reference`, and
`rubric` are exclusively for scoring and review.

## Frozen suite

- 42 cases: six each for ordinary conversation, limited reasoning, instruction
  following, conversation memory, persistent notes, grounding, and uncertainty.
- SHA256 of `heldout_v1.jsonl`:
  `95db43c50b26039207959b2b1fb842e83681baf2ee7278ff8d7d8e989ff2033b`.
- Authored independently of the existing Bliss training builders. Fictional names,
  numbers, and contexts test application of information supplied at inference.
- A read-only normalized exact-prompt check on 2026-09-15 found no matching user
  turns in any of the ten checked-in SFT v1-v5 train/validation JSONLs on
  mitch-xpt. This does not establish absence from other pretraining corpora,
  paraphrase absence, or complete independence from a public model's pretraining.
- The set is deliberately small. Publish individual answers and per-category
  results; do not call its mechanical total a comprehensive intelligence score.
- If failed rows are used to design new training examples, this suite becomes a
  development set. Freeze a fresh independently authored final test before making
  a subsequent generalization claim.

## Run the existing NCB baseline

The POSIX runner starts a fresh backend for each case, supplies an isolated
temporary notes file even when empty, and preserves generated assistant replies
between a case's user turns. Cases never share notes or conversation state.

```bash
python3 evaluate.py --validate
python3 evaluate.py \
  --binary /home/mitch/bliss-chat/build/nc_run_native \
  --model /home/mitch/bliss-chat/build/deploy/MODEL.NCB \
  --tokenizer /home/mitch/bliss-chat/build/deploy/TOKENIZER.NCT \
  --ctx 512 --max-tokens 128 --out /NEW_RUN_DIR/baseline_ncb
```

`--out` must be a new directory. Outputs include `predictions.jsonl`, every
generated conversation turn, metadata and model/binary/tokenizer hashes,
per-case backend stderr, a summary, and a `review.jsonl` with blank human scores.
`--categories` is for troubleshooting only; final comparisons use all 42 cases.

Native switches can be appended directly with repeatable `--runtime-arg`, for
example `--runtime-arg=--float-activations`. Use the equals form for values that
start with a dash. Each value is one literal argument, with no shell expansion.
The actual binary is hashed and `metadata.json` records `runtime_args`, avoiding
a wrapper script that would obscure which executable was measured. Omitting the
option preserves the existing invocation.

The controls sent after READY are `/maxtok 128`, `/info`, and `/template`.
Temperature 0 and seed 42 are passed on the command line. Existing other runtime
defaults remain active and are logged by `/info`; the baseline still has its
one-sentence system and stop policy. Use `--no-controls` for a replacement
sentinel backend that does not implement these commands, and record its actual
generation settings separately. A successful return without EOT is a protocol
failure, not a pass.

## Run the cached HF baseline or trained model

```bash
/home/mitch/bliss-runs/20260915-next/venv/bin/python hf_eval.py \
  --model /LOCAL_MODEL_DIRECTORY \
  --device cuda --dtype bfloat16 --ctx 512 --max-new-tokens 128 \
  --out /NEW_RUN_DIR/smol360_hf
```

This uses local cached files only, the tokenizer's chat template, a fixed short
Bliss system message, greedy generation, and actual generated conversation
history. It reserves the output budget within 512 tokens, discarding whole oldest
user/assistant pairs only if required and recording that event. It never supplies
gold assistant turns. BF16 GPU/HF timing is labeled separately from native timing.

An external backend can produce JSONL objects with `id` and `answer`, optionally
`turn_results`, then use the common scorer without running any model:

```bash
python3 evaluate.py --predictions /PREDICTIONS.jsonl --out /NEW_RUN_DIR/scored
```

Missing IDs fail. Duplicate IDs are rejected. Unknown extra IDs are recorded.

## Semantic review: the main coherence evidence

The mechanical gates catch obvious content and formatting failures, including
wrong-direction yes/no replies and stale memory values. They remain imperfect:
an answer can include all expected words while being contradictory, and a valid
paraphrase can miss a regex. Never equate regex pass with coherent conversation.

1. For each case, place baseline and candidate answers side by side with random
   labels A/B. Conceal model names, quantization, training status, and regex scores
   from the reviewer. Randomize which model appears first per case and keep the
   label mapping separate. Review all conversation turns, not just the recall.
2. Score each response on four dimensions, each 0, 1, or 2:
   - **Relevance:** 0 misses the request; 1 partially addresses it; 2 directly
     addresses what the user asked.
   - **Correctness/grounding:** 0 false or invented; 1 incomplete or ambiguous;
     2 consistent with the supplied facts and correct result, or appropriate
     uncertainty when the answer is unavailable.
   - **Coherence:** 0 incoherent, looping, or internally contradictory; 1 readable
     but awkward or needlessly repetitive; 2 clear, consistent, and natural.
   - **Instruction following:** 0 violates the principal constraint; 1 partly
     complies; 2 follows the requested output, brevity, and scope.
3. Add a short concrete note for every 0 and every disagreement with the regex
   gate. A correct alternative phrase can earn semantic credit without rewriting
   the frozen mechanical rubric after seeing model identity.
4. Report category means, baseline/candidate wins/ties/losses, severe failures,
   and representative verbatim answers. A candidate that learns a persona line
   but still invents memories or contradicts itself has not met the goal.
5. The 42-case suite supports a bounded improvement claim. For a strong release
   claim, add a fresh blinded ordinary-dialogue session after model selection and
   real-hardware checks below. Do not manufacture a percentage for subjective
   "massive improvement" without showing the underlying paired answers.

## Quantization, size, and speed

Compare full-precision/BF16, int8, and q4 exports from the **same frozen weights**,
with the same tokenizer, prompt rendering, context budget, and generation
controls. Retain each checkpoint and export hash. Existing MODEL_MX and MODEL_Q4
files are older than the current int8 file; equal architecture alone does not
prove equal checkpoint origin. Re-export when provenance is unavailable.

First establish full-precision native/HF numerical fidelity on identical token
prefixes (logits or next-token ranks within a documented numerical tolerance).
Then compare quantized-native semantic results against native full precision.
A changed greedy token is a divergence diagnostic, not automatically a quality
failure. Review new contradictions, wrong numbers, invented facts, lost recalls,
and malformed text separately. Keep the smaller export only when its observed
quality tradeoff is acceptable for the actual user-facing workload.

Record model bytes, total package bytes, peak backend memory, startup/prefill
time, visible time to first token, and end-to-end turn latency. The POSIX runner's
TTFT is first visible pipe output and includes prompt processing; its EOT token
count divided by elapsed time is **not** pure decode throughput. For latency A/B,
run each model serially under the same host load and repeat the same small prompt
set at least three times. Also time a fixed number of decode steps on identical
token prefixes: a model that emits shorter answers can look faster without having
a faster inference kernel. Do not mix GPU/HF timing with native CPU timing.

## Exact XP target and required hardware evidence

The specifically documented machine in `context/14-test-devices.md` is a Dell
Inspiron 8600 with Pentium M 1.40 GHz, 512 MB DDR333 RAM, and a GeForce FX Go5200
with 64 MB VRAM. The old note reports the installer runs but NC_RUN crashes;
SSE3 incompatibility and RAM pressure were hypotheses. Current `build-xp.sh`
already builds separate SSE2 and SSE3 backends, so that old build-flags diagnosis
is historical, not a current verification. `context/11-profiling.md` also records
older Pentium 4 timings, but those are a different CPU and cannot establish
Inspiron response speed.

For the actual Inspiron: verify a 32-bit XP-compatible SSE2 executable, supported
DLL imports, successful load, no illegal-instruction crash, stable multi-turn
chat, startup time, response latency, committed/working-set memory, and paging
under ordinary XP load. Test notebook AC power and CPU clock conditions and keep
them fixed. Measure the full GUI path, because its local arithmetic tools and
knowledge injection are outside `nc_run_native`.

Cross-compiling successfully, running on modern Linux, or using Wine/emulation
does not verify those properties on the physical XP computer. A model can be
fully trained and packaged while those final hardware claims remain unverified.

## Existing artifacts: read-only audit, 2026-09-15

Remote repository `/home/mitch/bliss-chat`, branch `feat/q4-experiment`, HEAD
`3c21d79`. No applicable AGENTS.md was found in the repo or its ancestors.
The existing untracked `bench/results_memory.csv`, `results_notes.csv`, and
`results_persona.csv` were preserved.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| MODEL.NCB | 292886664 | 11c82d07097f057dbb458a02e4bed8daf748f359b73dfd3abb4dd1df9dc679cc |
| MODEL_MX.NCB | 235477128 | 8e7eb54c1f1a4f64413fc6c44c87d6c8b315b9a681641ea1cb45a3b49438c133 |
| MODEL_Q4.NCB | 184158344 | 3e528333565f29617c2df8735c91e81370cd69019b7855955d5fa0ba1a9c2a28 |
| TOKENIZER.NCT | 477616 | 292796d0675114f3bcdf88a80127518fb4bcf365058cee446b3ff827024ebff3 |

The two existing native binaries are identical Linux x86-64 ELF files, SHA256
`f714ec5524e17cc8f8b7ab6a485e356cdcbdc47a4baf9455ac2f10cb1caca096`.
All five NCB headers report vocab32768, 12 layers, width768, six attention and
six KV heads, head dimension128, sequence length1024. Thus the current engine's
full FP32 KV cache is 72 MiB. Its `-c512` changes the logical context cutoff but
`state_init` still allocates from the header's1024 length; it does not halve that
allocation. A stale context document's36 MiB estimate assumes only three KV heads.

Existing CSVs record 35/40 memory,22/25 notes,7/8 persona, and74/100 keyword
correct general responses. They lack per-run model/config hashes and are not a
fresh baseline. Moreover v5 train contains exact matches for28/100 general and
6/8 persona prompts; v1 train contains36/100 and6/8 respectively. These existing
suites are useful regression checks, not clean held-out generalization evidence.

Source tests mostly assert presence of code strings, so passing them cannot
establish model coherence, arithmetic accuracy, quantization fidelity, or GUI
behavior. The new runner's chunked READY/EOT handling, visible-output timing,
token-count extraction, stderr drain, and unexpected-exit handling were verified
with a synthetic subprocess fixture; that is harness evidence only.
