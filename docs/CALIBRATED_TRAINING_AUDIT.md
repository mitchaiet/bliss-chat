# Calibrated training integration audit

## Result

**No material integration fault found in the inspected code and bounded CPU
contracts.** Explicit calibration retention flags survive the mixer and loader;
assistant labels and padding follow the expected boundaries; repeat counts and
live manifest settings reconcile. This audit did not change live training code,
run a model, use a GPU, or read benchmark files, acceptance suites, or predictions.

The inspected run was active. This is an interface and provenance audit, not a
claim that training completed or that its resulting model improved.

## Evidence and scope

Read-only source inspection:

| Source | SHA-256 |
| --- | --- |
| `tools/train_coherence.py` | `4cc4d60abfe698d0e8f79eeae76d5fb5f64df5f6f54fc990d600ab1bade12e90` |
| `tools/run_candidate.py` | `143a94cf814a3710127f7d2c2a8b297c5320bbb12fa0464ad9bf80baa8f2c0b0` |
| `tools/prepare_calibrated_mix.py` | `000a22c1854a4a4d72889e28579a750a593fa843aece6c797de2387202510649` |

Also inspected the independently authored `../calibration/v1` data. Read only
these two authorized live manifests on `mitch@100.106.68.91`:

- `/home/mitch/bliss-runs/20260915-next/lfm25-calibrated/training/training_manifest.json`
- `/home/mitch/bliss-runs/20260915-next/lfm25-calibrated/data/manifest.json`

The training manifest's script hash matches the inspected trainer. Its input
hashes match the mixture manifest's output hashes. The mixture's calibration
input hashes match the final local `v1` files:

| Artifact | SHA-256 |
| --- | --- |
| Calibration train | `498f777d7fbb62ee72d9888305f88594faed178218f7860882c89f621836a46c` |
| Calibration validation | `7ad16dd26c798aaaf1e7cb268fa6c24c0c68ae0a433d1b22eadf2002218b7a8c` |
| Mixed train | `27f95c480d48e44ea8dee70df7e40f65af75c7a3452e1160aca8e8b22b25ca00` |
| Mixed validation | `f0c34de6200eea87baa24207ee86ed2db67d5e4bc002d3ebeca1bb089d8b87f5` |

The mixed row files and foundation weights were not independently downloaded or
rehash-checked in this audit. The manifest chain and inspected runner's checksum
validation provide the evidence stated above.

CPU contract script: `../calibration/integration-audit/contracts.py`.
Machine-readable results: `../calibration/integration-audit/contracts-report.json`.
Its base-data fixture and runner dispatch are explicitly synthetic. The dispatch
double records arguments; it never starts the smoke trainer or real trainer.

## Retention flags and teacher anchoring

The calibration records set `retention: false`. The mixer weights records with
`dict(row, repeat_index=repeat)`, which preserves that boolean. Its validation
list appends the original calibration validation rows, also preserving the flag.
Public base rows are explicitly marked true and selected prior procedural rows
false.

`load_examples` uses an explicit boolean ahead of its source-name fallback.
Nonboolean values such as `"false"`, `0`, and `1` raise `ValueError` for usable
rows. Contracts verified these cases and all **2,368 real calibration records**.
The synthetic mixture loaded with exactly its two public fixtures retained and
every calibration copy unretained.

The legacy fallback recognizes `bliss-calibration-v1`, while this dataset's source
is `bliss-calibration-authored-v1`. That mismatch has **no effect on this run**
because the explicit false flag is present and preserved. Stripping that flag
would change the authored records to retention data; preserve it in future
transformations. An explicit JSON `null` is treated as unspecified/fallback.

The teacher pass uses `model.disable_adapter()` on the configured base model,
whose run path is `models/lfm25-350m`. No previous candidate path is used by this
run. This checks configuration and code behavior, not a fresh hash of foundation
weights. With KL enabled, only retention-marked rows are selected, at most four
per batch. Corrective rows still receive the normal assistant-token CE loss.

## Assistant labels, causal positions, and padding

`encode_messages` renders the whole conversation, then renders each assistant's
prompt prefix and completed prefix. Only `[prompt_end, assistant_end)` receives
token labels. Earlier system/user content stays at `-100`. Both prefixes must
match the corresponding full-conversation tokens; non-prefix templates are
rejected. Overlength rows are rejected without truncation.

The CPU tokenizer double gives role headers, content characters, and end tokens
distinct IDs. On all 2,368 calibration dialogues, exact expected masks matched:
assistant content and its end marker are supervised, while role headers and
system/user content are excluded. Tests also rejected overlength, no-assistant,
and deliberately rewritten-prefix examples.

The KL mask uses `labels[:, 1:]`. A supervised label at token position `t` is paired
with logit position `t-1`, consistent with causal next-token prediction. The row
and coordinate indexing in the teacher and student follows the same selected
examples. Static inspection confirms a maximum of 128 sampled assistant
positions for the FP32 KL calculation.

The collator contracts confirm padding to a multiple of eight, zero attention on
padding, `-100` padding labels, and a boolean retention mask. Using EOS as the pad
ID does not supervise pad positions because their labels remain ignored.

These contracts validate the algorithm with explicit doubles, not the real LFM
tokenizer or PyTorch kernels. The live manifest supplies complementary evidence:
**zero training and zero validation records were rejected** by actual tokenization
and prefix/length checks. No real tokenizer output was independently inspected in
this audit.

## Split isolation and intended weighting

The mixer checks all input messages for within-split duplicates and for exact
cross-split dialogue collisions **before** intentional training repetition. It
also requires disjoint calibration family IDs. Its validation output is not
repeated. The independent calibration generator separately checks disjoint
families/entities/wording and full user-context overlap; see `CALIBRATION_DATA.md`.

The bounded mixer contract used all real calibration rows plus a tiny synthetic
base. It verified:

- Exactly four copies of every calibration training dialogue.
- Repeat indices exactly `{0, 1, 2, 3}` for each dialogue.
- Preserved false retention flags on every copy.
- Unique validation dialogues and no cross-split full-message collisions.
- Correct category-balanced selection of the requested procedural fixture count.

The live mixture totals reconcile exactly:

| Component | Unique training rows | Rows per epoch | Share of epoch rows |
| --- | ---: | ---: | ---: |
| Public retention | 14,420 | 14,420 | 53.99% |
| Prior procedural | 4,096 | 4,096 | 15.34% |
| New calibration | 2,048 | 8,192 | 30.67% |
| Total | **20,564** | **26,708** | **100%** |

Prior procedural selection is 512 rows from each of eight categories. Validation
has **1,780 rows**, including the **320 unrepeated calibration rows**. Three full
scheduled epochs give each calibration row 12 scheduled presentations, compared
with three for each retained public/prior-procedural row; best-checkpoint
selection may select an earlier point. Row shares are not token-loss shares.

**Isolation limit:** the mixer's fingerprint includes assistant answers and
system text. By itself, it does not detect the same user context paired with a
different answer or system prompt. Existing source-level prompt deduplication and
the calibration generator's checks complement this safeguard. This audit did not
read all actual base/mixed rows, so it does not independently certify user-context
collision absence across those two source collections. No such collision was
observed in the bounded fixtures or calibration data.

## Live configuration and memory reporting

| Setting | Verified live manifest value |
| --- | --- |
| Foundation/tokenizer ID | `LiquidAI/LFM2.5-350M` |
| Pinned tokenizer revision | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |
| LoRA rank | 32 |
| LoRA alpha | 64, from `2 * rank` in the inspected trainer |
| LoRA dropout | 0.05 |
| Targets | `q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3` |
| Epochs | 3 |
| Batch / gradient accumulation | 32 / 2 |
| Effective batch on the configured one GPU | 64 examples, except final partial batch |
| Learning rate | `2e-5` |
| KL weight / maximum teacher examples | 2.0 / 4 |
| Maximum sequence length | 512 |
| Precision / optimizer | BF16 with TF32 allowed / fused AdamW |
| Validation and save interval | 100 steps; final-step save/evaluate callback |
| Training / supervised tokens | 5,683,116 / 2,394,319 |
| Retention rows | 14,420 |

Mocked runner dispatch verified that rank, epochs, batch, accumulation, learning
rate, KL, and all targets reach both smoke and full training. Smoke has a two-step
cap and separate output. Full training starts again from the configured foundation
and has no step cap, so smoke adaptation is not silently reused. The runner sets
`CUDA_VISIBLE_DEVICES=0` and `PYTORCH_ALLOC_CONF=expandable_segments:True`.

The trainer resets CUDA peak allocation statistics after baseline adaptation
validation, measures peak allocated bytes after `trainer.train()`, and records
`peak_training_gpu_allocated_bytes`. This interval includes training and its
scheduled validation/checkpoint selection. It excludes baseline validation and
the later final validation/merge/export. It measures **PyTorch allocated tensor
memory**, not total device memory, reserved allocator capacity, or other
processes. This audit did not read a completed metric or establish an actual peak.

Selection uses lowest adaptation `eval_loss`, which includes the configured
retention KL term where applicable. Neither the contracts nor that selection
metric establish broad semantic quality or final Windows XP hardware behavior.
