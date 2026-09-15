# Independent conversational calibration supplement

`tools/prepare_calibration_data.py` authors a small SFT supplement for Bliss. It
contains **2,048 training dialogues and 320 adaptation-validation dialogues**.
It does not run a model, use the GPU, or establish an improvement in model quality.

The final generated artifact directory is `../calibration/v1/` relative to this
repository. Files directly inside `../calibration/` are an earlier grammar-review
draft; use the `v1` subdirectory.

## Scope and provenance

These templates were freshly authored for this task without reading evaluation
files, acceptance suites, or model predictions. All personal, fictional, and
operational facts are supplied inside their dialogue. Arithmetic answers are
derived from the stated operands. Exact transformations operate on supplied text.
No factual knowledge is scraped or invented to fill missing information.

The generator reads only itself and two existing source files:

- `tools/prepare_coherence_data.py`: reads the literal `SYSTEM` assignment via AST,
  preserving the deployment instruction without importing or executing the file.
- `tools/train_coherence.py`: records its hash as training-interface provenance.

It reads no corpus, tokenizer, benchmark, evaluation result, or network resource.
The manifest records source SHA-256 hashes, seed, generation method, entity pools,
families, counts, checks, and output hashes. Seed: **2026091507**.

## Recipe

Each category contains **128 train + 20 validation** records. Training has two
scenario families per category, 64 examples per family. Validation has one
different scenario family per category, 20 examples per family. Each family uses
four explicitly authored wording forms. Total: **32 train / 16 validation families**
and **128 train / 64 validation wording forms**.

| Category | Training families | Held-out validation family |
| --- | --- | --- |
| Indirect requests | Polite rewrites; simplify verbose notes | Draft a message from an intent |
| Practical choices | Fit an available time; apply pickup rules | Choose a room satisfying two constraints |
| Conversational reference | Resolve plural references; follow same-location relations | Answer an elliptical follow-up about another person |
| Conversational repair | Replace a destination; correct which object was requested | Replace a preference with the latest choice |
| Personal information | Favorite snack; weekend activity | Event music choice |
| Fictional information | Story captain; title of a story's book | Courier destination in a made-up tale |
| Operational information | Parcel owner; supplied fictional cabinet code | Milestone date |
| Concise summaries | Combine two plans; report latest status | Summarize an event and its stated reason |
| Extract | Select note fields; retain matching rows | Filter marked lines and strip labels |
| Reorder | Reverse values; order by numerical rank | Order stops by 24-hour time |
| Exact copy | Preserve punctuation/case; preserve repeated spaces | Preserve a marked two-line payload |
| JSON | Person/room strings; item/integer count | Task string and readiness boolean |
| Requested-only outputs | Return a supplied letter code; exact-list membership | LEFT/RIGHT/TIE comparison |
| Logical relations | Transitive order; membership implications | Apply a one-way condition without reversing it |
| Two-step addition/subtraction | Inventory movements; active time after a break | Voucher balance |
| Two-step grouping | Multiply and add loose items; divide then add groups | Multiply tray contents and subtract removals |

Half the records cover ordinary conversation and grounded information; 31.25%
cover formatting, 12.5% cover two-step arithmetic, and 6.25% cover logical relations.
Responses are intentionally brief: the final recipe averages 5.70 assistant words
in training and 6.38 in validation. Conversational references and repairs have two
assistant turns; totals are 2,304 training and 360 validation assistant turns.

### Paired answerability

Personal, fictional, and operational categories use matched pairs. One example
contains the requested fact; its partner omits only that fact. Both retain the
same base context and question, and both stay in the same split. Missing answers
state the specific information absent, without adding a guessed value.

- Training: **192 answerable + 192 missing-information examples**, or 192 pairs.
- Validation: **30 answerable + 30 missing-information examples**, or 30 pairs.
- Logical membership/condition tasks also include unsupported conclusions.

The answerable counterparts prevent a blanket missing-information response from
being the correct training behavior. These pairs are intentionally similar
*within* a split; the purpose is to vary evidence while holding the question fixed.

### Split guarantees and limits

Train and validation have disjoint scenario-family IDs, wording forms, person
names, place names, item pools, color pools, snack pools, and activity pools.
Copy payloads use separate code prefixes. A whole matched pair is the split unit.
Shared common language, logical concepts, and arithmetic operands are expected;
numbers are not claimed to be disjoint.

The generator rejects duplicate dialogues, exact cross-split user-context
prompts, and cross-split prompt trigram Jaccard overlap at or above 0.65. The
observed maximum is **0.075**. This is a lexical overlap check, not a proof of
semantic independence from every possible dataset. No evaluation-set overlap
comparison is performed because evaluation content remains unread.

## Generation and checks

From the repository root:

```sh
python3 tools/prepare_calibration_data.py
```

The default writes `../calibration/v1/`. Existing nonempty output directories are
refused. For an independent regeneration, choose a new output directory:

```sh
python3 tools/prepare_calibration_data.py --output ../calibration/reproduced
```

Outputs:

- `train.jsonl` and `validation.jsonl`: standard `messages` records plus provenance,
  category, family, wording, pair, and verification metadata.
- `EXAMPLES.md`: a readable example from every family and both members of selected
  answerability pairs; exact-copy text retains its spaces and line breaks.
- `manifest.json`: provenance, recipe, split statistics, and audit results.
- `SHA256SUMS`: hashes of both data files, examples, and manifest.

All **2,368 record-level rule checks** pass, including arithmetic recomputation,
JSON parsing and value types, extraction, copy equality, ordering, restricted
outputs, correction anchors, logical closure/implication, and pair completeness.
27 generated collisions were deterministically resampled before writing. Basic
role/content validation and cross-split audits also pass. Authored example review
corrected singular/plural and a few phrasing issues before final generation.

An independent regeneration reproduced `train.jsonl`, `validation.jsonl`, and
`EXAMPLES.md` byte for byte. Both manifests differed only in the provenance hash
of `tools/train_coherence.py`, which was updated concurrently for integration;
the generator, SYSTEM input, recipe, checks, and data hashes were identical.
Every digest in the final `v1/SHA256SUMS` was independently verified.

Rule checks verify structured aspects and expected anchors; they do not constitute
a comprehensive semantic proof of every natural-language example. The artifact
is concise procedural calibration data, not a replacement for a broad and varied
conversation corpus.

## Training integration

The existing assistant-only training interface reads `messages`; extra metadata
is ignored. Every new record carries **`retention: false`** and the distinct
`source: bliss-calibration-authored-v1`. A loader must honor the explicit retention
flag. The legacy rule that recognizes only `bliss-procedural-v1` as procedural
would otherwise mistakenly apply retention KL to these new examples.

Use this as a bounded supplement to broader conversational retention data. The
generator does not choose mixture weights, duplicate rows for weighting, train,
or decide checkpoint selection. If a recipe repeats or weights these examples,
record the effective proportions in that training run's manifest.

No tokenizer was consumed. The longest complete dialogue has 495 characters in
training and 508 in validation, including the system message, but these are not
token counts. The selected model's normal tokenization/length and assistant-mask
checks must run at integration time. Adaptation validation can guide fitting; it
is not an untouched capability benchmark or evidence of Pentium M/Windows XP
memory fit, speed, or end-to-end usability.
