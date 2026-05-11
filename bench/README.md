# bliss-chat coherence benchmark

100 questions × 9 categories. Each model under test gets the same questions
through `nc_run_native` with `/reset` between turns, temperature 0.3, seed 42
for reproducibility.

## How to run

```bash
python3 bench/run_bench.py     # generates bench/results.csv
python3 bench/score.py         # prints summary, writes bench/scored.csv
```

## Rubrics (all 0/1)

| Rubric | Pass condition |
|---|---|
| `coherent` | length 2..600 chars, no bigram repeated 3+ times |
| `format`   | non-empty, no leaked `\nQ:` stop marker, has alphanumerics |
| `correct`  | at least one expected_keyword appears (case-insensitive) |

Note: `correct` is **keyword presence only** — it does not verify the *direction*
of the answer. "Fire is colder than ice" scores `correct=1` because the word
"ice" appears. A stricter rubric would need an LLM-as-judge or per-question
regex; both are deferred.

## Baseline — d12 base model, temp 0.3, seed 42 (2026-05-10)

| Category   | n  | Coherent | Format | Correct |
|------------|----|----------|--------|---------|
| factual    | 20 | 95%      | 100%   | 60%     |
| math       | 15 | 100%     | 100%   | **13%** |
| definition | 15 | 53%      | 100%   | 67%     |
| list       | 10 | 80%      | 100%   | 90%     |
| yesno      | 10 | 70%      | 100%   | 30%     |
| social     | 5  | 40%      | 100%   | 40%     |
| comparison | 10 | 90%      | 100%   | 90%     |
| time       | 5  | 100%     | 100%   | 20%     |
| causal     | 10 | 40%      | 100%   | 80%     |
| **TOTAL**  | 100| **77%**  | 100%   | **56%** |

## Failure modes observed

### Definition mode collapse
> Q: What is a cat?
> A: A cat is a cat. A cat is a cat. A cat is a cat.

Classic base-LM autoregressive loop. The few-shot prefix doesn't dampen it
because the next-token distribution after "A cat is a" collapses onto " cat".

### Math: template right, answer ignorant
> Q: What is 2 plus 2?
> A: 2 plus 2 equals 2.

> Q: What is 100 divided by 4?
> A: 100 divided by 4 equals 100.

The math few-shot taught it the shape (`X plus Y equals Z`), and it
faithfully fills Z with the most likely next token, which is the
operand the model saw most recently. Arithmetic is out of scope for a
110M base LM.

### Yes/no: gets the keyword, contradicts itself
> Q: Is fire hot?
> A: No, fire is not hot. Fire is a chemical reaction...

> Q: Can dogs fly?
> A: Yes, dogs can fly. They can fly at speeds of 20 miles per hour.

Yes/no questions have no template in the few-shot prefix, so the model
defaults to web-text-ish ramble. The `correct=30%` here is partly an
artifact of the lenient rubric.

### Comparison: surprisingly strong
> Q: Which is taller a giraffe or a dog?
> A: A giraffe is taller than a dog.

> Q: Which is sweeter sugar or salt?
> A: Sugar is sweeter than salt.

90% correct. The few-shot prefix doesn't even have a comparison example,
yet the model handles them well. World-knowledge from pretrain dominates.

## What this baseline tells us

* The model knows facts (60% on capitals/famous-people/etc.).
* The model can follow templates (math format adherence is 100%).
* The model **cannot** do arithmetic, definitions of common nouns
  ("water is water"), or yes/no with grounding.
* Mode collapse on definitions is the single biggest correctable issue —
  if we can make those answers shorter and stop them mid-sentence we'd
  bump coherence ~15 points without retraining.

## Next baselines to take

When we improve anything (new model, new prompt, working SFT), re-run
and replace the table above. Track:

* Per-category correct% delta
* Total coherent% (the most user-facing number)
* Average tokens per response (lower = less ramble usually)
