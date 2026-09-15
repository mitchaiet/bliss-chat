# Bliss final acceptance set, v2

This 48-case set was written independently for final acceptance on 2026-09-15.
Only the existing `evaluate.py` schema/scorer was consulted; prior held-out
questions and model predictions were not opened. No model was run. The set is
outside the training repository and must remain excluded from training inputs.

There are eight cases each for conversational semantics, arithmetic/logical
relations, instruction following, conversation corrections/memory, saved-note
grounding, and unknown information. The cases use practical hand-written
scenarios, not instances of the procedural training generator's templates.
Every case includes a reference and semantic rubric. Some exact-output gates
also deliberately test formatting. Open phrasing uses content gates plus a
rubric; matching a keyword does not establish semantic correctness.

Use the complete suite after selecting the candidate with the earlier selection
protocol. Do not use failures to adapt a model and then continue to describe this
same set as untouched final acceptance. Report results for all 48 cases and
review outputs for contradiction, fabrication, and rubric compliance.

Validation at freeze: all 48 reference answers passed the existing scorer;
96 deliberately incorrect answers and 48 empty answers were rejected. There
are at most two user turns per case, at most 43 input words including saved
notes, and at most nine reference-answer words. These size checks target a
512-token context and 128-token generation limit; they are word counts, not a
tokenizer or model execution check. The first turn in conversation-memory
cases requests a brief acknowledgment.

The cases were hand-authored without existing evaluation questions or output
feedback. This does not establish that equivalent facts/tasks were absent from
the upstream model's pretraining or every public instruction source. It is a
small acceptance sample, not evidence of broad reasoning competence or actual
Windows XP performance.

From this directory, reproduce schema/reference validation and verify the
frozen bytes:

```sh
python3 evaluate.py --cases heldout_v2_final.jsonl --validate
shasum -a 256 -c heldout_v2_final.sha256
```

`heldout_v2_final_validation.json` records the validation counts and frozen
hash. The JSONL is the authoritative suite; no inference or training belongs
in this validation step.
