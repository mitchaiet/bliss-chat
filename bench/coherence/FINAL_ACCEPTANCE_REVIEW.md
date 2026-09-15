# Bliss final acceptance review — 15 September 2026

## Conclusion

**The frozen LFM2.5-350M Q6 candidate substantially improves conversational understanding, short conversation recall, saved-note answers, and handling missing information over the original NCB backend. The full objective of improving all capabilities and response speed on the XP machine is not established.** Arithmetic and logic remain unreliable, several previously correct answers regress, and the candidate is 6.61 times slower in the serial Linux response test. Its model file is only 1.59% smaller. Actual XP performance remains unverified.

The candidate was selected before this fresh 48-case suite was run. This report accepts the evidence for bounded language and recall gains; it does **not** certify broad reasoning competence, universal improvement, or a faster XP release. Selection, weights, runtime, cases, and mechanical checks remain unchanged after final acceptance results.

## Overall results

| Measure | Original NCB | Selected LFM Q6 |
|---|---:|---:|
| Frozen mechanical checks | 3/48 | 24/48 |
| Fully correct final-answer content, semantic review | 7/48 | 26/48 |
| Relevance, mean /2 | 1.146 | 1.854 |
| Correctness/grounding, mean /2 | 0.458 | 1.167 |
| Coherence, mean /2 | 1.583 | 1.979 |
| Instruction following, mean /2 | 0.271 | 1.562 |
| Combined rubric, mean /8 | 3.458 | 6.562 |
| Protocol errors / empty user-turn answers | 0 / 0 | 0 / 0 |

Across paired combined scores, the selected candidate wins **37 cases**, loses **4**, and ties **7**. These are subjective ordinal rubric totals, not intelligence percentages. Clear, concise wrong answers can receive high coherence scores; correctness must be considered separately.

“Fully correct” counts final replies with correctness/grounding = 2. It does not require perfect formatting or compliance in every setup acknowledgment. All turns were visible to review, and coherence considers the whole exchange; content correctness and requested final-answer formatting chiefly concern the final reply. The selected model sometimes elaborates or invents dialogue when an initial acknowledgment should be only `OK`.

## Capability breakdown

Every category has eight cases. Semantic counts below mean fully correct final-answer content.

| Category | Mechanical, original → selected | Semantic, original → selected | Assessment |
|---|---:|---:|---|
| Conversational semantics | 1 → 5 | 2 → 6 | Substantial gain; negation and conditional choices still fail |
| Arithmetic and logic | 0 → 1 | 1 → 1 | No demonstrated correctness gain |
| Exact instruction following | 0 → 2 | 0 → 2 | Some gain; six of eight remain incompletely or incorrectly executed |
| Conversation memory | 0 → 4 | 0 → 5 | Substantial short-context gain; updates still unreliable |
| Saved-note grounding | 2 → 6 | 3 → 6 | Substantial gain; extra unsupported requirements still occur |
| Unknown information | 0 → 6 | 1 → 6 | Substantial gain; unsupported inference still occurs |

Arithmetic/logic correctness averages **0.375 → 0.250 /2**. Its combined score rises because the new answers are more direct and better formatted, despite being wrong. This is a material limitation, not evidence that reasoning improved.

The suite tests supplied facts and short exchanges. It does not establish comprehensive world knowledge, long-conversation retention, persistent memory reliability across application restarts, or nuanced open-ended dialogue over many turns.

## Representative gains and failures

These examples come from the complete preserved predictions; none were used for further adaptation or selection.

| Case | Expected behavior | Original answer | Selected answer |
|---|---|---|---|
| `conversation_semantics_03` | Understand an offer to carry a suitcase | “Jules would not carry the suitcase.” | “Jules is offering to carry the suitcase.” |
| `conversation_memory_07` | Keep parcel `BOX-73-K` despite a shelf distractor `BOX-37-K` | “Your parcel reference is BOX-37-K.” | “BOX-73-K” |
| `unknowns_05` | Conflicting door reports leave the location unresolved | “One can say the parcel was left at the front door.” | “UNRESOLVED” |
| `conversation_semantics_04` | Email stopped before Send was not sent | “No, she stopped before clicking Send.” | **“YES”** |
| `arithmetic_logic_01` | 14:35 plus 50 minutes = 15:25 | “It ends at 2:15, according to the workshop's schedule.” | **“00:00”** |
| `arithmetic_logic_06` | Receipt and photo ID are both required; ID is absent | “No, collection is not allowed.” | **“YES”** |
| `conversation_memory_02` | Swapping soup lunch and pasta dinner makes lunch pasta | “You have Swap the two meals, so what is planned for lunch?” | **“Soup.”** |
| `note_grounding_04` | Bring pencil and water; sheet music is provided | “bring a pencil and a bottle of water.” | **“Bring a pencil, a bottle of water, and the sheet music.”** |

All IDs have the prefix `v2final_` in the files. Other selected failures include 31 passengers in six-seat vans answered as 11 vans, inferring copper from roundness, reversing words incorrectly, unsorted filenames, and inferring current helmet use from bicycle ownership.

Mechanical gates are intentionally frozen and imperfect. The correct music-volume paraphrase misses a regex, and a correct updated list fails exact formatting. Conversely, the choir reply passes the mechanical gate while adding an unsupported required item. The semantic review records these differences without rewriting checks.

## Linux response time, memory, and size

Both backends ran serially on the same modern Linux workstation, with the parent confirming other owned CPU-heavy jobs had ended. Each case starts a fresh backend; multiple turns within a case reuse that process. There are 56 user turns per backend. Response time excludes process startup, which is reported separately.

| Measure | Original NCB | Selected LFM Q6 |
|---|---:|---:|
| Median user-turn completion | 0.692 s | 4.575 s |
| 95th percentile completion | 1.035 s | 7.280 s |
| Median first visible pipe output | 0.568 s | 3.667 s |
| Median process startup | 0.507 s | 4.959 s |
| Total reported generated tokens | 571 | 332 |
| Largest generated turn | 19 tokens | 34 tokens |
| Peak backend RSS | 291.02 MiB | 286.89 MiB |
| Model file | 292,886,664 bytes | 288,226,560 bytes |
| Tokenizer file | 477,616 bytes | 1,692,655 bytes |

The candidate takes **6.61× longer** per median response despite producing fewer tokens. The model alone shrinks **1.59%**; model plus tokenizer shrinks **1.17%**; observed peak process RSS shrinks **1.42%**. These are modest footprint savings. No answer reaches the 128-token limit.

This is an end-to-end native **Linux** comparison, including prefill and different runtime implementations. First output measures visible pipe bytes; token counts use each backend's tokenizer. These are not isolated decode throughput measurements and cannot be projected quantitatively onto a Pentium M.

The documented target is a **Dell Inspiron 8600, Pentium M 1.40 GHz, 512 MB DDR333 RAM, GeForce FX Go5200 64 MB, Windows XP**. Physical XP hardware was unavailable for this evaluation. Separate parent/native-agent Wine, memory-limit, IPC, and quantized-forward checks support compatibility and implementation evidence only; they do not establish hardware response time, GUI usability, total OS memory headroom, or real-device stability.

## Frozen protocol and review audit

- Independent final-suite author consulted the evaluator schema/scorer, without reading development questions or predictions. The suite contains 48 hand-written cases, eight per category, with at most two user turns each. The final suite remained excluded from training and candidate selection.
- Freeze validation accepted all 48 references and rejected 96 deliberately incorrect answers plus 48 empty answers. Equivalent upstream pretraining material cannot be ruled out.
- Selection was frozen at `2026-09-15T08:37:28.104813+00:00`. The original run executed `08:41:03–08:42:12 UTC`; selected ran next, `08:42:12–08:50:27 UTC`.
- Both used logical context 512, temperature 0, seed 42, and maximum generation 128. The selected direct ELF automatically uses FP32 activations for Q6; `runtime_args` is empty. No wrapper obscures its binary hash.
- Original deployment behavior is preserved: its Q/A template, punctuation/newline stopping, guard, and repetition penalty 1.12 differ from the candidate's ChatML and greedy generation. Note integration also belongs to the deployment stack. Therefore these results compare complete backends, not the isolated causal effect of training or quantization. Original NCB retains its historical larger KV allocation despite logical context 512.
- Model identity was randomly masked separately in every case. The review packet included all answers, prompts, notes, references, and rubrics, while excluding names, mechanical scores, and timing. All 48 A/B scores and explanations were written and validated **before** the mapping was opened.
- The same AI reviewer applied relevance, correctness/grounding, coherence, and instruction-following scores from 0 to 2. The reviewer had prior familiarity with model styles, so masking does not guarantee source recognition was impossible. There was no independent human rater.
- No scores, weights, or frozen checks were revised after identity disclosure. The final suite is now observed evaluation material and must not be reused as an untouched acceptance set after future tuning.

## Artifact identity and reproduction

| Artifact | SHA256 |
|---|---|
| Final suite | `9ef8940b4cacbcecd5d679e0eba90bfb5548eca488b7a37ae8fdcd63a9b1ca41` |
| Frozen selection file | `01d812d06245cb64a143d21764d092066e7ac4edaa7fa4a2a8b871ec743ec814` |
| Evaluation runner | `748b168cdbd1b52aad89b90ae9b68c928ee08247c8529fa7f811bbcd99373d31` |
| Original NCB model | `11c82d07097f057dbb458a02e4bed8daf748f359b73dfd3abb4dd1df9dc679cc` |
| Original native ELF | `f714ec5524e17cc8f8b7ab6a485e356cdcbdc47a4baf9455ac2f10cb1caca096` |
| Selected Q6 model | `ac4fe748bfa8f61af3b55f13e5435c299ca4c5abad71cd24c233f6ff24a33953` |
| Selected tokenizer | `2e9a8349d5242ff0070c060ac9e5d15a05739727f1cd4941d631b23bb881bb2d` |
| Selected actual native ELF | `128699c1e02a1f78c60b24af6437a6b58c91ae97b5c860300e41323ae338bf8e` |
| Masked scores before disclosure | `416a5899cb225fd87aeb0d28e8e4f4e5f4b4334e7aed77679335f01f326fd2f7` |

Selected source: `LiquidAI/LFM2.5-350M` revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`, first anchored training run checkpoint 413, Q6 group 64. The manifest records exact remote commands, timestamps, load averages, runner hash, and selection hash. Reproduction must use new output directories; preserve these original results.

All following artifacts are in this report's directory:

- `final48_run_manifest.json`: serial commands and provenance.
- `final48_original_ncb/` and `final48_selected_q6/`: all predictions, controls, stderr logs, hashes, timings, and mechanical summaries.
- `final48_original_ncb_qualitative_scores.json` and `final48_selected_q6_qualitative_scores.json`: every semantic grade and explanation.
- `final48_qualitative_summary.json` and `final48_performance_comparison.json`: aggregate results.
- `final48_masked_review.jsonl`, `final48_masked_scores.jsonl`, `final48_masked_mapping.json`, and `final48_unmask_audit.json`: identity masking and disclosure audit.
- `heldout_v2_final.jsonl`, its README, SHA file, and validation JSON: unchanged final cases and reference validation.
- `final48_acceptance_summary.json`: machine-readable conclusion, limitations, and artifact index.

The development-suite candidate history is preserved separately in `LFM_Q6_SELECTION.md`. It was used before this final test; final48 outcomes did not trigger a different candidate choice.
