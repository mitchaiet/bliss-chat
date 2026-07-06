# Bliss Chat XP v1.3.0 — memory

One self-contained portable `.exe` for Windows XP (Pentium 4 / Pentium M,
512 MB RAM). Copy to the XP machine, double-click, chat. No installer, no
separate model files, fully offline.

## Headline: Bliss now remembers

- **Persistent notes.** `/remember <fact>` stores a short note that survives
  restarts (`%APPDATA%\bliss-chat\MEMORY.TXT`); `/memories` lists, `/forget <n>`
  removes. Notes are injected into the model's system prefix as a `Notes:` line
  the new model was trained to read selectively.
- **New model: `bliss-d12-mem-c20-v2`.** Same d12 architecture, size, and
  Pentium 4 speed — retrained on a curated mixture that teaches multi-turn
  recall (names, pronouns, running lists, corrections), selective Notes recall,
  `Context:`-grounded answering for the Knowledge folder, and small-step
  reasoning.
- **Restored chats really resume.** Selecting a saved chat replays its recent
  turns into the model context (`/replay`), so follow-up questions work — the
  old behavior only re-rendered the text on screen.
- **Smarter context rollover.** The bounded thread summary now keeps both the
  questions and the answers.

## Also new

- **Real top-p (nucleus) sampling** — `/topp` and the presets now actually
  filter; previously temperature-only.
- **~72 MB less RAM.** The engine no longer keeps a second full copy of the
  KV cache for turn restarts (append-only insight); a d12 session now fits
  comfortably beside XP in 512 MB.
- **Knowledge folder retrieval v2** (GUI): term-overlap scoring across up to
  64 KB per file, match-centered snippets, and a single-line
  `Context: <snippets> <question>` prompt that matches the trained format;
  sources are listed under the answer.
- Memory UI in the Tools menu (View Memories, Forget, Open Knowledge Folder)
  and a per-message **Remember** button.

## Benchmarks (temp 0.3 bench / greedy evals, seed 42, native engine, same binary both sides)

| Metric | v1.2.3 (c20-v1) | v1.3.0 (mem-c20-v2a) |
|---|---|---|
| Correct (100-question bench) | 57% | **77%** |
| Coherent | 99% | **100%** |
| — math | 6.7% | **53.3%** |
| — factual | 60% | **95%** |
| — time | 20% | **100%** |
| Multi-turn memory (40 dialogs) | 27/40 | **37/40** |
| Notes selective recall (25 cases) | 21/25 | **22/25** |
| Persona probes | 5/8 | **8/8** |

Model: continued pretraining of `bliss-d12-curated-c20-v1` (+4,000 steps at
tail learning rate, curated-mix p=0.35 over the v2 mixture: 8,384 distinct
conversations across multi-turn recall, Notes recall, Context grounding,
reasoning, behavior, identity — plus the proven v1 doc set). Final validation
bpb 0.817234 (v1: 0.818168) — general language modeling did not regress.

## Files

- `bliss-chat-xp-v1.3.0-memory-portable.exe` — SHA-256: `TBD`

## Compatibility

Windows XP SP2+ · SSE2 required (SSE3 backend auto-selected when present) ·
512 MB RAM · ~300 MB free disk for extraction.
