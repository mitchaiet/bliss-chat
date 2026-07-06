# Bliss Chat upgrade roadmap

Goal: evolve Bliss Chat from an offline XP chatbot demo into a useful local XP assistant while preserving the current constraints: Windows XP, portable release, no cloud dependency, no modern runtime requirement, and explicit user control for automation.

## Phase 1 — Local assistant substrate

1. Chat history memory — done (shipped v1.3.0)
   - Existing native GUI already persists chats under `%APPDATA%\\bliss-chat\\chats`.
   - Next: add explicit `Remember this` notes and a small `memory.txt`/`memory.jsonl` store. — done (shipped v1.3.0: `/remember` notes, `memory.txt` via `-m`)
   - Retrieval: prepend 1-3 relevant memory snippets to the prompt. — done (shipped v1.3.0: notes render as a `Notes:` line in the system prefix)

2. RAG / Knowledge folder — done (shipped in C at v1.3.0 quality)
   - Drop `.txt`, `.md`, `.html` files into a local `knowledge/` folder.
   - Search lexical snippets before generation.
   - Prepend compact context to the prompt so the tiny model can answer from local facts.
   - First implementation lives in `server/bliss_xp_web_chat.py`; port the same logic to `XPCHAT.EXE` after behavior is proven. — done (ported to `XPCHAT.EXE`, shipped v1.3.0)

3. Tool calling
   - Safe direct tools first: calculator, date/time, file/folder listing, text search.
   - Automation tools later: open file, launch program, shell command.
   - Any write/launch/shell tool must show a preview and require explicit approval.

## Phase 2 — Runtime quality and speed

4. Sampling/runtime controls
   - Wire real top-p in `nc_run.c`. — done (shipped v1.3.0)
   - Add better anti-repeat and answer-length controls.
   - Keep `Coherent mode` as the default.

5. Quantization upgrades
   - Baseline: current per-row int8 `.NCB`.
   - Next: int8 matmul SIMD cleanup and benchmark.
   - Then: q4/groupwise weight-only experiment, likely as `MODEL_Q4.NCB` or `MODEL.TQB`.
   - Ship only if quality survives the fixed benchmark suite.

6. Speculative decoding
   - Use d6 draft model, d12 verifier.
   - Potential 2-4x speedup without retraining.

## Phase 3 — Native XP integration

7. Port web-proven RAG/tools into `XPCHAT.EXE`.
8. Add Knowledge folder UI: Open Folder, Re-index, Sources shown under answer. — partial (shipped v1.3.0: Open Folder + Sources footer under answers; Re-index still open)
9. Add Memory UI: Remember, Forget, View Memories. — done (shipped v1.3.0)
10. Add tool approval UI and audit log.

## Phase 4 — Model/data improvement

11. Improve distillation dataset with short XP-friendly answers.
12. Revisit LoRA/adapters instead of full SFT.
13. Tighten benchmark scoring before claiming quality gains.

## Current first step implemented

- Web harness now has a safe calculator tool.
- Web harness now has a local `knowledge/` RAG folder path.
- These are intentionally Python-side first so we can test behavior quickly before porting to Win32 C.
