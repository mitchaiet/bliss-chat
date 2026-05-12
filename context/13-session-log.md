# Session log — 2026-05-10

Chronological account of the long working session that brought
Bliss Chat from a working-but-rough demo to the v1.0.0 release.
Captures decisions, dead ends, and time stamps so any successor
can understand WHY the code looks how it looks.

Times are local. Commit SHAs are short.

---

## Phase 1 — Performance instrumentation (~14:00)

State at session start: d12 model running on the Pentium 4, ~0.9
tok/s, scalar matmul. User wanted to know where the time was going.

### Iter 1 — Profiler (`edddbec`)

Added `QueryPerformanceCounter`-based profiler to `nc_run.c`. Wraps
`forward_one`, `linear`, `rmsnorm`, `rope`, `softmax`, `ve_lookup`
with tick counters. Dumps `[prof turn]` lines on stderr per turn.

**XP P4 profile, d12 int8, 30 generated tokens:**
- 36 forward calls, 40.99 s total, **1.14 s/token**
- `linear`: **96.7 %**
- `softmax`: 1.5 %
- everything else: ~1.7 %

Conclusion: optimize matmul or nothing matters.

### Iter 2 — SSE2 matmul (`09aa226`)

Added SSE2 paths to `matmul_fp32` and `matmul_int8`. Both gated on
`__SSE2__` so macOS arm64 builds keep the scalar fallback.

- `matmul_fp32`: 8-lane fp32 mul/add with two parallel accumulators,
  horizontal sum at the end.
- `matmul_int8`: load 8 signed bytes via `_mm_loadl_epi64`, sign-extend
  via the cmpgt-zero trick (no SSSE3/SSE4.1 on P4), then int32 → fp32,
  multiply by x lanes, accumulate.

**XP P4 numbers:**
- `linear` avg/call: 15075 µs → 2765 µs (**5.45×**)
- forward avg: 1138.6 ms → 234.8 ms (**4.85×**)
- end-to-end: **0.88 → 4.26 tok/s**

Output bit-identical to scalar on greedy `"What is 2 plus 2?"`.

### Iter 6 — SSE2 attention helpers (`9cc9ae0`)

Extracted attention QK dot product and V weighted-sum into helper
functions with SSE2 fast paths. `expf` stays scalar (transcendental,
polynomial approximation not worth the precision risk).

**XP P4 numbers:**
- softmax block: 20.6 → 7.8 ms/forward (**2.6×** within the block)
- forward avg: 234.8 → 213.6 ms (+9 % on top of iter 2)
- cumulative end-to-end: **1138.6 → 213.6 ms = 5.33×, 4.68 tok/s**

Documented full breakdown in `context/11-profiling.md`.

---

## Phase 2 — Multi-turn KV cache (~15:00)

### Iter 3 — Multi-turn (`a51003e`)

Dropped the per-turn `state_restore_prefix()`. The KV cache now
persists across turns so the model "remembers" earlier exchanges.

Per-turn template:
- turn 0: `"<user>\nA:"` (prefix ends in `"Q: "`)
- turn N: `"\n\nQ: <user>\nA:"` (continue Q&A pattern)

Reset paths:
- `/reset` command — explicit reset, emits INFO + EOT 0
- `seq_pos + 64 >= ctx_max` — auto-reset before next turn would
  overflow, emits INFO

Default `ctx_max` changed from 256 to `model.sequence_len` (1024
for d12) so multi-turn has actual room.

Verified on XP: turn 1 "capital of France" → "Paris"; turn 2
"capital of Germany" → "Berlin". KV preserved.

### Iter 4 — Math few-shot + smarter stop (`08b229a`)

Added an arithmetic example to the few-shot prefix so the model
formats numeric answers as `"X plus Y equals Z"` (still gets the
math wrong, but template adherence becomes 100%).

Stop pattern: added `"\nA:"` in addition to `"\nQ:"`. The model
sometimes finishes a short answer and self-prompts as the assistant
again ("Hello! ...?\nA: I'm sorry, ..."). Catching `\nA:` clips that.

On trailing-flush, scan held bytes for any `\n` + `Q|A` substring
and truncate so `\n\nQ` artifacts don't leak when generation ends
via `assistant_end` before completing the stop pattern.

---

## Phase 3 — Models (~15:30)

### Iter 5 — Chinchilla d6 (`1414444`)

Trained d6 at ratio 20 (Chinchilla-optimal) vs the original ratio 12.
Wall clock: **9.1 min** on RTX 6000.

- val_bpb: 1.165 → **1.075** (8 % better)
- 30 M params, 75 MB int8, drop-in
- **~27 tok/s on the P4** — 5.85× faster than d12

Shipped alongside d12 as `MODEL_D6.NCB`. User picks based on
speed-vs-coherence trade-off.

### Iter 7 — SFT investigation, attempt 1 (`a3fa77c`)

Hypothesis 3 from `context/10-known-issues.md`: bf16 forward
overflow on long SmolTalk samples. Tested with
`NANOCHAT_DTYPE=float32`.

**Result: loss still NaN'd by step ~88.** fp32 was making the
explosion slower, not preventing it. Hypothesis 3 falsified.

---

## Phase 4 — GUI iterations (~16:00–18:00)

### Iter 8 — Slash commands (`05af659`)

Added `/help`, `/info` to backend. Slash commands route through the
existing `read_line()` loop with `if (!strcmp(line, "/foo")) {...}`
branches.

### Iter 9 — README polish (`9c0d361`)

Public-facing copy. Status table, slash-commands list, model variants,
download links direct to release assets.

### Iter 10 — Working stop button (`0658b98`)

GUI writes `\x01STOP\n` to backend stdin on Stop click. Backend's
per-token loop `PeekNamedPipe`'s stdin between tokens; on match,
consumes bytes, emits `INFO stopped by user`, returns to read-line.

Preserves the model load. Esc keyboard shortcut routes to the same
WM_COMMAND.

### Iter 11–13 — Sanitization + branding (~17:30)

- `ec26b7e` — refresh xpchat header + ABOUT dialog (purge stale
  llama.cpp / GGUF references).
- `365b4ae` — rename `logf` → `dbg_log` (collided with libm's `logf`,
  warning at build time).
- `742e87e` — Bliss Chat branding + PC-spec subtitle:
  - APP_NAME: "XP Tiny LLM" → "Bliss Chat" (kept lowercase for
    filesystem/registry paths to preserve user data).
  - New `assets/bliss_chat.ico` (hand-written multi-res BMP-format
    .ico via PIL; Pillow's default writer uses PNG-encoded entries
    which Windows XP can't read — discovered the hard way).
  - GUI subtitle now shows CPU name + cores + RAM, read from
    `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\
    ProcessorNameString` and `GlobalMemoryStatusEx`.

### Iter 14–18 — Creature comforts (`6cdffc8`)

Five small wins in one commit:

1. **Sticky-bottom auto-scroll** — turned out to be buggy because
   SCROLLINFO lags during fast streaming. Replaced with
   force-scroll-on-every-append (standard chat behavior).
2. **Esc = Stop, Enter = Send, Shift+Enter = newline** — input
   subclassing.
3. **Save Transcript** via `Ctrl+S` + File menu.
4. **Window position memory** — `HKCU\Software\bliss-chat\Window`.
5. **Live tok/s** in status bar — running `gRunChars / 4 /
   elapsed_s`, ticked every 500 ms.

### Iter 19 — Settings + New Chat menu (`c970a98`)

Modal Settings dialog (resource template `IDD_SETTINGS`) with
Temperature + Seed + 3 presets (Greedy/Balanced/Creative). Reset
Defaults. New `/temp <f>` and `/seed <int>` slash commands in
backend. Settings persist to `HKCU\Software\bliss-chat\Settings`.

New Chat menu item (Ctrl+N) — sends `/reset` AND clears the
on-screen transcript. Plain Clear still does the transcript only.

### Iter 20–21 — Chat list sidebar + manifest progress bar (`444cc53`)

Two things in one push:

1. **comctl32 v6 manifest** (`assets/xpchat.manifest`, embedded via
   `RT_MANIFEST` in resource.rc). Opts the EXE into XP's Luna theme.
   Progress bar becomes the iconic green segmented one. Buttons /
   edits / listbox pick up theming.
2. **ChatGPT-style sidebar.** Listbox + "+ New Chat" button. Chats
   stored as `%APPDATA%\bliss-chat\chats\<unix_ts>.txt`. Each turn
   auto-appended. First user message becomes the chat title.
   Clicking a sidebar item sends `/reset` and reloads that chat's
   transcript (KV cache dropped — v1 trade-off).

Window min size grew from 720×480 to 900×520 to fit the sidebar.

---

## Phase 5 — SFT, harder attempts (~19:00)

### Iter 22 — Grad-clip + NaN-skip in chat_sft.py (~20:30)

Patched `nanochat/scripts/chat_sft.py` with:

```python
if not torch.isfinite(loss):
    _bc_skip_step = True
else:
    loss.backward()
...
if not _bc_skip:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**Result: ran 7 steps cleanly (loss 1.32 → 2.33, climbing), then
NaN at step 8.** Cleaner than the bf16 attempt but still terminal.

### Iter 23 — Manual `.mul_` clip (no `clip_grad_norm_` 0×inf trap)

`clip_grad_norm_` silently does `0 * inf = NaN` when any grad is
inf, corrupting weights. Replaced with manual norm compute + mul,
with a finite-check that skips the optimizer step entirely on
non-finite norm.

**Result: identical 7-step run-up then NaN at step 8.** Optimizer
was processing as expected; the bad weights came from somewhere
else.

### Iter 24 — Vanilla AdamW SFT

Replaced nanochat's `MuonAdamW` with plain `torch.optim.AdamW`.
Tested the theory that Muon's Newton-Schultz orthogonalization
amplified the gradient.

**Result: byte-identical divergence trajectory.** 1.32 → 2.31 → NaN
at step 8. The optimizer is NOT the cause.

Logged in `context/10-known-issues.md`. Updated working theory:
the `d12_patched` checkpoint copies `<bos>`'s row into ALL chat-
special slots, so the model can't distinguish them, gradients
update them in lockstep, embedding space collapses.

Documented next-thing-to-try list:
1. Re-init special rows with random orthogonal vectors (give them
   a chance to differentiate).
2. Pretrain a fresh d12 from scratch with chat formatting baked in.
3. LoRA — never touch base weights.
4. Per-position cross-entropy logging at step 1.

---

## Phase 6 — Coherence benchmark (~21:00)

User asked for a numerical baseline. Built `bench/`:

- `questions.csv` — 100 questions across 9 categories (factual,
  math, definition, list, yesno, social, comparison, time, causal).
  Each has `expected_keywords` (pipe-separated, case-insensitive).
- `run_bench.py` — spawns `nc_run_native` once, drives all 100
  questions through stdin with `/reset` between turns. Captures
  response + token count + latency.
- `score.py` — three 0/1 rubrics:
  - `coherent`: 2..600 chars, no bigram repeated 3+ times
  - `format`: non-empty, no leaked `\nQ:`, has alphanumerics
  - `correct`: at least one expected keyword present

**Baseline — d12 base, temp 0.3, seed 42:**

| Category | n | Coh% | Fmt% | Cor% |
|---|---|---|---|---|
| factual | 20 | 95 | 100 | 60 |
| math | 15 | 100 | 100 | **13** |
| definition | 15 | 53 | 100 | 67 |
| list | 10 | 80 | 100 | 90 |
| yesno | 10 | 70 | 100 | 30 |
| social | 5 | 40 | 100 | 40 |
| comparison | 10 | 90 | 100 | 90 |
| time | 5 | 100 | 100 | 20 |
| causal | 10 | 40 | 100 | 80 |
| **TOTAL** | 100 | **77** | 100 | **56** |

Failure modes spelled out in `bench/README.md`. The biggest:
**definition mode collapse** ("A cat is a cat. A cat is a cat.")
which drives the 53 % coherence in that category.

---

## Phase 7 — Major UX overhaul via parallel agents (~22:00)

User wanted "feature completeness from modern LLM chat apps".
Launched **three general-purpose agents in parallel** via
`isolation: "worktree"`, each with a focused non-overlapping scope:

### Agent 1 (`30b60b2`) — Sidebar UX

- **Rename**: F2 / right-click → Rename. Modal `IDD_RENAME` dialog.
  Rewrites the `TITLE:` line of the chat file in place.
- **Delete**: Del / right-click → Delete. `MB_YESNO` confirm; if
  active chat was deleted, falls back to `/reset` + `chats_create_new()`.
- **Right-click context menu**: Rename / Delete / Save Transcript.
- **Search filter**: EDIT with cue banner "Search chats…". Case-
  insensitive substring filter via `gFilteredIndices[]` map.

Listbox subclassed (`chatlist_proc`) to capture VK_F2 / VK_DELETE /
WM_RBUTTONDOWN.

### Agent 2 (`32d0ab9`) — Per-message actions

Three flat-themed buttons between the toolbar and the transcript:

- **Copy last reply** — `EM_GETTEXTRANGE` over the assistant body's
  `[gLastAsstStart, gLastAsstEnd)` offsets, drops on `CF_TEXT`.
  Just the answer, no headers, no stats footer.
- **Regenerate** — sends `/reset` then re-fires `gPendingUser`
  through a shared `send_prompt_text()` path. New header reads
  "Assistant (regenerated) — h:mm".
- **Edit last prompt** — modal `IDD_EDITPROMPT` with multi-line
  EDIT prefilled with `gPendingUser`. OK = `/reset` + resend.

State resets on Clear and on sidebar chat switch.

### Agent 3 (`8016594`) — Settings v2 + Edit + Help menus

- **Settings dialog grew** to 260×280 with "Sampling" + "Advanced"
  groups. Top-P + Max tokens + multi-line System prompt + Reset All.
- **Three new backend slash commands**: `/topp <f>`, `/maxtok <int>`,
  `/system <text-with-\n-escapes>` (the last actually rewires the
  few-shot prefix and re-prefills).
- **New Edit menu**: Copy / Paste / Select All / Find (Ctrl+C/V/A/F).
  Find is a modal dialog with one EDIT + Find Next, wraps once.
- **New Help items**: Keyboard Shortcuts… (modal listing all keys),
  Slash Commands… (sends `/help`).
- **Empty-state hint** in `clear_transcript()` — muted gray "Welcome
  to Bliss Chat. Type a message below..." so the transcript is never
  blank on first launch.

### Integration

Sequential cherry-pick into master. **Two rounds of merge conflicts:**

- Both Agent 1 and Agent 2 reused resource IDs 210/211 for their
  new dialogs (Rename vs EditPrompt). Renumbered EditPrompt to 212/213.
- Agent 3 reused IDs 208–211 for new settings controls (top-p,
  max-tokens, system prompt, reset-all). Renumbered to 214–217 to
  coexist with Rename/EditPrompt.
- The toolbar layout was different between Agent 1 (added a search
  box above the listbox) and Agent 2 (kept the sidebar's vertical
  range full and added a button strip above the transcript).
  Merged manually to support both.

Result: `27050b7` final, all three feature sets integrated, build
clean.

---

## Phase 8 — XP Explorer toolbar + branding (~22:15)

User: "add a proper windows xp explorer like toolbar with icons".

- Generated `assets/toolbar_icons.bmp` via PIL — a 120×24 (later 264×24
  with 11 icons) strip, magenta (255,0,255) as the COMCTL32 transparency
  key. Custom hand-drawn icons (page+plus, floppy, red octagon, gear, etc.)
- Replaced the three flat text buttons (Save/Settings/Help) with a real
  `TOOLBARCLASSNAMEA` control, themed by the manifest, icons + labels,
  tooltips active. Buttons wired to existing `IDM_*` menu commands.
- Display rename to "Bliss Chat" (capitalized) everywhere user-facing.
  Filesystem and registry paths stay lowercase `bliss-chat` to preserve
  existing user data.
- Desktop shortcut now reads `Bliss Chat.lnk`; `make-shortcut.vbs` cleans
  up legacy names (`XP Tiny LLM.lnk`, `bliss-chat.lnk`).

Committed as `e5d04f3`, then `27050b7` cleaning up the accidentally-
committed agent worktree gitlinks.

---

## Phase 9 — Coherence push: d12 Chinchilla (~21:41 → 00:42)

User: "Can we train a better model?"

Analyzed options:
- (A) d12 at Chinchilla ratio 20 — same arch, ~3.4× more tokens, ~3 h
- (B) Bigger model (d14/d16) at ratio 20 — risky RAM, ~3 h
- (C) Mix in better data (UltraChat / WikiText) — requires data prep + run

User picked A and C. Started A immediately.

`scripts/train-d12-chinchilla.sh` launched at 21:41 via
`systemd-run --user --unit=d12-chinchilla --no-block`. Initial loss 7.7
declining smoothly. 33,600 iterations (ratio 20 auto-computed by
nanochat from `--target-param-data-ratio=20`).

At the time of the handoff write (22:33): 31.8 % done, loss 3.15,
ETA ~2 h 9 min → finish ~00:42.

When this lands, the plan is:
1. Export `d12_028320.pt` → `d12-chinchilla.ncb --int8` on the GPU box.
2. SCP back to Mac, drop in as `build/deploy/MODEL.NCB`.
3. Re-run `bench/run_bench.py` + `bench/score.py`.
4. If meaningful improvement, push to XP; otherwise pivot to (B) or
   the SFT-with-orthogonal-special-init experiment.

Option (C) intentionally deferred until (A)'s numbers are in. If A
moves the needle 56 % → 70 %, C is the obvious next step. If A moves
it 56 % → 58 %, the bottleneck is model SIZE, not training tokens,
and we go to B instead.

---

## Phase 10 — Handoff (~22:34)

Wrote `docs/HANDOFF.md` and this `context/13-session-log.md` plus
two new technical references (`context/01-data-formats.md` and
`context/05-sentinel-protocol.md`) and a GUI architecture doc
(`context/12-gui-architecture.md`).

The repo is shippable as-is. Next concrete event is the d12-chinchilla
training landing around 00:42 — that result decides direction (more
training, bigger model, or alternative SFT attempts).
