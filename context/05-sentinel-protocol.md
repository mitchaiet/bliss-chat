# Sentinel protocol — GUI ⇄ backend wire format

`XPCHAT.EXE` (GUI) spawns `NC_RUN.EXE` (backend) once at startup and
talks to it over Win32 anonymous pipes for the entire session. Stdout
+ stderr from the backend, stdin to the backend, plus a sideband
`backend-*.log` file collecting stderr (for the profiler dumps).

Wire format is **line-based with `\x01`-prefixed sentinels.**
Anything that does NOT start with `\x01` on stdout is user-visible
text (assistant output). Anything on stdin that doesn't start with
`/` or `\x01` is a user message.

This file is the canonical reference for what each sentinel means.

---

## Backend → GUI (stdout)

### `\x01READY\n`

Backend finished loading the model + prefilling the few-shot prefix.
The GUI flips `gBackendReady = 1`, enables the input box, and shows
"Ready" in the status bar.

Emitted exactly once per backend process, after `state_save_prefix()`
in `nc_run.c::main()`.

### `\x01INFO <text>\n`

Free-text status notice. The GUI routes these to either:

- **The model label** if `text` starts with `nanochat-` (the
  canonical model description format from startup).
- **The status bar** otherwise (e.g. `temperature = 0.700`,
  `conversation reset`, `context full, conversation reset`,
  `stopped by user`).

This routing fixed a bug where `/temp 0.7` was overwriting the model
label with "temperature = 0.700" because every INFO blindly updated it.

### `\x01PROG <pct>\n`

Progress sentinel, `pct ∈ [0, 100]`. Emitted during three phases:

1. Model file load (one PROG per N% of bytes read).
2. Few-shot prefix prefill (one PROG per N% of tokens prefilled).
3. Per-token generation (one PROG per token, mapping `gen_count`
   into the `budget` window).

GUI feeds these into `SendMessage(gProgress, PBM_SETPOS, ...)`. With
the comctl32 v6 manifest in place, the progress bar renders as the
Luna green segmented bar.

### `\x01EOT <count>\n`

End of one assistant turn. `count` is the number of generated tokens.
GUI:

- Computes elapsed s/tok and appends the `[N.N sec - X.XX sec/token
  - K tokens - HH:MM:SS]` stats footer.
- Re-enables Send, disables Stop.
- Appends the assistant text to the active chat's `.txt` file.
- Enables the Copy last reply / Regenerate / Edit last prompt buttons.

### `\x01ERR <text>\n`

Backend reported a per-turn error (rare — usually a malformed prompt
or a backend bug). GUI prints the message in red in the transcript
and re-enables Send.

---

## GUI → Backend (stdin)

### `<text>\n`

A single line, no leading `/`, no leading `\x01`. Treated as a user
turn:

- For the first turn after `/reset` or backend startup: prefill
  `"<text>\nA:"` against the few-shot prefix's "Q: " ending, then
  generate.
- For subsequent turns: prefill `"\n\nQ: <text>\nA:"` to continue
  the running Q&A pattern.

### `/reset\n`

Drop the conversation history. Backend restores the saved post-
prefix KV cache snapshot, resets `turn_idx = 0`, emits an INFO line
and an EOT 0.

### `/help\n`

Backend emits one INFO line per known slash command, then EOT 0.

### `/info\n`

Backend emits one INFO line summarizing model size, dtype, current
turn index, seq_pos / ctx_max, current temperature. Then EOT 0.

### `/temp <float>\n`

Sets the sampling temperature. `<float>` is clamped to `[0, 5]`.
Emits `INFO temperature = X.XXX` + EOT 0.

### `/seed <int>\n`

Re-seeds the RNG. Sampling becomes reproducible from this point
(for non-greedy temps). Emits `INFO seed = N` + EOT 0.

### `/topp <float>\n`

Sets the nucleus sampling cutoff. **Honored for real since v1.3.0**
— the sampler does full nucleus filtering (index sort of the vocab
by prob, truncate at cumulative `p`, renormalize). From v1.0 through
v1.2.3 the value was stored but ignored (temperature-only sampler).
The GUI's Settings dialog persists this to registry and pushes it on
every backend ready event.

### `/maxtok <int>\n`

Per-turn generation cap, clamped to `[0, 2048]`. `0` means "no cap,
use context budget" (the previous behavior). Applied as `min(budget,
maxtok)` inside the generation loop.

### `/rambleguard <0|1>\n`

Enables/disables the runtime loop suppressor (default on; since
v1.2.0): generation hard-stops when the same token repeats 4+ times
in a row, or when the just-finished 2/3/4-gram has already appeared
twice earlier in the answer. `0` restores the old let-it-ramble
behavior. Emits `INFO ramble_guard = N` + EOT 0.

### `/repeat <float>\n`

Sets the repetition penalty applied to tokens seen recently in the
generated answer (last 64 tokens; prompt/history logits untouched).
Clamped to `[1.0, 2.0]`; `1.0` disables it. Since v1.2.0. Emits
`INFO repeat_penalty = X.XXX` + EOT 0.

### `/system <one-line-text>\n`

Replace the few-shot prefix. The text uses `\n` (literal backslash-n)
as a newline escape since the wire format is line-oriented. Backend:

1. Resets state to a fresh (empty) KV cache.
2. Tokenizes the new text.
3. Prefills it through `forward_one()`.
4. Snapshots the resulting KV cache as the new "post-prefix" state.

Used by the GUI's Settings → System Prompt control. The default
prefix in `nc_run.c::FEWSHOT_PREFIX` stays the fallback if `/system`
is never sent.

### `/remember <fact>\n` (v1.3.0)

Stores a short persistent note (max 12 notes, 160 chars each). Notes
survive across conversations and app restarts: they're saved to the
notes file (the path given by `-m` or `/memfile`) and rendered as a
single `Notes: a; b; c` line inside the system prefix — the curated
training mixture teaches the model to read this exact `Notes:`
format. The new note takes effect in the prefilled prefix at the
next `/reset`, context rollover, or restart (the current
conversation already contains the fact naturally). Emits
`INFO noted (n/12): <fact>` + EOT 0 (or `memory full` /
`nothing to remember`).

### `/memories\n` (v1.3.0)

Lists stored notes, one `INFO <n>. <fact>` line per note (1-based
indexes), then EOT 0. Empty store gets a hint pointing at
`/remember`.

### `/forget <n>\n` (v1.3.0)

Removes note `n` (as numbered by `/memories`) and rewrites the notes
file. Emits `INFO forgot note N (K left)` + EOT 0, or
`INFO no note N - see /memories` if out of range.

### `/memfile <path>\n` (v1.3.0)

Points the backend at a different notes file, reloads it, and
immediately rebuilds + re-prefills the prefix (system text + fresh
`Notes:` line). Conversation history is dropped. Emits
`INFO memory file loaded (N notes)` + EOT 0.

The same file can be set at startup via the **`-m <memfile>` CLI
flag** — the GUI passes its notes file this way so stored notes are
in the prefix from the very first prompt.

### `/replay <user>\t<assistant>\n` (v1.3.0)

Prefills a stored exchange into the KV cache **without generating**.
The user and assistant halves are separated by a literal tab. The
pair is formatted through the normal Q/A turn template, forwarded
token by token, appended to the thread summary, and `turn_idx`
advances — so the model genuinely "remembers" the exchange. Used by
the GUI to restore a saved chat on sidebar switch. Replies
`INFO turn replayed` then EOT 0; if the context is nearly full the
replay is skipped with `INFO context full, replay skipped` + EOT 0.

### `\x01STOP\n`

Abort the current generation. Backend's per-token loop polls
`stdin` via `PeekNamedPipe` on every iteration; if it sees this
sentinel, it consumes the bytes, flushes whatever was held back,
emits `INFO stopped by user`, then EOT.

The model load is preserved — abort is cheap (just stops generating;
KV cache and weights stay live). Esc in the GUI also routes here.

### EOF on stdin

Backend exits cleanly. GUI sends this when it's tearing down (close
of the pipe via `CloseHandle(gBackendStdinW)` in `shutdown_backend()`).

---

## Hold-back buffer (transcript-side stop detection)

The backend emits raw tokens. The base d12 model often wants to
generate `\nQ:` to start a new turn — that's a clean stop signal
but the bytes shouldn't reach the user.

`nc_run.c` keeps a 3-byte hold buffer over emitted text. Older bytes
spill to stdout only when newer bytes push them out. If the buffer
becomes `['\n', 'Q', ':']` or `['\n', 'A', ':']`, the loop breaks
with `hit_stop=1` and the buffer is discarded (never emitted).

At end-of-loop, a partial stop pattern can still be in the buffer
(model emitted `\nQ` but generation ended before `:` via eos_id /
assistant_end). The trailing-flush logic scans the buffer for any
`\n` + `Q|A` substring and truncates at that point so we don't leak
`\nQ` to the GUI.

This is why you sometimes see "Q\n" trailing in older log files —
the hold-buffer rule predates the partial-stop scan.

---

## Why this protocol vs. JSON-RPC / similar

1. **Pipe-friendly.** Every sentinel is one ASCII line terminated
   by `\n`. No length prefixes, no escaping, no framing layer.
   The backend's stdin is whatever `getchar` gives us.
2. **Streamable.** Tokens go to stdout as bytes the instant they're
   generated. No buffering. The GUI sees partial text immediately.
3. **Mixed text + control.** The `\x01` prefix neatly separates
   user-visible tokens (no prefix) from machine-readable control
   (prefix), without needing a length frame or escape sequence.
   `\x01` is `SOH` — never appears in legitimate text.
4. **Simple to add commands.** A new slash command is one `if
   (!strncmp(line, "/foo ", 5))` branch in `nc_run.c::main()`.
