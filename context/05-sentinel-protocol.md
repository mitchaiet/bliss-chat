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

Sets the nucleus sampling cutoff. **Currently stored but not
honored** — the sampler is temperature-only. The GUI's Settings
dialog persists this to registry and pushes it on every backend
ready event, so the UI is honest about what the user picked.

### `/maxtok <int>\n`

Per-turn generation cap, clamped to `[0, 2048]`. `0` means "no cap,
use context budget" (the previous behavior). Applied as `min(budget,
maxtok)` inside the generation loop.

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
