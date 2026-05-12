# context/

Deep-dive technical documentation for Bliss Chat. Each file is a
standalone reference; together they cover the architecture from
the binary file formats up through the Win32 GUI.

For an at-a-glance project state, start with **`docs/HANDOFF.md`**
at the repo root.

## Files

| File | Topic |
|---|---|
| `00-overview.md` | Project pitch, pipeline, hardware targets |
| `01-data-formats.md` | NCB1 (model) and NCT1 (tokenizer) binary formats |
| `02-inference-engine.md` | `nc_run.c` internals, forward pass, KV cache |
| `05-sentinel-protocol.md` | GUI ⇄ backend wire format and slash commands |
| `09-training-history.md` | Every training run, recipe, val_bpb, sample output |
| `10-known-issues.md` | The SFT graveyard (10 falsified hypotheses) + open work |
| `11-profiling.md` | Per-Pentium-4-cycle perf breakdown, SSE2 deltas |
| `12-gui-architecture.md` | Win32 control hierarchy, dialogs, message routing |
| `13-session-log.md` | Chronological session log of the v1.0.0 push |

## Reading order

If you're new and want to understand the system:

1. `00-overview.md` — what is this project
2. `01-data-formats.md` — what's on disk
3. `02-inference-engine.md` — how the model runs
4. `05-sentinel-protocol.md` — how the GUI and backend talk
5. `12-gui-architecture.md` — what the UI is doing
6. `11-profiling.md` — where the time goes on a Pentium 4
7. `09-training-history.md` — how we got here
8. `10-known-issues.md` — what we tried and what didn't work
9. `13-session-log.md` — narrative log of the v1.0.0 push

## Reading order (if you're picking up after the handoff)

1. `docs/HANDOFF.md` at the repo root — current state, next moves
2. `13-session-log.md` — what just happened
3. `10-known-issues.md` — what's been ruled out for SFT
4. `09-training-history.md` — latest training run (d12 chinchilla)
   is "in progress" as of the session-log timestamp
