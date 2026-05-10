# Profiling

`nc_run.c` carries a small `QueryPerformanceCounter`-based profiler that
ticks the hot paths inside `forward_one` and dumps totals at the end of
each generation turn (the `[prof turn]` lines on stderr). Costs roughly
one nanosecond per tick — invisible relative to the workload.

Counters:
- `forward_calls / forward_ns` — wall time inside `forward_one`
- `linear_calls / linear_ns`   — every dense matmul (`linear()`)
- `rmsnorm_ns`                 — every `rmsnorm`
- `softmax_ns`                 — attention softmax block
- `rope_ns`                    — RoPE rotation
- `ve_lookup_ns`               — value-embed lookup

Reset at the top of every turn (after KV-prefix restore).

## Run 1 — XP Pentium 4 @ 3 GHz, d12 int8, 30 generated tokens (2026-05)

Prompt: `Tell me about cats`

```
[prof turn] forward_calls=36 total=40989.208ms avg=1138.589ms
[prof turn]   linear:  39618.143ms (2628 calls, avg 15075.397us)
[prof turn]   rmsnorm: 23.731ms     (6120 calls)
[prof turn]   rope:    38.590ms
[prof turn]   softmax: 615.962ms
[prof turn]   ve_look: 0.000ms
```

Share of `forward_ns`:

| op       | time     | %       |
|----------|----------|---------|
| linear   | 39.62 s  | **96.7 %** |
| softmax  | 0.62 s   | 1.5 %   |
| rope     | 0.04 s   | 0.09 %  |
| rmsnorm  | 0.02 s   | 0.06 %  |
| ve_look  | 0 s      | 0 %     |
| (other)  | 0.69 s   | 1.7 %   |

Per-token wall: **1.14 s** end-to-end → **0.88 tok/s**, matching the
~0.5–1 tok/s figure observed in the GUI. Each `forward_one` runs 73
linear ops (12 layers × 6 + lm_head + extras), each averaging 15 ms.

## Run 2 — Mac M3 native (sanity check), d12 int8, 22 forward calls

```
linear:  2625 ms / 2663 ms total  → 98.6 %
rmsnorm: 1 ms
softmax: 31 ms
rope:    0.17 ms
ve_look: 0 ms
```

Same shape, slightly higher linear share — ARM has a faster softmax
relative to its scalar matmul.

## Conclusion

`linear()` is essentially the entire CPU budget. Every other op is
noise. Optimizations only matter if they speed up dense matmul:

1. **SSE2 SIMD on the inner dot product** — P4 has SSE2 + SSE3, scalar
   today, expected 2-4× on dense fp32 lanes and 4× on int8.
2. Skipping unused KV positions in attention QK·V (already done via
   sliding window for short layers).
3. Cache-friendly layout / blocking — secondary; current matmul is
   already row-major-K friendly.

Softmax SIMD or rmsnorm SIMD: not worth the engineering effort
(combined < 2 % of forward).
