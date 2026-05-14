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

## Run 1 — XP Pentium 4-class machine, d12 int8, 30 generated tokens (2026-05)

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

## Run 2 — native sanity check, d12 int8, 22 forward calls

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

## Run 3 — XP Pentium 4, d12 int8, **SSE2 matmul** (2026-05)

After landing the SSE2 paths in `matmul_fp32` / `matmul_int8`
(8-lane fp32 add/mul; 8-byte int8 → cmpgt sign-extend → cvtepi32_ps).

```
[prof turn] forward_calls=91 total=21366.711ms avg=234.799ms
[prof turn]   linear:  18368.987ms (6643 calls, avg 2765.164us)
[prof turn]   rmsnorm: 59.359ms     (15470 calls)
[prof turn]   rope:    51.881ms
[prof turn]   softmax: 1872.144ms
[prof turn]   ve_look: 0.000ms
```

Speedups vs. Run 1 (scalar) — same hardware, same model:

| Op           | Scalar avg / call | SSE2 avg / call | **Speedup** |
|--------------|-------------------|-----------------|-------------|
| `linear`     | 15075 µs          | 2765 µs         | **5.45×**   |
| `forward_one`| 1138.6 ms         | 234.8 ms        | **4.85×**   |

End-to-end: **0.88 → 4.26 tok/s**. Greedy output identical to scalar
on `"What is 2 plus 2?"` (verified char-for-char vs. native arm64
scalar reference).

Softmax is now the second-largest cost (~9 %). RMSNorm/RoPE remain
in the noise. Further matmul wins (e.g. SSE2 lookup-table-free int8
× int8 with packed-x quantization) would be incremental; bigger
gains likely come from reducing call count (KV-cache reuse,
batched lm_head) than from making each call faster.

## Run 4 — XP Pentium 4, d12 int8, **SSE2 attention helpers** (2026-05)

Adding SSE2 to the QK·V dot product (`dot_fp32`) and the V weighted-sum
(`axpy_fp32`). `expf` left scalar (transcendental, polynomial approx
not worth the precision risk).

```
[prof turn] forward_calls=21 total=4486.136ms avg=213.626ms
[prof turn]   linear:  4001.548ms (1533 calls, avg 2610.273us)
[prof turn]   rmsnorm: 13.793ms     (3570 calls)
[prof turn]   rope:    5.497ms
[prof turn]   softmax: 163.967ms
[prof turn]   ve_look: 0.000ms
```

Softmax block (QK·V + softmax + V·weights):

| Stage         | Per-forward ms | Speedup |
|---------------|----------------|---------|
| Run 1 scalar  | ~17.1 ms       | 1×      |
| Run 3 SSE2 mm | 20.6 ms        | 0.8×    |
| Run 4 SSE2 attn | **7.8 ms**   | **2.2×** vs scalar |

End-to-end per forward: **234.8 → 213.6 ms** (+9 %). Cumulative
end-to-end vs the original scalar baseline: **1138.6 → 213.6 ms =
5.33× faster, ~4.68 tok/s on the Pentium 4.**
