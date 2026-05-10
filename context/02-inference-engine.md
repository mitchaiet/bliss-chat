# Inference engine: `nc_run.c`

This is the heart of the project. ~900 LOC of C99 in `src/nc_run.c` (plus
~600 LOC for the tokenizer in `src/nc_tokenizer.c`). It loads an `NCB1`
model file, runs inference one token at a time, prints results to stdout
using a sentinel-line protocol, and reads new prompts from stdin.

## High-level flow

```
main()
  parse args (-m model.ncb -t tokenizer.nct -c ctx -t temp -p top_p -s seed)
  model_load() ......... mmap-style load whole file into heap, parse NCB header
  nct_load() ........... load tokenizer file, build hash table
  state_init() ......... allocate KV cache + working buffers
  emit "INFO <model_desc>"
  emit "READY"
  for each line on stdin:
    state_reset()      <- KV cache wipe per turn (multi-turn future work)
    tokenize prompt    <- includes <bos><user_start>...<user_end><assistant_start>
    for each prompt token:
      forward_one(token_id)
    until generation done:
      next = sample(state, temp, top_p)
      forward_one(next)
      decode + emit token bytes
    emit "EOT"
```

## `forward_one(state, token_id)` — every step

This is the meat. Below is exactly what happens for one token, mapping
each step to nanochat's `gpt.py` so future-us can verify nothing drifts.

### 1. Token embedding lookup

```c
if (m->wte) {
    memcpy(s->x, m->wte + token_id*D, sizeof(float)*D);
} else {
    const int8_t *row = (int8_t*)m->wte_q + token_id*D;
    float scale = m->wte_scales[token_id];
    for (int i = 0; i < D; i++) s->x[i] = (float)row[i] * scale;
}
```

For int8 models we dequantize on lookup. The whole row gets pulled into
the working buffer `s->x` (one allocation, reused across tokens).

Maps to `gpt.py`: `x = self.transformer.wte(idx)`.

### 2. RMSNorm after embedding

```c
rmsnorm(s->xb, s->x, D);
memcpy(s->x, s->xb, sizeof(float)*D);
```

`rmsnorm` here uses no learnable scale, just `x[i] / sqrt(mean(x²) + 1e-6)`.

Maps to `gpt.py`: `x = norm(x)` after `wte`.

### 3. Smear

The "smear" is a cheap bigram trick: blend the previous token's pre-smear
embedding into the current via a sigmoid gate.

```c
if (s->has_prev_x) {
    double g = 0;
    for (int i = 0; i < 24; i++) g += smear_gate_w[i] * s->x[i];
    float gate = m->smear_lambda * sigmoidf((float)g);
    for (int i = 0; i < D; i++) s->x[i] += gate * s->prev_x_norm[i];
}
memcpy(s->prev_x_norm, s->xb, sizeof(float)*D);  // store PRE-smear normed embed
s->has_prev_x = 1;
```

The previous-embedding stored is the **pre-smear** (post-norm) one — this
matches nanochat's KV-cache decode path where it sets
`kv_cache.prev_embedding = x[:, -1:, :]` *before* applying smear (line 444
of `gpt.py`).

`smear_lambda` is a single scalar parameter. After pretraining it tends
toward zero, so smear is mostly a no-op early in training but available.

### 4. Save x0

```c
memcpy(s->x0, s->x, sizeof(float)*D);
```

Used by per-layer residual blending below.

### 5. Per-layer transformer block

For each of `n_layer` blocks:

#### 5a. Residual lambdas

```c
float r = m->resid_lambdas[li], x0l = m->x0_lambdas[li];
for (int i = 0; i < D; i++)
    s->x[i] = r * s->x[i] + x0l * s->x0[i];
```

`resid_lambdas[i]` typically initialized ~1.0 (decaying with depth),
`x0_lambdas[i]` typically ~0.0–0.2. These mix the residual stream with
the original embedding. Distinguishing feature of nanochat — not in
vanilla llama/GPT-2.

#### 5b. Attention

```c
rmsnorm(s->xb, s->x, D);   // norm input
linear(s->q_h, s->xb, H*HD, D, ...);  // Q projection
linear(s->k_h, s->xb, KH*HD, D, ...); // K projection
linear(s->v_h, s->xb, KH*HD, D, ...); // V projection
```

`linear()` dispatches on dtype: fp32 matmul or int8 matmul (which
multiplies the per-row scale once at the end).

Then the value-embedding mix (only on layers in `ve_layer_mask`):

```c
if (m->L[li].ve_fp32 || m->L[li].ve_q) {
    // Look up ve = value_embeds[layer][token_id], shape (KD)
    // Compute per-head gate from x[:12]
    for (int h = 0; h < KH; h++) {
        double a = sum(ve_gate_w[h] * x[:12]);
        float gate = 3.0f * sigmoid(a);
        for (int d = 0; d < HD; d++)
            v_h[h*HD + d] += gate * ve[h*HD + d];
    }
}
```

This is the "ResFormer" trick — gate-mixed extra value vector per layer.
Adds capacity without much compute. nanochat enables it on alternating
layers (last layer always included).

Then RoPE on Q and K, QK norm, scale by 1.2, write to KV cache:

```c
apply_rope_head(q_h, cos, sin, HD);
apply_rope_head(k_h, cos, sin, HD);
rmsnorm(q_h, q_h, HD);  // QK norm (per head)
rmsnorm(k_h, k_h, HD);
for (...) q_h[i] *= 1.2f;  // K *= 1.2 too
memcpy(kcache[li][pos], k_h, KD*sizeof(float));
memcpy(vcache[li][pos], v_h, KD*sizeof(float));
```

Sliding-window attention from `[max(0, pos - window + 1) .. pos]`:

```c
for each head h:
    h_kv = (h * KH / H);  // GQA mapping
    for t in [p0, pos]:
        scores[t] = (q_h[h] · k_cache[li][t][h_kv]) / sqrt(HD);
    softmax(scores);
    for t in [p0, pos]:
        out[h] += scores[t] * v_cache[li][t][h_kv];
```

Then output projection + residual:

```c
linear(s->xb, s->xb2, D, D, c_proj_w, ...);
for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
```

#### 5c. MLP

```c
rmsnorm(s->xb, s->x, D);
linear(s->ffn_h, s->xb, FF, D, c_fc_w, ...);  // FF = 4*D
for (int i = 0; i < FF; i++) {
    float v = s->ffn_h[i];
    if (v < 0) v = 0;
    s->ffn_h[i] = v * v;             // ReLU^2
}
linear(s->xb, s->ffn_h, D, FF, c_proj_w, ...);
for (int i = 0; i < D; i++) s->x[i] += s->xb[i];  // residual
```

ReLU² is the non-standard part — most modern LLMs use SwiGLU. Cheaper
forward, slightly worse model. Karpathy's trade.

#### 5d. Backout snapshot

At the midpoint layer (`n_layer/2`), save `x` for backout.

### 6. Backout subtract + final norm

```c
for (int i = 0; i < D; i++)
    s->x[i] -= m->backout_lambda * s->x_backout[i];
rmsnorm(s->xb, s->x, D);
```

Removes mid-layer features that the model decided to ignore at the top.
Another nanochat-specific trick.

### 7. LM head + softcap

```c
linear(s->logits, s->xb, pad_vocab, D, lm_head_w, ...);
for (int i = 0; i < vocab_size; i++) {
    float v = s->logits[i] / softcap;  // softcap=15
    s->logits[i] = softcap * tanhf(v);
}
```

The softcap (`15 * tanh(x/15)`) prevents logit blow-up during training and
is kept for inference for fidelity.

### 8. State advance

`s->seq_pos++`. Done.

## Sampling

`sample(state, temperature, top_p)`:

- **Greedy** (`temperature == 0`): argmax of logits.
- **Temperature**: divide logits by `T`, softmax to get probs.
- **Top-p**: if `p < 1.0`, sort indices by prob descending, take the
  smallest prefix whose cumulative prob ≥ `p`, sample from that prefix
  weighted by their original probs.
- **Pure** sampling (`p >= 1.0`): inverse-CDF sample.

RNG is xorshift32 keyed off the `-s` flag. Deterministic given seed.

## Working buffer layout

All buffers are heap allocations done once at `state_init`:

| Buffer | Size | Used for |
|---|---|---|
| `kcache` | `n_layer × seq_len × kv_dim × 4` | K cache |
| `vcache` | `n_layer × seq_len × kv_dim × 4` | V cache (post-VE blend) |
| `prev_x_norm` | `D × 4` | smear's prev pre-smear normed embed |
| `x` | `D × 4` | residual stream activation |
| `x0` | `D × 4` | initial post-smear/post-norm (for x0_lambdas) |
| `xb`, `xb2` | `D × 4` × 2 | scratch for norm output, attn output |
| `q_h`, `k_h`, `v_h` | `KD × 4` (×3) | per-token Q/K/V |
| `attn_scores` | `H × seq_len × 4` | softmax scratch per head |
| `ffn_h` | `4D × 4` | MLP hidden activation |
| `logits` | `pad_vocab × 4` | output logits |
| `probs`, `ranking` | `vocab × 4` (×2) | sampling |

For d12 (n_layer=12, n_embd=768, kv_dim=384, seq_len=1024):
- KV cache: 12 × 1024 × 384 × 4 × 2 = 36 MB
- Other buffers: ~250 KB

So inference adds ~36 MB on top of the ~280 MB int8 weights.

## What's intentionally missing

- **No SIMD intrinsics**. The P4's SSE3 is too weak to beat scalar for
  these tiny models. Tested both with and without `__SSE3__` paths in
  ggml; scalar won for d6.
- **No threading**. Single-threaded keeps things simple and the P4's
  single-core hyperthreading didn't help in practice (memory-bandwidth
  bound, not compute bound).
- **No mmap**. `fread` the whole file into one heap allocation. Simpler
  on Windows where the mmap APIs are different from POSIX. The model
  fits in RAM anyway.
- **No graph compilation**. Just plain C function calls.

## Validation

The native build on Mac generates **identical** output (greedy, fixed seed)
to a PyTorch reference run for the same model. Spot-checked on:
- `"The capital of France is"` → same exact tokens
- `"Once upon a time"` → same exact tokens
- `"hi"` → same exact tokens

This confirms the architecture port is numerically faithful.
