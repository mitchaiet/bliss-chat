# Data formats

Two custom binary formats live in this repo: **NCB1** (model weights)
and **NCT1** (tokenizer). Both are designed for the inference engine
on the XP machine to `mmap()` and walk linearly — no decompression,
no JSON, no Python dependency on the inference side.

Canonical writers: `tools/export_ncb.py`, `tools/export_tokenizer.py`.
Canonical reader: `src/nc_run.c::model_load()` and
`src/nc_tokenizer.c::nct_load()`.

---

## NCB1 — model weights

### Header (256 bytes, packed)

```
char     magic[8];        // "NCB1\0\0\0\0"
int32_t  version;         // currently 1
int32_t  dtype_code;      // 0 = fp32, 1 = int8 per-row symmetric
int32_t  vocab_size;
int32_t  pad_vocab_size;  // rounded up so lm_head has a friendly shape
int32_t  n_layer;
int32_t  n_head;
int32_t  n_kv_head;
int32_t  n_embd;
int32_t  head_dim;
int32_t  sequence_len;
uint64_t window_pattern_mask;   // bit i = 1 means layer i is full attention
uint64_t value_embed_mask;      // bit i = 1 means layer i has a value embedding
int32_t  short_window;
int32_t  long_window;
float    rotary_base;
float    softcap;               // 15.0 in nanochat
int32_t  rotary_seq_len;        // typically sequence_len * 10
... padding to 256 bytes ...
```

### Tensors (in fixed order, immediately after header)

```
wte                          // [pad_vocab_size, n_embd]
lm_head                      // [pad_vocab_size, n_embd]
per-layer (for li in 0..n_layer-1):
    q_w                      // [n_head*head_dim, n_embd]
    k_w                      // [n_kv_head*head_dim, n_embd]
    v_w                      // [n_kv_head*head_dim, n_embd]
    o_w                      // [n_embd, n_head*head_dim]
    fc_w                     // [4*n_embd, n_embd]   (ReLU^2 MLP)
    pj_w                     // [n_embd, 4*n_embd]
    resid_lambdas            // [n_layer]   per-layer residual gain
    x0_lambdas               // [n_layer]   per-layer x0 add gain
    smear                    // small per-token mixing scalars
    backout                  // mid-layer residual subtract
    (and if bit li in value_embed_mask is set:)
        value_embeds[li]     // [pad_vocab_size, n_kv_head*head_dim]
        ve_gate[li]          // [n_kv_head*head_dim] — STAYS fp32 even in int8 mode (tiny)
rotary_cos                   // [rotary_seq_len, head_dim/2]
rotary_sin                   // [rotary_seq_len, head_dim/2]
```

### Quantization (int8 mode, `dtype_code == 1`)

For each weight matrix `W` of shape `[rows, cols]`:

```
int8   weights[rows][cols]    // signed bytes, row-major
fp32   scales[rows]           // dequant scale per row: W ≈ weights * scales[r]
```

Per-row symmetric (zero-mean) quantization. No zero-point, no group quant.
Computing the scales: `scales[r] = max(|W[r, :]|) / 127.0`. We snap weights
to int8 by rounding `W[r,c] / scales[r]`.

Embeddings (`wte`, `value_embeds`) AND the LM head (`lm_head`) all use
this scheme. Per-layer scalars (resid_lambdas, x0_lambdas, smear, backout)
and rotary tables stay fp32 — they're tiny.

### Sizes

| Model | n_layer | n_embd | params | fp32 size | int8 size |
|---|---|---|---|---|---|
| d6 (chinchilla) | 6 | 384 | ~30 M | ~295 MB | **75 MB** |
| d12 | 12 | 768 | ~110 M | ~1.1 GB | **280 MB** |

The target XP-era machines have limited RAM. The d12 model leaves enough headroom
for KV cache + OS only because of int8 quantization. d20 (~250 M params,
~700 MB int8) would not fit.

---

## NCT1 — tokenizer

### Header

```
char     magic[8];         // "NCT1\0\0\0\0"
uint32_t version;
uint32_t vocab_size;       // total tokens including specials
uint32_t n_regular;        // count of regular (mergeable_ranks) tokens
uint32_t n_specials;       // count of special tokens
```

### Tables

```
Regular tokens (in id order, 0..n_regular-1):
    repeat n_regular times:
        uint32_t nbytes
        bytes    byte_sequence[nbytes]

Special tokens (after the regular table):
    repeat n_specials times:
        uint32_t id              // assigned token ID
        uint32_t nbytes
        bytes    byte_sequence[nbytes]   // e.g. "<|bos|>"
```

### Notes

- Token IDs for special tokens are the ones tiktoken assigned (not
  necessarily contiguous with `n_regular`). The 9 chat-specials in
  nanochat are `32760..32768`.
- The C loader (`nc_tokenizer.c::nct_load`) builds an in-memory
  hash table at startup mapping `byte_sequence → id` for encode, and
  walks a flat array for `id → byte_sequence` decode.
- Pre-tokenization regex is an ASCII subset of tiktoken's GPT-4 split
  pattern (`\p{L}` → `[A-Za-z]`, `\p{N}` → `[0-9]`). Non-ASCII text
  still encodes via byte-level fallback (each byte its own token, no
  merges across non-ASCII bytes), so English chat encodes optimally
  but other languages take more tokens than tiktoken would produce.

### Size

For a 32K-token nanochat vocab, NCT1 lands at **~480 KB**. Tiny.

---

## Why custom formats vs. GGUF / safetensors / etc.

GGUF was the obvious choice and we tried it (llama.cpp XP port).
Two problems:

1. **GGUF dispatch overhead** is significant on a single-core P4.
   The fused ops, dispatcher tables, and quant variants add per-call
   cost that dwarfs the actual matmul on a small model.
2. **Quant flavors we don't need.** GGUF supports K-quants, AWQ, etc.;
   we only need int8 per-row. A simpler format keeps the C loader
   one screen of code and the decoder branch-free.

End result: same hardware, ~100× speed difference (0.04 → 4.7 tok/s).

---

## Format evolution

`NCB1` and `NCT1` are versioned via the `version` field. The C loader
currently only accepts `version == 1`. If the layout changes:

1. Bump the version constant in both writer and reader.
2. Keep the magic the same so the loader's "is this even a model
   file?" check still works.
3. Document the diff in this file under a new `## NCBn` section.

No backwards compatibility is built in — the deployed XP machine
re-downloads a fresh model file from the HTTP server each time.
