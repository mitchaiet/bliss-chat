# Known issues

## SFT diverges to NaN — unresolved

### Symptoms

The standard nanochat SFT pipeline (`scripts/chat_sft.py`) produces NaN loss
within 5 training steps when applied to either our d6 or d12 base model.
Resulting checkpoint is unusable — generation produces gibberish.

Loss trajectory before NaN (consistent across all variants tried):

```
step 1: ~1.2
step 2: ~1.6
step 3: ~1.8
step 4: ~2.0
step 5: nan  (and remains nan for the rest of training)
```

### What we tried

| Variant | Result |
|---|---|
| Default SFT settings | NaN @ step 3 (d6), step 5 (d12) |
| 10× lower LR | NaN |
| 30× lower LR (matrix-lr=0.001) | NaN |
| Cold-start optimizer (`--load-optimizer=0`) | NaN |
| Identity-conversations only (patched `chat_sft.py`) | NaN |
| `--init-lr-frac=0.05 --warmup-ratio=0.2` (effective matrix LR ~1e-4) | NaN @ step 5 |
| Pre-init special-token wte/lm_head/value_embeds from `<\|bos\|>`'s trained values | **NaN @ step 5 still** |
| `NANOCHAT_DTYPE=float32` (force fp32 forward — falsifies hypothesis 3) | NaN by step ~88 |
| Grad-finite check + manual `mul_` clip (max_norm=1.0) + skip-on-NaN | Trained 7 steps cleanly (loss 1.32→2.33), then **NaN by step 8** |
| Vanilla `torch.optim.AdamW` (replace Muon entirely) + lr 5e-5 + same guards | **Identical divergence pattern**: trained 7 steps (loss 1.32→2.31), NaN at step 8 |

### Diagnosis attempts

**Hypothesis 1**: Special tokens at random init cause exploding logits.

The chat-special tokens (`<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, etc., ids 32760–32767) are never observed during pretraining. Their wte rows stay at the random init from `init_weights()` (uniform-with-std=0.8) instead of the trained magnitude (std~5.5).

We patched the checkpoint to copy `<|bos|>`'s well-trained embeddings into all the other special-token slots (wte + lm_head + value_embeds). **Did not fix the NaN.**

**Hypothesis 2**: LR too high.

With matrix-lr=0.002 × init_lr_frac=0.05 × lrm-at-step-5=0.07 = effective LR ≈ 7e-6. Already absurdly small. NaN persists. So the issue isn't the optimizer step magnitude.

**Hypothesis 3 (FALSIFIED 2026-05-10)**: bf16 forward overflow on long SmolTalk samples.

Tested with `NANOCHAT_DTYPE=float32`. Loss still NaN'd by step ~88
(later than bf16's step 5, but still terminal). bf16 was making the
explosion faster, not causing it.

**Hypothesis 4 (FALSIFIED 2026-05-10)**: malformed all-masked samples produce
0.0-with-no-grad cross-entropy that corrupts grads. Hypothesis 5: subsequent
inf-grad steps cause `clip_grad_norm_` to do `0 * inf = NaN` and corrupt
weights.

Both were partially right, neither fully solves it. Patched `chat_sft.py`
with all three guards (skip non-grad/non-finite loss, manual `.mul_()` clip
that avoids the inf-times-zero trap, skip optimizer step on non-finite grad
norm). Result: trains 7 steps cleanly, loss climbs 1.32 → 2.33 (i.e. it is
DIVERGING under the constraints), then walls at NaN by step 8.

**Working theory (UPDATED 2026-05-10)**: Optimizer is not the cause.

Swapped Muon → vanilla `torch.optim.AdamW` with lr 5e-5 + all the
NaN-skip / grad-clip guards still in place. **Result: the divergence
trajectory is byte-identical** — loss 1.32 → 2.31 over 7 steps, NaN at
step 8. AdamW with lr 5e-5 has a per-step update bounded at ~5e-5,
which cannot explain weights going off-cliff in 7 steps. So whatever's
driving divergence is either:

* in the data — SFT data (SmolTalk-style) has a structural issue
  with how special-token positions are masked / labeled, OR
* in the patched checkpoint — `d12_patched` copies `<bos>`'s row into
  every chat-special slot, so WTE[user_start] ≡ WTE[bos] etc. The
  model can't distinguish them, gradients update them all in lock-
  step, and after a few updates the embedding space collapses.

The second is the more plausible explanation given the consistency of
the 7-step run-up. Loss climbs each step (1.32, 1.71, 2.02, 2.13, 2.20,
2.28, 2.31) — the model is _learning to be worse_ in a predictable way.

**Next things to try** (when revisiting):

1. **Re-init the special-token rows with random orthogonal vectors instead
   of copying `<bos>`** (i.e. give the model a chance to differentiate them
   instead of feeding it identical rows for 9 different roles).
2. **Pretrain a fresh d12 from scratch on chat-formatted data** so the
   special tokens are seen continuously during base training; no SFT needed.
3. **LoRA / adapter** — freeze every base param, only learn a small
   adapter. Loss landscape changes; might dodge the collapse.
4. **Inspect a single sample's gradient direction**: is the model being
   asked to learn an impossible mapping (e.g. all special-token labels
   point to the same row, gradient becomes self-canceling).
5. Patch nanochat's loss to log per-position cross-entropy at step 1; see
   which positions contribute most to the climbing loss — likely the
   special-token positions.

These are research-grade and out of scope for the v1 ship. The base
model ramble is what users will see for now.

**Hypothesis 4 (untested)**: Loss computation NaN from a malformed sample.

The mask is set per-token; if a sample has `mask=1` everywhere or
`-100` (ignore_index) everywhere, cross-entropy can divide by 0 → NaN.

### Possible fixes (future work)

1. Force fp32 forward + fp32 model (set `COMPUTE_DTYPE=torch.float32` before
   loading the model). Costs memory but eliminates bf16 overflow.
2. Add gradient clipping. nanochat doesn't enable it by default.
   `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` in
   the training loop.
3. Skip steps with NaN loss (`if torch.isnan(loss): optimizer.zero_grad(); continue`).
   Pragmatic but ugly.
4. Pretrain a smaller seq_len (256 or 512) so SmolTalk samples truncate
   more aggressively.
5. Train SFT-from-scratch — start with a freshly-init small model and
   pretrain on a chat-formatted mix from the beginning. The chat-specials
   would learn organically.

### Workaround currently shipped

The base d12 model (no SFT) is what's deployed on the XP machine. It
generates fluent multi-sentence English but doesn't follow the
`<|user_start|>...` chat template, so it just continues your prompt
rather than answering it. Users frame prompts as completion-style
("The capital of France is", "Once upon a time") rather than question-style.

## Multi-turn KV cache — TODO

`nc_run.c::main()` calls `state_reset(&S)` at the top of each user-message
loop iteration. This wipes the KV cache, so the model sees only the
current message — no conversation memory.

To enable multi-turn:
1. Don't reset state between turns.
2. The chat template formatting needs to NOT re-emit `<|bos|>` after the
   first turn.
3. Manage seq_len overflow: when `seq_pos + tokens_to_add > sequence_len`,
   either error out or implement a sliding-window context shift.

Estimated work: ~30 LOC + careful testing.

## Stop button is a no-op

`xpchat.c::stop_generation()` is wired but the button is permanently
disabled. To support stopping mid-generation:

Option A: Send a `\x01STOP\n` line to backend stdin. `nc_run.c` polls stdin
non-blockingly between tokens; if it sees STOP, breaks out of the
generation loop and emits `\x01EOT\n`.

Option B: GUI kills `NC_RUN.EXE` and respawns it. Loses the model load
time (a few seconds for the int8 model), but simpler.

Option B is what we'd ship. Estimated ~50 LOC in `xpchat.c`.

## ASCII-only pre-tokenizer

`nc_tokenizer.c` implements an ASCII subset of tiktoken's GPT-4-style
split pattern (`\p{L}` → `[A-Za-z]`, `\p{N}` → `[0-9]`). Non-ASCII text
still encodes via byte-level fallback tokens (each byte is its own token,
no merges across non-ASCII bytes).

Result: English chat encodes optimally. Other languages still encode but
with more tokens than tiktoken would produce.

To fix: would need a Unicode-aware regex implementation or a small lookup
table for `\p{L}`/`\p{N}` based on Unicode categories. ~200 LOC + Unicode
table data. Not a priority for v1.

## Sliding window may be off-by-one

`nc_run.c` uses `p0 = pos - window + 1` for the inclusive lower bound of
the window. nanochat's flash_attention uses `(left_window, 0)` where
`left_window` is the number of *tokens* before the current position. We
might be including one extra token at the boundary (or missing one). For
small contexts this is invisible; for ctx near `long_window` it could
cause minor scoring drift.

## Memory headroom on the XP machine

The d12 int8 model (~280 MB) plus ~36 MB of KV cache plus ~150 MB of OS
overhead leaves limited free RAM on the XP test class. We've seen the system
function under this pressure but it's tight. Pushing to a `--depth=14`
or `d20` model would not fit.

If we needed more headroom: `q4` quantization (4-bit per weight, ~140 MB
for d12) would help, at some quality loss. Implementation cost: ~100 LOC
in `export_ncb.py` and `nc_run.c::matmul_int4`.
