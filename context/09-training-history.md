# Training history

Chronological record of every training run, configuration, and result.
Useful for reproducing or for future-us looking back to figure out why
something is shaped a particular way.

## Hardware

NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- 96 GB VRAM
- Compute capability 12.0
- CUDA driver 580.95.05
- ~1.5–2× slower than H100 for our workloads

Linux Ubuntu 24.04, kernel 6.8.0, Python 3.10.19, PyTorch 2.9.1+cu128.

Training framework: [Karpathy nanochat](https://github.com/karpathy/nanochat)
@ commit `dc54a1a` (early 2026).

## Run 1: d6 base pretrain — **shipped to XP**

**Goal**: smallest viable nanochat — bring up the export → C engine → XP
pipeline end to end.

**Config** (`scripts/train-d6.sh`):

```
--depth=6 --head-dim=64 --window-pattern=L
--max-seq-len=512 --device-batch-size=32 --total-batch-size=16384
--num-iterations=5000 --eval-every=500 --sample-every=500
```

**Architecture**: 6 layers × 384 dim × 6 heads (no GQA), full-attention
window only (no sliding), 32K vocab, no dropout.

**Numbers**:
- Total params: **30M** (rounded; exactly 73.5M counting `value_embeds`,
  but 38M of those are unused embedding tables that just sit in memory
  during inference).
- Tokens trained: 16384 × 5000 = **82M** tokens of ClimbMix web data.
- Wall clock: **1.6 minutes** (lol).
- Final val_bpb: **1.165**.
- Sampled outputs: still mostly repetitive — "the capital of France is the
  capital of France, and the capital of France is the capital of France."

**Export**: 75 MB int8 NCB1 file via `tools/export_ncb.py --int8`.

**XP performance**: ~3 tok/s on the Pentium 4. Output is fluent English
that rambles incoherently — typical undertrained tiny base LM behavior.

## Run 2: d6 SFT — **failed (NaN)**

**Goal**: chat-tune the d6 base model. Default nanochat SFT pipeline.

**Result**: loss 2.28 (step 1) → 3.07 (step 2) → **NaN** (step 3).
Continued ticking to completion at step 1500 with NaN loss the whole way.
The saved checkpoint is unusable.

**Variants attempted**, all NaN'd by step 5:
- Default LR (`embedding-lr=0.3`, `matrix-lr=0.02`, etc.).
- 10× lower LR.
- 30× lower LR (`matrix-lr=0.001`).
- Cold-start optimizer (`--load-optimizer=0`).
- Identity-only data mix (patched `chat_sft.py`).
- `--init-lr-frac=0.05` + `--warmup-ratio=0.2`.

Went on to d12 to get a properly coherent base model.

## Run 3: d12 base pretrain — **shipped to XP**

**Goal**: a real base LM. d12 is the smallest size in nanochat's
`scaling_laws.sh` and the first tier where outputs are genuinely coherent
sentences.

**Config** (`scripts/train-d12.sh`):

```
--depth=12 --max-seq-len=1024
--device-batch-size=32 --total-batch-size=65536
--num-iterations=10000 --eval-every=1000 --sample-every=1000
```

**Architecture**: 12 layers × 768 dim × 6 heads, GQA (n_kv_head=6, same as
n_head, so no GQA effective), `SSSL` window pattern (3 short layers, 1
long, repeated), 32K vocab.

**Numbers**:
- Total params: **~110M** (transformer matrices + lm_head + wte + value_embeds × 6 layers).
- Tokens trained: 65536 × 10000 = **655M** tokens (~6× more than d6).
- Wall clock: **53.95 minutes**.
- Final val_bpb: **0.862** (much better than d6's 1.165).
- Sampled outputs:
  - "The chemical symbol of gold is Au. It is a metal..."
  - "The planets of the solar system are: Earth, Venus, Mars, Jupiter, Saturn, Uranus, Neptune"
  - "My favorite color is blue. It's the color of the sky, the color of the ocean,"

**Export**: **280 MB int8 NCB1**.

**XP performance**: ~0.5–1 tok/s on the Pentium 4. Output is dramatically
more coherent than d6 — multi-sentence English, occasional persona drift
("My name is Marie and I am a science teacher..."), but recognizable
real-world knowledge.

## Run 4: d12 SFT (initial attempt) — **failed (NaN)**

**Goal**: chat-tune d12 with full nanochat data mix.

**Config**:

```
--max-seq-len=1024 --device-batch-size=16 --total-batch-size=32768
--num-iterations=1500 --mmlu-epochs=1 --gsm8k-epochs=1
```

(Default LRs.)

**Result**: NaN at step 5. Same explosion pattern as d6:
1.14 → 1.50 → 1.71 → 1.79 → NaN.

**Diagnosis**: The chat-special tokens (`<|user_start|>`, `<|assistant_start|>`, etc., ids 32760–32767) were never observed during pretraining. Their wte
embeddings are still at random init (std 0.8) while real tokens are at
trained magnitudes (std 5.5). When SFT loss tries to predict them from
random embeddings, gradients explode.

`<|bos|>` (id 32759) IS well-trained because it's used as a doc separator
during pretraining.

## Run 5: d12 SFT with embedding patch — **in progress**

**Pre-step**: `tools/patch_specials.py` (see `/tmp/patch_specials.py`):
- Copy `wte[<|bos|>]` row into the other 8 special-token rows.
- Same for `lm_head` rows.
- Same for each `value_embeds[layer]` table at VE layers.

This bootstraps the special-token embeddings to a trained magnitude
matching `<|bos|>` (which was actually pretrained), so initial logits
aren't garbage.

**Then SFT** with conservative LRs (see `/tmp/d12p_sft2.sh`):

```
--embedding-lr=0.03 --unembedding-lr=0.0008 --matrix-lr=0.002
--init-lr-frac=0.05 --warmup-ratio=0.1
```

(In-progress at the time of writing.)

## Run 6: d6 Chinchilla — **alternative shipped**

**Goal**: a properly-trained tiny model — same 30M params as d6, but
trained at Chinchilla-optimal data:param = 20 instead of the original's
2.7 ratio. Hypothesis: more tokens of training will close most of the
quality gap to d12, while staying small enough to be markedly faster
on the Pentium 4.

**Config** (`scripts/train-d6-chinchilla.sh`):

```
--depth=6 --head-dim=64 --window-pattern=L
--max-seq-len=512 --device-batch-size=32 --total-batch-size=16384
--target-param-data-ratio=20
```

nanochat computed `num_iterations=28320` from the ratio (~5.7× the
original d6 run).

**Numbers**:
- Total params: **30M** (same as Run 1).
- Tokens trained: 16384 × 28320 = **464M tokens** (~5.7× more than Run 1).
- Wall clock: **9.1 minutes**.
- Final val_bpb: **1.075** (vs. Run 1's 1.165 — ~8 % bpb improvement).
- Sampled outputs (still rambly — d6 size, not training, is the cap):
  - "The capital of France is the capital of the world..."
  - "The chemical symbol of gold is gold. It is a metal that is used in the manufacture of jewelry..."

**Export**: 75 MB int8 NCB1 file via `tools/export_ncb.py --int8`.
Same byte size as Run 1 — quantization is per-row.

**XP performance**: 36.5 ms/forward, ~**27 tok/s** — **5.85× faster
than d12** (which is at 213.6 ms/forward = 4.68 tok/s). Linear is
685 µs/call vs d12's 2610 µs/call (smaller `n_embd` and fewer calls).

**Status**: shipped as `MODEL_D6.NCB` on the XP machine alongside
`MODEL.NCB` (d12). User can choose which to load — d6 is faster but
less coherent on free-form prompts; d12 is slower but produces fuller
multi-sentence answers.

## Future runs to consider

- **d20** for "real GPT-2-grade" coherence. Would take ~2 hours on the RTX 6000.
  280 MB → ~700 MB int8. Would NOT fit on the Dell's 512 MB RAM.
- **d8 / d10 at Chinchilla ratio 20** — somewhere between d6's speed
  and d12's coherence.
- **Smaller seq_len**: 256 or even 128 for production deployment to XP,
  saves KV cache memory.

## Lessons

- **For tiny demo tier on a laptop**, use `runs/runcpu.sh` config (d6, 30 min
  on M3 Max). For real coherence, d12+ and ~50 min on a Blackwell.
- **The default nanochat SFT pipeline NaN's on pretrained checkpoints
  whose chat-special-token embeddings are at random init.** This isn't
  documented anywhere in the repo as far as I can tell. The fix is to
  pre-initialize those embeddings (copy `<|bos|>`'s trained values).
- **`int8` per-row quantization** preserves quality essentially perfectly
  on these small models — both d6 and d12 produce identical greedy output
  fp32 vs int8 in our spot-checks.
- **CPU inference scales badly with model size on the P4** — d6 (30M) at
  3 tok/s, d12 (110M) at ~0.7 tok/s. The bottleneck is memory bandwidth
  on the DDR-400 RAM, not compute. Larger models won't get faster with
  any reasonable optimization on this hardware.
