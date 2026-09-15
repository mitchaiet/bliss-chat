# Bliss coherence adaptation recipe

The main improvement should come from replacing the undertrained nanochat base
with an established small instruction model. This adaptation teaches concise
responses and use of conversation/notes while preserving that model's general
conversation ability. Improvement over the unchanged instruction checkpoint
must be measured; fine-tuning can also make it worse.

## Sources and output

Run `tools/prepare_coherence_data.py` with Python containing `pyarrow`,
`tokenizers`, and `jinja2`. The defaults read already cached files on mitch-xpt:

```sh
python tools/prepare_coherence_data.py \
  --out /home/mitch/bliss-runs/20260915-next/coherence-data
```

The script writes `train.jsonl`, `val.jsonl`, and `manifest.json`. Every record
contains `messages`, source provenance, a content hash, and rendered token
counts. The manifest hashes the generator, all input parquet files, tokenizer
files, and output JSONL files. It refuses a nonempty output directory.

Public retention data is [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, published under Apache-2.0.
Its authors used it to train SmolLM2-135M/360M-Instruct. It is retention material,
not new proof of general knowledge. Keep its upstream license/attribution with
any redistributed derived dataset.

Default selection requests 16,000 public training conversations and 800 public
test conversations. Public quotas by examples are 65% smol-magpie-ultra-short,
10% everyday-conversations, 10% smol-summarize-20k, 10% smollm-rewrite-30k, and
5% smol-contraints (the upstream spelling). A public row may supply a complete
prefix of up to four assistant turns. No assistant response is clipped, rewritten,
or stripped to ASCII. Each assistant turn has at most 192 tokens, and the entire
rendered conversation fits 512 tokens. Unfilled source quotas are reported.
The build fails if fewer than 70% of the requested public training rows remain.

New procedural examples make up approximately 25% of rendered tokens, computed
after public selection. They cover distractor-resistant recall, corrections,
selective notes, admitting missing supplied facts, addition, subtraction,
ordering, and simple stated rules. Arithmetic uses the same counted item/unit
throughout. Facts are supplied in each example; no invented claim is presented
as outside knowledge. The generator reads neither the legacy benchmarks nor
the independent evaluation suite. Legacy Bliss curated datasets are excluded.

## Split guarantees and limits

Public train/test retain their upstream split. Exact first-user prompt groups
appearing in eligible public test rows are excluded from training, and exact
conversation duplicates are rejected. This blocks straightforward local
contamination. It does not establish that the upstream pretrained model never
saw the public test split or semantically similar examples.

Procedural training and validation use separate name, place, item, and numeric
operand pools, and different prompt wording families. Task types intentionally
overlap, so procedural validation measures transfer to new entities/phrasings.
It does not independently establish broader reasoning skill. The final sealed
evaluation must remain separate from both splits and from checkpoint selection.

Public task-specific system instructions are retained. Examples without them
and all procedural examples receive the explicit Bliss system message saved in
the manifest. Native SmolLM2 chat formatting is retained. Do not train this model
on nanochat's plain `Q:/A:` or interchange token IDs from the old tokenizer.

## Conservative training proposal

Use SmolLM2-360M-Instruct revision
`a10cc1512eabd3dde888204e902eca88bddb4951` as the frozen bf16 backbone. On a 96GB
GPU, ordinary LoRA avoids the unnecessary precision/runtime complexity of
4-bit training. Quantization belongs after merging the selected adapter.

- LoRA rank 16, alpha 32, dropout 0.05, no trainable bias.
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`; freeze embeddings, final head, and normalization weights.
- AdamW learning rate `5e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight
  decay `0.01`; maximum gradient norm `0.5`; warm up for 5% of optimizer steps
  and cosine-decay to zero.
- One pass only initially. Batch 16 conversations, accumulation 4 (64
  conversations per optimizer step), context 512, dynamic padding and attention
  masks; no cross-conversation packing. Effective steps are
  `ceil(training_rows / 64)`, normally a few hundred. The produced manifest,
  not this estimate, supplies actual token totals.
- Use assistant-only cross entropy, including the genuine assistant end token.
  Compute masks from the native chat boundaries, verify several decoded masks,
  and never use the reporting `assistant_token_count` to create labels. Padding
  labels are `-100`; attention masks distinguish padding even if EOS doubles as
  the pad token. Reject an example with no supervised target.
- Evaluate before training and every 50 optimizer steps; save at those points.
  Report public-retention and procedural losses separately. Stop on nonfinite
  loss/gradients or a clear sustained regression; do not update through NaNs.
- Keep the unchanged model as a selection candidate. Choose using held-out
  adaptation loss plus a separate development behavior set. Run the final sealed
  suite only after selection, and compare the exact quantized runtime artifact
  against the untuned quantized baseline at identical generation settings.

The response cap and 512-token training context are practical XP constraints.
The native port still needs its own tokenizer parity, logits, generation,
memory, SSE2, and actual XP hardware checks. Successful GPU training alone does
not prove these.
