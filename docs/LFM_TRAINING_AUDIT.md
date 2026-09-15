# LFM2.5 training compatibility audit

The CPU-only audit passed against `train_coherence.py` SHA256
`1c7ed07b5c0c6e8dcbb5e9e65c8fc2de793a51a3dede039f3c1fd2313fb9e2f5`.
Full results, input hashes, and dependency versions are in
`LFM_TRAINING_AUDIT.json`.

- The actual training-main tokenizer loader worked with the publisher's
  TokenizersBackend metadata and the `extra_special_tokens={}` compatibility
  override under transformers 4.57.6.
- A deterministic sample of 340 training rows, drawn across every source,
  matched both exact token prefixes and the official template's assistant
  generation masks: 452 assistant boundaries and 32,540 supervised tokens.
- All 26,471 training rows were scanned for thinking/tool/continuation fields
  that could make this template alter earlier assistant content. None occurred.
  Such content in future data requires another audit or explicit rejection;
  the result does not establish prefix invariance for arbitrary conversations.
- A 22,832-parameter, randomly initialized LFM model with one convolution layer
  and one attention layer exercised `q_proj,v_proj,out_proj` PEFT adapters and
  the actual `FiniteTrainer.compute_loss`. Combined CE/KL and KL-only backward
  passes produced finite nonzero gradients in all eight adapter tensors.
  Frozen base parameters had no gradients, and disabling adapters recovered
  exactly unchanged teacher logits.
- The independently computed next-token CE plus sampled assistant-position KL
  matched the training loss exactly. An intentionally wrong teacher shift gave
  a different KL. The 128-position sampling branch was exercised.
- Merging the tiny adapters changed logits by at most `1.79e-7` in FP32.
  CUDA was not initialized. No trained weights or evaluation cases were read.

## Reproduce

With the repository checked out beneath the run directory on the training host:

```sh
/home/mitch/bliss-runs/20260915-next/venv/bin/python \
  tests/test_lfm_training_compatibility.py \
  --model-dir /home/mitch/bliss-runs/20260915-next/models/lfm25-350m \
  --training-script tools/train_coherence.py \
  --train-jsonl /home/mitch/bliss-runs/20260915-next/coherence-data/train.jsonl \
  --sample-per-source 64
```

Run from the repository root. Optional `--output new-report.json` refuses to
overwrite an existing report. The test hides CUDA, uses CPU tensors, and never
loads a pretrained model. It checks training mechanics, not BF16 GPU stability,
the full pretrained model's quality, native inference, or Windows XP behavior.
