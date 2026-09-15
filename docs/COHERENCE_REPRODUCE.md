# Reproducing the XP coherence candidate

The training machine is a Linux host with an RTX PRO 6000 Blackwell GPU.
Inference uses the separate native C engine and requires no Python on XP.
Final model selection, measurements and limitations belong in the accompanying
evaluation report and model card; a successful training run alone is not acceptance.

## Inputs and environment

Use Python 3.10, PyTorch 2.9.1+cu128 and `requirements-coherence.txt`.
The actual run used an isolated environment and local, verified model files.
Never run training on the target XP machine.

Pinned publisher inputs:

- `LiquidAI/LFM2.5-350M`, revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
  Publisher weight SHA256:
  `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`.
- `HuggingFaceTB/smol-smoltalk`, revision
  `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`.

LFM's publisher tokenizer metadata targets a newer Transformers loader. The
scripts explicitly use `PreTrainedTokenizerFast` with an empty dictionary for
`extra_special_tokens` under Transformers 4.57.6. Publisher files are preserved;
no remote Python code is trusted or executed.

## Selected training run

Replace the capitalized paths below with your local verified files. Choose new
output directories. Data preparation and training reject output collisions;
the exporter can overwrite files, so always give it a fresh directory.

```sh
python tools/prepare_coherence_data.py \
  --public-data PUBLIC_PARQUET_DATA --tokenizer LFM_MODEL \
  --tokenizer-id LiquidAI/LFM2.5-350M \
  --tokenizer-revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f \
  --out PREPARED_DATA
python tools/run_candidate.py \
  --model LFM_MODEL --model-id LiquidAI/LFM2.5-350M \
  --revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f \
  --prepared-data PREPARED_DATA --run NEW_SELECTED_RUN \
  --batch-size 32 --accumulation 2 --epochs 1 --rank 16 \
  --learning-rate 2e-5 --kl-weight 1.0 \
  --targets q_proj,v_proj,out_proj
```

This is the selected `lfm25-anchored` recipe. It has 26,430 training rows and
1,460 validation rows; the original run selected checkpoint 413. Its exact
executed source and manifests are preserved in the evidence archive. The
merged directory includes the saved tokenizer with the publisher's vocabulary
and template. Export into a fresh directory:

```sh
python tools/export_slm.py --model NEW_SELECTED_RUN/training/merged \
  --out NEW_NATIVE_EXPORT --bits 6 --group 64 --context 512 --scale-method maxabs
```

Q6 uses FP32 activations automatically. The native tokenizer should match
the publisher-derived hash recorded in the model card. Fixed seeds preserve
the recipe; bitwise training reproducibility across hardware/software changes
is not guaranteed.

## Broader calibrated comparison (not selected)

The following independent data supplement and three-epoch comparison produced
different strengths and regressions. Its native coherence score was lower,
so it is preserved as an experiment rather than used in the selected package.
Use the same `PREPARED_DATA` created above and new directories for this run.

```sh
python tools/prepare_calibration_data.py --output CALIBRATION_DATA
python tools/prepare_calibrated_mix.py \
  --base PREPARED_DATA --calibration CALIBRATION_DATA --out MIXED_DATA
python tools/run_candidate.py \
  --model LFM_MODEL --model-id LiquidAI/LFM2.5-350M \
  --revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f \
  --prepared-data MIXED_DATA --run NEW_RUN \
  --batch-size 32 --accumulation 2 --epochs 3 --rank 32 \
  --learning-rate 2e-5 --kl-weight 2.0 \
  --targets q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3
```

The mixture has 20,564 unique training dialogues: 14,420 public retention rows,
4,096 earlier procedural rows and 2,048 new calibration rows. Four explicit
copies of each new calibration row give 26,708 training rows per epoch.
The 1,780 validation rows are not repeated. The resulting training file has
5,683,116 rendered tokens and 2,394,319 supervised assistant tokens per epoch.

Public rows use retention KL against the unchanged foundation. Corrective rows
explicitly disable that anchor so the adapter can learn answers the base model
does not already produce. The selected checkpoint minimizes the adaptation
validation objective. The independent evaluation suites are never training inputs.

`state.json`, `training/training_manifest.json`, `training/progress.jsonl`,
`training/completed.json`, the adapter and merged weights preserve each run.
The peak GPU field measures allocated training tensors, not total VRAM, and
excludes initial validation and final export.

## XP build and validation

Use MinGW's `i686-w64-mingw32-gcc` and `windres`:

```sh
bash scripts/build-coherence-xp.sh
python tools/audit_xp_binary.py --help
python tools/export_slm.py --help
python tools/package_coherence.py --help
```

The build targets 32-bit PE, subsystem 5.1, Pentium M/SSE2 and the legacy
MSVCRT runtime. The native tests cover tokenizer parity, matrix kernels,
forward-pass parity, context resets, bounded memory and saved-note IPC.
GUI helper tests cover UTF-8 transport, control acknowledgments and settings.
`validate_wine_runtime.py` separately checks the real PE backend under an
isolated Wine prefix. Wine does not establish physical XP startup, laptop
memory pressure, sustained response speed or GUI usability.

Preserve publisher licenses and modification notices in every packaged model.
The LFM Open License is distinct from the Apache-2.0 dataset license.
