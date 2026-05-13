# Bliss Chat XP v1.1.0

Model version: bliss-d12-curated-c20-v1

This release packages the Windows XP chat app with the first explicitly versioned Bliss-native model.

## Included files

- XPCHAT.EXE — Windows XP GUI chat app
- NC_RUN.EXE — Windows XP console/runtime backend
- MODEL.NCB — int8 Bliss model export
- TOKENIZER.NCT — tokenizer export
- MODEL_VERSION.txt — model identity and checksums

## Model notes

- Trained from the start with plain Q:/A: style data mixed into base training.
- Avoids a separate unstable special-token SFT path.
- Training completed 33,600 steps.
- Final validation bpb: 0.818168.

## Install on Windows XP

Copy all included files into:

C:\xp-llm\

Launch XPCHAT.EXE from that folder or from a shortcut pointing there.
