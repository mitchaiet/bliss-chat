# Bliss Chat 2.0.0-rc.2 — native XP, fully local

RC2 makes the Windows application easier to use with local documents and saved
memories. The model, tokenizer and SSE2 inference backend are unchanged from
RC1. There is no internet lookup or remote helper.

## Changes

- Standard Windows buttons, compact toolbar, native dialogs and more space for
  conversation. Diagnostics are optional; the window fits an 800x600 desktop
  at normal scaling and follows the installed XP theme.
- **Documents:** import local text/Markdown/HTML, inspect files, preview matching
  passages, remove library copies and toggle automatic retrieval. Files now
  persist in application data when using the self-extracting EXE.
- Retrieval scores whole words within passages, supplies up to two small
  snippets and identifies source filenames. Limits remain modest for 512 MB:
  128 files scanned, 64 KiB each; no extra model or vector database.
- **Memories:** inspect, add and forget actual persistent backend notes.
  Remember opens a review dialog instead of silently truncating a message.
- Keyboard navigation and Ctrl+D/Ctrl+M shortcuts; Enter during generation
  preserves the next draft.

## Download and use

Run the portable EXE or extract the entire folder ZIP and launch `XPCHAT.EXE`.
An optional small GUI-update ZIP is provided for existing RC1 folder installs:
close Bliss, keep a backup of the old `XPCHAT.EXE`, and replace that one file.
For the RC1 single-file launcher, download the new portable EXE instead.

Chats, memories and imported documents live in `%APPDATA%\bliss-chat`. The
portable EXE only extracts program/model files into temporary storage. The
original files you import are preserved. Documents disabled or notes forgotten
may still appear in earlier chat context; start a new chat to clear that context.

See [Local workspace guide](docs/XP_LOCAL_WORKSPACE.md) for supported text
encodings, limits, shortcuts and memory behavior.

## Validation and limits

The release includes a GUI validation receipt and a PE/import/CPU audit. Checks
cover UTF-8 boundaries, file limits, retrieval, existing GUI transport/settings,
and native dialogs with real model inference under an isolated Wine XP prefix.
Physical XP startup, Luna appearance, laptop speed and combined OS/application
RAM pressure remain unverified. This remains a prerelease; v1.3.0 remains the
latest stable release.

The model's measured gains and failures are unchanged; see the
[RC1 evaluation](RELEASE_v2.0.0-rc.1.md). Local documents and memories supply
context, but cannot guarantee factual answers or fix the model's reasoning.
