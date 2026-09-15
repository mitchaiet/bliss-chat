# Native XP local workspace (2.0.0-rc.2)

This update keeps the RC1 LFM2.5-350M Q6 weights, tokenizer and SSE2 inference
executable byte-for-byte. It changes the Windows front end and document
retrieval. **The application remains fully local. No internet lookup, network
helper, server, account or additional model is required.**

## Native interface

The app uses the installed Windows theme: XP Luna when visual styles are
enabled, or Windows Classic when they are disabled. Send/Stop and reply actions
are standard Win32 buttons. A compact common-controls toolbar, Tahoma/system
fonts, native menus, common file picker and dialogs replace custom drawing.
Diagnostics are available under **View > Show Diagnostics**. The conversation
has more space and the default window is clamped to the screen's work area;
800x600 is a supported layout target at normal 96-DPI scaling.

**Enter** sends; **Shift+Enter** inserts a new line; **Escape** stops a reply.
Pressing Enter during generation preserves the next draft. **Ctrl+D** opens
Documents and **Ctrl+M** opens Memories. Existing save, search, regenerate,
edit-last-prompt and SAPI read-aloud features remain available.

## Local documents (lightweight RAG)

RAG means finding relevant reference text and supplying it with a question.
Open **Documents**, import a file, and optionally preview a question's matching
passages. The importer copies the original and never overwrites an existing
library filename. Select a document to inspect it or remove the library copy.

The persistent library is `%APPDATA%\bliss-chat\Knowledge`. This fixes the
portable launcher's former temporary-folder problem. An older `Knowledge`
folder beside the executable is still searched as a compatibility fallback;
its files are managed in Explorer, and are not listed as imported library files.

- Supported contents: UTF-8/ASCII `.txt`, `.md`, `.html`, `.htm`.
- Maximum: 128 files scanned per question, 64 KiB per file. Larger, binary and
  invalid UTF-8 files are excluded. Subfolders, PDF and Word are not parsed.
- Matching uses distinct whole words and overlapping passages, avoiding hits
  such as `cat` inside `concatenate`. It works best with English search words.
- At most two 240-byte snippets are supplied. The augmented prompt is capped
  at 1,023 bytes; a long question may leave room for fewer or no snippets.
  The runtime still enforces its separate 512-token context limit.
- The preview uses the same retrieval code. Replies list the filenames supplied
  as sources. This identifies context, not proof that the model used it correctly.
- **Use documents in chats** persists between launches. Turning it off affects
  future retrieval; start a new chat to discard previously supplied context.

No embeddings model or vector database runs on the laptop. This intentionally
bounded lexical search costs little memory, but is not semantic search across
large collections. HTML handling removes tags; it is not a full browser/parser.
Filesystem APIs retain the existing Windows ANSI filename limitation: use
filenames representable in the machine's code page. Chat and document contents
are UTF-8.

## Memories

**Memories** lists the actual backend notes. Add a brief fact or select one to
forget. **Remember...** beside a reply opens this dialog with the last user
message for review; it no longer silently saves a truncated message.

The backend stores up to 12 notes, each under 160 UTF-8 bytes, in
`%APPDATA%\bliss-chat\MEMORY.TXT`. It automatically supplies up to three matching
notes when relevant. Add/forget operations use the existing backend protocol
and atomic save path, so the UI does not edit an out-of-date second copy.
The dialog waits for command acknowledgements and shows persistence errors.
Forgetting a note does not erase existing chat files or already supplied model
context; start a new chat when that matters. To change a fact, forget the old
note and add the replacement.

## Build and evidence

```sh
bash scripts/build-coherence-gui-xp.sh
python3 -m unittest discover -s tests -p 'test_xpchat_*.py'
python3 -m unittest discover -s tests -p test_coherence_config_source.py
```

The GUI-only build preserves the frozen inference executable. The normal
`build-coherence-xp.sh` remains available for building both components.
The release includes executable/import checks and a GUI validation receipt.
Native dialogs and real model inference were exercised under isolated Wine
with an XP-configured prefix and a headless display. This is development-host
validation, **not physical XP hardware validation**. Luna appearance, actual
laptop speed, high-DPI layouts and total XP/GUI/model memory pressure still
require a run on the target machine. The underlying model's RC1 evaluation and
known reasoning limitations remain applicable; no new training is claimed.
