# Side-by-side Bliss chat

One browser input sends each message to both native models. Answers stream
independently, and each model retains its own conversation until New chat.
This is a modern browser tool for a Linux model host, not part of the XP EXE.

## Run on the model host

Python 3.10+ and the native executables are required. Set `BLISS_RUN_ROOT` to
the preserved training-run directory containing:

```
artifacts/nc_run_native
artifacts/MODEL.NCB
artifacts/TOKENIZER.NCT
native/build/slm/variants/128699c1e02a1f78/slm_run
native/exports/lfm350-trained-q6/MODEL.SLM
native/exports/lfm350-trained-q6/TOKENIZER.SLT
```

The exported `.SLM` and `.SLT` files are the same bytes as the release's
`MODEL.NCB` and `TOKENIZER.NCT`, with native format identification in the headers.
The original model is available from the v1.3.0 release. Build the appropriate
Linux backends from their corresponding source revisions. The original runner
source is retained in this repository; new inference is `src/slm_run.c`.

```sh
BLISS_RUN_ROOT=/path/to/run python3 apps/bliss-compare/server.py
```

The service binds only to `127.0.0.1:8791`. For a remote host, create a private
SSH tunnel from the browsing computer (replace `USER@HOST`):

```sh
ssh -N -L 127.0.0.1:8877:127.0.0.1:8791 USER@HOST
```

Open `http://127.0.0.1:8877/`. When browsing on the model host itself, open
`http://127.0.0.1:8791/`. No extra web packages are needed.

Enter sends; Shift+Enter adds a line break. Each model uses its own prompt and
stopping defaults, context 512, temperature zero, seed 42 and a 128-token reply
limit. Refresh starts a new browser conversation; idle sessions expire after
30 minutes. No transcripts are deliberately saved. Temporary notes and backend
stderr are cleaned up on reset/expiry. Model replies render as text.

`verification.json` records the live browser smoke test, including follow-up
responses, Enter-to-send and resetting both panels. It measures interface
operation, not answer accuracy. This tool does not start or replace a GPU server.
