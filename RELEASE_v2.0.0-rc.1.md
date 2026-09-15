# Bliss Chat XP v2.0.0-rc.1

A new offline conversation model, native SSE2 inference engine and Unicode
chat interface, packaged as one portable Windows EXE.

## Download and run

Download **bliss-chat-xp-v2.0.0-rc.1-portable.exe**, place it in a convenient
folder and double-click. It extracts the model and application into a temporary
folder and launches Bliss Chat. Keep the EXE for future launches. Extraction
can take time on an old hard drive; the progress window stays visible.
The folder ZIP is also available for users who prefer to extract once and run
`XPCHAT.EXE` directly. No Python, GPU, account or internet is required on XP.

The intended target is Pentium M/SSE2, 512 MB RAM, Windows XP SP2/SP3. This is
a **release candidate**: physical XP startup, complete system memory pressure
and response speed have not been verified. Keep v1.3.0 available while testing.
The executable is not Authenticode-signed.

## What changed

- Adapted LiquidAI LFM2.5-350M foundation with native six-bit weights, replacing
  the previous model in this package. Model weights: 274.87 MiB.
- Native C inference for attention and short-convolution layers, accurate
  tokenization, bounded context and conversation replay.
- Unicode chat transport, corrected control acknowledgments and validated GUI
  settings. Context defaults to 512 tokens and answers to at most 128 tokens.
- Browser comparison tool in `apps/bliss-compare`, plus training, conversion,
  build and evaluation tools with pinned provenance.

## Results and remaining weaknesses

On a fresh 48-case comparison frozen after model selection:

| Measure | Previous Bliss | Bliss 2 Q6 |
|---|---:|---:|
| Correct final-answer content, reviewed | 7/48 | 26/48 |
| Combined relevance/correctness/coherence/instruction score | 3.458/8 | 6.562/8 |
| Coherence score | 1.583/2 | 1.979/2 |
| Mechanical checks | 3/48 | 24/48 |

The new candidate won 37 paired rubric comparisons, lost four and tied seven.
Scores came from a small AI-reviewed suite with masked model labels; the
reviewer had prior familiarity with the models. Content correctness does not
mean perfect formatting. These results compare complete deployment backends,
not the isolated effect of adaptation.

Arithmetic and logic remain weak: **both models got only 1/8 fully correct**.
The later ten-question demonstration also exposed regressions: the new model
answered 17+26 as 39, invented a cat's name, and wrongly claimed local language
models require internet access. Those outputs are included without correction
in `bench/coherence/ten-question-demonstration.json`.

Median native Linux response time increased from 0.692 to 4.575 seconds,
**6.61 times slower**, on the modern workstation. This is not Pentium M timing.
The selected backend completed a full 512-token forward pass under a 384 MiB
virtual-memory limit, with about 294.6 MiB peak Linux RSS, excluding the OS/GUI.
The Windows backend passed separate Wine checks; physical XP remains untested.

## Package identity and validation

RC1 wraps the exact tested `2.0.0-candidate.20260915` GUI, backend and model.
Their embedded version strings are preserved so validation remains applicable.
The portable wrapper identifies itself as `2.0.0-rc.1`; no weights changed for
this release. The wrapper uses an 8 MiB LZMA dictionary for extraction.

See the attached `release-validation.json`, `SHA256SUMS.txt`, model card and
full evidence archive. The model, tokenizer and every packaged executable are
checked against the frozen selected artifact hashes. The installer payload is
extracted independently and compared byte-for-byte with the verified folder.
Static PE checks and Wine evidence do not establish physical XP behavior.

The EXE also supports extraction without launch, into a **new** directory:

```
bliss-chat-xp-v2.0.0-rc.1-portable.exe /S --extract-only /D=C:\Bliss2-RC1
```

`/D` must be the last argument. An existing nonempty directory is rejected.

## License and reproduction

The model uses **LFM Open License v1.0**, including its commercial-use revenue
condition. Exact terms and modification notices are included in the EXE and
folder ZIP. Public adaptation data uses Apache-2.0. GPT-6 Astra assisted the
engineering; its weights are not included.

See `MODEL_CARD.md`, `docs/COHERENCE_REPRODUCE.md`,
`bench/coherence/FINAL_ACCEPTANCE_REVIEW.md` and the evidence archive for the
training recipe, candidate comparisons, limitations and reproducibility data.
Older reports describe the pre-publication state; this release publishes that
preserved work with the browser tool and portable packaging added.
