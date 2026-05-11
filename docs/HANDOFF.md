# Bliss Chat — handoff

Snapshot for whoever is picking this up cold. Captures **what's live**,
**what's running**, **what's tried**, and **what's worth trying next**.
Written 2026-05-10 22:33 local.

Pair this with `context/00-overview.md` (project pitch + arch),
`context/02-inference-engine.md` (NC_RUN internals), and
`context/10-known-issues.md` (the SFT graveyard with hypothesis log).

---

## 1. What this project is, in one paragraph

**Bliss Chat** is a real chat LLM that runs natively on a Pentium 4
(2004 Dell Dimension 4700, 512 MB RAM, Windows XP SP3). Custom
nanochat-architecture transformer (~110 M params) trained on an RTX
6000 Blackwell, exported to a custom int8 binary format, served by a
hand-written ~1500 LOC C99 inference engine with SSE2 SIMD, fronted
by a Win32 GUI. **~4.7 tok/s on the P4** (vs llama.cpp's 0.04 tok/s on
the same box). End-to-end local: no internet, no emulation.

Public: <https://github.com/mitchaiet/bliss-chat>. Two portable
self-extracting EXEs in the v1.0.0 release.

---

## 2. Live state (as of 2026-05-10 22:33)

| Surface | State |
|---|---|
| GitHub repo `master` | `27050b7` ("chore: gitignore .claude/ worktree cache") |
| GitHub release `v1.0.0` | Live, both portable EXEs current |
| XP machine (`192.168.1.31`, `xpt`/`xpt-dimension`) | Latest XPCHAT.EXE + NC_RUN.EXE deployed, desktop shortcut "Bliss Chat" with new icon |
| GPU box (`100.106.68.91`, ssh `mitch`/`experiential`) | **`systemd --user` unit `d12-chinchilla.service` is RUNNING**, step 9611/33600 (28.6%), ETA 2 h 9 min, finishes ~00:42 |
| Working tree | clean |

`mitch@100.106.68.91` SSH has flaked on Tailscale auth a few times during this session — if it hangs, just retry; do **not** modify the user's tailscale config.

---

## 3. What's shipped (feature inventory)

The repo log is the authoritative answer. Highlights:

### Performance (CPU)
- SSE2 matmul (fp32 + int8) — 5.45× faster `linear()`
- SSE2 attention helpers (dot, axpy) — 2.6× faster softmax block
- Cumulative on Pentium 4: **0.88 → 4.68 tok/s** (5.33×)
- Profiler instrumentation via `QueryPerformanceCounter`; `[prof turn]` lines in stderr

### Model files on disk
- `MODEL.NCB`     d12 (110 M, web data, ratio ~6), 280 MB int8, deployed default
- `MODEL_D6.NCB`  d6 chinchilla (30 M, ratio 20), 75 MB int8 — **~27 tok/s on P4**, less coherent
- New: **d12 chinchilla** (110 M, ratio 20) — training right now

### Backend (`src/nc_run.c`)
- Multi-turn KV cache, `/reset` slash command, auto-reset near `ctx_max`
- Slash commands: `/help`, `/info`, `/reset`, `/temp <f>`, `/seed <int>`, `/topp <f>` (stored but sampler is still temp-only), `/maxtok <int>`, `/system <text-with-\n-escapes>` (rewires few-shot prefix + re-prefills)
- Sentinel I/O on stdout: `\x01READY\n`, `\x01INFO ...\n`, `\x01PROG N\n`, `\x01EOT <count>\n`, `\x01ERR ...\n`
- Sentinel I/O on stdin: line-of-text turns, `\x01STOP\n` for mid-turn abort
- Few-shot prefix (4 examples now: capital/list/math/joke) prefilled once, snapshotted, restored each turn
- Hold-back buffer strips trailing "\nQ:" / "\nA:" stop patterns from emitted output

### GUI (`src/xpchat.c` ~ 2200 LOC, Win32)
- COMCTL32 v6 manifest embedded (Luna theme; green segmented progress bar)
- Top: title strip with app icon (Bliss hill + chat bubble), name, host PC specs, model description
- XP Explorer-style toolbar with 24×24 icons + labels (New Chat / Save / Stop / Settings / Help)
- Per-message action row: Copy last reply / Regenerate / Edit last prompt
- Left sidebar with chat list, search filter (cue banner), right-click context menu (Rename / Delete / Save Transcript), F2 to rename, Del to delete
- RichEdit transcript with sticky-bottom auto-scroll, timestamped headers ("You — 8:26 PM")
- Settings dialog (modal): Sampling group (temp, seed, 3 presets) + Advanced group (top-p, max tokens, multi-line system prompt) + Reset All Settings
- Edit menu: Copy / Paste / Select All / Find (Ctrl+C/V/A/F, accelerator table in WinMain)
- Help menu: About / Keyboard Shortcuts (modal) / Slash Commands (sends `/help`)
- Window position + size persisted to `HKCU\Software\bliss-chat\Window`
- Settings persisted to `HKCU\Software\bliss-chat\Settings` (TempMilli, Seed, TopPMilli, MaxTok, SystemPrompt)
- Chat history persisted to `%APPDATA%\bliss-chat\chats\<unix_ts>.txt` with `TITLE: ...` header
- Esc = Stop, Enter = Send, Shift+Enter = newline, Ctrl+N = New Chat, Ctrl+S = Save Transcript
- Live tok/s estimate in status bar (chars / 4 / elapsed) during generation

### Tooling
- `bench/` — 100-question coherence benchmark + scorer. Baseline: **56 % correct, 77 % coherent, 100 % format** on the deployed d12. Run with `python3 bench/run_bench.py && python3 bench/score.py`.
- `scripts/build-xp.sh` — Linux mingw cross-compile to two EXEs + NSIS portable builds
- `scripts/push-to-xp.sh` — telnet+HTTP push (configure via env vars; never write a password into the script)
- `scripts/train-{d6,d12,d6-chinchilla,d8-chinchilla,d12-chinchilla}.sh` — every training recipe used
- `scripts/sft-fp32.sh` + `scripts/sft-adamw.sh` — SFT graveyard (both NaN; see § 4)
- `assets/make_icon.py` — generates `bliss_chat.ico` (BMP-format multi-res ICO; **do not use Pillow's default ICO writer**, XP can't read its PNG-in-ICO output)
- `assets/make_toolbar_icons.py` — generates the toolbar BMP strip with magenta transparency key
- `installer/portable.nsi` — single-EXE self-extract installer

---

## 4. The SFT graveyard — what we tried and why it didn't work

**Symptom**: Training NaN'd by step ~8 across every variant.

Hypotheses tested:
1. ❌ Default LR — NaN at step 3
2. ❌ 10× lower LR — NaN
3. ❌ 30× lower LR — NaN
4. ❌ Cold-start optimizer (`--load-optimizer=0`) — NaN
5. ❌ Identity-only data — NaN
6. ❌ `--init-lr-frac=0.05 --warmup-ratio=0.2` — NaN
7. ❌ Pre-init special-token embeddings from `<bos>` — still NaN
8. ❌ `NANOCHAT_DTYPE=float32` (no bf16 overflow) — NaN at step 88 (later but terminal)
9. ❌ Grad-finite check + manual `mul_` clip + skip-on-NaN — clean for 7 steps (loss 1.32 → 2.33), then NaN at step 8
10. ❌ **Vanilla `torch.optim.AdamW` + grad-finite + clip** — byte-identical 1.32 → 2.31 → NaN trajectory

The byte-identical AdamW/Muon trajectories **rule out the optimizer** as the cause.

**Current working theory** (see `context/10-known-issues.md`): the `d12_patched` checkpoint copies `<bos>`'s row into all 9 chat-special slots, so `WTE[user_start] ≡ WTE[bos] ≡ WTE[assistant_start] ≡ …`. The model literally cannot distinguish them; gradients update them in lockstep; embedding space collapses after a handful of updates. The 7-step run-up is the model "learning to be worse" predictably.

**Next things worth trying** (in priority order):
1. **Re-init the special-token rows with random orthogonal vectors** (give them a chance to differentiate)
2. Pretrain a fresh d12 from scratch on chat-formatted text — no SFT needed
3. LoRA / adapter — freeze every base param, only learn a small adapter on chat data
4. Inspect per-position cross-entropy at step 1 to confirm which positions drive the climbing loss

---

## 5. Open threads right now

| Thread | Owner | State |
|---|---|---|
| `d12-chinchilla` training | `systemd-run --user` on GPU box | 28.6 % done, ETA ~00:42 local |
| Three abandoned agent worktrees | `.claude/worktrees/agent-*` | Already merged via cherry-pick; locked by agent PIDs; will auto-clean. `git worktree list` shows them. |
| `nanochat` repo on GPU box | `~/nanochat/scripts/chat_sft.py` | **Reverted to original** (the v2 / v3 SFT patches were applied in-place and rolled back via `cp /tmp/chat_sft.py.orig`). Untouched on disk. |

### When d12-chinchilla finishes (~00:42)

```bash
# On GPU box:
ssh mitch@100.106.68.91 'cd ~/nanochat && source .venv/bin/activate && \
  python ~/export_ncb.py --src ~/.cache/nanochat/base_checkpoints/d12 \
                         --out ~/d12-chinchilla.ncb --int8'

# On Mac:
scp mitch@100.106.68.91:~/d12-chinchilla.ncb /tmp/
cp /tmp/d12-chinchilla.ncb build/deploy/MODEL.NCB   # replaces deployed d12

# Re-bench:
python3 bench/run_bench.py
python3 bench/score.py

# If numbers go up materially, push to XP via the usual flow.
# If not, the d12 size is the bottleneck — go to Section 6's "next moves".
```

There's a risk the chinchilla checkpoint **overwrites** the legacy d12 in `~/.cache/nanochat/base_checkpoints/d12/`. Confirm the model file you copy out is `model_033600.pt`, not the old `model_010000.pt`. The exporter defaults to highest step; should pick the new one.

---

## 6. Concrete next moves, ranked

1. **Bench the new d12-chinchilla** when training lands (1 h work after it finishes; deterministic value)
2. **Re-init special-token rows orthogonally and retry SFT** — the embedding-collapse theory is the most plausible un-ruled-out one. ~30 min of Python.
3. **Speculative decoding (d6 → d12 verify)** — 2–4× speed on d12, zero training risk. ~half a day in `nc_run.c`. Architecturally clean: d6 generates K tokens, d12 verifies in one batched forward, accept the longest matching prefix.
4. **LoRA adapter** for SFT — never touches base weights; loss landscape changes; might dodge the collapse pattern.
5. **Wire `top_p` sampling in `nc_run.c`** — backend already accepts `/topp` and persists it; sampler still ignores. Partial sort over top-K candidates needed; ~100 LOC.
6. **Train d14 or d16 chinchilla** — middle size class between d12 and d20. Estimate 400–500 MB int8, **just barely fits** in the Dell's 512 MB RAM. Worth a careful size check first.
7. **Stricter bench scoring** — current `correct%` rubric is keyword-presence only. "Fire is colder than ice" scores correct because "ice" appears. LLM-as-judge or per-question regex would tighten this before using the number for serious tracking.
8. **Launch / write-up** — explicitly DEFERRED at the user's request. Don't do this without permission.

---

## 7. Quirks and gotchas (the painful lessons)

**Cross-build is the only test surface for `xpchat.c`.** No POSIX build of the GUI. Run `bash scripts/build-xp.sh` to verify — it MUST be zero warnings (the build has been kept warning-clean and we want to keep that bar).

**clang on macOS warns about `'windows.h' file not found` in `xpchat.c`.** Pure false positive from the user's editor; ignore. The mingw cross is the real compile.

**Pillow's default ICO writer uses PNG-encoded entries which Windows XP cannot load** (PNG-in-ICO support is Vista+). `assets/make_icon.py` writes BMP-format entries by hand. Don't switch back to `img.save(..., format="ICO")`.

**Windows XP icon cache (`IconCache.db`) is sticky.** When the EXE's embedded icon changes, the desktop shortcut may keep the old icon until the cache is wiped AND explorer restarts. The fix sequence: kill `XPCHAT.EXE` / `NC_RUN.EXE`, delete the new shortcut, delete `IconCache.db`, force a logoff (`shutdown -l` from telnet) — user reconnects via RDP, fresh logon, explorer rebuilds cache. See `xp-deploy/make-shortcut.vbs`.

**`make-shortcut.vbs` cleans up `XP Tiny LLM.lnk` and `bliss-chat.lnk`** (legacy names) before creating `Bliss Chat.lnk`. Don't break that array.

**XP telnet sessions die after ~5 min idle.** Long-running NC_RUN.EXE invocations from expect will outlive the connection. Pattern used throughout: split commands across two telnet sessions — kick off the long job in one (let it disconnect), then reconnect to read result files.

**The HTTP server on `192.168.1.43:8088` serves from `build/deploy/`.** Past confusion: there were two stage dirs at one point. cwd of the http.server process is what matters; check with `lsof -p $(pgrep http.server) | grep cwd`.

**Filesystem / registry paths use lowercase `bliss-chat`** even though the display name is now "Bliss Chat":
- `%APPDATA%\bliss-chat\chats\`
- `HKCU\Software\bliss-chat\Settings`
- `HKCU\Software\bliss-chat\Window`
- Save-transcript filename pattern: `bliss-chat-YYYYMMDD-HHMMSS.txt`
- `C:\xp-llm\` on the XP machine itself (legacy path; do not migrate without good reason — users' existing chats live there)

**Resource IDs in `src/resource.rc` are sparse on purpose:**
- 200–207: Settings dialog and its base controls
- 210/211: Rename dialog
- 212/213: Edit prompt dialog
- 214–217: Settings v2 extras (top-p, max-tokens, system prompt, reset all)
- 220–222: Find dialog
- 230/231: Keyboard Shortcuts dialog
- Don't reuse 208–213 without checking; collisions caused real merge-conflict pain.

**Comctl32 v6 manifest is embedded.** `Edit_SetCueBannerText`, themed buttons, segmented progress bar, themed toolbar all rely on it. The manifest file is `assets/xpchat.manifest`, packed via `1 RT_MANIFEST` in `resource.rc`.

**The `nanochat` repo on the GPU box was patched in-place several times** during SFT attempts (`scripts/chat_sft.py`). The original was preserved at `/tmp/chat_sft.py.orig` and restored at session end. If `/tmp` is wiped, the original lives in the upstream nanochat repo.

**`tools/patch_specials.py` was referenced in `09-training-history.md` Run 5 but only ever existed at `/tmp/patch_specials.py`** on the GPU box. If you need to reproduce the `d12_patched` checkpoint, that script is gone — `d12_patched/` itself still has the snapshot but the production-recipe isn't in version control. Worth re-creating if you want to revisit.

---

## 8. File map

```
xp-llm/                           (repo root, GH = mitchaiet/bliss-chat)
├── README.md                     download links + perf table + slash commands
├── HANDOFF.md → docs/HANDOFF.md  ← this file
├── src/
│   ├── nc_run.c                  inference engine, ~1700 LOC
│   ├── nc_tokenizer.c            BPE + ASCII pre-tokenizer
│   ├── xpchat.c                  Win32 GUI, ~2200 LOC
│   └── resource.rc               icon + manifest + dialog templates
├── tools/
│   ├── export_ncb.py             PyTorch checkpoint → NCB1
│   └── export_tokenizer.py       tiktoken pickle → NCT1
├── server/
│   └── nc_dashboard.py           training progress dashboard
├── installer/
│   ├── portable.nsi              single-EXE self-extract recipe
│   ├── xp-tiny-llm.nsi           legacy install-style recipe
│   └── xp-tiny-llm-chat.nsi      legacy
├── assets/
│   ├── bliss_chat.ico            multi-res app icon (BMP-format)
│   ├── make_icon.py              ico generator
│   ├── toolbar_icons.bmp         5×24px toolbar strip
│   ├── make_toolbar_icons.py     toolbar bmp generator
│   └── xpchat.manifest           comctl32 v6 manifest
├── scripts/                      build + deploy + train
├── bench/                        100-Q coherence benchmark
├── context/                      deep architecture docs
│   ├── 00-overview.md
│   ├── 02-inference-engine.md
│   ├── 09-training-history.md
│   ├── 10-known-issues.md        SFT graveyard
│   └── 11-profiling.md           per-Pentium-4 cycle breakdown
└── docs/
    ├── ARCHITECTURE.md           pre-existing deep dive
    └── HANDOFF.md                this file
```

---

## 9. Environment + credentials cheat sheet

| Resource | Where | Auth |
|---|---|---|
| XP machine | `192.168.1.31` on LAN | telnet `xpt` / `xpt-dimension` |
| Mac (this box) | `192.168.1.43` on LAN | n/a (local) |
| HTTP server for XP push | `http://192.168.1.43:8088/` | none, LAN only |
| GPU box | `100.106.68.91` (Tailscale) | ssh `mitch` / `experiential` |
| Training dashboard | `http://100.106.68.91:8899/` | none |
| GitHub | `gh` CLI logged in as `mitchaiet` (also `experientialtech`, inactive) | persisted in keychain |
| nanochat checkpoints | `~/.cache/nanochat/base_checkpoints/d{6,12,12_patched}/` on GPU box | n/a |

Never commit creds. `scripts/push-to-xp.sh` requires `XP_HOST`, `XP_USER`, `XP_PASS`, `HOST_IP` env vars and refuses to default.

---

## 10. How to verify things still work

```bash
# Cross-build (must be zero warnings)
bash scripts/build-xp.sh

# Native sanity (POSIX build of backend only)
clang -O2 -std=c99 -o build/nc_run_native src/nc_run.c src/nc_tokenizer.c -lm
printf "What is the capital of France?\n" | \
  build/nc_run_native build/deploy/MODEL.NCB build/deploy/TOKENIZER.NCT -t 0 -s 7

# Coherence bench
python3 bench/run_bench.py && python3 bench/score.py

# Deploy to XP (binaries only)
cp build/NC_RUN.EXE build/XPCHAT.EXE build/deploy/
expect -c '<see end of scripts/push-to-xp.sh for the exact pattern>'

# Rebuild portable EXEs for release
cd installer
makensis -DMODEL=../build/deploy/MODEL_D6.NCB \
         -DOUTFILE=../build/bliss-chat-fast-portable.exe portable.nsi
makensis -DMODEL=../build/deploy/MODEL.NCB    \
         -DOUTFILE=../build/bliss-chat-portable.exe      portable.nsi

# Upload to GitHub
gh release upload v1.0.0 --clobber \
  build/bliss-chat-fast-portable.exe build/bliss-chat-portable.exe
```

---

## 11. Recently-completed iterations (last 5 commits)

```
27050b7 chore: gitignore .claude/ worktree cache
e5d04f3 feat: XP Explorer-style toolbar with icons + rename to "Bliss Chat"
0125805 feat: Settings v2 (top-p/max-tokens/system prompt), Edit + Help menus, Find
32d0ab9 feat: per-message Copy / Regenerate / Edit buttons above transcript
fdebb0a feat: rename / delete / right-click menu / search filter in sidebar
```

All three of the parallel-agent feature commits (`fdebb0a`, `32d0ab9`,
`0125805`) came from `isolation: "worktree"` Agent calls and were
integrated via `git cherry-pick` with hand-resolved ID conflicts.

---

End of handoff. The d12-chinchilla bench will be the first concrete next datapoint.
