# GUI architecture (xpchat.c)

`XPCHAT.EXE` is a single-file Win32 program. ~2200 LOC of C99 + a
small `resource.rc` for dialog templates, icon, manifest, and the
toolbar bitmap. No MFC, no .NET, no XAML, no WTL — straight `windows.h`.

This file documents the non-obvious patterns. Pair with `src/xpchat.c`
as the authoritative source.

---

## Main process structure

```
WinMain
├── settings_load()                   // HKCU\Software\bliss-chat\Settings
├── win_load_placement()              // HKCU\Software\bliss-chat\Window
├── RegisterClassA                    // "BlissChatMain" window class
├── CreateWindowExA                   // main hwnd
├── ShowWindow / UpdateWindow
├── CreateAcceleratorTableA           // Ctrl+S/N/C/V/A/F
└── message loop: GetMessage → TranslateAccelerator → TranslateMessage → Dispatch

WM_CREATE handler
├── log_init()                        // C:\xp-llm\xpchat-YYYYMMDD-HHMMSS.log
├── make_menu()                       // File / Edit / Conversation / Help
├── create_fonts()                    // Tahoma 8/14, monospace Tahoma 14
├── create_controls()                 // ALL HWNDs created here
├── subclass_input()                  // Hook Enter / Shift+Enter / Esc
├── layout_controls()                 // Initial positioning
├── chats_resolve_dir()               // %APPDATA%\bliss-chat\chats
├── chats_index_disk()                // Scan + populate sidebar
├── chats_create_new()                // Start a fresh chat for this session
├── clear_transcript()                // Welcome + model + empty-state hint
└── spawn_backend()                   // CreateProcess NC_RUN.EXE + pipes

Backend reader thread (started by spawn_backend)
└── reads stdout/stderr line-by-line, posts WM_BACKEND_* messages to gMain
```

---

## Control hierarchy

```
gMain (top-level)
├── gIcon          STATIC (SS_ICON)           // 32x32 Bliss icon, top-left
├── gTitle         STATIC                     // "Bliss Chat" big bold
├── gSubtitle      STATIC                     // CPU + RAM specs
├── gModel         STATIC                     // "Model: nanochat-d12 ..."
├── gToolbar       TOOLBARCLASSNAMEA          // XP Explorer-style toolbar
│                                             // Buttons: New Chat / Save / Stop / Settings / Help
├── gSearch        EDIT (cue banner)          // "Search chats..."
├── gChatList      LISTBOX (LBS_NOTIFY)       // Sidebar of chats
├── gNewChatBtn    BUTTON                     // "+ New Chat"
├── gCopyLastBtn   BUTTON (BS_FLAT)           // Per-message action strip
├── gRegenBtn      BUTTON (BS_FLAT)
├── gEditLastBtn   BUTTON (BS_FLAT)
├── gTranscript    RICHEDIT20A                // Main transcript view
├── gInput         EDIT (multi-line, subclassed for Enter/Esc)
├── gSend          BUTTON                     // Right column under input
├── gStop          BUTTON
├── gClear         BUTTON
├── gStatus        STATIC                     // Bottom status text
└── gProgress      PROGRESS_CLASSA            // Bottom progress bar
```

Layout is recomputed entirely in `layout_controls(hwnd)` on every
`WM_SIZE`. No anchoring system — just hand-arithmetic.

---

## Custom messages (`WM_APP + N`)

Posted by the backend-reader thread, dispatched on the main thread.
All `lparam` strings are `malloc`'d in the reader and `free`'d in
the main-thread handler.

| ID | Name | Payload |
|---|---|---|
| `WM_APP + 1` | `WM_APPEND_TEXT` | streamed assistant text chunk (lparam = char*) |
| `WM_APP + 2` | `WM_RUN_DONE` | end of turn (wparam = token count, lparam = full reply text) |
| `WM_APP + 3` | `WM_BACKEND_READY` | model loaded + prefix prefilled |
| `WM_APP + 4` | `WM_BACKEND_DEAD` | child process exited |
| `WM_APP + 5` | `WM_BACKEND_INFO` | INFO sentinel (lparam = info text) |
| `WM_APP + 6` | `WM_RUN_ERR` | ERR sentinel (lparam = error text) |
| `WM_APP + 7` | `WM_BACKEND_PROG` | PROG sentinel (wparam = pct 0..100) |

Reason for cross-thread `PostMessage` instead of doing UI updates
from the reader: RichEdit and other Win32 controls are not
thread-safe. All UI mutation happens on the main message loop.

---

## Resource IDs

Living in `src/resource.rc`. Spread out on purpose so the three
parallel UX agent worktrees didn't collide:

| Range | Group |
|---|---|
| 101 | App icon (RT_ICON) |
| 102 | Toolbar bitmap (RT_BITMAP) |
| 200–207 | Settings dialog base controls |
| 210–211 | Rename Chat dialog |
| 212–213 | Edit Last Prompt dialog |
| 214–217 | Settings v2 extras (top-p, max-tokens, system prompt, reset all) |
| 220–222 | Find dialog |
| 230–231 | Keyboard Shortcuts dialog |
| 1001–1022 | Main-window child control IDs (IDC_*) |
| 2001–2025 | Menu command IDs (IDM_*) |
| 3001 | Timer (TIMER_STATUS, 500 ms tick during generation) |

When adding new dialogs / controls, **add them in a fresh range**.
Reusing IDs already caused real merge-conflict pain during the
parallel-agent integration.

---

## Dialogs

Five modal dialogs, all template-based in `resource.rc`:

1. **`IDD_SETTINGS`** — Sampling group (temp, seed, presets) +
   Advanced group (top-p, max tokens, system prompt). `settings_dlg_proc`.
2. **`IDD_RENAME`** — One EDIT, OK/Cancel. `rename_dlg_proc`.
3. **`IDD_EDITPROMPT`** — Multi-line EDIT for re-send. `edit_prompt_dlg_proc`.
4. **`IDD_FIND`** — Find Next over transcript. `find_dlg_proc`.
5. **`IDD_SHORTCUTS`** — Read-only EDIT listing keybindings. `shortcuts_dlg_proc`.

All invoked via `DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_*),
hwnd, dlg_proc, 0)`.

---

## Input subclassing

`gInput` (the user's message box) is subclassed via
`SetWindowLongPtrA(GWLP_WNDPROC, input_proc)`. The custom proc
intercepts:

- **Enter** (without Shift) → `send_prompt()`
- **Shift+Enter** → pass through (insert newline)
- **Esc** → if generating, post `WM_COMMAND(IDC_STOP, BN_CLICKED)`
- everything else → call `gOldInputProc`

The chat-list sidebar is also subclassed (`chatlist_proc` /
`gOldChatListProc`) for F2 / Del / right-click.

---

## Custom drawing & themed controls

Mostly avoided. The comctl32 v6 manifest gives us themed buttons,
edits, listbox, and progress bar for free. The toolbar uses
`TBSTYLE_FLAT | TBSTYLE_TOOLTIPS | TBSTYLE_LIST | CCS_TOP | CCS_NODIVIDER`
which the manifest renders as the XP Luna look.

One place we draw colored text manually: the transcript headers.
`rich_append_color(text, color, bold)` builds a `CHARFORMAT2`
with `CFM_COLOR | CFM_BOLD`, applies it via `EM_SETCHARFORMAT
SCF_SELECTION`, then `EM_REPLACESEL` inserts the text with that
format. Used for "You — 8:26 PM" (blue), "Assistant — 8:26 PM"
(green), error lines (red), and the stats footer (gray).

---

## Auto-scroll

After every transcript append, force `WM_VSCROLL SB_BOTTOM +
EM_SCROLLCARET`. Originally tried sticky-bottom (only scroll if
the user was already at the bottom via `GetScrollInfo`), but
SCROLLINFO lags during fast streaming and the user got stuck at
the top of generation. Standard chat-app behavior wins.

If the user scrolls up to read history mid-generation, they lose
their position. Acceptable trade-off given how the user sees this
app being used.

---

## Window state persistence

Two registry keys:

- `HKCU\Software\bliss-chat\Window` — `X`, `Y`, `W`, `H` (REG_DWORD).
  Saved on `WM_CLOSE` via `GetWindowPlacement(... rcNormalPosition)`.
  Loaded at `WinMain` startup; coordinates sanity-clamped against
  the current screen so an unplugged second monitor doesn't strand
  the window off-screen.
- `HKCU\Software\bliss-chat\Settings` — `TempMilli`, `Seed`,
  `TopPMilli`, `MaxTok` (REG_DWORD) and `SystemPrompt` (REG_SZ).
  Pushed to backend on `WM_BACKEND_READY` via `settings_apply_to_backend()`.

Chat history lives outside the registry, in `%APPDATA%\bliss-chat\chats\
<unix_ts>.txt`. See `context/05-sentinel-protocol.md` for the file
format.

---

## What the GUI does NOT do

- **Render Markdown.** RichEdit can do bold/italic via CHARFORMAT, but
  the model rarely emits markdown and detecting it would be fragile.
- **Highlight code.** No syntax detection; if the model emits a code
  block it's just plain text.
- **Spell-check.** Win32 EDIT has no built-in spellcheck on XP.
- **Token streaming animation.** Tokens just appear as they arrive.
  No "thinking" indicator beyond the progress bar.
- **Voice / image input.** Out of scope for a local LLM.

These are deliberate omissions to keep the EXE small (now ~720 KB)
and the build dependency-free.
