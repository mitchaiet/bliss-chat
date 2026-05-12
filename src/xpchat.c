// bliss-chat — XP-native LLM front-end.
// Front-end for the custom NC_RUN.EXE inference backend (Karpathy nanochat
// architecture). Loads MODEL.NCB once and stays resident. Sentinel protocol:
//   stdout: "\x01READY\n"        -> backend has loaded the model
//   stdout: "\x01INFO <text>\n"  -> backend status / banner / notice
//   stdout: "\x01PROG <pct>\n"   -> progress 0..100 during load/prefill/gen
//   stdout: "\x01EOT <count>\n"  -> end of one assistant turn (token count)
//   stdout: "\x01ERR <text>\n"   -> backend reported an error this turn
//   stdin:  "<text>\n"           -> one user turn (single line, no embedded newlines)
//   stdin:  "\x01STOP\n"         -> abort current generation
//   stdin:  EOF                  -> backend exits

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0501

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <richedit.h>
#include <shellapi.h>
#include <shlobj.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define APP_NAME "Bliss Chat"
#define APP_DISPLAY_NAME "Bliss Chat"
#define APP_WINDOW_TITLE "Bliss Chat - Local Chat Assistant"
#define APP_TAGLINE "Local AI conversation utility"
#define IDI_APP 101

#define IDC_TRANSCRIPT 1001
#define IDC_INPUT      1002
#define IDC_SEND       1003
#define IDC_STOP       1004
#define IDC_CLEAR      1005
#define IDC_STATUS     1006
#define IDC_MODEL      1010
#define IDC_TITLE      1011
#define IDC_SUBTITLE   1012
#define IDC_APPICON    1013
#define IDC_PROGRESS   1014
#define IDC_CHATLIST   1015
#define IDC_NEWCHATBTN 1016
#define IDC_SEARCH     1017
#define IDC_DIAGNOSTICS 1018
#define IDC_TOOLBAR    1020
#define IDC_COPY_LAST  1030
#define IDC_REGEN      1031
#define IDC_EDIT_LAST  1032
#define IDC_TASK_SAVE   1040
#define IDC_TASK_EXPORT 1041
#define IDC_TASK_IMPORT 1042
#define IDC_TASK_CLEAR  1043
#define IDC_TASKLIST    1044
#define IDC_BTN_BOLD    1050
#define IDC_BTN_ITALIC  1051
#define IDC_BTN_SMILE   1052
#define IDC_BTN_ATTACH  1053
#define IDC_BTN_PROMPTS 1054
#define IDC_IDENTITY_FRAME 1060
#define IDC_CPU_LABEL      1061
#define IDC_CPU_VALUE      1062
#define IDC_MEMORY_LABEL   1063
#define IDC_MEMORY_VALUE   1064
#define IDC_THREADS_LABEL  1065
#define IDC_THREADS_VALUE  1066
#define IDC_MODEL_GROUP    1067
#define IDC_MODEL_STATE    1068
#define IDC_TASKPANE_TASKS 1070
#define IDC_TASKPANE_RECENT 1071
#define IDC_CONV_GROUP     1072
#define IDC_DIAG_GROUP     1073
#define IDC_INPUT_GROUP    1074
#define IDC_INPUT_LABEL    1075
#define IDC_INPUT_HINT     1076

#define IDM_EXIT       2001
#define IDM_CLEAR      2002
#define IDM_SEND       2003
#define IDM_STOP       2004
#define IDM_ABOUT      2005
#define IDM_SAVE       2006
#define IDM_NEWCHAT    2007
#define IDM_SETTINGS   2008
#define IDM_RENAME     2009
#define IDM_DELETE     2010
#define IDM_CHATSAVE   2011
#define IDM_COPY       2020
#define IDM_PASTE      2021
#define IDM_SELECTALL  2022
#define IDM_FIND       2023
#define IDM_SHORTCUTS  2024
#define IDM_SLASHHELP  2025
#define IDM_OPEN       2026
#define IDM_EXPORT     2027
#define IDM_PRINT      2028
#define IDM_MODELINFO  2029
#define IDM_PERF       2030
#define IDM_TEMPLATES  2031
#define IDM_HELPTOPICS 2032

// Resource IDs from resource.rc — must stay in sync.
#define IDD_SETTINGS         200
#define IDC_TEMP_EDIT        201
#define IDC_SEED_EDIT        202
#define IDC_SEED_RANDOM      203
#define IDC_PRESET_GREEDY    204
#define IDC_PRESET_BALANCED  205
#define IDC_PRESET_CREATIVE  206
#define IDC_DEFAULTS         207
#define IDC_TOPP_EDIT        214
#define IDC_MAXTOK_EDIT      215
#define IDC_SYSPROMPT_EDIT   216
#define IDC_RESET_ALL        217

#define IDD_RENAME           210
#define IDC_RENAME_EDIT      211

#define IDD_EDITPROMPT       212
#define IDC_EDITPROMPT_TEXT  213

#define IDD_FIND             220
#define IDC_FIND_EDIT        221
#define IDC_FIND_NEXT        222

#define IDD_SHORTCUTS        230
#define IDC_SHORTCUTS_TEXT   231

// Defaults that match nc_run.c's CLI defaults.
#define DEFAULT_TEMP   0.0f
#define DEFAULT_SEED   0  /* 0 = "use clock" */
#define DEFAULT_TOPP   0.95f
#define DEFAULT_MAXTOK 128

#define WM_APPEND_TEXT    (WM_APP + 1)
#define WM_RUN_DONE       (WM_APP + 2)   // wparam = generated_token_count, lparam = response text
#define WM_BACKEND_READY  (WM_APP + 3)
#define WM_BACKEND_DEAD   (WM_APP + 4)
#define WM_BACKEND_INFO   (WM_APP + 5)
#define WM_RUN_ERR        (WM_APP + 6)   // lparam = error text
#define WM_BACKEND_PROG   (WM_APP + 7)   // wparam = pct (0..100)
#define TIMER_STATUS      3001

#define PROMPT_MAX     8192
#define BACKEND_EXE    "NC_RUN.EXE"
#define MODEL_FILE     "MODEL.NCB"
#define TOKENIZER_FILE "TOKENIZER.NCT"
#define MODEL_LABEL    "(loading...)"
#define APP_SUBTITLE   ""  /* set at startup from get_pc_specs() */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} Buffer;

static HINSTANCE gInstance;
static HWND gMain;
static HWND gTranscript;
static HWND gInput;
static HWND gSend;
static HWND gStop;
static HWND gClear;
static HWND gStatus;
static HWND gModel;
static HWND gModelState;
static HWND gTitle;
static HWND gSubtitle;
static HWND gIcon;
static HWND gProgress;
static HWND gIdentityFrame;
static HWND gCpuLabel;
static HWND gCpuValue;
static HWND gMemoryLabel;
static HWND gMemoryValue;
static HWND gThreadsLabel;
static HWND gThreadsValue;
static HWND gModelGroup;
static HWND gTaskPane;
static HWND gRecentPane;
static HWND gConversationGroup;
static HWND gDiagnosticsGroup;
static HWND gInputGroup;
static HWND gInputLabel;
static HWND gInputHint;
static HWND gChatList;     // sidebar listbox of saved chats
static HWND gNewChatBtn;   // "+ New Chat" button under the listbox
static HWND gSearch;       // search/filter EDIT above the listbox
static HWND gDiagnostics;  // lower diagnostics log pane
static HWND gTaskSaveBtn;
static HWND gTaskExportBtn;
static HWND gTaskImportBtn;
static HWND gTaskClearBtn;
static HWND gTaskList;
static HWND gBoldBtn;
static HWND gItalicBtn;
static HWND gSmileBtn;
static HWND gAttachBtn;
static HWND gPromptToolsBtn;
static WNDPROC gOldChatListProc;  // original listbox WNDPROC (for subclass)
static char gSearchText[128];     // current filter; "" = show all
static HWND gToolbar;      // COMCTL32 toolbar below the title strip
static HWND gCopyLastBtn;  // per-message action strip above the transcript
static HWND gRegenBtn;
static HWND gEditLastBtn;
static HFONT gUiFont;
static HFONT gTitleFont;
static HFONT gPaneFont;
static HFONT gMonoFont;
static HICON gSendIcon;
static HICON gStopIcon;
static HIMAGELIST gToolbarImages;
static HMODULE gRichEdit;
static WNDPROC gOldInputProc;

static RECT gRcToolbarBand;
static RECT gRcIdentityBand;
static RECT gRcModelBox;
static RECT gRcTaskPane;
static RECT gRcRecentPane;
static RECT gRcConversationPane;
static RECT gRcDiagnosticsPane;
static RECT gRcInputPane;
static RECT gRcStatusBar;
static char gPcCpu[160] = "Intel(R) Pentium(R) 4 CPU 3.00GHz";
static char gPcMemory[64] = "512 MB RAM";
static char gPcThreads[32] = "1";
static char gModelName[160] = "(loading...)";
static char gStatusText[160] = "Loading model...";

static char gAppDir[MAX_PATH];
static char gPendingUser[PROMPT_MAX];
static char gLogPath[MAX_PATH];
static FILE * gLogFile = NULL;
static CRITICAL_SECTION gLogLock;
static int gLogInited = 0;

// Chat persistence — see big block below for the data model and helpers.
// Forward declarations so send_prompt / WM_RUN_DONE can call them.
#define MAX_CHATS 200
#define CHAT_TITLE_MAX 80
typedef struct {
    char path[MAX_PATH];
    char title[CHAT_TITLE_MAX];
    SYSTEMTIME stamp;
} ChatEntry;
static ChatEntry gChats[MAX_CHATS];
static int       gChatCount = 0;
static int       gActiveIdx = -1;
static char      gChatsDir[MAX_PATH];
// Listbox row -> gChats index mapping. With search filtering active, only
// a subset of gChats is visible; gFilteredIndices[row] resolves the row to
// the underlying gChats slot. Length is the listbox item count.
static int       gFilteredIndices[MAX_CHATS];
static int       gFilteredCount = 0;
static void chats_create_new(void);
static void chats_append_turn(const char *role, const char *text);
static void chats_set_title_from(const char *user_text);
static void chats_repopulate_listbox(void);
static void save_transcript_dialog(HWND parent);
static BOOL backend_send_line(const char *line);

static volatile LONG gRunning;
static volatile LONG gBackendReady;
static DWORD gRunStarted;

// Per-message action state. The most-recent-assistant-turn body lives in
// the RichEdit between [gLastAsstStart, gLastAsstEnd) — character offsets,
// not bytes. gLastAsstStart is set when we begin appending the assistant
// header (we capture the cpMax AFTER appending the header), and
// gLastAsstEnd is set when WM_RUN_DONE fires (right before the stats
// footer is appended). gHasAsstTurn flips to TRUE once at least one
// assistant turn has completed, gating the action buttons.
static LONG gLastAsstStart = -1;
static LONG gLastAsstEnd   = -1;
static int  gHasAsstTurn   = 0;
// Set TRUE by the Regenerate action so the next assistant header reads
// "Assistant (regenerated) ..." instead of "Assistant ...". Cleared by
// send_prompt() after one use.
static int  gRegenLabel = 0;
// Settings — mirror what we last sent to the backend so the dialog can
// pre-populate. Persisted in HKCU\Software\bliss-chat\Settings.
static float gTemp     = DEFAULT_TEMP;
static DWORD gSeed     = DEFAULT_SEED;
static float gTopP     = DEFAULT_TOPP;
static DWORD gMaxTok   = DEFAULT_MAXTOK;
static char  gSysPrompt[2048] = "";  // empty = use backend default

// Last find text from the Find dialog. Static so "Find Next" stays useful
// when the dialog is dismissed (and reusable if we ever wire F3).
static char  gLastFind[256] = "";
// Running count of *assistant* characters emitted in the current turn,
// reset when a new turn starts. Used to compute live tok/s in the
// status bar between EOT events. Reasonable approximation: tokens
// average ~4 chars for English in our BPE.
static volatile LONG gRunChars;

static HANDLE gBackendProcess;
static HANDLE gBackendStdinW;   // GUI writes here -> backend reads as stdin
static HANDLE gBackendStdoutR;  // GUI reads here  <- backend writes to stdout
static HANDLE gReaderThread;
static HANDLE gWaitThread;

// ---------- logging ----------
static void log_init(void) {
    SYSTEMTIME st;
    if (gLogInited) return;
    InitializeCriticalSection(&gLogLock);
    GetLocalTime(&st);
    snprintf(gLogPath, sizeof(gLogPath),
        "%s\\xpchat-%04d%02d%02d-%02d%02d%02d.log",
        gAppDir, st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    gLogFile = fopen(gLogPath, "w");
    if (gLogFile) {
        fprintf(gLogFile, "=== Bliss Chat session log ===\n");
        fprintf(gLogFile, "log file: %s\n", gLogPath);
        fflush(gLogFile);
    }
    gLogInited = 1;
}

static void dbg_log(const char * tag, const char * fmt, ...) {
    SYSTEMTIME st;
    va_list ap;
    if (!gLogFile) return;
    GetLocalTime(&st);
    EnterCriticalSection(&gLogLock);
    fprintf(gLogFile, "[%02d:%02d:%02d.%03d] [%s] ",
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, tag);
    va_start(ap, fmt);
    vfprintf(gLogFile, fmt, ap);
    va_end(ap);
    fputc('\n', gLogFile);
    fflush(gLogFile);
    LeaveCriticalSection(&gLogLock);
}

static void log_bytes(const char * tag, const char * data, size_t len) {
    SYSTEMTIME st;
    size_t i;
    if (!gLogFile) return;
    GetLocalTime(&st);
    EnterCriticalSection(&gLogLock);
    fprintf(gLogFile, "[%02d:%02d:%02d.%03d] [%s] (%lu bytes) ",
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, tag, (unsigned long)len);
    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)data[i];
        if (c == '\n') fputs("\\n", gLogFile);
        else if (c == '\r') fputs("\\r", gLogFile);
        else if (c < 0x20 || c == 0x7f) fprintf(gLogFile, "\\x%02x", c);
        else fputc(c, gLogFile);
    }
    fputc('\n', gLogFile);
    fflush(gLogFile);
    LeaveCriticalSection(&gLogLock);
}

static void log_close(void) {
    if (gLogFile) {
        dbg_log("GUI", "session ending");
        fclose(gLogFile);
        gLogFile = NULL;
    }
    if (gLogInited) DeleteCriticalSection(&gLogLock);
    gLogInited = 0;
}

// ---------- buffer helpers ----------
static char * dup_text_len(const char *text, size_t len) {
    char *copy = (char *)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, len);
    copy[len] = 0;
    return copy;
}

static char * dup_text(const char *text) {
    return dup_text_len(text ? text : "", strlen(text ? text : ""));
}

static void buffer_init(Buffer *b) { b->data = NULL; b->len = 0; b->cap = 0; }

static int buffer_reserve(Buffer *b, size_t need) {
    char *next;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) cap *= 2;
    next = (char *)realloc(b->data, cap);
    if (!next) return 0;
    b->data = next;
    b->cap  = cap;
    return 1;
}

static int buffer_append(Buffer *b, const char *text, size_t len) {
    if (!buffer_reserve(b, b->len + len + 1)) return 0;
    memcpy(b->data + b->len, text, len);
    b->len += len;
    b->data[b->len] = 0;
    return 1;
}

static void normalize_newlines(Buffer *out, const char *text, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        char c = text[i];
        if (c == '\r') {
            if (i + 1 < len && text[i + 1] == '\n') i++;
            buffer_append(out, "\r\n", 2);
        } else if (c == '\n') {
            buffer_append(out, "\r\n", 2);
        } else {
            buffer_append(out, &c, 1);
        }
    }
}

// ---------- app dir ----------
static void get_app_dir(void) {
    char *slash;
    GetModuleFileNameA(NULL, gAppDir, sizeof(gAppDir));
    slash = strrchr(gAppDir, '\\');
    if (slash) *slash = 0;
}

static int file_exists_in_app_dir(const char *name) {
    char path[MAX_PATH * 2];
    DWORD attr;
    snprintf(path, sizeof(path), "%s\\%s", gAppDir, name);
    attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

// ---------- transcript helpers ----------
static void update_status_bar(void) {
    char model_part[220];
    char thread_part[64];
    if (!gStatus) return;
    snprintf(model_part, sizeof(model_part), "Model: %s", gModelName);
    snprintf(thread_part, sizeof(thread_part), "%s thread", gPcThreads);
    SendMessageA(gStatus, SB_SETTEXTA, 0, (LPARAM)gStatusText);
    SendMessageA(gStatus, SB_SETTEXTA, 1, (LPARAM)model_part);
    SendMessageA(gStatus, SB_SETTEXTA, 2, (LPARAM)thread_part);
    SendMessageA(gStatus, SB_SETTEXTA, 3, (LPARAM)"0.0 tok/s");
    SendMessageA(gStatus, SB_SETTEXTA, 4, (LPARAM)gPcMemory);
}

static void update_model_box(void) {
    BOOL ready = InterlockedCompareExchange(&gBackendReady, 0, 0) != 0;
    BOOL named = strcmp(gModelName, "(loading...)") != 0;
    if (gModelState) SetWindowTextA(gModelState, ready ? "Model loaded" : "Loading model");
    if (gModel) SetWindowTextA(gModel, named ? gModelName : MODEL_LABEL);
    if (gProgress) ShowWindow(gProgress, (ready || named) ? SW_HIDE : SW_SHOW);
    update_status_bar();
}

static void set_status(const char *text) {
    snprintf(gStatusText, sizeof(gStatusText), "%s", text ? text : "");
    update_status_bar();
}

static void diagnostics_appendf(const char *fmt, ...) {
    char body[512];
    char line[640];
    SYSTEMTIME st;
    va_list ap;
    if (!gDiagnostics || !fmt) return;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);
    GetLocalTime(&st);
    snprintf(line, sizeof(line), "[%02d:%02d:%02d]  %s\r\n",
             st.wHour, st.wMinute, st.wSecond, body);
    SendMessageA(gDiagnostics, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
    SendMessageA(gDiagnostics, EM_REPLACESEL, FALSE, (LPARAM)line);
    SendMessageA(gDiagnostics, WM_VSCROLL, SB_BOTTOM, 0);
}

static void rich_set_format(COLORREF color, BOOL bold) {
    CHARFORMAT2A cf;
    ZeroMemory(&cf, sizeof(cf));
    cf.cbSize = sizeof(cf);
    cf.dwMask = CFM_COLOR | CFM_BOLD;
    cf.crTextColor = color;
    cf.dwEffects = bold ? CFE_BOLD : 0;
    SendMessageA(gTranscript, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
}

static void rich_append_color(const char *text, COLORREF color, BOOL bold) {
    CHARRANGE range;
    Buffer norm;
    if (!gTranscript || !text || !*text) return;
    buffer_init(&norm);
    normalize_newlines(&norm, text, strlen(text));
    if (!norm.data) return;
    range.cpMin = -1;
    range.cpMax = -1;
    SendMessageA(gTranscript, EM_EXSETSEL, 0, (LPARAM)&range);
    rich_set_format(color, bold);
    SendMessageA(gTranscript, EM_REPLACESEL, FALSE, (LPARAM)norm.data);
    // Always scroll to the new end. The previous "sticky-bottom" version
    // checked SCROLLINFO before appending and only scrolled if the user
    // was already at the bottom; SCROLLINFO lags during fast streaming
    // and the result was that mid-generation text never followed.
    // Standard chat-app behavior wins: auto-scroll on every append.
    SendMessageA(gTranscript, WM_VSCROLL, SB_BOTTOM, 0);
    SendMessageA(gTranscript, EM_SCROLLCARET, 0, 0);
    free(norm.data);
}

// Return the character offset of the RichEdit's end-of-text. RichEdit
// offsets are in UTF-16 code units, but our content is ASCII-only and
// CRLF-normalized, so EM_EXSETSEL(-1,-1) followed by EM_EXGETSEL gives
// us a usable position for EM_GETTEXTRANGE later.
static LONG rich_end_pos(void) {
    CHARRANGE r;
    if (!gTranscript) return 0;
    r.cpMin = -1; r.cpMax = -1;
    SendMessageA(gTranscript, EM_EXSETSEL, 0, (LPARAM)&r);
    SendMessageA(gTranscript, EM_EXGETSEL, 0, (LPARAM)&r);
    return r.cpMax;
}

// Pull a character range out of the transcript as ANSI text. Caller frees.
// Returns NULL on failure or empty range.
static char * rich_get_range(LONG start, LONG end) {
    TEXTRANGEA tr;
    LONG n;
    char *buf;
    if (!gTranscript || end <= start) return NULL;
    n = end - start;
    // EM_GETTEXTRANGE writes at most cpMax-cpMin chars plus a NUL, but
    // because RichEdit positions count UTF-16 units we add slack to
    // be safe against multi-byte expansion (we expect 1:1 ASCII, but
    // belt-and-braces costs nothing).
    buf = (char *)malloc((size_t)n * 2 + 8);
    if (!buf) return NULL;
    tr.chrg.cpMin = start;
    tr.chrg.cpMax = end;
    tr.lpstrText  = buf;
    SendMessageA(gTranscript, EM_GETTEXTRANGE, 0, (LPARAM)&tr);
    buf[(size_t)n * 2 + 7] = 0;
    return buf;
}

// Put a NUL-terminated ANSI string on the clipboard as CF_TEXT.
// Returns 1 on success, 0 on failure. text is copied into a moveable
// HGLOBAL; SetClipboardData transfers ownership.
static int clipboard_set_text(HWND owner, const char *text) {
    HGLOBAL hg;
    size_t len;
    char *dst;
    if (!text) return 0;
    len = strlen(text);
    if (!OpenClipboard(owner)) return 0;
    EmptyClipboard();
    hg = GlobalAlloc(GMEM_MOVEABLE, len + 1);
    if (!hg) { CloseClipboard(); return 0; }
    dst = (char *)GlobalLock(hg);
    if (!dst) { GlobalFree(hg); CloseClipboard(); return 0; }
    memcpy(dst, text, len);
    dst[len] = 0;
    GlobalUnlock(hg);
    if (!SetClipboardData(CF_TEXT, hg)) {
        GlobalFree(hg);
        CloseClipboard();
        return 0;
    }
    CloseClipboard();
    return 1;
}

static void input_toggle_charformat(DWORD mask, DWORD effect) {
    CHARFORMAT2A cf;
    if (!gInput) return;
    ZeroMemory(&cf, sizeof(cf));
    cf.cbSize = sizeof(cf);
    cf.dwMask = mask;
    SendMessageA(gInput, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    cf.cbSize = sizeof(cf);
    cf.dwMask = mask;
    cf.dwEffects = (cf.dwEffects & effect) ? 0 : effect;
    SendMessageA(gInput, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    SetFocus(gInput);
}

static void input_insert_text(const char *text) {
    if (!gInput || !text) return;
    SendMessageA(gInput, EM_REPLACESEL, TRUE, (LPARAM)text);
    SetFocus(gInput);
}

// Enable/disable the three per-message action buttons based on whether
// we have a recorded assistant turn and the backend is idle.
static void update_msg_actions(void) {
    BOOL idle = !InterlockedCompareExchange(&gRunning, 0, 0);
    BOOL ready = InterlockedCompareExchange(&gBackendReady, 0, 0) != 0;
    BOOL have_asst = gHasAsstTurn ? TRUE : FALSE;
    BOOL have_user = gPendingUser[0] != 0 ? TRUE : FALSE;
    if (gCopyLastBtn) EnableWindow(gCopyLastBtn, have_asst);
    if (gRegenBtn)    EnableWindow(gRegenBtn,    have_asst && have_user && ready && idle);
    if (gEditLastBtn) EnableWindow(gEditLastBtn, have_user && ready && idle);
}

static void clear_transcript(void) {
    char model_text[256];
    // Reset per-message action state alongside the visible text.
    gLastAsstStart = -1;
    gLastAsstEnd   = -1;
    gHasAsstTurn   = 0;
    gPendingUser[0] = 0;
    update_msg_actions();
    SetWindowTextA(gTranscript, "");
    rich_append_color("Bliss Chat\r\n", RGB(0, 128, 0), TRUE);
    if (gModel && GetWindowTextA(gModel, model_text, sizeof(model_text)) > 0) {
        rich_append_color(model_text, RGB(96, 96, 96), FALSE);
        rich_append_color("\r\n", RGB(96, 96, 96), FALSE);
    } else {
        rich_append_color(MODEL_LABEL "\r\n", RGB(96, 96, 96), FALSE);
    }
    if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
        rich_append_color("Note: this clears the on-screen transcript only. Backend conversation memory persists until exit.\r\n\r\n", RGB(96, 96, 96), FALSE);
    } else {
        rich_append_color("Loading model...\r\n\r\n", RGB(96, 96, 96), FALSE);
    }
    // Empty-state welcome line. Renders in muted gray so it reads as a
    // hint, not a turn. It stays until the first user turn pushes it
    // out of view; clear_transcript() rewrites it on Clear / New Chat.
    rich_append_color(
        "Welcome to Bliss Chat. Type a message below to start, or try a slash command like /help.\r\n\r\n",
        RGB(128, 128, 128), FALSE);
}

static void set_running(BOOL running) {
    BOOL ready = InterlockedCompareExchange(&gBackendReady, 0, 0) != 0;
    EnableWindow(gSend,  ready && !running);
    EnableWindow(gInput, ready);
    EnableWindow(gStop,  ready && running);
    EnableWindow(gClear, !running);
    // Mirror Stop enabledness onto the toolbar button.
    if (gToolbar) {
        SendMessageA(gToolbar, TB_SETSTATE, IDM_STOP,
                     (ready && running) ? TBSTATE_ENABLED : 0);
    }
    // Per-message buttons follow the same idle/ready rules.
    update_msg_actions();
    // Reset bar at phase transitions; backend will fill it via PROG messages.
    if (gProgress) SendMessageA(gProgress, PBM_SETPOS, 0, 0);
    if (running) SetTimer(gMain, TIMER_STATUS, 500, NULL);
    else KillTimer(gMain, TIMER_STATUS);
}

static void post_chunk(const char *text, size_t len) {
    char *copy;
    if (!text || len == 0) return;
    copy = dup_text_len(text, len);
    if (!copy) return;
    if (!PostMessageA(gMain, WM_APPEND_TEXT, 0, (LPARAM)copy)) free(copy);
}

// ---------- backend reader ----------
// Pulls bytes from gBackendStdoutR forever, splits on \x01 sentinels, posts
// regular response text via WM_APPEND_TEXT and end-of-turn / ready / dead via
// dedicated messages.
static DWORD WINAPI reader_thread(LPVOID param) {
    char chunk[1024];
    Buffer line; // buffer for the current sentinel line being assembled
    Buffer turn; // accumulated assistant text for the current turn
    DWORD got;
    int in_sentinel = 0;

    (void)param;
    buffer_init(&line);
    buffer_init(&turn);

    while (ReadFile(gBackendStdoutR, chunk, sizeof(chunk), &got, NULL) && got > 0) {
        size_t i;
        size_t flush_from = 0;
        for (i = 0; i < got; i++) {
            unsigned char c = (unsigned char)chunk[i];
            if (in_sentinel) {
                if (c == '\n') {
                    line.data[line.len] = 0;
                    if (strncmp(line.data, "READY", 5) == 0) {
                        dbg_log("BACKEND", "READY received");
                        InterlockedExchange(&gBackendReady, 1);
                        PostMessageA(gMain, WM_BACKEND_READY, 0, 0);
                    } else if (strncmp(line.data, "INFO ", 5) == 0) {
                        dbg_log("BACKEND", "INFO: %s", line.data + 5);
                        char *t = dup_text(line.data + 5);
                        PostMessageA(gMain, WM_BACKEND_INFO, 0, (LPARAM)t);
                    } else if (strncmp(line.data, "EOT", 3) == 0) {
                        // Form: "EOT <token_count>"
                        int tcount = 0;
                        if (line.data[3] == ' ') tcount = atoi(line.data + 4);
                        dbg_log("BACKEND", "EOT received, %d tokens, %lu bytes", tcount, (unsigned long)turn.len);
                        char *t = dup_text_len(turn.data ? turn.data : "", turn.len);
                        PostMessageA(gMain, WM_RUN_DONE, (WPARAM)tcount, (LPARAM)t);
                        turn.len = 0;
                        if (turn.data) turn.data[0] = 0;
                    } else if (strncmp(line.data, "PROG ", 5) == 0) {
                        int pct = atoi(line.data + 5);
                        if (pct < 0) pct = 0; if (pct > 100) pct = 100;
                        PostMessageA(gMain, WM_BACKEND_PROG, (WPARAM)pct, 0);
                    } else if (strncmp(line.data, "ERR", 3) == 0) {
                        dbg_log("BACKEND", "ERR sentinel: %s", line.data + 3);
                        char *t = dup_text(line.data + 3);
                        PostMessageA(gMain, WM_RUN_ERR, 0, (LPARAM)t);
                        turn.len = 0;
                        if (turn.data) turn.data[0] = 0;
                    } else {
                        dbg_log("BACKEND", "unknown sentinel: %s", line.data);
                    }
                    line.len = 0;
                    if (line.data) line.data[0] = 0;
                    in_sentinel = 0;
                    flush_from = i + 1;
                } else {
                    char b = (char)c;
                    buffer_append(&line, &b, 1);
                }
            } else if (c == 0x01) {
                // flush any pending response bytes before the sentinel
                if (i > flush_from) {
                    size_t n = i - flush_from;
                    buffer_append(&turn, chunk + flush_from, n);
                    post_chunk(chunk + flush_from, n);
                }
                in_sentinel = 1;
                flush_from = i + 1;
            }
        }
        if (!in_sentinel && got > flush_from) {
            size_t n = got - flush_from;
            buffer_append(&turn, chunk + flush_from, n);
            log_bytes("TOKEN", chunk + flush_from, n);
            post_chunk(chunk + flush_from, n);
        }
    }
    dbg_log("READER", "ReadFile loop ended (pipe closed or error)");

    free(line.data);
    free(turn.data);
    PostMessageA(gMain, WM_BACKEND_DEAD, 0, 0);
    return 0;
}

// ---------- backend launcher ----------
static int launch_backend(void) {
    SECURITY_ATTRIBUTES sa;
    HANDLE in_r = NULL, in_w = NULL, out_r = NULL, out_w = NULL;
    HANDLE backend_err = INVALID_HANDLE_VALUE;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    char command[1024];
    char err_buf[256];
    char err_log_path[MAX_PATH];

    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;

    if (!CreatePipe(&in_r, &in_w, &sa, 0))  goto fail_msg_pipe;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) goto fail_msg_pipe;

    // Don't let the child inherit the parent ends.
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    // Backend stderr goes to its own log file so llama.cpp diagnostics survive.
    {
        SYSTEMTIME st;
        GetLocalTime(&st);
        snprintf(err_log_path, sizeof(err_log_path),
            "%s\\backend-%04d%02d%02d-%02d%02d%02d.log",
            gAppDir, st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    }
    backend_err = CreateFileA(err_log_path, GENERIC_WRITE, FILE_SHARE_READ,
        &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    snprintf(command, sizeof(command),
        "\"%s\\%s\" \"%s\\%s\" \"%s\\%s\" -c 256 -t 0.8 -p 0.95",
        gAppDir, BACKEND_EXE, gAppDir, MODEL_FILE, gAppDir, TOKENIZER_FILE);

    dbg_log("GUI", "spawning backend: %s", command);
    dbg_log("GUI", "backend stderr -> %s", err_log_path);

    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags     = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdInput   = in_r;
    si.hStdOutput  = out_w;
    si.hStdError   = backend_err != INVALID_HANDLE_VALUE ? backend_err : out_w;
    si.wShowWindow = SW_HIDE;

    if (!CreateProcessA(NULL, command, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, gAppDir, &si, &pi)) {
        DWORD err = GetLastError();
        snprintf(err_buf, sizeof(err_buf), "Could not start " BACKEND_EXE ". Windows error %lu.", (unsigned long)err);
        dbg_log("GUI", "CreateProcess failed: %lu", (unsigned long)err);
        MessageBoxA(gMain, err_buf, APP_NAME, MB_ICONERROR | MB_OK);
        goto fail;
    }
    dbg_log("GUI", "backend pid=%lu started", (unsigned long)pi.dwProcessId);

    // Close the child ends in this process.
    CloseHandle(in_r); in_r = NULL;
    CloseHandle(out_w); out_w = NULL;
    if (backend_err != INVALID_HANDLE_VALUE) {
        CloseHandle(backend_err);
        backend_err = INVALID_HANDLE_VALUE;
    }

    gBackendProcess = pi.hProcess;
    CloseHandle(pi.hThread);
    gBackendStdinW   = in_w;
    gBackendStdoutR  = out_r;

    gReaderThread = CreateThread(NULL, 0, reader_thread, NULL, 0, NULL);
    if (!gReaderThread) {
        TerminateProcess(gBackendProcess, 1);
        CloseHandle(gBackendProcess);
        gBackendProcess = NULL;
        goto fail;
    }

    return 1;

fail_msg_pipe:
    MessageBoxA(gMain, "Could not create stdio pipes for backend.", APP_NAME, MB_ICONERROR | MB_OK);
fail:
    if (in_r)  CloseHandle(in_r);
    if (in_w)  CloseHandle(in_w);
    if (out_r) CloseHandle(out_r);
    if (out_w) CloseHandle(out_w);
    if (backend_err != INVALID_HANDLE_VALUE) CloseHandle(backend_err);
    return 0;
}

static void shutdown_backend(void) {
    if (gBackendStdinW) {
        // Closing stdin signals EOF -> backend will exit cleanly.
        CloseHandle(gBackendStdinW);
        gBackendStdinW = NULL;
    }
    if (gBackendProcess) {
        // Give it a couple seconds to exit on its own, else kill it.
        if (WaitForSingleObject(gBackendProcess, 2000) != WAIT_OBJECT_0) {
            TerminateProcess(gBackendProcess, 1);
        }
        CloseHandle(gBackendProcess);
        gBackendProcess = NULL;
    }
    if (gBackendStdoutR) {
        CloseHandle(gBackendStdoutR);
        gBackendStdoutR = NULL;
    }
    if (gReaderThread) {
        WaitForSingleObject(gReaderThread, 1000);
        CloseHandle(gReaderThread);
        gReaderThread = NULL;
    }
}

// ---------- send a turn ----------
static int input_has_text(const char *text) {
    const unsigned char *p = (const unsigned char *)text;
    while (*p) { if (*p > ' ') return 1; p++; }
    return 0;
}

static void sanitize_user(char *text) {
    char *p;
    for (p = text; *p; p++) {
        if (*p == '\r' || *p == '\n' || *p == '\t') *p = ' ';
    }
}

// Internal: send a fully-prepared prompt through the same path the Send
// button uses. user_prompt has already been sanitized. If show_user_header
// is true, a fresh "You — h:mm" header + the user text is appended to the
// transcript; Regenerate sets this to false so the prior user header isn't
// duplicated. The assistant header reads "Assistant (regenerated) — h:mm"
// when gRegenLabel is set (cleared here after use).
static void send_prompt_text(const char *user_prompt, int show_user_header) {
    DWORD written;
    char with_newline[PROMPT_MAX + 4];

    if (!InterlockedCompareExchange(&gBackendReady, 0, 0)) {
        MessageBeep(MB_ICONWARNING);
        return;
    }
    if (InterlockedCompareExchange(&gRunning, 0, 0)) return;
    if (!user_prompt || !input_has_text(user_prompt)) return;

    snprintf(gPendingUser, sizeof(gPendingUser), "%s", user_prompt);

    // Persist this turn to the active chat file. Backend slash commands
    // (lines that start with '/') are runtime-only — don't store them.
    if (user_prompt[0] != '/') {
        if (gActiveIdx < 0) chats_create_new();
        chats_append_turn("USER", user_prompt);
        chats_set_title_from(user_prompt);
    }

    {
        SYSTEMTIME st;
        GetLocalTime(&st);
        int h12 = st.wHour % 12; if (h12 == 0) h12 = 12;
        char hdr[64];
        const char *ampm = (st.wHour < 12 ? "AM" : "PM");
        if (show_user_header) {
            snprintf(hdr, sizeof(hdr), "You  -  %d:%02d %s\r\n",
                     h12, st.wMinute, ampm);
            rich_append_color(hdr, RGB(0, 84, 166), TRUE);
            rich_append_color(user_prompt, RGB(0, 0, 0), FALSE);
            snprintf(hdr, sizeof(hdr), "\r\n\r\n");
            rich_append_color(hdr, RGB(0, 0, 0), FALSE);
        }
        if (gRegenLabel) {
            snprintf(hdr, sizeof(hdr), "Assistant (regenerated)  -  %d:%02d %s\r\n",
                     h12, st.wMinute, ampm);
            gRegenLabel = 0;
        } else {
            snprintf(hdr, sizeof(hdr), "Assistant  -  %d:%02d %s\r\n",
                     h12, st.wMinute, ampm);
        }
        rich_append_color(hdr, RGB(0, 128, 0), TRUE);
        // The assistant *body* starts right after this header. Capture
        // the current end-of-text as the body's start cpMin.
        gLastAsstStart = rich_end_pos();
        gLastAsstEnd   = gLastAsstStart;  // will move forward as tokens arrive
    }

    InterlockedExchange(&gRunning, 1);
    InterlockedExchange(&gRunChars, 0);
    gRunStarted = GetTickCount();
    set_running(TRUE);
    set_status("Generating...");
    diagnostics_appendf("Generating response...");

    dbg_log("USER", "%s", user_prompt);
    snprintf(with_newline, sizeof(with_newline), "%s\n", user_prompt);
    if (!WriteFile(gBackendStdinW, with_newline, (DWORD)strlen(with_newline), &written, NULL)) {
        dbg_log("GUI", "WriteFile to backend stdin failed");
        rich_append_color("[backend write failed]\r\n", RGB(192, 0, 0), TRUE);
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
        set_status("Backend error");
    }
}

static void send_prompt(void) {
    char user_prompt[PROMPT_MAX];
    GetWindowTextA(gInput, user_prompt, sizeof(user_prompt));
    sanitize_user(user_prompt);
    if (!input_has_text(user_prompt)) return;
    SetWindowTextA(gInput, "");
    send_prompt_text(user_prompt, 1);
}

// ---------- menu / fonts / controls ----------
static void make_menu(HWND hwnd) {
    HMENU menu = CreateMenu();
    HMENU file = CreatePopupMenu();
    HMENU edit = CreatePopupMenu();
    HMENU convo = CreatePopupMenu();
    HMENU model = CreatePopupMenu();
    HMENU tools = CreatePopupMenu();
    HMENU help = CreatePopupMenu();

    AppendMenuA(file, MF_STRING, IDM_NEWCHAT, "New Chat\tCtrl+N");
    AppendMenuA(file, MF_STRING, IDM_OPEN, "Open...");
    AppendMenuA(file, MF_SEPARATOR, 0, NULL);
    AppendMenuA(file, MF_STRING, IDM_SAVE, "Save Transcript...\tCtrl+S");
    AppendMenuA(file, MF_STRING, IDM_EXPORT, "Export Transcript...");
    AppendMenuA(file, MF_STRING, IDM_PRINT, "Print...");
    AppendMenuA(file, MF_SEPARATOR, 0, NULL);
    AppendMenuA(file, MF_STRING, IDM_EXIT, "Exit");

    AppendMenuA(edit, MF_STRING, IDM_COPY,      "Copy\tCtrl+C");
    AppendMenuA(edit, MF_STRING, IDM_PASTE,     "Paste\tCtrl+V");
    AppendMenuA(edit, MF_STRING, IDM_SELECTALL, "Select All\tCtrl+A");
    AppendMenuA(edit, MF_SEPARATOR, 0, NULL);
    AppendMenuA(edit, MF_STRING, IDM_FIND,      "Find...\tCtrl+F");

    AppendMenuA(convo, MF_STRING, IDM_NEWCHAT, "New Chat\tCtrl+N");
    AppendMenuA(convo, MF_STRING, IDM_SEND,    "Send");
    AppendMenuA(convo, MF_SEPARATOR, 0, NULL);
    AppendMenuA(convo, MF_STRING, IDM_SETTINGS, "Settings...");
    AppendMenuA(convo, MF_SEPARATOR, 0, NULL);
    AppendMenuA(convo, MF_STRING, IDM_CLEAR,   "Clear Conversation");

    AppendMenuA(model, MF_STRING, IDM_MODELINFO, "Model Info");
    AppendMenuA(model, MF_STRING, IDM_PERF, "Performance");
    AppendMenuA(model, MF_SEPARATOR, 0, NULL);
    AppendMenuA(model, MF_STRING, IDM_SETTINGS, "Model Settings...");

    AppendMenuA(tools, MF_STRING, IDM_TEMPLATES, "Templates");
    AppendMenuA(tools, MF_STRING, IDM_FIND, "Find...\tCtrl+F");

    AppendMenuA(help, MF_STRING, IDM_HELPTOPICS, "Help Topics");
    AppendMenuA(help, MF_STRING, IDM_SHORTCUTS, "Keyboard Shortcuts...");
    AppendMenuA(help, MF_STRING, IDM_SLASHHELP, "Slash Commands...");
    AppendMenuA(help, MF_SEPARATOR, 0, NULL);
    AppendMenuA(help, MF_STRING, IDM_ABOUT, "About");

    AppendMenuA(menu, MF_POPUP, (UINT_PTR)file, "File");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)edit, "Edit");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)convo, "Conversation");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)model, "Model");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)tools, "Tools");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)help, "Help");
    SetMenu(hwnd, menu);
}

static void create_fonts(void) {
    NONCLIENTMETRICSA ncm;
    LOGFONTA lf;
    ZeroMemory(&ncm, sizeof(ncm));
    ncm.cbSize = sizeof(ncm);
    SystemParametersInfoA(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0);

    gUiFont = CreateFontIndirectA(&ncm.lfMessageFont);

    lf = ncm.lfMessageFont;
    lf.lfHeight = -22;
    lf.lfWeight = FW_BOLD;
    strcpy(lf.lfFaceName, "Tahoma");
    gTitleFont = CreateFontIndirectA(&lf);

    lf = ncm.lfMessageFont;
    lf.lfHeight = -13;
    lf.lfWeight = FW_BOLD;
    strcpy(lf.lfFaceName, "Tahoma");
    gPaneFont = CreateFontIndirectA(&lf);

    lf = ncm.lfMessageFont;
    lf.lfHeight = -14;
    lf.lfWeight = FW_NORMAL;
    strcpy(lf.lfFaceName, "Tahoma");
    gMonoFont = CreateFontIndirectA(&lf);

}

static HWND make_control(const char *klass, const char *text, DWORD style, DWORD exstyle, int id, HWND parent) {
    HWND hwnd = CreateWindowExA(exstyle, klass, text, style | WS_CHILD | WS_VISIBLE,
        0, 0, 10, 10, parent, (HMENU)(INT_PTR)id, gInstance, NULL);
    if (hwnd && gUiFont) SendMessageA(hwnd, WM_SETFONT, (WPARAM)gUiFont, TRUE);
    return hwnd;
}

static int add_toolbar_icon(HIMAGELIST hil, const char *dll_name, int icon_index, LPCSTR stock_icon) {
    HICON icon = NULL;
    int shared = 0;
    int slot = -1;
    if (dll_name && *dll_name) {
        ExtractIconExA(dll_name, icon_index, &icon, NULL, 1);
    }
    if (!icon && stock_icon) {
        icon = (HICON)LoadImageA(NULL, stock_icon, IMAGE_ICON, 32, 32,
                                 LR_DEFAULTCOLOR | LR_SHARED);
        shared = icon ? 1 : 0;
    }
    if (!icon) {
        icon = (HICON)LoadImageA(gInstance, MAKEINTRESOURCEA(IDI_APP),
                                 IMAGE_ICON, 32, 32, LR_DEFAULTCOLOR);
    }
    if (icon) {
        slot = ImageList_AddIcon(hil, icon);
        if (!shared) DestroyIcon(icon);
    }
    return slot;
}

static HICON load_extracted_icon(const char *dll_name, int icon_index, int size) {
    HICON large = NULL, small = NULL, icon = NULL;
    if (!dll_name || !*dll_name) return NULL;
    ExtractIconExA(dll_name, icon_index, &large, &small, 1);
    icon = small ? small : large;
    if (icon && size != 16) {
        HICON copy = (HICON)CopyImage(icon, IMAGE_ICON, size, size, LR_COPYFROMRESOURCE);
        if (copy) {
            if (small && small != copy) DestroyIcon(small);
            if (large && large != small && large != copy) DestroyIcon(large);
            return copy;
        }
    }
    if (large && large != icon) DestroyIcon(large);
    return icon;
}

static HICON make_fallback_command_icon(int stop_icon, int size) {
    HDC screen = GetDC(NULL);
    HDC color_dc = CreateCompatibleDC(screen);
    HDC mask_dc = CreateCompatibleDC(screen);
    HBITMAP color_bmp = CreateCompatibleBitmap(screen, size, size);
    HBITMAP mask_bmp = CreateBitmap(size, size, 1, 1, NULL);
    HBITMAP old_color = NULL;
    HBITMAP old_mask = NULL;
    HICON icon = NULL;
    ICONINFO ii;
    RECT rc;

    if (!screen || !color_dc || !mask_dc || !color_bmp || !mask_bmp) goto done;

    old_color = (HBITMAP)SelectObject(color_dc, color_bmp);
    old_mask = (HBITMAP)SelectObject(mask_dc, mask_bmp);
    SetRect(&rc, 0, 0, size, size);
    FillRect(color_dc, &rc, GetSysColorBrush(COLOR_BTNFACE));
    FillRect(mask_dc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));

    if (stop_icon) {
        HPEN red = CreatePen(PS_SOLID, size >= 24 ? 5 : 3, RGB(190, 0, 0));
        HPEN dark = CreatePen(PS_SOLID, 1, RGB(96, 0, 0));
        HPEN old = (HPEN)SelectObject(color_dc, dark);
        MoveToEx(color_dc, size / 4, size / 4, NULL);
        LineTo(color_dc, size - size / 4, size - size / 4);
        MoveToEx(color_dc, size - size / 4, size / 4, NULL);
        LineTo(color_dc, size / 4, size - size / 4);
        SelectObject(color_dc, red);
        MoveToEx(color_dc, size / 4 + 1, size / 4 + 1, NULL);
        LineTo(color_dc, size - size / 4 - 1, size - size / 4 - 1);
        MoveToEx(color_dc, size - size / 4 - 1, size / 4 + 1, NULL);
        LineTo(color_dc, size / 4 + 1, size - size / 4 - 1);
        SelectObject(color_dc, old);
        DeleteObject(red);
        DeleteObject(dark);
    } else {
        POINT pts[7];
        HBRUSH green = CreateSolidBrush(RGB(31, 168, 38));
        HPEN outline = CreatePen(PS_SOLID, 1, RGB(0, 102, 0));
        HGDIOBJ old_brush = SelectObject(color_dc, green);
        HGDIOBJ old_pen = SelectObject(color_dc, outline);
        pts[0].x = size / 6;     pts[0].y = size / 3;
        pts[1].x = size / 2;     pts[1].y = size / 3;
        pts[2].x = size / 2;     pts[2].y = size / 6;
        pts[3].x = size - 3;     pts[3].y = size / 2;
        pts[4].x = size / 2;     pts[4].y = size - size / 6;
        pts[5].x = size / 2;     pts[5].y = size - size / 3;
        pts[6].x = size / 6;     pts[6].y = size - size / 3;
        Polygon(color_dc, pts, 7);
        SelectObject(color_dc, old_pen);
        SelectObject(color_dc, old_brush);
        DeleteObject(outline);
        DeleteObject(green);
    }

    ZeroMemory(&ii, sizeof(ii));
    ii.fIcon = TRUE;
    ii.hbmColor = color_bmp;
    ii.hbmMask = mask_bmp;
    icon = CreateIconIndirect(&ii);

done:
    if (old_color) SelectObject(color_dc, old_color);
    if (old_mask) SelectObject(mask_dc, old_mask);
    if (color_bmp) DeleteObject(color_bmp);
    if (mask_bmp) DeleteObject(mask_bmp);
    if (color_dc) DeleteDC(color_dc);
    if (mask_dc) DeleteDC(mask_dc);
    if (screen) ReleaseDC(NULL, screen);
    return icon;
}

static HICON load_command_button_icon(int stop_icon) {
    static const struct { const char *dll_name; int index; } send_icons[] = {
        { "browseui.dll", 32 }, { "browseui.dll", 33 },
        { "shdocvw.dll", 32 },  { "shell32.dll", 138 },
        { "shell32.dll", 146 }
    };
    static const struct { const char *dll_name; int index; } stop_icons[] = {
        { "browseui.dll", 26 }, { "browseui.dll", 27 },
        { "shdocvw.dll", 26 },  { "shell32.dll", 131 },
        { "shell32.dll", 109 }
    };
    int i;
    const int size = 24;
    if (stop_icon) {
        for (i = 0; i < (int)(sizeof(stop_icons) / sizeof(stop_icons[0])); i++) {
            HICON icon = load_extracted_icon(stop_icons[i].dll_name, stop_icons[i].index, size);
            if (icon) return icon;
        }
        {
            HICON shared = (HICON)LoadImageA(NULL, IDI_ERROR, IMAGE_ICON, size, size,
                                             LR_DEFAULTCOLOR | LR_SHARED);
            if (shared) {
                HICON icon = (HICON)CopyImage(shared, IMAGE_ICON, size, size, 0);
                if (icon) return icon;
            }
        }
    } else {
        for (i = 0; i < (int)(sizeof(send_icons) / sizeof(send_icons[0])); i++) {
            HICON icon = load_extracted_icon(send_icons[i].dll_name, send_icons[i].index, size);
            if (icon) return icon;
        }
    }
    return make_fallback_command_icon(stop_icon, size);
}

static HIMAGELIST load_builtin_toolbar_imagelist(void) {
    static const struct {
        const char *dll_name;
        int icon_index;
        LPCSTR stock_icon;
    } icons[] = {
        { "shell32.dll",  1,  IDI_APPLICATION }, // document
        { "shell32.dll",  4,  IDI_APPLICATION }, // open folder
        { "shell32.dll",  6,  IDI_APPLICATION }, // floppy disk
        { "shell32.dll",  1,  IDI_APPLICATION }, // document/export
        { "shell32.dll", 15,  IDI_APPLICATION }, // printer
        { "shell32.dll", 21,  IDI_APPLICATION }, // settings/properties
        { "shell32.dll", 16,  IDI_INFORMATION }, // computer/info
        { "shell32.dll", 22,  IDI_APPLICATION }, // search/performance
        { "shell32.dll",  1,  IDI_APPLICATION }, // template document
        { NULL,           0,  IDI_QUESTION    }, // help
        { NULL,           0,  IDI_INFORMATION }  // about
    };
    HIMAGELIST hil = ImageList_Create(32, 32, ILC_COLOR32 | ILC_MASK,
                                      (int)(sizeof(icons) / sizeof(icons[0])), 0);
    int i;
    if (!hil) return NULL;
    for (i = 0; i < (int)(sizeof(icons) / sizeof(icons[0])); i++) {
        if (add_toolbar_icon(hil, icons[i].dll_name, icons[i].icon_index,
                             icons[i].stock_icon) < 0) {
            ImageList_Destroy(hil);
            return NULL;
        }
    }
    return hil;
}

static void get_pc_specs_detail(char *cpu_out, int cpu_sz,
                                char *mem_out, int mem_sz,
                                char *threads_out, int threads_sz) {
    char cpu[128] = "Unknown CPU";
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_QUERY_VALUE, &key) == ERROR_SUCCESS) {
        DWORD type = 0;
        DWORD sz = (DWORD)sizeof(cpu);
        if (RegQueryValueExA(key, "ProcessorNameString", NULL, &type,
                             (BYTE *)cpu, &sz) == ERROR_SUCCESS) {
            cpu[sizeof(cpu) - 1] = 0;
            // Collapse runs of whitespace — Intel's strings have lots of padding.
            char *r = cpu, *w = cpu; int sp = 0;
            while (*r) {
                if (*r == ' ' || *r == '\t') { if (!sp) *w++ = ' '; sp = 1; }
                else { *w++ = *r; sp = 0; }
                r++;
            }
            if (w > cpu && w[-1] == ' ') w--;
            *w = 0;
        }
        RegCloseKey(key);
    }

    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    unsigned int ram_mb = 0;
    if (GlobalMemoryStatusEx(&mem)) {
        // Round to nearest 64 MB so we don't print "511 MB" on a 512 MB box.
        unsigned long long mb = mem.ullTotalPhys / (1024ULL * 1024ULL);
        ram_mb = (unsigned int)(((mb + 32) / 64) * 64);
    }

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    unsigned int cores = (unsigned int)si.dwNumberOfProcessors;

    snprintf(cpu_out, cpu_sz, "%s", cpu);
    if (ram_mb > 0) {
        snprintf(mem_out, mem_sz, "%u MB RAM", ram_mb);
    } else {
        snprintf(mem_out, mem_sz, "Memory unknown");
    }
    snprintf(threads_out, threads_sz, "%u", cores);
    cpu_out[cpu_sz - 1] = 0;
    mem_out[mem_sz - 1] = 0;
    threads_out[threads_sz - 1] = 0;
}

// Pull a one-line PC-spec banner. Falls back to generic strings if the
// registry/system call fails.
static void get_pc_specs(char *out, int outsz) {
    char cpu[160], mem[64], threads[32];
    get_pc_specs_detail(cpu, sizeof(cpu), mem, sizeof(mem),
                        threads, sizeof(threads));
    snprintf(out, outsz, "%s   %s thread   %s", cpu, threads, mem);
    out[outsz - 1] = 0;
}

static void create_controls(HWND hwnd) {
    HICON icon;
    INITCOMMONCONTROLSEX icc;

    icc.dwSize = sizeof(icc);
    icc.dwICC  = ICC_BAR_CLASSES | ICC_LISTVIEW_CLASSES |
                 ICC_PROGRESS_CLASS | ICC_LINK_CLASS;
    InitCommonControlsEx(&icc);

    gIdentityFrame = make_control("STATIC", "", SS_ETCHEDFRAME, 0, IDC_IDENTITY_FRAME, hwnd);
    gModelGroup = make_control("BUTTON", "System", BS_GROUPBOX, 0, IDC_MODEL_GROUP, hwnd);
    gConversationGroup = make_control("BUTTON", "Conversation", BS_GROUPBOX, 0, IDC_CONV_GROUP, hwnd);
    gDiagnosticsGroup = make_control("BUTTON", "Diagnostics", BS_GROUPBOX, 0, IDC_DIAG_GROUP, hwnd);
    gInputGroup = make_control("BUTTON", "Message", BS_GROUPBOX, 0, IDC_INPUT_GROUP, hwnd);
    if (gModelGroup) SendMessageA(gModelGroup, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gConversationGroup) SendMessageA(gConversationGroup, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gDiagnosticsGroup) SendMessageA(gDiagnosticsGroup, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gInputGroup) SendMessageA(gInputGroup, WM_SETFONT, (WPARAM)gPaneFont, TRUE);

    gTaskPane = make_control("BUTTON", "Chat Tasks", BS_GROUPBOX, 0, IDC_TASKPANE_TASKS, hwnd);
    gRecentPane = make_control("BUTTON", "Recent Chats", BS_GROUPBOX, 0, IDC_TASKPANE_RECENT, hwnd);
    if (gTaskPane) SendMessageA(gTaskPane, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gRecentPane) SendMessageA(gRecentPane, WM_SETFONT, (WPARAM)gPaneFont, TRUE);

    gIcon = make_control("STATIC", "", SS_ICON, 0, IDC_APPICON, hwnd);
    icon = (HICON)LoadImageA(gInstance, MAKEINTRESOURCEA(IDI_APP), IMAGE_ICON, 48, 48, LR_DEFAULTCOLOR);
    if (icon) SendMessageA(gIcon, STM_SETICON, (WPARAM)icon, 0);
    gTitle = make_control("STATIC", APP_DISPLAY_NAME, SS_LEFT, 0, IDC_TITLE, hwnd);
    SendMessageA(gTitle, WM_SETFONT, (WPARAM)gTitleFont, TRUE);
    gSubtitle = make_control("STATIC", APP_TAGLINE, SS_LEFT, 0, IDC_SUBTITLE, hwnd);
    get_pc_specs_detail(gPcCpu, sizeof(gPcCpu), gPcMemory, sizeof(gPcMemory),
                        gPcThreads, sizeof(gPcThreads));
    gCpuLabel = make_control("STATIC", "Computer:", SS_LEFT, 0, IDC_CPU_LABEL, hwnd);
    gCpuValue = make_control("STATIC", gPcCpu, SS_LEFT, 0, IDC_CPU_VALUE, hwnd);
    gMemoryLabel = make_control("STATIC", "Memory:", SS_LEFT, 0, IDC_MEMORY_LABEL, hwnd);
    gMemoryValue = make_control("STATIC", gPcMemory, SS_LEFT, 0, IDC_MEMORY_VALUE, hwnd);
    gThreadsLabel = make_control("STATIC", "Threads:", SS_LEFT, 0, IDC_THREADS_LABEL, hwnd);
    gThreadsValue = make_control("STATIC", gPcThreads, SS_LEFT, 0, IDC_THREADS_VALUE, hwnd);
    if (gCpuLabel) SendMessageA(gCpuLabel, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gMemoryLabel) SendMessageA(gMemoryLabel, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    if (gThreadsLabel) SendMessageA(gThreadsLabel, WM_SETFONT, (WPARAM)gPaneFont, TRUE);
    gModelState = make_control("STATIC", "Loading model", SS_LEFT, 0, IDC_MODEL_STATE, hwnd);
    gModel = make_control("STATIC", MODEL_LABEL, SS_LEFT, 0, IDC_MODEL, hwnd);

    gTranscript = make_control("RichEdit20A", "",
        ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_TABSTOP,
        WS_EX_CLIENTEDGE, IDC_TRANSCRIPT, hwnd);
    SendMessageA(gTranscript, WM_SETFONT, (WPARAM)gMonoFont, TRUE);
    SendMessageA(gTranscript, EM_SETBKGNDCOLOR, 0, RGB(255, 255, 255));
    SendMessageA(gTranscript, EM_EXLIMITTEXT, 0, 524288);

    gInput = make_control("RichEdit20A", "",
        ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | WS_TABSTOP,
        WS_EX_CLIENTEDGE, IDC_INPUT, hwnd);
    SendMessageA(gInput, EM_SETLIMITTEXT, PROMPT_MAX - 1, 0);
    SendMessageA(gInput, EM_SETBKGNDCOLOR, 0, RGB(255, 255, 255));

    gSend  = make_control("BUTTON", "", BS_DEFPUSHBUTTON | BS_ICON | WS_TABSTOP, 0, IDC_SEND, hwnd);
    gStop  = make_control("BUTTON", "", BS_PUSHBUTTON | BS_ICON | WS_TABSTOP,    0, IDC_STOP, hwnd);
    gSendIcon = load_command_button_icon(0);
    gStopIcon = load_command_button_icon(1);
    if (gSendIcon) SendMessageA(gSend, BM_SETIMAGE, IMAGE_ICON, (LPARAM)gSendIcon);
    if (gStopIcon) SendMessageA(gStop, BM_SETIMAGE, IMAGE_ICON, (LPARAM)gStopIcon);
    gClear = make_control("BUTTON", "Clear", BS_PUSHBUTTON | WS_TABSTOP,    0, IDC_CLEAR, hwnd);
    ShowWindow(gClear, SW_HIDE);
    gStatus = CreateWindowExA(0, STATUSCLASSNAMEA, "",
        WS_CHILD | WS_VISIBLE | SBARS_SIZEGRIP,
        0, 0, 10, 10, hwnd, (HMENU)(INT_PTR)IDC_STATUS, gInstance, NULL);

    gDiagnostics = make_control("EDIT", "",
        ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL,
        WS_EX_CLIENTEDGE, IDC_DIAGNOSTICS, hwnd);
    SendMessageA(gDiagnostics, WM_SETFONT, (WPARAM)gMonoFont, TRUE);
    SendMessageA(gDiagnostics, EM_SETLIMITTEXT, 32768, 0);
    SendMessageA(gDiagnostics, EM_SETBKGNDCOLOR, 0, RGB(255, 255, 255));

    // Standard segmented XP progress bar. The v6 manifest gives it Luna skin.
    gProgress = CreateWindowExA(0, PROGRESS_CLASSA, NULL,
        WS_CHILD | WS_VISIBLE,
        0, 0, 10, 16, hwnd, (HMENU)(INT_PTR)IDC_PROGRESS, gInstance, NULL);
    SendMessageA(gProgress, PBM_SETRANGE32, 0, 100);
    SendMessageA(gProgress, PBM_SETSTEP, 4, 0);
    SendMessageA(gProgress, PBM_SETPOS, 0, 0);

    // Built-in XP shell icons. These are OS-owned icon resources loaded from
    // system DLLs into a toolbar image list; no app-generated bitmap strip.
    {
        gToolbar = CreateWindowExA(0, TOOLBARCLASSNAMEA, NULL,
            WS_CHILD | WS_VISIBLE
              | TBSTYLE_FLAT | TBSTYLE_TOOLTIPS
              | CCS_TOP | CCS_NODIVIDER,
            0, 0, 0, 0, hwnd, (HMENU)(INT_PTR)IDC_TOOLBAR, gInstance, NULL);
        SendMessageA(gToolbar, TB_BUTTONSTRUCTSIZE, sizeof(TBBUTTON), 0);
        SendMessageA(gToolbar, TB_SETBITMAPSIZE, 0, MAKELPARAM(32, 32));
        SendMessageA(gToolbar, TB_SETBUTTONSIZE, 0, MAKELPARAM(86, 62));
        if (!gToolbarImages) gToolbarImages = load_builtin_toolbar_imagelist();
        if (gToolbarImages) {
            SendMessageA(gToolbar, TB_SETIMAGELIST, 0, (LPARAM)gToolbarImages);
        }

        TBBUTTON tb_btns[] = {
            { 0, IDM_NEWCHAT,  TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"New Chat" },
            { 1, IDM_OPEN,     TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Open"     },
            { 2, IDM_SAVE,     TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Save"     },
            { 3, IDM_EXPORT,   TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Export"   },
            { 4, IDM_PRINT,    TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Print"    },
            { 0, 0,           TBSTATE_ENABLED, BTNS_SEP,                       {0}, 0, 0                  },
            { 5, IDM_SETTINGS, TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Settings" },
            { 6, IDM_MODELINFO,TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Model Info" },
            { 7, IDM_PERF,     TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Performance" },
            { 8, IDM_TEMPLATES,TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Templates" },
            { 0, 0,           TBSTATE_ENABLED, BTNS_SEP,                       {0}, 0, 0                  },
            { 9, IDM_HELPTOPICS,TBSTATE_ENABLED,BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"Help Topics" },
            {10, IDM_ABOUT,    TBSTATE_ENABLED, BTNS_AUTOSIZE | BTNS_SHOWTEXT, {0}, 0, (INT_PTR)"About"    },
        };
        SendMessageA(gToolbar, TB_ADDBUTTONS,
                     (WPARAM)(sizeof(tb_btns) / sizeof(tb_btns[0])),
                     (LPARAM)tb_btns);
    }

    // Per-message action strip — sits just above the transcript. Acts on
    // the most recent assistant turn / last user prompt. Disabled until
    // an assistant turn has completed.
    gCopyLastBtn = make_control("BUTTON", "Copy last reply", BS_PUSHBUTTON | BS_FLAT | WS_TABSTOP, 0, IDC_COPY_LAST, hwnd);
    gRegenBtn    = make_control("BUTTON", "Regenerate",      BS_PUSHBUTTON | BS_FLAT | WS_TABSTOP, 0, IDC_REGEN,     hwnd);
    gEditLastBtn = make_control("BUTTON", "Edit last prompt",BS_PUSHBUTTON | BS_FLAT | WS_TABSTOP, 0, IDC_EDIT_LAST, hwnd);
    EnableWindow(gCopyLastBtn, FALSE);
    EnableWindow(gRegenBtn,    FALSE);
    EnableWindow(gEditLastBtn, FALSE);
    ShowWindow(gCopyLastBtn, SW_HIDE);
    ShowWindow(gRegenBtn, SW_HIDE);
    ShowWindow(gEditLastBtn, SW_HIDE);

    // Chat history sidebar. Layout (top -> bottom):
    //   [search edit]    <- filter; placeholder cue banner
    //   [chat list]
    //   [+ New Chat]
    gSearch = make_control("EDIT", "",
        ES_AUTOHSCROLL,
        WS_EX_CLIENTEDGE, IDC_SEARCH, hwnd);
    SendMessageA(gSearch, EM_SETLIMITTEXT, sizeof(gSearchText) - 1, 0);
    // Cue banner: lpwstr is required to be a wide string, so transcode.
    {
        WCHAR cue[64];
        MultiByteToWideChar(CP_UTF8, 0, "Search chats\xe2\x80\xa6",
                            -1, cue, (int)(sizeof(cue) / sizeof(cue[0])));
        // 1 = show cue even when focused (Edit_SetCueBannerTextFocused)
        SendMessageW(gSearch, EM_SETCUEBANNER, (WPARAM)TRUE, (LPARAM)cue);
    }
    gChatList   = make_control("LISTBOX", "",
        LBS_NOTIFY | LBS_NOINTEGRALHEIGHT | WS_VSCROLL | WS_BORDER | WS_TABSTOP,
        0, IDC_CHATLIST, hwnd);
    gNewChatBtn = make_control("BUTTON", "New Chat", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_NEWCHATBTN, hwnd);
    gTaskSaveBtn = make_control("BUTTON", "Save Conversation", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_TASK_SAVE, hwnd);
    gTaskExportBtn = make_control("BUTTON", "Export Transcript...", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_TASK_EXPORT, hwnd);
    gTaskImportBtn = make_control("BUTTON", "Import Conversation...", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_TASK_IMPORT, hwnd);
    gTaskClearBtn = make_control("BUTTON", "Clear Conversation", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_TASK_CLEAR, hwnd);

    gBoldBtn = make_control("BUTTON", "B", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_BTN_BOLD, hwnd);
    gItalicBtn = make_control("BUTTON", "I", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_BTN_ITALIC, hwnd);
    gSmileBtn = make_control("BUTTON", ":)", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_BTN_SMILE, hwnd);
    gAttachBtn = make_control("BUTTON", "Attach...", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_BTN_ATTACH, hwnd);
    gPromptToolsBtn = make_control("BUTTON", "Prompt Tools  v", BS_PUSHBUTTON | WS_TABSTOP, 0, IDC_BTN_PROMPTS, hwnd);
    ShowWindow(gBoldBtn, SW_HIDE);
    ShowWindow(gItalicBtn, SW_HIDE);
    ShowWindow(gSmileBtn, SW_HIDE);
    ShowWindow(gAttachBtn, SW_HIDE);
    ShowWindow(gPromptToolsBtn, SW_HIDE);
    gInputLabel = make_control("STATIC", "Type your message below:", SS_LEFT, 0, IDC_INPUT_LABEL, hwnd);
    gInputHint = make_control("STATIC", "Press Enter to send.  Press Shift+Enter for a new line.",
        SS_LEFT, 0, IDC_INPUT_HINT, hwnd);
    if (gInputLabel) SendMessageA(gInputLabel, WM_SETFONT, (WPARAM)gPaneFont, TRUE);

    EnableWindow(gSend,  FALSE);
    EnableWindow(gInput, FALSE);
    EnableWindow(gStop,  FALSE);
    update_model_box();
    update_status_bar();
}

static void layout_controls(HWND hwnd) {
    RECT rc;
    int w, h;
    int pad = 8;
    int compact;
    int toolbar_h;
    int main_top;
    int status_h = 23;
    int input_h;
    int gap = 8;
    int left_w;
    int content_x;
    int main_bottom;
    int diag_h;
    int info_w;
    int task_h;
    int parts[5];

    GetClientRect(hwnd, &rc);
    w = rc.right - rc.left;
    h = rc.bottom - rc.top;
    if (w < 900) w = 900;
    compact = h < 650;
    toolbar_h = compact ? 58 : 64;
    input_h = compact ? 86 : 96;
    diag_h = compact ? 104 : 112;
    task_h = compact ? 172 : 186;
    left_w = (w < 1040) ? 248 : 270;
    info_w = (w < 1040) ? 292 : 340;

    SetRect(&gRcToolbarBand, 0, 0, w, toolbar_h);
    SetRect(&gRcIdentityBand, 0, 0, 0, 0);

    main_top = toolbar_h + gap;
    main_bottom = h - status_h - pad;
    SetRect(&gRcStatusBar, 0, h - status_h, w, h);

    content_x = pad + left_w + 16;

    SetRect(&gRcTaskPane, pad, main_top, pad + left_w, main_top + task_h);
    SetRect(&gRcRecentPane, pad, gRcTaskPane.bottom + gap,
            pad + left_w, main_bottom);

    SetRect(&gRcInputPane, content_x, main_bottom - input_h,
            w - pad, main_bottom);
    SetRect(&gRcModelBox, content_x, gRcInputPane.top - gap - diag_h,
            content_x + info_w, gRcInputPane.top - gap);
    SetRect(&gRcDiagnosticsPane, gRcModelBox.right + gap, gRcModelBox.top,
            w - pad, gRcModelBox.bottom);
    SetRect(&gRcConversationPane, content_x, main_top,
            w - pad, gRcModelBox.top - gap);
    if (gRcConversationPane.bottom < gRcConversationPane.top + 138) {
        gRcConversationPane.bottom = gRcConversationPane.top + 138;
        gRcModelBox.top = gRcConversationPane.bottom + gap;
        gRcModelBox.bottom = gRcModelBox.top + diag_h;
        gRcModelBox.right = gRcModelBox.left + info_w;
        gRcDiagnosticsPane.left = gRcModelBox.right + gap;
        gRcDiagnosticsPane.top = gRcModelBox.top;
        gRcDiagnosticsPane.right = w - pad;
        gRcDiagnosticsPane.bottom = gRcModelBox.bottom;
        gRcInputPane.top = gRcDiagnosticsPane.bottom + gap;
        gRcInputPane.bottom = main_bottom;
    }

    ShowWindow(gIdentityFrame, SW_HIDE);
    ShowWindow(gIcon, SW_HIDE);
    ShowWindow(gTitle, SW_HIDE);
    ShowWindow(gSubtitle, SW_HIDE);

    MoveWindow(gModelGroup, gRcModelBox.left, gRcModelBox.top,
               gRcModelBox.right - gRcModelBox.left,
               gRcModelBox.bottom - gRcModelBox.top, TRUE);
    {
        int label_x = gRcModelBox.left + 12;
        int value_x = gRcModelBox.left + 86;
        int row_y = gRcModelBox.top + 22;
        int value_w = gRcModelBox.right - value_x - 12;
        MoveWindow(gCpuLabel, label_x, row_y, 68, 18, TRUE);
        MoveWindow(gCpuValue, value_x, row_y, value_w, 18, TRUE);
        row_y += 20;
        MoveWindow(gMemoryLabel, label_x, row_y, 68, 18, TRUE);
        MoveWindow(gMemoryValue, value_x, row_y, value_w, 18, TRUE);
        row_y += 20;
        MoveWindow(gThreadsLabel, label_x, row_y, 68, 18, TRUE);
        MoveWindow(gThreadsValue, value_x, row_y, value_w, 18, TRUE);
        row_y += 20;
        MoveWindow(gModelState, label_x, row_y, 86, 18, TRUE);
        MoveWindow(gModel, value_x, row_y, value_w, 18, TRUE);
        MoveWindow(gProgress, value_x, row_y + 1, value_w, 16, TRUE);
    }
    update_model_box();

    // COMCTL32 toolbar handles its own sizing; we just place + autosize it.
    if (gToolbar) {
        MoveWindow(gToolbar, 0, 0, w, toolbar_h, TRUE);
        SendMessageA(gToolbar, TB_AUTOSIZE, 0, 0);
    }

    MoveWindow(gTaskPane, gRcTaskPane.left, gRcTaskPane.top,
               gRcTaskPane.right - gRcTaskPane.left,
               gRcTaskPane.bottom - gRcTaskPane.top, TRUE);
    {
        int x = gRcTaskPane.left + 12;
        int y = gRcTaskPane.top + 24;
        int bw = gRcTaskPane.right - gRcTaskPane.left - 24;
        MoveWindow(gNewChatBtn, x, y, bw, 24, TRUE);
        y += 28;
        MoveWindow(gTaskSaveBtn, x, y, bw, 24, TRUE);
        y += 28;
        MoveWindow(gTaskExportBtn, x, y, bw, 24, TRUE);
        y += 28;
        MoveWindow(gTaskImportBtn, x, y, bw, 24, TRUE);
        y += 28;
        MoveWindow(gTaskClearBtn, x, y, bw, 24, TRUE);
    }

    {
        int y = gRcRecentPane.top + 24;
        int x = gRcRecentPane.left + 12;
        int bw = gRcRecentPane.right - gRcRecentPane.left - 24;
        int search_h = 24;
        MoveWindow(gRecentPane, gRcRecentPane.left, gRcRecentPane.top,
                   gRcRecentPane.right - gRcRecentPane.left,
                   gRcRecentPane.bottom - gRcRecentPane.top, TRUE);
        MoveWindow(gSearch, x, y, bw, search_h, TRUE);
        y += search_h + 8;
        MoveWindow(gChatList, x, y, bw,
                   gRcRecentPane.bottom - y - 12, TRUE);
    }

    MoveWindow(gConversationGroup, gRcConversationPane.left, gRcConversationPane.top,
               gRcConversationPane.right - gRcConversationPane.left,
               gRcConversationPane.bottom - gRcConversationPane.top, TRUE);
    {
        int action_y = gRcConversationPane.top + 21;
        int action_x = gRcConversationPane.left + 10;
        int action_h = 24;
        int action_gap = 6;
        int transcript_y = action_y + action_h + 7;
        MoveWindow(gCopyLastBtn, action_x, action_y, 104, action_h, TRUE);
        action_x += 104 + action_gap;
        MoveWindow(gRegenBtn, action_x, action_y, 92, action_h, TRUE);
        action_x += 92 + action_gap;
        MoveWindow(gEditLastBtn, action_x, action_y, 104, action_h, TRUE);
        ShowWindow(gCopyLastBtn, SW_SHOW);
        ShowWindow(gRegenBtn, SW_SHOW);
        ShowWindow(gEditLastBtn, SW_SHOW);
        MoveWindow(gTranscript,
            gRcConversationPane.left + 10, transcript_y,
            gRcConversationPane.right - gRcConversationPane.left - 20,
            gRcConversationPane.bottom - transcript_y - 10, TRUE);
    }

    MoveWindow(gDiagnosticsGroup, gRcDiagnosticsPane.left, gRcDiagnosticsPane.top,
               gRcDiagnosticsPane.right - gRcDiagnosticsPane.left,
               gRcDiagnosticsPane.bottom - gRcDiagnosticsPane.top, TRUE);
    MoveWindow(gDiagnostics,
        gRcDiagnosticsPane.left + 10, gRcDiagnosticsPane.top + 22,
        gRcDiagnosticsPane.right - gRcDiagnosticsPane.left - 20,
        gRcDiagnosticsPane.bottom - gRcDiagnosticsPane.top - 32, TRUE);

    {
        int input_x = gRcInputPane.left + 12;
        int input_y = gRcInputPane.top + 39;
        int input_box_h = compact ? 30 : 34;
        int icon_sz = input_box_h;
        int button_gap = 6;
        int buttons_w = icon_sz * 2 + button_gap;
        int button_x = gRcInputPane.right - buttons_w - 14;
        int input_w = button_x - input_x - 12;
        MoveWindow(gInputGroup, gRcInputPane.left, gRcInputPane.top,
                   gRcInputPane.right - gRcInputPane.left,
                   gRcInputPane.bottom - gRcInputPane.top, TRUE);
        MoveWindow(gInputLabel, gRcInputPane.left + 12, gRcInputPane.top + 19,
                   220, 18, TRUE);
        ShowWindow(gBoldBtn, SW_HIDE);
        ShowWindow(gItalicBtn, SW_HIDE);
        ShowWindow(gSmileBtn, SW_HIDE);
        ShowWindow(gAttachBtn, SW_HIDE);
        ShowWindow(gPromptToolsBtn, SW_HIDE);
        ShowWindow(gClear, SW_HIDE);

        if (input_w < 160) input_w = 160;
        MoveWindow(gInput, input_x, input_y, input_w, input_box_h, TRUE);
        MoveWindow(gSend, button_x, input_y, icon_sz, input_box_h, TRUE);
        MoveWindow(gStop, button_x + icon_sz + button_gap, input_y,
                   icon_sz, input_box_h, TRUE);
        MoveWindow(gInputHint, gRcInputPane.left + 12, input_y + input_box_h + 9,
                   gRcInputPane.right - gRcInputPane.left - 24, 16, TRUE);
    }

    SetWindowPos(gIdentityFrame, HWND_BOTTOM, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    SetWindowPos(gModelGroup, HWND_BOTTOM, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    SetWindowPos(gConversationGroup, HWND_BOTTOM, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    SetWindowPos(gDiagnosticsGroup, HWND_BOTTOM, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    SetWindowPos(gInputGroup, HWND_BOTTOM, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

    MoveWindow(gStatus, gRcStatusBar.left, gRcStatusBar.top,
               gRcStatusBar.right - gRcStatusBar.left,
               gRcStatusBar.bottom - gRcStatusBar.top, TRUE);
    parts[0] = 190;
    parts[1] = (w > 1000) ? 430 : 360;
    parts[2] = parts[1] + 110;
    parts[3] = parts[2] + 120;
    parts[4] = -1;
    SendMessageA(gStatus, SB_SETPARTS, 5, (LPARAM)parts);
    update_status_bar();

    RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE | RDW_ALLCHILDREN);
}

// =============================================================
//  Chat persistence + sidebar
// =============================================================
//
// Each chat is one .txt file in %APPDATA%\bliss-chat\chats\ named
// <unix_timestamp>.txt. Format:
//
//   TITLE: <first user message, truncated>
//   TIME: <ISO 8601 local timestamp>
//   ---
//   [USER]
//   <text>
//   [ASSISTANT]
//   <text>
//   ...
//
// gChats[] mirrors the on-disk list, sorted newest first. gActiveIdx
// is the chat to which new turns get appended. Selecting a different
// chat in the sidebar switches active (sends /reset to the backend so
// the cache matches the empty state, and replays the historical text
// into the transcript view — but we do NOT replay tokens through the
// model, so previous-turn context is lost; this is the v1 trade-off).

// (gChats / gChatCount / gActiveIdx / gChatsDir declared near the top.)

// Resolve %APPDATA%\bliss-chat\chats and create it if missing.
static void chats_resolve_dir(void) {
    char appdata[MAX_PATH];
    if (SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, appdata) != S_OK) {
        snprintf(gChatsDir, sizeof(gChatsDir), "%s\\chats", gAppDir);
    } else {
        snprintf(gChatsDir, sizeof(gChatsDir), "%s\\bliss-chat\\chats", appdata);
    }
    char parent[MAX_PATH];
    snprintf(parent, sizeof(parent), "%s\\bliss-chat", appdata);
    CreateDirectoryA(parent, NULL);          // ok if it exists
    CreateDirectoryA(gChatsDir, NULL);
}

// Truncate a line to fit, dropping a trailing newline if present.
static void chat_chomp_title(char *s) {
    char *p = strchr(s, '\n'); if (p) *p = 0;
    p = strchr(s, '\r'); if (p) *p = 0;
    int n = (int)strlen(s);
    if (n > CHAT_TITLE_MAX - 4) {
        s[CHAT_TITLE_MAX - 4] = 0;
        strcat(s, "...");
    }
}

// Read the TITLE: header line of an existing chat file.
static void chat_read_title(const char *path, char *out, int outsz) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { snprintf(out, outsz, "(unreadable)"); return; }
    char line[CHAT_TITLE_MAX + 32];
    out[0] = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (!strncmp(line, "TITLE: ", 7)) {
            snprintf(out, outsz, "%s", line + 7);
            chat_chomp_title(out);
            break;
        }
        if (!strncmp(line, "---", 3)) break;
    }
    fclose(fp);
    if (!out[0]) snprintf(out, outsz, "(empty chat)");
}

static int chat_file_has_turns(const char *path) {
    FILE *fp = fopen(path, "rb");
    char line[256];
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp)) {
        if (!strncmp(line, "[USER]", 6) || !strncmp(line, "[ASSISTANT]", 11)) {
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);
    return 0;
}

// Append "<title>" to the listbox in the conventional "M/D h:mm — title"
// shape. Done at the head of the list because we sort newest first.
static void chats_listbox_insert(int idx, const ChatEntry *e) {
    if (!gChatList) return;
    char hour12 = (char)((e->stamp.wHour % 12) ? (e->stamp.wHour % 12) : 12);
    char ampm   = (char)((e->stamp.wHour < 12) ? 'a' : 'p');
    char display[200];
    snprintf(display, sizeof(display), "%d/%d %d:%02d%c   %s",
             e->stamp.wMonth, e->stamp.wDay,
             hour12, e->stamp.wMinute, ampm,
             e->title);
    SendMessageA(gChatList, LB_INSERTSTRING, idx, (LPARAM)display);
}

// Compare for sort (newest first).
static int chat_cmp_desc(const void *a, const void *b) {
    const ChatEntry *ea = (const ChatEntry *)a;
    const ChatEntry *eb = (const ChatEntry *)b;
    FILETIME fa, fb;
    SystemTimeToFileTime(&ea->stamp, &fa);
    SystemTimeToFileTime(&eb->stamp, &fb);
    return CompareFileTime(&fb, &fa);
}

// Case-insensitive ASCII substring test. Returns 1 if `needle` is "" or
// is found inside `haystack`, 0 otherwise. The on-disk chat titles are
// only loosely sanitized (user text), so we tolerate any 8-bit input.
static int ascii_istrstr(const char *haystack, const char *needle) {
    size_t nlen, i;
    if (!needle || !*needle) return 1;
    if (!haystack) return 0;
    nlen = strlen(needle);
    for (i = 0; haystack[i]; i++) {
        size_t j;
        for (j = 0; j < nlen; j++) {
            unsigned char a = (unsigned char)haystack[i + j];
            unsigned char b = (unsigned char)needle[j];
            if (!a) return 0;
            if (a >= 'A' && a <= 'Z') a = (unsigned char)(a + 32);
            if (b >= 'A' && b <= 'Z') b = (unsigned char)(b + 32);
            if (a != b) break;
        }
        if (j == nlen) return 1;
    }
    return 0;
}

// Repopulates the listbox from gChats[], filtering by the current
// gSearchText (case-insensitive substring). Builds gFilteredIndices[]
// so click handlers can map row -> gChats slot.
static void chats_repopulate_listbox(void) {
    if (!gChatList) return;
    SendMessageA(gChatList, LB_RESETCONTENT, 0, 0);
    gFilteredCount = 0;
    int active_row = -1;
    for (int i = 0; i < gChatCount; i++) {
        if (gSearchText[0] && !ascii_istrstr(gChats[i].title, gSearchText)) continue;
        gFilteredIndices[gFilteredCount] = i;
        chats_listbox_insert(gFilteredCount, &gChats[i]);
        if (i == gActiveIdx) active_row = gFilteredCount;
        gFilteredCount++;
    }
    if (active_row >= 0) {
        SendMessageA(gChatList, LB_SETCURSEL, (WPARAM)active_row, 0);
    }
}

// Scan gChatsDir, fill gChats[], populate the sidebar.
static void chats_index_disk(void) {
    gChatCount = 0;
    char glob[MAX_PATH];
    snprintf(glob, sizeof(glob), "%s\\*.txt", gChatsDir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(glob, &fd);
    if (h == INVALID_HANDLE_VALUE) goto done;
    do {
        ChatEntry entry;
        if (gChatCount >= MAX_CHATS) break;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ZeroMemory(&entry, sizeof(entry));
        snprintf(entry.path, sizeof(entry.path), "%s\\%s", gChatsDir, fd.cFileName);
        FileTimeToSystemTime(&fd.ftLastWriteTime, &entry.stamp);
        SystemTimeToTzSpecificLocalTime(NULL, &entry.stamp, &entry.stamp);
        chat_read_title(entry.path, entry.title, sizeof(entry.title));
        if (!strcmp(entry.title, "(new chat)") && !chat_file_has_turns(entry.path)) continue;
        gChats[gChatCount++] = entry;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
done:
    qsort(gChats, gChatCount, sizeof(ChatEntry), chat_cmp_desc);
    chats_repopulate_listbox();
}

// Create a new on-disk chat with empty body. Sets gActiveIdx to it
// and refreshes the listbox. Title starts as "(new chat)" and is
// rewritten on the first user turn.
static void chats_create_new(void) {
    if (gChatCount >= MAX_CHATS) return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    time_t now = time(NULL);
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s\\%llu.txt", gChatsDir, (unsigned long long)now);

    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "TITLE: (new chat)\r\n");
    fprintf(fp, "TIME: %04d-%02d-%02dT%02d:%02d:%02d\r\n---\r\n",
            st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    fclose(fp);

    // Insert at front (newest first).
    memmove(&gChats[1], &gChats[0], sizeof(ChatEntry) * (size_t)gChatCount);
    gChatCount++;
    snprintf(gChats[0].path, sizeof(gChats[0].path), "%s", path);
    snprintf(gChats[0].title, sizeof(gChats[0].title), "(new chat)");
    gChats[0].stamp = st;
    gActiveIdx = 0;
    chats_repopulate_listbox();
}

// Append a turn to the active chat file.
static void chats_append_turn(const char *role, const char *text) {
    if (gActiveIdx < 0 || gActiveIdx >= gChatCount) return;
    FILE *fp = fopen(gChats[gActiveIdx].path, "ab");
    if (!fp) return;
    fprintf(fp, "[%s]\r\n%s\r\n", role, text);
    fclose(fp);
}

// Rewrite the TITLE: line of the chat at gChats[idx] in place. Returns 1
// on success. Used by both the auto-title-on-first-turn path and the
// explicit Rename action. `new_title` is copied into gChats[idx].title
// (length-clamped) before being written to the file.
static int chats_write_title(int idx, const char *new_title) {
    if (idx < 0 || idx >= gChatCount) return 0;
    char title[CHAT_TITLE_MAX];
    snprintf(title, sizeof(title), "%s", new_title ? new_title : "");
    chat_chomp_title(title);

    // Read full file, swap TITLE: line, write back.
    FILE *fp = fopen(gChats[idx].path, "rb");
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return 0; }
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return 0; }
    fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[sz] = 0;

    fp = fopen(gChats[idx].path, "wb");
    if (!fp) { free(buf); return 0; }
    fprintf(fp, "TITLE: %s\r\n", title);
    // Skip the original TITLE: line (whatever's up to the first \n).
    char *body = strchr(buf, '\n');
    if (body) fwrite(body + 1, 1, strlen(body + 1), fp);
    free(buf);
    fclose(fp);

    snprintf(gChats[idx].title, sizeof(gChats[idx].title), "%s", title);
    return 1;
}

// On the first user message of a chat, rewrite the TITLE: line so the
// sidebar shows something meaningful.
static void chats_set_title_from(const char *user_text) {
    if (gActiveIdx < 0) return;
    if (strcmp(gChats[gActiveIdx].title, "(new chat)") != 0) return;
    if (chats_write_title(gActiveIdx, user_text)) {
        chats_repopulate_listbox();
    }
}

// Delete a chat: unlink the file, remove from gChats[], refresh.
// Returns 1 if the deleted entry was the active chat, 0 otherwise.
static int chats_delete(int idx) {
    int was_active;
    if (idx < 0 || idx >= gChatCount) return 0;
    was_active = (idx == gActiveIdx);
    DeleteFileA(gChats[idx].path);
    if (idx + 1 < gChatCount) {
        memmove(&gChats[idx], &gChats[idx + 1],
                sizeof(ChatEntry) * (size_t)(gChatCount - idx - 1));
    }
    gChatCount--;
    if (was_active) {
        gActiveIdx = -1;
    } else if (gActiveIdx > idx) {
        gActiveIdx--;
    }
    chats_repopulate_listbox();
    return was_active;
}

// Modal dialog state for the rename dialog. The new title flows back
// through gRenameBuf on IDOK; on IDCANCEL the buffer is left empty.
static char gRenameBuf[CHAT_TITLE_MAX];

static INT_PTR CALLBACK rename_dlg_proc(HWND dlg, UINT msg, WPARAM wparam, LPARAM lparam) {
    (void)lparam;
    switch (msg) {
    case WM_INITDIALOG:
        SetDlgItemTextA(dlg, IDC_RENAME_EDIT, gRenameBuf);
        // Select all so the user can just start typing.
        SendDlgItemMessageA(dlg, IDC_RENAME_EDIT, EM_SETSEL, 0, -1);
        SetFocus(GetDlgItem(dlg, IDC_RENAME_EDIT));
        return FALSE;  // we set focus manually
    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDOK:
            GetDlgItemTextA(dlg, IDC_RENAME_EDIT, gRenameBuf, (int)sizeof(gRenameBuf));
            EndDialog(dlg, IDOK);
            return TRUE;
        case IDCANCEL:
            gRenameBuf[0] = 0;
            EndDialog(dlg, IDCANCEL);
            return TRUE;
        }
        return FALSE;
    case WM_CLOSE:
        gRenameBuf[0] = 0;
        EndDialog(dlg, IDCANCEL);
        return TRUE;
    }
    return FALSE;
}

// Prompt the user for a new title for gChats[idx]. Empty/whitespace ==
// cancel. Persists the new title and refreshes the listbox.
static void chats_rename_interactive(HWND parent, int idx) {
    if (idx < 0 || idx >= gChatCount) return;
    snprintf(gRenameBuf, sizeof(gRenameBuf), "%s", gChats[idx].title);
    INT_PTR rc = DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_RENAME),
                                 parent, rename_dlg_proc, 0);
    if (rc != IDOK) return;
    if (!input_has_text(gRenameBuf)) return;  // empty/whitespace -> cancel
    if (chats_write_title(idx, gRenameBuf)) {
        chats_repopulate_listbox();
    }
}

// Confirm + delete gChats[idx]. If the deleted chat was active, fall back
// to a fresh new chat (matches the existing IDC_NEWCHATBTN behavior).
static void chats_delete_interactive(HWND parent, int idx) {
    if (idx < 0 || idx >= gChatCount) return;
    char prompt[CHAT_TITLE_MAX + 64];
    snprintf(prompt, sizeof(prompt),
             "Delete chat \"%s\"?\n\nThis cannot be undone.",
             gChats[idx].title);
    if (MessageBoxA(parent, prompt, APP_NAME,
                    MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2) != IDYES) return;
    int was_active = chats_delete(idx);
    if (was_active) {
        if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
            backend_send_line("/reset");
        }
        chats_create_new();
        clear_transcript();
    }
}

// Render a chat file into the transcript pane, replacing whatever is
// there. [USER] turns are blue/bold "You" headers; [ASSISTANT] turns
// are green/bold "Assistant" headers.
static void chats_load_into_view(int idx) {
    if (idx < 0 || idx >= gChatCount) return;
    FILE *fp = fopen(gChats[idx].path, "rb");
    if (!fp) return;

    // Reset per-message action state — the on-disk replay does not
    // populate gLastAsst{Start,End} (those track only this-session
    // turns), and gPendingUser belonged to the previous chat.
    gLastAsstStart = -1;
    gLastAsstEnd   = -1;
    gHasAsstTurn   = 0;
    gPendingUser[0] = 0;
    update_msg_actions();

    SetWindowTextA(gTranscript, "");
    rich_append_color("Bliss Chat\r\n", RGB(0, 128, 0), TRUE);
    rich_append_color(gChats[idx].title, RGB(96, 96, 96), FALSE);
    rich_append_color("\r\n\r\n", RGB(96, 96, 96), FALSE);

    char line[8192];
    int in_user = 0, in_asst = 0, past_header = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (!past_header) {
            if (!strncmp(line, "---", 3)) past_header = 1;
            continue;
        }
        if (!strncmp(line, "[USER]", 6)) {
            in_user = 1; in_asst = 0;
            rich_append_color("You\r\n", RGB(0, 84, 166), TRUE);
            continue;
        }
        if (!strncmp(line, "[ASSISTANT]", 11)) {
            in_user = 0; in_asst = 1;
            rich_append_color("\r\nAssistant\r\n", RGB(0, 128, 0), TRUE);
            continue;
        }
        if (in_user || in_asst) {
            rich_append_color(line, RGB(0, 0, 0), FALSE);
        }
    }
    fclose(fp);
    rich_append_color("\r\n", RGB(0, 0, 0), FALSE);
}

// Send a single line to the backend's stdin (the line should NOT include
// a trailing newline — we add it). Returns TRUE on success.
// Buffer is sized to hold the longest realistic command: `/system` plus a
// ~2k system prompt with escaped newlines.
static BOOL backend_send_line(const char *line) {
    if (!line || !gBackendStdinW) return FALSE;
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s\n", line);
    DWORD wrote = 0;
    return WriteFile(gBackendStdinW, buf, (DWORD)strlen(buf), &wrote, NULL);
}

// HKCU\Software\bliss-chat\Settings holds the persisted Temp/Seed.
#define BLISS_REG_SETTINGS "Software\\bliss-chat\\Settings"

static void settings_load(void) {
    HKEY key;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, BLISS_REG_SETTINGS, 0,
                      KEY_READ, &key) != ERROR_SUCCESS) return;
    DWORD type = 0, sz;
    DWORD t_milli = 0;     // store temp as int(*1000) so REG_DWORD fits cleanly
    sz = sizeof(t_milli);
    if (RegQueryValueExA(key, "TempMilli", NULL, &type, (BYTE *)&t_milli, &sz) == ERROR_SUCCESS) {
        gTemp = (float)t_milli / 1000.0f;
    }
    DWORD seed = 0;
    sz = sizeof(seed);
    if (RegQueryValueExA(key, "Seed", NULL, &type, (BYTE *)&seed, &sz) == ERROR_SUCCESS) {
        gSeed = seed;
    }
    // Top-P stored same as Temp: int(*1000) for REG_DWORD cleanliness.
    DWORD p_milli = 0;
    sz = sizeof(p_milli);
    if (RegQueryValueExA(key, "TopPMilli", NULL, &type, (BYTE *)&p_milli, &sz) == ERROR_SUCCESS) {
        gTopP = (float)p_milli / 1000.0f;
    }
    DWORD maxtok = 0;
    sz = sizeof(maxtok);
    if (RegQueryValueExA(key, "MaxTok", NULL, &type, (BYTE *)&maxtok, &sz) == ERROR_SUCCESS) {
        gMaxTok = maxtok;
    }
    // System prompt is REG_SZ. May be empty (treated as "use backend default").
    sz = sizeof(gSysPrompt);
    if (RegQueryValueExA(key, "SystemPrompt", NULL, &type, (BYTE *)gSysPrompt, &sz) != ERROR_SUCCESS) {
        gSysPrompt[0] = 0;
    } else {
        gSysPrompt[sizeof(gSysPrompt) - 1] = 0;
    }
    RegCloseKey(key);
}

static void settings_save(void) {
    HKEY key;
    if (RegCreateKeyExA(HKEY_CURRENT_USER, BLISS_REG_SETTINGS, 0, NULL,
                        REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL,
                        &key, NULL) != ERROR_SUCCESS) return;
    DWORD t_milli = (DWORD)(gTemp * 1000.0f + 0.5f);
    DWORD p_milli = (DWORD)(gTopP * 1000.0f + 0.5f);
    RegSetValueExA(key, "TempMilli",  0, REG_DWORD, (BYTE *)&t_milli,  sizeof(t_milli));
    RegSetValueExA(key, "Seed",       0, REG_DWORD, (BYTE *)&gSeed,    sizeof(gSeed));
    RegSetValueExA(key, "TopPMilli",  0, REG_DWORD, (BYTE *)&p_milli,  sizeof(p_milli));
    RegSetValueExA(key, "MaxTok",     0, REG_DWORD, (BYTE *)&gMaxTok,  sizeof(gMaxTok));
    RegSetValueExA(key, "SystemPrompt", 0, REG_SZ,
                   (BYTE *)gSysPrompt, (DWORD)(strlen(gSysPrompt) + 1));
    RegCloseKey(key);
}

// Wipe the whole HKCU\Software\bliss-chat\Settings subkey and revert
// the in-memory mirrors to defaults. Used by "Reset All Settings".
static void settings_reset_all(void) {
    gTemp   = DEFAULT_TEMP;
    gSeed   = DEFAULT_SEED;
    gTopP   = DEFAULT_TOPP;
    gMaxTok = DEFAULT_MAXTOK;
    gSysPrompt[0] = 0;
    // Delete the subkey so a stale value can't be resurrected.
    RegDeleteKeyA(HKEY_CURRENT_USER, BLISS_REG_SETTINGS);
}

// Escape newlines in a system-prompt string so it survives the single-line
// stdin transport to the backend. Backend re-inflates "\\n" -> '\n'.
static void escape_newlines(const char *in, char *out, size_t outsz) {
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 3 < outsz; i++) {
        if (in[i] == '\r') continue;
        if (in[i] == '\n') {
            out[o++] = '\\';
            out[o++] = 'n';
        } else if (in[i] == '\\') {
            out[o++] = '\\';
            out[o++] = '\\';
        } else {
            out[o++] = in[i];
        }
    }
    out[o] = 0;
}

// Push the current settings down to the backend via slash commands.
// Safe to call any time — backend echoes an INFO + EOT for each.
static void settings_apply_to_backend(void) {
    char buf[3072];
    snprintf(buf, sizeof(buf), "/temp %.3f", gTemp);
    backend_send_line(buf);
    snprintf(buf, sizeof(buf), "/topp %.3f", gTopP);
    backend_send_line(buf);
    snprintf(buf, sizeof(buf), "/maxtok %lu", (unsigned long)gMaxTok);
    backend_send_line(buf);
    if (gSeed != 0) {
        snprintf(buf, sizeof(buf), "/seed %lu", (unsigned long)gSeed);
        backend_send_line(buf);
    }
    if (gSysPrompt[0]) {
        char esc[2200];
        escape_newlines(gSysPrompt, esc, sizeof(esc));
        snprintf(buf, sizeof(buf), "/system %s", esc);
        backend_send_line(buf);
    }
}

// Dialog proc for the Settings dialog. Loads gTemp / gSeed / gTopP /
// gMaxTok / gSysPrompt into the controls on init, writes them back on
// OK, and handles the preset / random-seed / reset buttons.
static INT_PTR CALLBACK settings_dlg_proc(HWND dlg, UINT msg, WPARAM wparam, LPARAM lparam) {
    char buf[64];
    (void)lparam;
    switch (msg) {
    case WM_INITDIALOG:
        snprintf(buf, sizeof(buf), "%.2f", gTemp);
        SetDlgItemTextA(dlg, IDC_TEMP_EDIT, buf);
        snprintf(buf, sizeof(buf), "%lu", (unsigned long)gSeed);
        SetDlgItemTextA(dlg, IDC_SEED_EDIT, buf);
        snprintf(buf, sizeof(buf), "%.2f", gTopP);
        SetDlgItemTextA(dlg, IDC_TOPP_EDIT, buf);
        snprintf(buf, sizeof(buf), "%lu", (unsigned long)gMaxTok);
        SetDlgItemTextA(dlg, IDC_MAXTOK_EDIT, buf);
        SetDlgItemTextA(dlg, IDC_SYSPROMPT_EDIT, gSysPrompt);
        return TRUE;

    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDOK: {
            float t, p;
            DWORD maxtok;
            char sysbuf[2048];
            GetDlgItemTextA(dlg, IDC_TEMP_EDIT, buf, sizeof(buf));
            t = (float)atof(buf);
            if (t < 0.0f) t = 0.0f;
            if (t > 5.0f) t = 5.0f;
            gTemp = t;
            GetDlgItemTextA(dlg, IDC_SEED_EDIT, buf, sizeof(buf));
            gSeed = (DWORD)strtoul(buf, NULL, 10);
            GetDlgItemTextA(dlg, IDC_TOPP_EDIT, buf, sizeof(buf));
            p = (float)atof(buf);
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            gTopP = p;
            GetDlgItemTextA(dlg, IDC_MAXTOK_EDIT, buf, sizeof(buf));
            maxtok = (DWORD)strtoul(buf, NULL, 10);
            if (maxtok > 2048) maxtok = 2048;
            gMaxTok = maxtok;
            GetDlgItemTextA(dlg, IDC_SYSPROMPT_EDIT, sysbuf, sizeof(sysbuf));
            snprintf(gSysPrompt, sizeof(gSysPrompt), "%s", sysbuf);
            settings_save();
            settings_apply_to_backend();
            EndDialog(dlg, IDOK);
            return TRUE;
        }

        case IDCANCEL:
            EndDialog(dlg, IDCANCEL);
            return TRUE;

        case IDC_PRESET_GREEDY:
            SetDlgItemTextA(dlg, IDC_TEMP_EDIT, "0.00");
            return TRUE;

        case IDC_PRESET_BALANCED:
            SetDlgItemTextA(dlg, IDC_TEMP_EDIT, "0.70");
            return TRUE;

        case IDC_PRESET_CREATIVE:
            SetDlgItemTextA(dlg, IDC_TEMP_EDIT, "1.20");
            return TRUE;

        case IDC_SEED_RANDOM: {
            DWORD r = GetTickCount() ^ ((DWORD)(uintptr_t)dlg);
            snprintf(buf, sizeof(buf), "%lu", (unsigned long)r);
            SetDlgItemTextA(dlg, IDC_SEED_EDIT, buf);
            return TRUE;
        }

        case IDC_DEFAULTS:
            // "Reset Defaults" — only clears the Sampling group, leaves
            // the Advanced fields alone (system prompt etc.).
            SetDlgItemTextA(dlg, IDC_TEMP_EDIT, "0.80");
            SetDlgItemTextA(dlg, IDC_SEED_EDIT, "0");
            return TRUE;

        case IDC_RESET_ALL:
            // "Reset All Settings" — wipe every field AND clear the
            // registry subkey so a stale value can't sneak back.
            if (MessageBoxA(dlg,
                "Reset every setting (temperature, seed, top-p, max tokens, system prompt) to defaults and clear the saved registry values?",
                "Reset all settings", MB_ICONQUESTION | MB_YESNO) == IDYES) {
                settings_reset_all();
                SetDlgItemTextA(dlg, IDC_TEMP_EDIT, "0.80");
                SetDlgItemTextA(dlg, IDC_SEED_EDIT, "0");
                SetDlgItemTextA(dlg, IDC_TOPP_EDIT, "0.95");
                SetDlgItemTextA(dlg, IDC_MAXTOK_EDIT, "0");
                SetDlgItemTextA(dlg, IDC_SYSPROMPT_EDIT, "");
            }
            return TRUE;
        }
        return FALSE;

    case WM_CLOSE:
        EndDialog(dlg, IDCANCEL);
        return TRUE;
    }
    return FALSE;
}

// Edit-prompt dialog. lParam at DialogBoxParam time is a char* holding the
// initial text; on OK we write the edited text back to that same buffer
// (sized PROMPT_MAX). Cancel leaves it untouched.
static INT_PTR CALLBACK editprompt_dlg_proc(HWND dlg, UINT msg, WPARAM wparam, LPARAM lparam) {
    static char *out_buf;
    switch (msg) {
    case WM_INITDIALOG:
        out_buf = (char *)lparam;
        if (out_buf) {
            SetDlgItemTextA(dlg, IDC_EDITPROMPT_TEXT, out_buf);
            // Select all so the user can just start typing to replace,
            // or Shift+End / arrow to extend the selection.
            SendDlgItemMessageA(dlg, IDC_EDITPROMPT_TEXT, EM_SETSEL, 0, -1);
        }
        SetFocus(GetDlgItem(dlg, IDC_EDITPROMPT_TEXT));
        return FALSE;  // we set focus manually

    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDOK: {
            if (out_buf) {
                GetDlgItemTextA(dlg, IDC_EDITPROMPT_TEXT, out_buf, PROMPT_MAX);
            }
            EndDialog(dlg, IDOK);
            return TRUE;
        }
        case IDCANCEL:
            EndDialog(dlg, IDCANCEL);
            return TRUE;
        }
        return FALSE;

    case WM_CLOSE:
        EndDialog(dlg, IDCANCEL);
        return TRUE;
    }
    return FALSE;
}

// HKCU\Software\bliss-chat\Window holds the last main-window placement
// (x/y/w/h) so the next launch comes up where you left it.
#define BLISS_REG_KEY "Software\\bliss-chat"

static void win_save_placement(HWND hwnd) {
    WINDOWPLACEMENT wp;
    wp.length = sizeof(wp);
    if (!GetWindowPlacement(hwnd, &wp)) return;
    if (wp.showCmd == SW_SHOWMINIMIZED) return;  // don't persist iconic state
    RECT r = wp.rcNormalPosition;
    HKEY key;
    if (RegCreateKeyExA(HKEY_CURRENT_USER, BLISS_REG_KEY, 0, NULL,
                        REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL,
                        &key, NULL) != ERROR_SUCCESS) return;
    DWORD x = (DWORD)r.left, y = (DWORD)r.top;
    DWORD w = (DWORD)(r.right - r.left), h = (DWORD)(r.bottom - r.top);
    RegSetValueExA(key, "X", 0, REG_DWORD, (BYTE *)&x, sizeof(x));
    RegSetValueExA(key, "Y", 0, REG_DWORD, (BYTE *)&y, sizeof(y));
    RegSetValueExA(key, "W", 0, REG_DWORD, (BYTE *)&w, sizeof(w));
    RegSetValueExA(key, "H", 0, REG_DWORD, (BYTE *)&h, sizeof(h));
    RegCloseKey(key);
}

// Returns TRUE and fills *out if a saved placement exists. Coordinates
// are sanity-clamped against the current virtual desktop, so unplugging
// a second monitor doesn't strand the window off-screen.
static BOOL win_load_placement(RECT *out) {
    HKEY key;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, BLISS_REG_KEY, 0,
                      KEY_READ, &key) != ERROR_SUCCESS) return FALSE;
    DWORD x = 0, y = 0, w = 0, h = 0, type = 0, sz;
    sz = sizeof(x); RegQueryValueExA(key, "X", NULL, &type, (BYTE *)&x, &sz);
    sz = sizeof(y); RegQueryValueExA(key, "Y", NULL, &type, (BYTE *)&y, &sz);
    sz = sizeof(w); RegQueryValueExA(key, "W", NULL, &type, (BYTE *)&w, &sz);
    sz = sizeof(h); RegQueryValueExA(key, "H", NULL, &type, (BYTE *)&h, &sz);
    RegCloseKey(key);
    if (w < 320 || h < 240) return FALSE;
    int sw = GetSystemMetrics(SM_CXSCREEN);
    int sh = GetSystemMetrics(SM_CYSCREEN);
    if ((int)x < -((int)w - 80) || (int)x > sw - 80) x = 40;
    if ((int)y < 0              || (int)y > sh - 80) y = 40;
    out->left   = (LONG)x;
    out->top    = (LONG)y;
    out->right  = (LONG)(x + w);
    out->bottom = (LONG)(y + h);
    return TRUE;
}

// Save the current transcript to a user-chosen .txt file. The Common
// Dialog handles browse / overwrite confirm; we just dump the
// RichEdit's plain text plus a small header.
static void save_transcript_dialog(HWND parent) {
    OPENFILENAMEA ofn;
    char path[MAX_PATH];
    SYSTEMTIME st;
    GetLocalTime(&st);
    snprintf(path, sizeof(path),
             "bliss-chat-%04d%02d%02d-%02d%02d%02d.txt",
             st.wYear, st.wMonth, st.wDay,
             st.wHour, st.wMinute, st.wSecond);

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = parent;
    ofn.lpstrFilter = "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile   = path;
    ofn.nMaxFile    = sizeof(path);
    ofn.lpstrTitle  = "Save Transcript";
    ofn.lpstrDefExt = "txt";
    ofn.Flags       = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (!GetSaveFileNameA(&ofn)) return;

    int n = GetWindowTextLengthA(gTranscript);
    if (n <= 0) {
        MessageBoxA(parent, "Transcript is empty.", APP_NAME, MB_ICONINFORMATION | MB_OK);
        return;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) return;
    GetWindowTextA(gTranscript, buf, n + 1);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        free(buf);
        MessageBoxA(parent, "Could not open file for writing.", APP_NAME, MB_ICONERROR | MB_OK);
        return;
    }
    char header[160];
    snprintf(header, sizeof(header),
             "bliss-chat transcript -- saved %04d-%02d-%02d %02d:%02d:%02d\r\n"
             "----------------------------------------------------------------\r\n",
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    fwrite(header, 1, strlen(header), fp);
    fwrite(buf, 1, (size_t)n, fp);
    fwrite("\r\n", 1, 2, fp);
    fclose(fp);
    free(buf);
}

// ---------- Edit-menu plumbing ----------
//
// Routes Copy / Paste / Select All to the right control:
//  - Copy: prefers the transcript (it's read-only but holds the visible
//    chat history); falls back to whatever has focus so input-box copies
//    still work.
//  - Paste: targets the input box.
//  - Select All: always targets the transcript.

static void do_copy(void) {
    HWND focus = GetFocus();
    HWND target = gTranscript;
    // If user has the input focused (e.g. selecting their own draft),
    // let WM_COPY go there instead of the transcript.
    if (focus == gInput) target = gInput;
    if (target) SendMessageA(target, WM_COPY, 0, 0);
}

static void do_paste(void) {
    if (gInput) {
        SetFocus(gInput);
        SendMessageA(gInput, WM_PASTE, 0, 0);
    }
}

static void do_select_all(void) {
    if (!gTranscript) return;
    SetFocus(gTranscript);
    // RichEdit honors plain EM_SETSEL just fine.
    SendMessageA(gTranscript, EM_SETSEL, 0, -1);
}

// Case-insensitive substring search of the transcript starting from
// the position right after the current selection. Returns 0 if not
// found, 1 if a hit was selected.
static int do_find_in_transcript(const char *needle) {
    if (!gTranscript || !needle || !*needle) return 0;
    int total = GetWindowTextLengthA(gTranscript);
    if (total <= 0) return 0;
    char *hay = (char *)malloc((size_t)total + 1);
    if (!hay) return 0;
    GetWindowTextA(gTranscript, hay, total + 1);

    // Where to start: just after the current selection end. If nothing
    // selected, start at 0. If we've already scanned past the end on a
    // previous "Find Next", wrap to the top.
    DWORD sel_lo = 0, sel_hi = 0;
    SendMessageA(gTranscript, EM_GETSEL, (WPARAM)&sel_lo, (LPARAM)&sel_hi);
    int start = (int)sel_hi;
    if (start < 0 || start >= total) start = 0;

    size_t nlen = strlen(needle);
    int found_at = -1;
    for (int pass = 0; pass < 2 && found_at < 0; pass++) {
        int from = (pass == 0) ? start : 0;
        int to   = (pass == 0) ? total : start;
        for (int i = from; i + (int)nlen <= to; i++) {
            int ok = 1;
            for (size_t j = 0; j < nlen; j++) {
                char a = hay[i + j], b = needle[j];
                if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
                if (b >= 'A' && b <= 'Z') b = (char)(b + 32);
                if (a != b) { ok = 0; break; }
            }
            if (ok) { found_at = i; break; }
        }
    }
    free(hay);
    if (found_at < 0) return 0;
    SendMessageA(gTranscript, EM_SETSEL, (WPARAM)found_at,
                 (LPARAM)(found_at + (int)nlen));
    SendMessageA(gTranscript, EM_SCROLLCARET, 0, 0);
    return 1;
}

static INT_PTR CALLBACK find_dlg_proc(HWND dlg, UINT msg, WPARAM wparam, LPARAM lparam) {
    (void)lparam;
    switch (msg) {
    case WM_INITDIALOG:
        SetDlgItemTextA(dlg, IDC_FIND_EDIT, gLastFind);
        // Move caret to end of edit so typing replaces.
        SendDlgItemMessageA(dlg, IDC_FIND_EDIT, EM_SETSEL, 0, -1);
        return TRUE;

    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDC_FIND_NEXT:
        case IDOK:
            GetDlgItemTextA(dlg, IDC_FIND_EDIT, gLastFind, sizeof(gLastFind));
            if (gLastFind[0]) {
                if (!do_find_in_transcript(gLastFind)) {
                    MessageBoxA(dlg, "Not found.", APP_NAME,
                                MB_ICONINFORMATION | MB_OK);
                }
            }
            return TRUE;

        case IDCANCEL:
            EndDialog(dlg, IDCANCEL);
            return TRUE;
        }
        return FALSE;

    case WM_CLOSE:
        EndDialog(dlg, IDCANCEL);
        return TRUE;
    }
    return FALSE;
}

static INT_PTR CALLBACK shortcuts_dlg_proc(HWND dlg, UINT msg, WPARAM wparam, LPARAM lparam) {
    (void)lparam;
    switch (msg) {
    case WM_INITDIALOG: {
        static const char SHORTCUTS_BODY[] =
            "Keyboard shortcuts\r\n"
            "\r\n"
            "  Ctrl+N      New Chat\r\n"
            "  Ctrl+S      Save Transcript\r\n"
            "  Ctrl+C      Copy\r\n"
            "  Ctrl+V      Paste\r\n"
            "  Ctrl+A      Select All (transcript)\r\n"
            "  Ctrl+F      Find in transcript\r\n"
            "  Esc         Stop generation\r\n"
            "  Enter       Send message\r\n"
            "  Shift+Enter Insert newline in input\r\n"
            "  F2          Rename chat (if available)\r\n"
            "  Del         Delete chat (if available)\r\n";
        SetDlgItemTextA(dlg, IDC_SHORTCUTS_TEXT, SHORTCUTS_BODY);
        return TRUE;
    }

    case WM_COMMAND:
        if (LOWORD(wparam) == IDOK || LOWORD(wparam) == IDCANCEL) {
            EndDialog(dlg, IDOK);
            return TRUE;
        }
        return FALSE;

    case WM_CLOSE:
        EndDialog(dlg, IDOK);
        return TRUE;
    }
    return FALSE;
}

static LRESULT CALLBACK input_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    // Enter sends; Shift+Enter inserts a newline (passthrough).
    if (msg == WM_KEYDOWN && wparam == VK_RETURN) {
        if (GetKeyState(VK_SHIFT) & 0x8000) {
            return CallWindowProcA(gOldInputProc, hwnd, msg, wparam, lparam);
        }
        send_prompt();
        return 0;
    }
    if (msg == WM_CHAR && wparam == VK_RETURN && !(GetKeyState(VK_SHIFT) & 0x8000)) {
        return 0;
    }
    // Esc aborts an in-flight generation. Forwarded to the window
    // proc so the same handler the Stop button uses fires.
    if (msg == WM_KEYDOWN && wparam == VK_ESCAPE) {
        if (InterlockedCompareExchange(&gRunning, 0, 0) && gMain) {
            PostMessageA(gMain, WM_COMMAND, MAKEWPARAM(IDC_STOP, BN_CLICKED), 0);
            return 0;
        }
    }
    return CallWindowProcA(gOldInputProc, hwnd, msg, wparam, lparam);
}

static void subclass_input(void) {
    gOldInputProc = (WNDPROC)SetWindowLongPtrA(gInput, GWLP_WNDPROC, (LONG_PTR)input_proc);
}

// Resolve the listbox row at point `pt` (client coords of gChatList). The
// LB_ITEMFROMPOINT result encodes the row in LOWORD and "is the point
// inside an item?" in HIWORD (0 = inside, 1 = below last item).
static int chatlist_row_at(POINT pt_client) {
    LRESULT r;
    int row;
    int outside;
    r = SendMessageA(gChatList, LB_ITEMFROMPOINT, 0, MAKELPARAM(pt_client.x, pt_client.y));
    row = (int)(short)LOWORD(r);
    outside = (int)HIWORD(r);
    if (outside) return -1;
    if (row < 0 || row >= gFilteredCount) return -1;
    return row;
}

// Pop up the right-click context menu for chat row `row` (listbox row,
// already validated). Items: Rename / Delete / --- / Save Transcript...
// Coordinates are in screen space (TrackPopupMenu expects that).
static void chatlist_popup_menu(HWND parent, int row, int screen_x, int screen_y) {
    HMENU menu = CreatePopupMenu();
    if (!menu) return;
    AppendMenuA(menu, MF_STRING, IDM_RENAME,   "Rename\tF2");
    AppendMenuA(menu, MF_STRING, IDM_DELETE,   "Delete\tDel");
    AppendMenuA(menu, MF_SEPARATOR, 0, NULL);
    AppendMenuA(menu, MF_STRING, IDM_CHATSAVE, "Save Transcript...");
    // Select the row so visually it's clear which one the menu targets.
    SendMessageA(gChatList, LB_SETCURSEL, (WPARAM)row, 0);
    int cmd = (int)TrackPopupMenu(menu,
        TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN,
        screen_x, screen_y, 0, parent, NULL);
    DestroyMenu(menu);
    if (cmd == 0) return;
    if (row < 0 || row >= gFilteredCount) return;
    int chat_idx = gFilteredIndices[row];
    switch (cmd) {
    case IDM_RENAME:
        chats_rename_interactive(parent, chat_idx);
        break;
    case IDM_DELETE:
        chats_delete_interactive(parent, chat_idx);
        break;
    case IDM_CHATSAVE:
        save_transcript_dialog(parent);
        break;
    }
}

static void run_task_list_item(int item) {
    UINT command = 0;
    switch (item) {
    case 0: command = IDM_NEWCHAT; break;
    case 1: command = IDM_SAVE; break;
    case 2: command = IDM_EXPORT; break;
    case 3: command = IDM_OPEN; break;
    case 4: command = IDM_CLEAR; break;
    default: return;
    }
    PostMessageA(gMain, WM_COMMAND, MAKEWPARAM(command, 0), 0);
}

// Listbox subclass: handle F2 (rename), Delete (delete), and right-click
// (context menu). Everything else falls through to the default listbox
// behavior (selection, scroll, etc).
static LRESULT CALLBACK chatlist_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (msg == WM_KEYDOWN) {
        if (wparam == VK_F2) {
            int row = (int)SendMessageA(hwnd, LB_GETCURSEL, 0, 0);
            if (row >= 0 && row < gFilteredCount) {
                chats_rename_interactive(gMain, gFilteredIndices[row]);
            }
            return 0;
        }
        if (wparam == VK_DELETE) {
            int row = (int)SendMessageA(hwnd, LB_GETCURSEL, 0, 0);
            if (row >= 0 && row < gFilteredCount) {
                chats_delete_interactive(gMain, gFilteredIndices[row]);
            }
            return 0;
        }
    }
    if (msg == WM_RBUTTONDOWN) {
        POINT pt;
        pt.x = (short)LOWORD(lparam);
        pt.y = (short)HIWORD(lparam);
        int row = chatlist_row_at(pt);
        if (row >= 0) {
            POINT screen = pt;
            ClientToScreen(hwnd, &screen);
            chatlist_popup_menu(gMain, row, screen.x, screen.y);
        }
        return 0;
    }
    return CallWindowProcA(gOldChatListProc, hwnd, msg, wparam, lparam);
}

static void subclass_chatlist(void) {
    gOldChatListProc = (WNDPROC)SetWindowLongPtrA(gChatList, GWLP_WNDPROC, (LONG_PTR)chatlist_proc);
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case WM_CREATE:
        gMain = hwnd;
        log_init();
        dbg_log("GUI", "WM_CREATE, app dir = %s", gAppDir);
        make_menu(hwnd);
        create_fonts();
        create_controls(hwnd);
        subclass_input();
        subclass_chatlist();
        layout_controls(hwnd);
        chats_resolve_dir();
        chats_index_disk();
        clear_transcript();
        diagnostics_appendf("Waiting for model...");
        if (!file_exists_in_app_dir(BACKEND_EXE) || !file_exists_in_app_dir(MODEL_FILE)
            || !file_exists_in_app_dir(TOKENIZER_FILE)) {
            dbg_log("GUI", "missing files: %s=%d %s=%d %s=%d",
                BACKEND_EXE, file_exists_in_app_dir(BACKEND_EXE),
                MODEL_FILE, file_exists_in_app_dir(MODEL_FILE),
                TOKENIZER_FILE, file_exists_in_app_dir(TOKENIZER_FILE));
            diagnostics_appendf("Required backend/model/tokenizer files are missing.");
            MessageBoxA(hwnd,
                BACKEND_EXE ", " MODEL_FILE ", or " TOKENIZER_FILE " is missing from the application folder.",
                APP_NAME, MB_ICONERROR | MB_OK);
        } else if (!launch_backend()) {
            set_status("Backend failed to start");
            diagnostics_appendf("Backend failed to start.");
        } else {
            diagnostics_appendf("Backend process launched.");
        }
        {
            char msg[MAX_PATH + 64];
            snprintf(msg, sizeof(msg), "Log: %s\r\n", gLogPath);
            rich_append_color(msg, RGB(96, 96, 96), FALSE);
        }
        return 0;

    case WM_SIZE:
        layout_controls(hwnd);
        return 0;

    case WM_ERASEBKGND: {
        RECT rc;
        GetClientRect(hwnd, &rc);
        FillRect((HDC)wparam, &rc, GetSysColorBrush(COLOR_BTNFACE));
        return 1;
    }

    case WM_CTLCOLORSTATIC:
        SetBkColor((HDC)wparam, GetSysColor(COLOR_BTNFACE));
        SetTextColor((HDC)wparam, GetSysColor(COLOR_WINDOWTEXT));
        return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);

    case WM_GETMINMAXINFO: {
        MINMAXINFO *mmi = (MINMAXINFO *)lparam;
        mmi->ptMinTrackSize.x = 900;
        mmi->ptMinTrackSize.y = 560;
        return 0;
    }

    case WM_NOTIFY: {
        LPNMHDR hdr = (LPNMHDR)lparam;
        if (hdr && (hdr->code == NM_CLICK || hdr->code == NM_RETURN)) {
            int command = 0;
            if (hdr->hwndFrom == gNewChatBtn) command = IDC_NEWCHATBTN;
            else if (hdr->hwndFrom == gTaskSaveBtn) command = IDC_TASK_SAVE;
            else if (hdr->hwndFrom == gTaskExportBtn) command = IDC_TASK_EXPORT;
            else if (hdr->hwndFrom == gTaskImportBtn) command = IDC_TASK_IMPORT;
            else if (hdr->hwndFrom == gTaskClearBtn) command = IDC_TASK_CLEAR;
            if (command) {
                SendMessageA(hwnd, WM_COMMAND, MAKEWPARAM(command, 0), (LPARAM)hdr->hwndFrom);
                return 0;
            }
        }
        if (hdr && gTaskList && hdr->hwndFrom == gTaskList &&
            (hdr->code == NM_CLICK || hdr->code == NM_DBLCLK || hdr->code == LVN_ITEMACTIVATE)) {
            LPNMITEMACTIVATE act = (LPNMITEMACTIVATE)lparam;
            int item = act && act->iItem >= 0
                ? act->iItem
                : ListView_GetNextItem(gTaskList, -1, LVNI_SELECTED);
            if (item >= 0) run_task_list_item(item);
            return 0;
        }
        break;
    }

    case WM_TIMER:
        if (wparam == TIMER_STATUS && InterlockedCompareExchange(&gRunning, 0, 0)) {
            DWORD elapsed_ms = GetTickCount() - gRunStarted;
            double elapsed_s = elapsed_ms / 1000.0;
            LONG chars = InterlockedCompareExchange(&gRunChars, 0, 0);
            // Tokens-per-second from BPE-typical ~4 chars/token. Rough,
            // but matches the EOT footer's measured tok/s within ~10%.
            double approx_tps = 0.0;
            if (elapsed_s > 0.25 && chars > 0) {
                approx_tps = (double)chars / 4.0 / elapsed_s;
            }
            char text[96];
            if (approx_tps > 0.05) {
                snprintf(text, sizeof(text),
                         "Generating... %.1f sec, %.1f tok/s",
                         elapsed_s, approx_tps);
            } else {
                snprintf(text, sizeof(text),
                         "Generating... %.1f sec", elapsed_s);
            }
            set_status(text);
            return 0;
        }
        break;

    case WM_BACKEND_READY:
        InterlockedExchange(&gBackendReady, 1);
        set_running(FALSE);
        set_status("Ready");
        diagnostics_appendf("Backend ready.");
        layout_controls(hwnd);
        InvalidateRect(hwnd, &gRcModelBox, TRUE);
        InvalidateRect(hwnd, &gRcStatusBar, TRUE);
        rich_append_color("Backend ready. Type a message and press Enter.\r\n\r\n", RGB(0, 128, 0), TRUE);
        // Push the persisted Settings down to the new backend.
        settings_apply_to_backend();
        SetFocus(gInput);
        return 0;

    case WM_BACKEND_PROG: {
        if (gProgress) SendMessageA(gProgress, PBM_SETPOS, (WPARAM)wparam, 0);
        return 0;
    }

    case WM_BACKEND_INFO: {
        char *info = (char *)lparam;
        if (info && *info) {
            // Only update the "Model: ..." header for actual model-name
            // banners. Other INFO events — temp changes, /reset notices,
            // etc. — flash through the status bar instead so the model
            // identity stays visible.
            if (!strncmp(info, "Bliss ", 6) || !strncmp(info, "nanochat-", 9)) {
                char label[256];
                snprintf(label, sizeof(label), "%s", info);
                snprintf(gModelName, sizeof(gModelName), "%s", info);
                SetWindowTextA(gModel, label);
                diagnostics_appendf("Model loaded: %s", info);
                layout_controls(hwnd);
                InvalidateRect(hwnd, &gRcModelBox, TRUE);
                InvalidateRect(hwnd, &gRcStatusBar, TRUE);
            } else {
                set_status(info);
                diagnostics_appendf("%s", info);
            }
        }
        if (info) free(info);
        return 0;
    }

    case WM_BACKEND_DEAD:
        InterlockedExchange(&gBackendReady, 0);
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
        set_status("Backend exited");
        diagnostics_appendf("Backend exited.");
        InvalidateRect(hwnd, &gRcModelBox, TRUE);
        InvalidateRect(hwnd, &gRcStatusBar, TRUE);
        rich_append_color("\r\n[backend process exited]\r\n", RGB(192, 0, 0), TRUE);
        return 0;

    case WM_APPEND_TEXT: {
        char *text = (char *)lparam;
        if (text) InterlockedExchangeAdd(&gRunChars, (LONG)strlen(text));
        rich_append_color(text, RGB(0, 0, 0), FALSE);
        free(text);
        return 0;
    }

    case WM_RUN_ERR: {
        char *message = (char *)lparam;
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
        if (message && *message) {
            rich_append_color("\r\n[error: ", RGB(192, 0, 0), TRUE);
            rich_append_color(message, RGB(192, 0, 0), FALSE);
            rich_append_color("]\r\n\r\n", RGB(192, 0, 0), TRUE);
        }
        if (message) free(message);
        set_status("Error");
        diagnostics_appendf("Backend error.");
        SetFocus(gInput);
        return 0;
    }

    case WM_RUN_DONE: {
        char *message = (char *)lparam;
        int  tcount   = (int)wparam;
        DWORD elapsed_ms = GetTickCount() - gRunStarted;
        double elapsed_s = elapsed_ms / 1000.0;
        BOOL was_running = InterlockedCompareExchange(&gRunning, 0, 0) != 0;
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
        if (!was_running) {
            if (message) free(message);
            if (InterlockedCompareExchange(&gBackendReady, 0, 0) != 0) set_status("Ready");
            return 0;
        }
        // Persist the assistant turn to the active chat file (skip if the
        // user just sent a slash command — backend may emit an empty EOT).
        if (message && *message && tcount > 0) {
            chats_append_turn("ASSISTANT", message);
        }
        // Lock in the assistant body's end offset before the trailing
        // newline + stats footer are appended. This is what Copy uses.
        gLastAsstEnd = rich_end_pos();
        if (tcount > 0 && gLastAsstEnd > gLastAsstStart) gHasAsstTurn = 1;
        rich_append_color("\r\n", RGB(0, 0, 0), FALSE);
        // Stats footer + timestamp.
        {
            SYSTEMTIME st;
            GetLocalTime(&st);
            char foot[160];
            if (tcount > 0) {
                snprintf(foot, sizeof(foot),
                    "[%.1f sec - %.2f sec/token - %d tokens - %02d:%02d:%02d]\r\n\r\n",
                    elapsed_s, elapsed_s / (double)tcount, tcount,
                    st.wHour, st.wMinute, st.wSecond);
            } else {
                snprintf(foot, sizeof(foot),
                    "[%.1f sec - no tokens generated - %02d:%02d:%02d]\r\n\r\n",
                    elapsed_s, st.wHour, st.wMinute, st.wSecond);
            }
            rich_append_color(foot, RGB(96, 96, 96), FALSE);
        }
        if (message) free(message);
        set_status("Ready");
        update_msg_actions();
        if (tcount > 0) diagnostics_appendf("%d tokens generated.", tcount);
        else diagnostics_appendf("No tokens generated.");
        SetFocus(gInput);
        return 0;
    }

    case WM_COMMAND:
        switch (LOWORD(wparam)) {
        case IDC_SEND:
        case IDM_SEND:
            send_prompt();
            return 0;
        case IDC_STOP:
        case IDM_STOP: {
            // Send a STOP sentinel down the backend's stdin pipe. NC_RUN
            // polls stdin between tokens; if it sees "\x01STOP\n" mid-
            // generation, it breaks out, emits an INFO + EOT, and
            // returns to the read-line loop without losing the model.
            if (InterlockedCompareExchange(&gRunning, 0, 0) && gBackendStdinW) {
                static const char STOP_SENTINEL[] = "\x01STOP\n";
                DWORD wrote = 0;
                WriteFile(gBackendStdinW, STOP_SENTINEL, (DWORD)sizeof(STOP_SENTINEL) - 1, &wrote, NULL);
                set_status("Stopping...");
            }
            return 0;
        }
        case IDC_CLEAR:
        case IDM_CLEAR:
        case IDC_TASK_CLEAR:
            clear_transcript();
            return 0;
        case IDC_COPY_LAST: {
            // Pull just the body of the most recent assistant turn out of
            // the RichEdit — no "Assistant — h:mm" header, no stats footer.
            // The range was snapshotted in WM_RUN_DONE.
            if (!gHasAsstTurn || gLastAsstEnd <= gLastAsstStart) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            char *body = rich_get_range(gLastAsstStart, gLastAsstEnd);
            if (!body) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            // Trim a single trailing CRLF (the streaming token loop usually
            // leaves the cursor at end-of-line; users don't want it pasted).
            size_t bl = strlen(body);
            while (bl > 0 && (body[bl - 1] == '\n' || body[bl - 1] == '\r')) {
                body[--bl] = 0;
            }
            if (clipboard_set_text(hwnd, body)) {
                set_status("Copied last reply to clipboard.");
            } else {
                set_status("Could not access clipboard.");
            }
            free(body);
            return 0;
        }
        case IDC_REGEN: {
            // Re-run the last user prompt. We send /reset to the backend
            // first to drop the just-finished turn from its KV cache, then
            // re-send gPendingUser through the regular send path. The
            // transcript gets a fresh "Assistant (regenerated) — h:mm"
            // header followed by the new body; we do NOT try to in-place
            // replace the prior body.
            if (!gHasAsstTurn || gPendingUser[0] == 0) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            if (!InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            if (InterlockedCompareExchange(&gRunning, 0, 0)) return 0;
            char prompt_copy[PROMPT_MAX];
            snprintf(prompt_copy, sizeof(prompt_copy), "%s", gPendingUser);
            backend_send_line("/reset");
            // Backend's /reset is synchronous from the GUI's POV in the
            // sense that the next stdin line we write enters a clean
            // context. (NC_RUN handles /reset, emits an INFO + EOT, then
            // returns to the read-line loop.) We don't wait for an event
            // here — WriteFile ordering on the pipe is enough.
            gRegenLabel = 1;
            send_prompt_text(prompt_copy, 0);
            return 0;
        }
        case IDC_EDIT_LAST: {
            // Pop a modal dialog prefilled with the last user prompt. On
            // OK we send /reset to drop the prior turn pair from the KV
            // cache, then re-send the edited prompt through the regular
            // send path (which appends a fresh "You — h:mm" header).
            if (gPendingUser[0] == 0) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            if (!InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                MessageBeep(MB_ICONWARNING);
                return 0;
            }
            if (InterlockedCompareExchange(&gRunning, 0, 0)) return 0;
            char edited[PROMPT_MAX];
            snprintf(edited, sizeof(edited), "%s", gPendingUser);
            INT_PTR r = DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_EDITPROMPT),
                                        hwnd, editprompt_dlg_proc, (LPARAM)edited);
            if (r != IDOK) return 0;
            sanitize_user(edited);
            if (!input_has_text(edited)) return 0;
            backend_send_line("/reset");
            send_prompt_text(edited, 1);
            return 0;
        }
        case IDM_ABOUT:
            MessageBoxA(hwnd,
                APP_NAME "\n\nReal LLM running natively on Windows XP.\n"
                "Backend: NC_RUN.EXE — custom C99 inference engine\n"
                "Architecture: Karpathy nanochat (RMSNorm, RoPE, QK-norm,\n"
                "  ReLU\xb2 MLP, value embeddings, sliding-window attention)\n"
                "SIMD: SSE2 / SSE3 (Pentium 4 compatible)\n"
                "Model file: MODEL.NCB (custom NCB1 format, int8 quantized)\n"
                "\n"
                "Slash commands: /help /info /reset /temp /topp /seed /maxtok /system",
                APP_NAME, MB_ICONINFORMATION | MB_OK);
            return 0;
        case IDM_EXIT:
            PostMessageA(hwnd, WM_CLOSE, 0, 0);
            return 0;
        case IDM_SAVE:
        case IDC_TASK_SAVE:
            save_transcript_dialog(hwnd);
            return 0;
        case IDM_EXPORT:
        case IDC_TASK_EXPORT:
            save_transcript_dialog(hwnd);
            return 0;
        case IDM_OPEN:
        case IDC_TASK_IMPORT:
            MessageBoxA(hwnd,
                "Import is reserved for saved bliss-chat conversation files.\n\n"
                "For now, use Recent Chats to reopen conversations already stored on this XP profile.",
                APP_NAME, MB_ICONINFORMATION | MB_OK);
            return 0;
        case IDM_PRINT:
            MessageBoxA(hwnd,
                "Printing will be wired to the transcript view in a later pass.",
                APP_NAME, MB_ICONINFORMATION | MB_OK);
            return 0;
        case IDM_NEWCHAT:
        case IDC_NEWCHATBTN:
            // Drops backend conversation history, creates a fresh on-disk
            // chat, and clears the on-screen transcript.
            if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                backend_send_line("/reset");
            }
            chats_create_new();
            clear_transcript();
            return 0;
        case IDC_CHATLIST:
            // LBN_SELCHANGE fires on every selection change. Switch which
            // chat new turns are appended to, and reload that chat's
            // transcript into the view. Note: the listbox row is into the
            // *filtered* view; map through gFilteredIndices[].
            if (HIWORD(wparam) == LBN_SELCHANGE) {
                int row = (int)SendMessageA(gChatList, LB_GETCURSEL, 0, 0);
                if (row >= 0 && row < gFilteredCount) {
                    int sel = gFilteredIndices[row];
                    if (sel >= 0 && sel < gChatCount && sel != gActiveIdx) {
                        if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                            backend_send_line("/reset");
                        }
                        gActiveIdx = sel;
                        chats_load_into_view(sel);
                    }
                }
            }
            return 0;
        case IDC_SEARCH:
            // EN_CHANGE on the search edit -> re-filter the listbox by
            // the current text content (case-insensitive substring).
            if (HIWORD(wparam) == EN_CHANGE) {
                GetWindowTextA(gSearch, gSearchText, (int)sizeof(gSearchText));
                chats_repopulate_listbox();
            }
            return 0;
        case IDM_SETTINGS:
            DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_SETTINGS), hwnd,
                            settings_dlg_proc, 0);
            return 0;
        case IDM_MODELINFO: {
            char msg[640];
            snprintf(msg, sizeof(msg),
                "Model: %s\n"
                "Backend: NC_RUN.EXE\n"
                "Tokenizer: TOKENIZER.NCT\n"
                "Computer: %s\n"
                "Memory: %s\n"
                "Threads: %s",
                gModelName, gPcCpu, gPcMemory, gPcThreads);
            MessageBoxA(hwnd, msg, APP_NAME " - Model Info",
                        MB_ICONINFORMATION | MB_OK);
            return 0;
        }
        case IDM_PERF:
            MessageBoxA(hwnd,
                "Performance telemetry is shown in the status bar while tokens stream.\n\n"
                "The backend also writes profiler lines to stderr when built with profiling enabled.",
                APP_NAME " - Performance", MB_ICONINFORMATION | MB_OK);
            return 0;
        case IDM_TEMPLATES:
        case IDC_BTN_PROMPTS:
            if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                backend_send_line("/help");
                diagnostics_appendf("Prompt tools requested slash-command help.");
            } else {
                MessageBoxA(hwnd, "Backend not ready yet.", APP_NAME,
                            MB_ICONINFORMATION | MB_OK);
            }
            return 0;
        case IDC_BTN_BOLD:
            input_toggle_charformat(CFM_BOLD, CFE_BOLD);
            return 0;
        case IDC_BTN_ITALIC:
            input_toggle_charformat(CFM_ITALIC, CFE_ITALIC);
            return 0;
        case IDC_BTN_SMILE:
            input_insert_text(":)");
            return 0;
        case IDC_BTN_ATTACH:
            set_status("Plain-text prompt mode");
            return 0;
        case IDM_COPY:
            do_copy();
            return 0;
        case IDM_PASTE:
            do_paste();
            return 0;
        case IDM_SELECTALL:
            do_select_all();
            return 0;
        case IDM_FIND:
            DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_FIND), hwnd,
                            find_dlg_proc, 0);
            return 0;
        case IDM_SHORTCUTS:
        case IDM_HELPTOPICS:
            DialogBoxParamA(gInstance, MAKEINTRESOURCEA(IDD_SHORTCUTS), hwnd,
                            shortcuts_dlg_proc, 0);
            return 0;
        case IDM_SLASHHELP:
            // Backend already lists slash commands for `/help` — route the
            // menu item through the same path the toolbar Help button uses.
            if (InterlockedCompareExchange(&gBackendReady, 0, 0)) {
                backend_send_line("/help");
            } else {
                MessageBoxA(hwnd, "Backend not ready yet.", APP_NAME,
                            MB_ICONINFORMATION | MB_OK);
            }
            return 0;
        }
        break;

    case WM_CLOSE:
        win_save_placement(hwnd);
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        dbg_log("GUI", "WM_DESTROY");
        shutdown_backend();
        if (gUiFont)    DeleteObject(gUiFont);
        if (gTitleFont) DeleteObject(gTitleFont);
        if (gPaneFont)  DeleteObject(gPaneFont);
        if (gMonoFont)  DeleteObject(gMonoFont);
        if (gSendIcon)  DestroyIcon(gSendIcon);
        if (gStopIcon)  DestroyIcon(gStopIcon);
        if (gToolbarImages) ImageList_Destroy(gToolbarImages);
        if (gRichEdit)  FreeLibrary(gRichEdit);
        log_close();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wparam, lparam);
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prev, LPSTR cmdline, int show) {
    WNDCLASSA wc;
    HWND hwnd;
    MSG msg;
    HICON icon;

    (void)prev;
    (void)cmdline;
    gInstance = instance;
    get_app_dir();
    settings_load();

    gRichEdit = LoadLibraryA("RICHED20.DLL");
    if (!gRichEdit) {
        MessageBoxA(NULL, "RICHED20.DLL could not be loaded.", APP_NAME, MB_ICONERROR | MB_OK);
        return 1;
    }

    icon = (HICON)LoadImageA(instance, MAKEINTRESOURCEA(IDI_APP), IMAGE_ICON, 32, 32, LR_DEFAULTCOLOR);

    ZeroMemory(&wc, sizeof(wc));
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = instance;
    wc.hIcon = icon ? icon : LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = "XPTinyLLMWindow";

    if (!RegisterClassA(&wc)) return 1;

    // Restore last window placement if we have one in the registry,
    // otherwise fall back to a sensible default size centered by Windows.
    int win_x = CW_USEDEFAULT, win_y = CW_USEDEFAULT;
    int win_w = 1180, win_h = 760;
    {
        RECT saved;
        if (win_load_placement(&saved)) {
            win_x = (int)saved.left;
            win_y = (int)saved.top;
            win_w = (int)(saved.right - saved.left);
            win_h = (int)(saved.bottom - saved.top);
        }
    }
    hwnd = CreateWindowExA(0, wc.lpszClassName, APP_WINDOW_TITLE,
        WS_OVERLAPPEDWINDOW,
        win_x, win_y, win_w, win_h,
        NULL, NULL, instance, NULL);
    if (!hwnd) return 1;

    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    // Tiny accelerator table.
    ACCEL accels[] = {
        { FCONTROL | FVIRTKEY, 'S', IDM_SAVE },
        { FCONTROL | FVIRTKEY, 'N', IDM_NEWCHAT },
        { FCONTROL | FVIRTKEY, 'C', IDM_COPY },
        { FCONTROL | FVIRTKEY, 'V', IDM_PASTE },
        { FCONTROL | FVIRTKEY, 'A', IDM_SELECTALL },
        { FCONTROL | FVIRTKEY, 'F', IDM_FIND },
    };
    HACCEL haccel = CreateAcceleratorTableA(accels, (int)(sizeof(accels) / sizeof(accels[0])));

    while (GetMessageA(&msg, NULL, 0, 0)) {
        if (haccel && TranslateAcceleratorA(hwnd, haccel, &msg)) continue;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    if (haccel) DestroyAcceleratorTable(haccel);
    return (int)msg.wParam;
}
