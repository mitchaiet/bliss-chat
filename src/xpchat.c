// XP Tiny LLM (real chat edition).
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
#include <richedit.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define APP_NAME "XP Tiny LLM"
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

#define IDM_EXIT       2001
#define IDM_CLEAR      2002
#define IDM_SEND       2003
#define IDM_STOP       2004
#define IDM_ABOUT      2005

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
#define MODEL_LABEL    "Model: (loading...)"
#define APP_SUBTITLE   "Chat-tuned LLM running locally on Windows XP"

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
static HWND gTitle;
static HWND gSubtitle;
static HWND gIcon;
static HWND gProgress;
static HFONT gUiFont;
static HFONT gTitleFont;
static HFONT gMonoFont;
static HMODULE gRichEdit;
static WNDPROC gOldInputProc;

static char gAppDir[MAX_PATH];
static char gPendingUser[PROMPT_MAX];
static char gLogPath[MAX_PATH];
static FILE * gLogFile = NULL;
static CRITICAL_SECTION gLogLock;
static int gLogInited = 0;

static volatile LONG gRunning;
static volatile LONG gBackendReady;
static DWORD gRunStarted;

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
        fprintf(gLogFile, "=== XP Tiny LLM session log ===\n");
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
static void set_status(const char *text) {
    if (gStatus) SetWindowTextA(gStatus, text);
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
    SendMessageA(gTranscript, EM_SCROLLCARET, 0, 0);
    free(norm.data);
}

static void clear_transcript(void) {
    char model_text[256];
    SetWindowTextA(gTranscript, "");
    rich_append_color("XP Tiny LLM\r\n", RGB(0, 128, 0), TRUE);
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
}

static void set_running(BOOL running) {
    BOOL ready = InterlockedCompareExchange(&gBackendReady, 0, 0) != 0;
    EnableWindow(gSend,  ready && !running);
    EnableWindow(gInput, ready);
    EnableWindow(gStop,  ready && running);
    EnableWindow(gClear, !running);
    // Reset bar at phase transitions; backend will fill it via PROG messages.
    if (gProgress) SendMessageA(gProgress, PBM_SETPOS, 0, 0);
    if (running) SetTimer(gMain, TIMER_STATUS, 1000, NULL);
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

static void send_prompt(void) {
    char user_prompt[PROMPT_MAX];
    DWORD written;
    char with_newline[PROMPT_MAX + 4];

    if (!InterlockedCompareExchange(&gBackendReady, 0, 0)) {
        MessageBeep(MB_ICONWARNING);
        return;
    }
    if (InterlockedCompareExchange(&gRunning, 0, 0)) return;

    GetWindowTextA(gInput, user_prompt, sizeof(user_prompt));
    sanitize_user(user_prompt);
    if (!input_has_text(user_prompt)) return;

    snprintf(gPendingUser, sizeof(gPendingUser), "%s", user_prompt);

    rich_append_color("You\r\n", RGB(0, 84, 166), TRUE);
    rich_append_color(user_prompt, RGB(0, 0, 0), FALSE);
    rich_append_color("\r\n\r\nAssistant\r\n", RGB(0, 128, 0), TRUE);
    SetWindowTextA(gInput, "");

    InterlockedExchange(&gRunning, 1);
    gRunStarted = GetTickCount();
    set_running(TRUE);
    set_status("Generating...");

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

// ---------- menu / fonts / controls ----------
static void make_menu(HWND hwnd) {
    HMENU menu = CreateMenu();
    HMENU file = CreatePopupMenu();
    HMENU convo = CreatePopupMenu();
    HMENU help = CreatePopupMenu();

    AppendMenuA(file, MF_STRING, IDM_EXIT, "Exit");
    AppendMenuA(convo, MF_STRING, IDM_SEND, "Send");
    AppendMenuA(convo, MF_SEPARATOR, 0, NULL);
    AppendMenuA(convo, MF_STRING, IDM_CLEAR, "Clear transcript");
    AppendMenuA(help, MF_STRING, IDM_ABOUT, "About XP Tiny LLM");

    AppendMenuA(menu, MF_POPUP, (UINT_PTR)file, "File");
    AppendMenuA(menu, MF_POPUP, (UINT_PTR)convo, "Conversation");
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

static void create_controls(HWND hwnd) {
    HICON icon;

    gIcon = make_control("STATIC", "", SS_ICON, 0, IDC_APPICON, hwnd);
    icon = (HICON)LoadImageA(gInstance, MAKEINTRESOURCEA(IDI_APP), IMAGE_ICON, 32, 32, LR_DEFAULTCOLOR);
    if (icon) SendMessageA(gIcon, STM_SETICON, (WPARAM)icon, 0);

    gTitle = make_control("STATIC", APP_NAME, SS_LEFT, 0, IDC_TITLE, hwnd);
    SendMessageA(gTitle, WM_SETFONT, (WPARAM)gTitleFont, TRUE);
    gSubtitle = make_control("STATIC", APP_SUBTITLE, SS_LEFT, 0, IDC_SUBTITLE, hwnd);
    gModel = make_control("STATIC", MODEL_LABEL, SS_LEFT, 0, IDC_MODEL, hwnd);

    gTranscript = make_control("RichEdit20A", "",
        ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_TABSTOP,
        WS_EX_CLIENTEDGE, IDC_TRANSCRIPT, hwnd);
    SendMessageA(gTranscript, WM_SETFONT, (WPARAM)gMonoFont, TRUE);
    SendMessageA(gTranscript, EM_SETBKGNDCOLOR, 0, RGB(255, 255, 255));
    SendMessageA(gTranscript, EM_EXLIMITTEXT, 0, 524288);

    gInput = make_control("EDIT", "",
        ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | WS_TABSTOP,
        WS_EX_CLIENTEDGE, IDC_INPUT, hwnd);
    SendMessageA(gInput, EM_SETLIMITTEXT, PROMPT_MAX - 1, 0);

    gSend  = make_control("BUTTON", "Send",  BS_DEFPUSHBUTTON | WS_TABSTOP, 0, IDC_SEND, hwnd);
    gStop  = make_control("BUTTON", "Stop",  BS_PUSHBUTTON | WS_TABSTOP,    0, IDC_STOP, hwnd);
    gClear = make_control("BUTTON", "Clear", BS_PUSHBUTTON | WS_TABSTOP,    0, IDC_CLEAR, hwnd);
    gStatus = make_control("STATIC", "Loading model...", SS_LEFT, 0, IDC_STATUS, hwnd);

    {
        INITCOMMONCONTROLSEX icc;
        icc.dwSize = sizeof(icc);
        icc.dwICC  = ICC_PROGRESS_CLASS;
        InitCommonControlsEx(&icc);
        // Use the standard segmented bar (works on classic-themed XP without
        // a COMCTL32 v6 manifest — PBS_MARQUEE silently no-ops without themes).
        // We animate it manually via TIMER_PROGRESS sweeping 0->100 repeatedly.
        gProgress = CreateWindowExA(0, PROGRESS_CLASSA, NULL,
            WS_CHILD | WS_VISIBLE,
            0, 0, 10, 16, hwnd, (HMENU)(INT_PTR)IDC_PROGRESS, gInstance, NULL);
        SendMessageA(gProgress, PBM_SETRANGE32, 0, 100);
        SendMessageA(gProgress, PBM_SETSTEP, 4, 0);
        SendMessageA(gProgress, PBM_SETPOS, 0, 0);
    }

    EnableWindow(gSend,  FALSE);
    EnableWindow(gInput, FALSE);
    EnableWindow(gStop,  FALSE);
}

static void layout_controls(HWND hwnd) {
    RECT rc;
    int w, h;
    int pad = 12;
    int header_h = 62;
    int inner_x;
    int inner_w;
    int action_w = 92;
    int transcript_y;
    int transcript_h;
    int input_y;
    int input_h;
    int status_y;
    int clear_y;

    GetClientRect(hwnd, &rc);
    w = rc.right - rc.left;
    h = rc.bottom - rc.top;
    if (w < 720) w = 720;
    if (h < 480) h = 480;

    MoveWindow(gIcon,     pad,         pad,       36,  36, TRUE);
    MoveWindow(gTitle,    pad + 48,    pad - 2,  300,  28, TRUE);
    MoveWindow(gSubtitle, pad + 50,    pad + 28, 480,  20, TRUE);
    MoveWindow(gModel,    w - pad - 320, pad + 6, 320, 20, TRUE);

    inner_x = pad;
    inner_w = w - pad * 2;
    status_y = h - pad - 20;
    input_h = h < 560 ? 62 : 96;
    input_y = status_y - input_h - 10;
    transcript_y = pad + header_h;
    transcript_h = input_y - transcript_y - 10;
    if (transcript_h < 160) transcript_h = 160;

    MoveWindow(gTranscript, inner_x, transcript_y, inner_w, transcript_h, TRUE);
    MoveWindow(gInput,      inner_x, input_y, inner_w - action_w - 14, input_h, TRUE);
    MoveWindow(gSend,       inner_x + inner_w - action_w, input_y,        action_w, 28, TRUE);
    MoveWindow(gStop,       inner_x + inner_w - action_w, input_y + 34,   action_w, 28, TRUE);
    clear_y = input_y + 68;
    MoveWindow(gClear,      inner_x + inner_w - action_w, clear_y,        action_w, 28, TRUE);
    // Status text on the left, progress bar on the right of the same row
    {
        int prog_w = 220;
        int status_w = inner_w - prog_w - 10;
        if (status_w < 100) status_w = 100;
        MoveWindow(gStatus,   inner_x,                    status_y, status_w, 20, TRUE);
        MoveWindow(gProgress, inner_x + status_w + 10,    status_y, prog_w,   16, TRUE);
    }

    if (input_h < 96) ShowWindow(gClear, SW_HIDE);
    else ShowWindow(gClear, SW_SHOW);
}

static LRESULT CALLBACK input_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
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
    return CallWindowProcA(gOldInputProc, hwnd, msg, wparam, lparam);
}

static void subclass_input(void) {
    gOldInputProc = (WNDPROC)SetWindowLongPtrA(gInput, GWLP_WNDPROC, (LONG_PTR)input_proc);
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
        layout_controls(hwnd);
        clear_transcript();
        if (!file_exists_in_app_dir(BACKEND_EXE) || !file_exists_in_app_dir(MODEL_FILE)
            || !file_exists_in_app_dir(TOKENIZER_FILE)) {
            dbg_log("GUI", "missing files: %s=%d %s=%d %s=%d",
                BACKEND_EXE, file_exists_in_app_dir(BACKEND_EXE),
                MODEL_FILE, file_exists_in_app_dir(MODEL_FILE),
                TOKENIZER_FILE, file_exists_in_app_dir(TOKENIZER_FILE));
            MessageBoxA(hwnd,
                BACKEND_EXE ", " MODEL_FILE ", or " TOKENIZER_FILE " is missing from the application folder.",
                APP_NAME, MB_ICONERROR | MB_OK);
        } else if (!launch_backend()) {
            set_status("Backend failed to start");
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
        RECT fill;
        GetClientRect(hwnd, &fill);
        FillRect((HDC)wparam, &fill, GetSysColorBrush(COLOR_BTNFACE));
        return 1;
    }

    case WM_CTLCOLORSTATIC:
        SetBkColor((HDC)wparam, GetSysColor(COLOR_BTNFACE));
        return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);

    case WM_GETMINMAXINFO: {
        MINMAXINFO *mmi = (MINMAXINFO *)lparam;
        mmi->ptMinTrackSize.x = 720;
        mmi->ptMinTrackSize.y = 480;
        return 0;
    }

    case WM_TIMER:
        if (wparam == TIMER_STATUS && InterlockedCompareExchange(&gRunning, 0, 0)) {
            DWORD elapsed = (GetTickCount() - gRunStarted) / 1000;
            char text[96];
            snprintf(text, sizeof(text), "Generating... %lu sec elapsed", (unsigned long)elapsed);
            set_status(text);
            return 0;
        }
        break;

    case WM_BACKEND_READY:
        InterlockedExchange(&gBackendReady, 1);
        set_running(FALSE);
        set_status("Ready");
        rich_append_color("Backend ready. Type a message and press Enter.\r\n\r\n", RGB(0, 128, 0), TRUE);
        SetFocus(gInput);
        return 0;

    case WM_BACKEND_PROG: {
        if (gProgress) SendMessageA(gProgress, PBM_SETPOS, (WPARAM)wparam, 0);
        return 0;
    }

    case WM_BACKEND_INFO: {
        char *info = (char *)lparam;
        if (info && *info) {
            char label[256];
            snprintf(label, sizeof(label), "Model: %s", info);
            SetWindowTextA(gModel, label);
        }
        if (info) free(info);
        return 0;
    }

    case WM_BACKEND_DEAD:
        InterlockedExchange(&gBackendReady, 0);
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
        set_status("Backend exited");
        rich_append_color("\r\n[backend process exited]\r\n", RGB(192, 0, 0), TRUE);
        return 0;

    case WM_APPEND_TEXT: {
        char *text = (char *)lparam;
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
        SetFocus(gInput);
        return 0;
    }

    case WM_RUN_DONE: {
        char *message = (char *)lparam;
        int  tcount   = (int)wparam;
        DWORD elapsed_ms = GetTickCount() - gRunStarted;
        double elapsed_s = elapsed_ms / 1000.0;
        InterlockedExchange(&gRunning, 0);
        set_running(FALSE);
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
            clear_transcript();
            return 0;
        case IDM_ABOUT:
            MessageBoxA(hwnd,
                APP_NAME "\n\nReal LLM running natively on Windows XP.\n"
                "Backend: NC_RUN.EXE — custom C99 inference engine\n"
                "Architecture: Karpathy nanochat (RMSNorm, RoPE, QK-norm,\n"
                "  ReLU\xb2 MLP, value embeddings, sliding-window attention)\n"
                "SIMD: SSE2 / SSE3 (Pentium 4 compatible)\n"
                "Model file: MODEL.NCB (custom NCB1 format, int8 quantized)\n"
                "\n"
                "Slash commands: /help /info /reset",
                APP_NAME, MB_ICONINFORMATION | MB_OK);
            return 0;
        case IDM_EXIT:
            PostMessageA(hwnd, WM_CLOSE, 0, 0);
            return 0;
        }
        break;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        dbg_log("GUI", "WM_DESTROY");
        shutdown_backend();
        if (gUiFont)    DeleteObject(gUiFont);
        if (gTitleFont) DeleteObject(gTitleFont);
        if (gMonoFont)  DeleteObject(gMonoFont);
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

    hwnd = CreateWindowExA(0, wc.lpszClassName, APP_NAME,
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 940, 660,
        NULL, NULL, instance, NULL);
    if (!hwnd) return 1;

    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    while (GetMessageA(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return (int)msg.wParam;
}
