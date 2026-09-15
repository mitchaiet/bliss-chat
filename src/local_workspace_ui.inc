/* Native, offline-only document and memory dialogs. Included after GUI helpers. */
static void local_workspace_init(void) {
    char base[MAX_PATH], parent[MAX_PATH];
    HKEY key;
    DWORD type, bytes, enabled;
    if (SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, base) != S_OK)
        snprintf(base, sizeof(base), "%s", gAppDir);
    snprintf(parent, sizeof(parent), "%s\\bliss-chat", base);
    CreateDirectoryA(parent, NULL);
    snprintf(gKnowledgeDir, sizeof(gKnowledgeDir), "%s\\Knowledge", parent);
    CreateDirectoryA(gKnowledgeDir, NULL);
    if (RegOpenKeyExA(HKEY_CURRENT_USER, "Software\\bliss-chat\\LocalWorkspace", 0, KEY_READ, &key) == ERROR_SUCCESS) {
        bytes = sizeof(enabled);
        if (RegQueryValueExA(key, "UseDocuments", NULL, &type, (BYTE *)&enabled, &bytes) == ERROR_SUCCESS &&
            type == REG_DWORD && bytes == sizeof(enabled) && enabled <= 1) gUseKnowledge = (int)enabled;
        RegCloseKey(key);
    }
}

static void knowledge_list(HWND dlg) {
    WIN32_FIND_DATAA fd;
    char glob[MAX_PATH * 2], status[400];
    HANDLE h;
    int count = 0;
    SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_RESETCONTENT, 0, 0);
    snprintf(glob, sizeof(glob), "%s\\*", gKnowledgeDir);
    h = FindFirstFileA(glob, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || !know_extension(fd.cFileName)) continue;
            SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_ADDSTRING, 0, (LPARAM)fd.cFileName);
            count++;
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    snprintf(status, sizeof(status), "%d local document%s. Up to 128 files, 64 KiB each. UTF-8 text, Markdown or HTML.", count, count == 1 ? "" : "s");
    set_dialog_utf8(dlg, IDC_KNOW_STATUS, status);
}

static int knowledge_selected_path(HWND dlg, char *path, size_t cap) {
    LRESULT row = SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_GETCURSEL, 0, 0);
    char name[MAX_PATH];
    LRESULT size;
    if (row == LB_ERR) return 0;
    size = SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_GETTEXTLEN, row, 0);
    if (size < 0 || size >= MAX_PATH) return 0;
    SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_GETTEXT, row, (LPARAM)name);
    return snprintf(path, cap, "%s\\%s", gKnowledgeDir, name) < (int)cap;
}

static void knowledge_import(HWND dlg) {
    OPENFILENAMEA ofn;
    char source[MAX_PATH] = "", destination[MAX_PATH * 2];
    const char *name;
    char *text;
    if (SendDlgItemMessageA(dlg, IDC_KNOW_FILES, LB_GETCOUNT, 0, 0) >= KNOW_FILES_MAX) {
        set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Library full. Remove a document before importing another."); return;
    }
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn); ofn.hwndOwner = dlg;
    ofn.lpstrFilter = "Local documents (*.txt;*.md;*.html;*.htm)\0*.txt;*.md;*.html;*.htm\0\0";
    ofn.lpstrFile = source; ofn.nMaxFile = sizeof(source);
    ofn.lpstrTitle = "Copy a document into your local library";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;
    if (!GetOpenFileNameA(&ofn)) return;
    text = know_extension(source) ? know_read_file(source) : NULL;
    if (!text) {
        set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Import failed: choose UTF-8/ASCII text, Markdown or HTML no larger than 64 KiB."); return;
    }
    free(text);
    name = strrchr(source, '\\'); name = name ? name + 1 : source;
    snprintf(destination, sizeof(destination), "%s\\%s", gKnowledgeDir, name);
    /* CopyFile's fail-if-exists flag preserves both originals and library copies. */
    if (!CopyFileA(source, destination, TRUE)) {
        set_dialog_utf8(dlg, IDC_KNOW_STATUS, GetLastError() == ERROR_FILE_EXISTS ?
            "That filename is already in your library. Rename the new file to keep both copies." :
            "Could not copy the document. Check the folder and available disk space."); return;
    }
    knowledge_list(dlg);
    set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Imported a local copy. The original file was kept. Select it to preview, or search below.");
}

static INT_PTR CALLBACK knowledge_dlg_proc(HWND dlg, UINT msg, WPARAM wp, LPARAM lp) {
    (void)lp;
    switch (msg) {
    case WM_INITDIALOG: {
        char query[PROMPT_MAX];
        CheckDlgButton(dlg, IDC_KNOW_ENABLED, gUseKnowledge ? BST_CHECKED : BST_UNCHECKED);
        SendDlgItemMessageA(dlg, IDC_KNOW_PREVIEW, EM_SETLIMITTEXT, KNOW_FILE_MAX + 1024, 0);
        knowledge_list(dlg);
        if (get_window_utf8(gInput, query, sizeof(query)) > 0) set_dialog_utf8(dlg, IDC_KNOW_QUERY, query);
        set_dialog_utf8(dlg, IDC_KNOW_PREVIEW,
            "Import a small reference document, then enter a question below to see the passages Bliss would receive.\r\n\r\n"
            "Matching passages are supplied locally with each relevant message while Use documents in chats is checked. "
            "Replies show the names of the documents supplied; a listed source does not guarantee a correct answer.\r\n\r\n"
            "Files stay on this computer. The model uses short passages that match your question.");
        return TRUE;
    }
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_KNOW_IMPORT: knowledge_import(dlg); return TRUE;
        case IDC_KNOW_FILES:
            if (HIWORD(wp) == LBN_SELCHANGE) {
                char path[MAX_PATH * 2];
                char *text = knowledge_selected_path(dlg, path, sizeof(path)) ? know_read_file(path) : NULL;
                set_dialog_utf8(dlg, IDC_KNOW_PREVIEW, text ? text : "Cannot preview: file is missing, too large, or not UTF-8 text.");
                free(text);
            }
            return TRUE;
        case IDC_KNOW_SEARCH: {
            char query[1024], preview[2048] = "";
            KnowledgeHit best[2];
            int i;
            if (get_dialog_utf8(dlg, IDC_KNOW_QUERY, query, sizeof(query)) < 0 || !input_has_text(query)) {
                set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Enter a short question to preview matching passages."); return TRUE;
            }
            knowledge_matches(query, best);
            for (i = 0; i < 2; i++) if (best[i].score > 0) {
                size_t used = strlen(preview);
                snprintf(preview + used, sizeof(preview) - used, "%s\r\n%s\r\n\r\n", best[i].name, best[i].snippet);
            }
            set_dialog_utf8(dlg, IDC_KNOW_PREVIEW, preview[0] ? preview : "No matching passages. Try words that appear in your documents.");
            set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Preview only. This does not send a chat message or change your conversation.");
            return TRUE;
        }
        case IDC_KNOW_ENABLED: {
            HKEY key;
            DWORD value = IsDlgButtonChecked(dlg, IDC_KNOW_ENABLED) == BST_CHECKED;
            gUseKnowledge = (int)value;
            if (RegCreateKeyExA(HKEY_CURRENT_USER, "Software\\bliss-chat\\LocalWorkspace", 0, NULL, 0, KEY_WRITE, NULL, &key, NULL) == ERROR_SUCCESS) {
                RegSetValueExA(key, "UseDocuments", 0, REG_DWORD, (BYTE *)&value, sizeof(value)); RegCloseKey(key);
            }
            set_dialog_utf8(dlg, IDC_KNOW_STATUS, value ? "Document retrieval is on for future messages." : "Document retrieval is off for future messages. Start a new chat to clear previously supplied context.");
            return TRUE;
        }
        case IDC_KNOW_REMOVE: {
            char path[MAX_PATH * 2];
            if (!knowledge_selected_path(dlg, path, sizeof(path))) return TRUE;
            if (MessageBoxW(dlg, L"Remove this copy from the local library? The file you originally imported will be kept.",
                    L"Remove Document", MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) != IDYES) return TRUE;
            if (DeleteFileA(path)) {
                knowledge_list(dlg);
                set_dialog_utf8(dlg, IDC_KNOW_PREVIEW, "Document removed from the library. Start a new chat to clear previously supplied context.");
            } else set_dialog_utf8(dlg, IDC_KNOW_STATUS, "Could not remove this file.");
            return TRUE;
        }
        case IDC_KNOW_FOLDER:
            ShellExecuteA(dlg, "open", gKnowledgeDir, NULL, NULL, SW_SHOWNORMAL); return TRUE;
        case IDCANCEL: EndDialog(dlg, 0); return TRUE;
        }
        break;
    case WM_CLOSE: EndDialog(dlg, 0); return TRUE;
    }
    return FALSE;
}

static void memory_buttons(HWND dlg) {
    BOOL ready = !gMemoryPending && InterlockedCompareExchange(&gBackendReady, 0, 0);
    EnableWindow(GetDlgItem(dlg, IDC_MEM_ADD), ready);
    EnableWindow(GetDlgItem(dlg, IDC_MEM_REMOVE), ready);
    EnableWindow(GetDlgItem(dlg, IDC_MEM_REFRESH), ready);
    EnableWindow(GetDlgItem(dlg, IDCANCEL), !gMemoryPending);
}

static void memory_request(HWND dlg, const char *command) {
    gMemoryFailed = 0;
    SendDlgItemMessageA(dlg, IDC_MEM_LIST, LB_RESETCONTENT, 0, 0);
    gMemoryPending = 0;
    if (command && backend_send_line(command)) gMemoryPending++;
    if (backend_send_line("/memories")) gMemoryPending++;
    set_dialog_utf8(dlg, IDC_MEM_STATUS, gMemoryPending ? "Reading saved memories..." : "Could not contact the model.");
    memory_buttons(dlg);
}

static void memory_info(const char *info) {
    char *end;
    long number = strtol(info, &end, 10);
    if (number > 0 && number <= 12 && !strncmp(end, ". ", 2)) {
        WCHAR *wide = utf8_to_wide(end + 2);
        if (wide) {
            LRESULT row = SendDlgItemMessageW(gMemoryDialog, IDC_MEM_LIST, LB_ADDSTRING, 0, (LPARAM)wide);
            if (row >= 0) SendDlgItemMessageA(gMemoryDialog, IDC_MEM_LIST, LB_SETITEMDATA, row, number);
        }
        free(wide);
    }
    if (!gMemoryFailed) set_dialog_utf8(gMemoryDialog, IDC_MEM_STATUS, info);
}

static INT_PTR CALLBACK memories_dlg_proc(HWND dlg, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_INITDIALOG:
        gMemoryDialog = dlg;
        set_dialog_utf8(dlg, IDC_MEM_TEXT, lp ? (const char *)lp : "");
        memory_request(dlg, NULL);
        return TRUE;
    case WM_MEMORY_ACK:
        if (gMemoryPending > 0) gMemoryPending--;
        if (!gMemoryPending && !gMemoryFailed) {
            char status[160];
            LRESULT count = SendDlgItemMessageA(dlg, IDC_MEM_LIST, LB_GETCOUNT, 0, 0);
            snprintf(status, sizeof(status), "%ld of 12 memories saved on this computer. Relevant notes are recalled automatically.", (long)count);
            set_dialog_utf8(dlg, IDC_MEM_STATUS, status);
        }
        memory_buttons(dlg);
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_MEM_ADD: {
            char fact[160], command[180];
            if (gMemoryPending) return TRUE;
            if (get_dialog_utf8(dlg, IDC_MEM_TEXT, fact, sizeof(fact)) < 0) {
                set_dialog_utf8(dlg, IDC_MEM_STATUS, "This memory is too long. Shorten it to one brief fact and try again."); return TRUE;
            }
            sanitize_user(fact);
            if (!input_has_text(fact)) return TRUE;
            snprintf(command, sizeof(command), "/remember %s", fact);
            memory_request(dlg, command);
            return TRUE;
        }
        case IDC_MEM_REMOVE: {
            LRESULT row, number;
            char command[32];
            if (gMemoryPending) return TRUE;
            row = SendDlgItemMessageA(dlg, IDC_MEM_LIST, LB_GETCURSEL, 0, 0);
            if (row == LB_ERR) return TRUE;
            number = SendDlgItemMessageA(dlg, IDC_MEM_LIST, LB_GETITEMDATA, row, 0);
            if (number < 1 || number > 12) return TRUE;
            snprintf(command, sizeof(command), "/forget %ld", (long)number);
            memory_request(dlg, command);
            return TRUE;
        }
        case IDC_MEM_REFRESH: if (!gMemoryPending) memory_request(dlg, NULL); return TRUE;
        case IDCANCEL: if (!gMemoryPending) EndDialog(dlg, 0); return TRUE;
        }
        break;
    case WM_CLOSE: if (!gMemoryPending) EndDialog(dlg, 0); return TRUE;
    case WM_DESTROY: gMemoryDialog = NULL; return TRUE;
    }
    return FALSE;
}

static void open_memories(HWND parent, const char *prefill) {
    if (!InterlockedCompareExchange(&gBackendReady, 0, 0) || InterlockedCompareExchange(&gRunning, 0, 0) ||
        InterlockedCompareExchange(&gRequestRead, 0, 0) != InterlockedCompareExchange(&gRequestWritten, 0, 0)) {
        MessageBoxW(parent, L"Wait for the current response or command to finish, then open Memories.", L"Memories", MB_OK | MB_ICONINFORMATION);
        return;
    }
    DialogBoxParamW(gInstance, MAKEINTRESOURCEW(IDD_MEMORIES), parent, memories_dlg_proc, (LPARAM)prefill);
}
