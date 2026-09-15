BLISS CHAT 2.0.0-rc.2 - FULLY LOCAL

Run the portable EXE, or extract the entire folder ZIP and run XPCHAT.EXE.
Keep MODEL.NCB, TOKENIZER.NCT and the NC_RUN executables beside XPCHAT.EXE.
There is no account, network helper, internet lookup or additional download.
Target: Windows XP SP2/SP3, Pentium M / SSE2, 512 MB RAM.

NEW NATIVE UI
The app uses your Windows theme, standard buttons and native dialogs.
The compact toolbar leaves more room for chat. View > Show Diagnostics
shows the technical details when needed. Normal 96-DPI scaling is intended.
Enter sends; Shift+Enter adds a new line; Escape stops the current reply.
Ctrl+D opens Documents. Ctrl+M opens Memories. Ctrl+N starts a new chat.

LOCAL DOCUMENTS (RAG)
Click Documents > Import to copy a small reference file into the library.
Supported: UTF-8/ASCII text, Markdown and HTML (.txt, .md, .html, .htm).
Limit: 64 KiB per file; 128 files scanned per question. No PDF/Word parser.
Select a file to preview it, or enter a question and click Find passages.
With Use documents in chats checked, up to two relevant passages are
supplied to the model and the reply lists the source filenames.
Matching uses words, so use terms that appear in the document.
Source labels identify supplied files; they do not guarantee a correct answer.

Imported copies persist in %APPDATA%\bliss-chat\Knowledge, including when
running the portable EXE. Your original imported file is kept. Duplicate
filenames are not overwritten. Existing Knowledge folders beside a folder
installation are also searched; manage those older files in Explorer.
Use filenames supported by the Windows machine's language/code page.

MEMORIES
Click Memories to review saved facts, add a short fact, or forget one.
Remember... beside a reply lets you review the last message before saving.
The model recalls relevant notes across chats. Maximum: 12 short notes;
each must be under 160 UTF-8 bytes. Notes persist in:
%APPDATA%\bliss-chat\MEMORY.TXT
To change a fact, forget the old note and add its replacement.
Turning off documents or forgetting notes affects future retrieval.
Earlier conversation text is kept; start New Chat to clear that context.

UPGRADE
For an extracted RC1 folder, the small GUI-update ZIP can replace XPCHAT.EXE
while keeping the original model/runtime files. Close Bliss and keep a backup
of the old XPCHAT.EXE first. For the RC1 single-file launcher, use the new EXE.

VALIDATION
The model, tokenizer and inference backend are unchanged from RC1.
The updated GUI passed development-host tests and ran with the real model
under Wine using an isolated XP prefix. It remains a release candidate:
physical XP startup, laptop speed, Luna appearance and total OS/application
RAM pressure are not yet verified. Close other large applications.
The small model can still give incorrect answers. See MODEL_CARD.md for
measured results and limitations, and MODEL-LICENSE.txt for model licensing.
