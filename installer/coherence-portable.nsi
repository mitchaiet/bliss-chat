; Build the self-contained Bliss 2 XP release from a verified package folder.
; makensis -DPAYLOAD=<folder> -DOUTFILE=<exe> coherence-portable.nsi
Unicode false
!include "FileFunc.nsh"
!ifndef PAYLOAD
  !error "Pass -DPAYLOAD=<verified BlissChat2-XP folder>."
!endif
!ifndef OUTFILE
  !define OUTFILE "bliss-chat-xp-v2.0.0-rc.2-portable.exe"
!endif
!ifndef ICON
  !define ICON "..\assets\bliss_chat.ico"
!endif
!ifndef RELEASE_VERSION
  !define RELEASE_VERSION "2.0.0"
!endif
!ifndef RELEASE_NAME
  !define RELEASE_NAME "Bliss Chat 2"
!endif
!ifndef FILE_VERSION
  !define FILE_VERSION "2.0.0.0"
!endif
Name "${RELEASE_NAME}"
Caption "Bliss Chat 2"
OutFile "${OUTFILE}"
Icon "${ICON}"
RequestExecutionLevel user
AutoCloseWindow true
ShowInstDetails nevershow
SetCompressor /SOLID lzma
; Keep decompression memory modest on the 512 MB target.
SetCompressorDictSize 8
InstallDir "$TEMP\BlissChat2-Extracted"
VIProductVersion "${FILE_VERSION}"
VIAddVersionKey "ProductName" "Bliss Chat XP"
VIAddVersionKey "FileDescription" "${RELEASE_NAME} - offline LFM2.5-350M"
VIAddVersionKey "FileVersion" "${RELEASE_VERSION}"
VIAddVersionKey "ProductVersion" "${RELEASE_VERSION}"
VIAddVersionKey "LegalCopyright" "See MODEL-LICENSE.txt and NOTICE.txt in the extracted payload."
Var ExtractOnly

Section
  StrCpy $ExtractOnly "0"
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "--extract-only" $R1
  IfErrors launch_mode
    ; Validation/portable-folder mode. /D must be the last CLI argument.
    StrCpy $ExtractOnly "1"
    IfFileExists "$INSTDIR\*.*" output_exists
    SetOutPath "$INSTDIR"
    Goto payload
  output_exists:
    SetErrorLevel 2
    Quit
  launch_mode:
    InitPluginsDir
    SetOutPath "$PLUGINSDIR"
  payload:
    File "${PAYLOAD}\XPCHAT.EXE"
    File "${PAYLOAD}\NC_RUN.EXE"
    File "${PAYLOAD}\NC_RUN_SSE2.EXE"
    File "${PAYLOAD}\NC_RUN_SSE3.EXE"
    File "${PAYLOAD}\MODEL.NCB"
    File "${PAYLOAD}\TOKENIZER.NCT"
    File "${PAYLOAD}\MODEL_VERSION.txt"
    File "${PAYLOAD}\MODEL_CARD.md"
    File "${PAYLOAD}\MODEL-LICENSE.txt"
    File "${PAYLOAD}\DATA-LICENSE.txt"
    File "${PAYLOAD}\NOTICE.txt"
    File "${PAYLOAD}\README.TXT"
    File "${PAYLOAD}\release-manifest.json"
    IfErrors extraction_failed
    StrCmp $ExtractOnly "1" finished
    ExecWait '"$PLUGINSDIR\XPCHAT.EXE"' $R2
    IfErrors extraction_failed
    Goto finished
  extraction_failed:
    SetErrorLevel 1
    Quit
  finished:
    SetErrorLevel 0
SectionEnd
