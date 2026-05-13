; bliss-chat portable build.
; Produces ONE self-contained .exe. User double-clicks, it extracts XPCHAT.EXE,
; NC_RUN.EXE, MODEL.NCB and TOKENIZER.NCT into a temp directory, runs the GUI,
; and cleans the temp dir up when the GUI exits.
;
; Build with:  makensis -DMODEL=path/to/MODEL.NCB installer/portable.nsi
;
; The MODEL define lets you point at either the small d6 model (~75 MB, fast on
; the Pentium 4) or the bigger d12 model (~280 MB, more coherent).

!ifndef OUTFILE
  !define OUTFILE "bliss-chat-xp-v1.1.0-bliss-d12-curated-c20-v1-portable.exe"
!endif

!ifndef MODEL
  !error "Pass -DMODEL=<path to .NCB> on the makensis command line."
!endif

!ifndef BUILDDIR
  !define BUILDDIR "..\build"
!endif

!ifndef DEPLOYDIR
  !define DEPLOYDIR "..\build\deploy"
!endif

Name "bliss-chat"
OutFile "${OUTFILE}"

; ----- Self-extract-and-run-with-no-UI mode -----
SilentInstall silent
RequestExecutionLevel user           ; no admin needed (XP runs as Admin by default anyway)
SetCompressor /SOLID lzma            ; tightest single-pass compression
SetCompressorDictSize 96             ; 96 MB dict — fine for the 280 MB model

; Minimum target: Windows XP SP2 (5.1)
; (NSIS itself runs fine on XP/2000/Win9x; we only need our payload there.)

; The XP About dialog uses these via VerInfoKey
VIProductVersion "1.1.0.0"
VIAddVersionKey "ProductName"     "Bliss Chat XP"
VIAddVersionKey "FileDescription" "Bliss Chat XP portable - bliss-d12-curated-c20-v1"
VIAddVersionKey "LegalCopyright"  ""
VIAddVersionKey "FileVersion"     "1.1.0"
VIAddVersionKey "ProductVersion"  "1.1.0"
VIAddVersionKey "ModelName"       "bliss-d12-curated-c20-v1"

Section
  ; $PLUGINSDIR is a per-invocation temp dir that NSIS cleans up on exit.
  InitPluginsDir
  SetOutPath $PLUGINSDIR

  ; Payload — must all land in the same dir as XPCHAT.EXE because
  ; XPCHAT.EXE locates NC_RUN.EXE/MODEL.NCB/TOKENIZER.NCT via its own
  ; module path.
  File "${BUILDDIR}\XPCHAT.EXE"
  File "${BUILDDIR}\NC_RUN.EXE"
  File "${DEPLOYDIR}\TOKENIZER.NCT"
  File "${DEPLOYDIR}\MODEL_VERSION.txt"
  File "${DEPLOYDIR}\release-manifest.json"

  ; Model file gets renamed to MODEL.NCB at extract time (the GUI is
  ; hardcoded to that name; this lets a single .nsi build either flavor).
  File /oname=MODEL.NCB "${MODEL}"

  ; Run the GUI and block until it exits.
  ExecWait '"$PLUGINSDIR\XPCHAT.EXE"'

  ; PLUGINSDIR is auto-deleted when this process ends — nothing left
  ; behind on the user's machine.
SectionEnd
