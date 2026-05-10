Unicode false

!define APP_NAME "XP Tiny LLM"
!define APP_EXE "XPCHAT.EXE"
!define APP_VERSION "1.0.0"

Name "${APP_NAME}"
OutFile "/Users/mitchaiet/VMs/WindowsXP/XP-Tiny-LLM-Setup.exe"
InstallDir "$PROGRAMFILES\XP Tiny LLM"
InstallDirRegKey HKCU "Software\XP Tiny LLM" "InstallDir"
RequestExecutionLevel user
Icon "/Users/mitchaiet/VMs/WindowsXP/app-src/xp_tiny_llm.ico"
UninstallIcon "/Users/mitchaiet/VMs/WindowsXP/app-src/xp_tiny_llm.ico"

Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  File "/Users/mitchaiet/VMs/WindowsXP/package/XPCHAT.EXE"
  File "/Users/mitchaiet/VMs/WindowsXP/package/RUN.EXE"
  File "/Users/mitchaiet/VMs/WindowsXP/package/STORIES.BIN"
  File "/Users/mitchaiet/VMs/WindowsXP/package/TOKENIZER.BIN"
  File "/Users/mitchaiet/VMs/WindowsXP/package/XP_TINY_LLM.ICO"
  File "/Users/mitchaiet/VMs/WindowsXP/package/README.TXT"

  CreateDirectory "$SMPROGRAMS\XP Tiny LLM"
  CreateShortCut "$SMPROGRAMS\XP Tiny LLM\XP Tiny LLM.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\XP_TINY_LLM.ICO"
  CreateShortCut "$SMPROGRAMS\XP Tiny LLM\Uninstall XP Tiny LLM.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortCut "$DESKTOP\XP Tiny LLM.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\XP_TINY_LLM.ICO"

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\XP Tiny LLM" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM" "Publisher" "XP Tiny LLM"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM" "DisplayIcon" "$INSTDIR\XP_TINY_LLM.ICO"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM" "UninstallString" "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\XP Tiny LLM.lnk"
  Delete "$SMPROGRAMS\XP Tiny LLM\XP Tiny LLM.lnk"
  Delete "$SMPROGRAMS\XP Tiny LLM\Uninstall XP Tiny LLM.lnk"
  RMDir "$SMPROGRAMS\XP Tiny LLM"

  Delete "$INSTDIR\XPCHAT.EXE"
  Delete "$INSTDIR\RUN.EXE"
  Delete "$INSTDIR\STORIES.BIN"
  Delete "$INSTDIR\TOKENIZER.BIN"
  Delete "$INSTDIR\XP_TINY_LLM.ICO"
  Delete "$INSTDIR\README.TXT"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  DeleteRegKey HKCU "Software\XP Tiny LLM"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\XP Tiny LLM"
SectionEnd
