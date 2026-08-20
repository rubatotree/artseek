@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\artseek.exe" (
  echo ArtSeek 尚未安装。请先在 PowerShell 运行 scripts\setup_windows.ps1
  pause
  exit /b 1
)
".venv\Scripts\artseek.exe" serve

