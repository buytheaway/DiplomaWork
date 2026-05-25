@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_scale_full.ps1"

echo.
echo Launcher finished. Backend and Desktop windows are separate.
pause
