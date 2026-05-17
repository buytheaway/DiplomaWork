@echo off
setlocal

cd /d "%~dp0.."

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_full_demo.ps1"

echo.
echo Launcher finished. Backend and Desktop windows should stay open.
pause
