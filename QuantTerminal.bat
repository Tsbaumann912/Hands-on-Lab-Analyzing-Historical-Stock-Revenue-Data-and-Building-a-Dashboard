@echo off
rem ---- QuantTerminal launcher (Windows) ----
rem Double-click to open the terminal in its own window.
rem (The desktop shortcut created by install_windows.bat does the
rem  same thing; this file is a fallback that works from the repo.)

cd /d "%~dp0"

rem Prefer pythonw so no console window stays open.
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw desktop.py
    exit /b 0
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 desktop.py
    exit /b %errorlevel%
)

python desktop.py
