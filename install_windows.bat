@echo off
setlocal
title QuantTerminal - Windows Setup

echo ============================================================
echo   QuantTerminal - Quantitative Futures Trading Terminal
echo   One-time Windows setup
echo ============================================================
echo.

rem ---- Locate Python 3 (py launcher first, then python on PATH) ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3 was not found on this computer.
    echo.
    echo Opening the Python download page. During install, tick
    echo   [x] Add python.exe to PATH
    echo then run this file again.
    start https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

echo Using Python: %PY%
echo.
echo [1/2] Installing dependencies (this can take a few minutes)...
%PY% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo Dependency installation failed - see the messages above.
    pause
    exit /b 1
)

echo.
echo [2/2] Creating the desktop shortcut...
%PY% "%~dp0install_desktop_app.py"
if errorlevel 1 (
    echo.
    echo Shortcut creation failed - see the messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done! Double-click "QuantTerminal" on your desktop
echo   to open the trading terminal.
echo ============================================================
pause
