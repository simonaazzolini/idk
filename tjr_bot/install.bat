@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   TJR Bot Installer
echo   MetaTrader 5 Expert Advisor + Local Dashboard
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
pip install -r dashboard\requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo      Done.
echo.

echo [2/4] Finding MetaTrader 5 terminal data folders...
set MT5_FOUND=0

:: Common MT5 install paths
for /d %%D in ("%APPDATA%\MetaQuotes\Terminal\*") do (
    set MT5_PATH=%%D
    if exist "%%D\MQL5\Experts" (
        echo      Found MT5 terminal: %%D
        echo      Copying TJR_EA.mq5...
        xcopy /Y "MT5\TJR_EA.mq5" "%%D\MQL5\Experts\" >nul
        if !errorlevel! == 0 (
            echo      Copied to: %%D\MQL5\Experts\TJR_EA.mq5
            set MT5_FOUND=1
        )
    )
)

:: Also copy to Common files folder for MT5 file access
if exist "%APPDATA%\MetaQuotes\Terminal\Common\Files" (
    echo      Common Files folder found — dashboard JSON/CSV will go here.
) else (
    mkdir "%APPDATA%\MetaQuotes\Terminal\Common\Files" 2>nul
)

if %MT5_FOUND% == 0 (
    echo.
    echo [WARN] No MT5 terminal found. Copy MT5\TJR_EA.mq5 manually to:
    echo        %%APPDATA%%\MetaQuotes\Terminal\^<ID^>\MQL5\Experts\
    echo.
)
echo.

echo [3/4] Checking directory structure...
if not exist "dashboard\frontend\index.html" (
    echo [ERROR] Dashboard frontend files not found.
    echo         Run this installer from the tjr_bot\ directory.
    pause
    exit /b 1
)
echo      OK.
echo.

echo [4/4] Setup complete!
echo.
echo ============================================================
echo   NEXT STEPS:
echo ============================================================
echo.
echo   IN METATRADER 5:
echo   1. Open MetaEditor (press F4 in MT5)
echo   2. Open TJR_EA.mq5 from Experts folder
echo   3. Press F7 to compile — should show 0 errors
echo   4. Drag TJR_EA from Navigator ^> Expert Advisors onto:
echo      - XAUUSD M5 chart (recommended primary)
echo      - NAS100 M5 chart (optional second)
echo      - EURUSD M5 chart (optional third)
echo   5. Enable "Allow Algo Trading" in EA settings
echo   6. Set ServerUTC_Offset to match your broker:
echo      e.g. if broker server is UTC+3, set to 3
echo.
echo   DASHBOARD:
echo   7. Open a terminal in this folder and run:
echo         cd dashboard
echo         python server.py
echo   8. Open browser: http://localhost:8000
echo.
echo   IMPORTANT:
echo   - Use ECN/RAW spread account for best fills
echo   - The EA writes data to MQL5\Files\tjr_live_data.json
echo   - Dashboard reads this file via MT5 Python API
echo   - Keep MT5 running while dashboard is active
echo.
echo ============================================================
pause
