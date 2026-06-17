@echo off
chcp 65001 >nul
title Z-Image Launcher

echo ========================================
echo      Z-Image WebUI  Starting...
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Python not found, trying auto-install...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0启动webui.ps1'"
    if %errorlevel% neq 0 (
        echo [!] Failed. Please install Python 3.10+ from python.org
        pause
        exit /b 1
    )
    pause
    exit /b 0
)

echo [2/3] Launching WebUI...
echo.
echo Open browser at: http://localhost:7860
echo Press Ctrl+C to stop
echo.

start http://localhost:7860
python webui.py

echo.
pause
