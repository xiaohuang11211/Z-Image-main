@echo off
chcp 65001 >nul
title Z-Image Quick

echo ========================================
echo      Z-Image WebUI  Starting...
echo ========================================
echo.

cd /d "%~dp0"
echo Open browser at: http://localhost:7860
echo.

start http://localhost:7860
python webui.py

echo.
pause >nul
