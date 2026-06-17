@echo off
chcp 65001 >nul
title Z-Image 文生图

echo ╔══════════════════════════════════════╗
echo ║      ⚡ Z-Image 文生图  启动中...    ║
echo ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0"
echo 🚀 启动 WebUI: http://localhost:7860
echo.

start http://localhost:7860
python webui.py

echo.
echo 按任意键退出...
pause >nul
