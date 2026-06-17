@echo off
chcp 65001 >nul
title Z-Image 文生图

echo ╔══════════════════════════════════════╗
echo ║      ⚡ Z-Image 文生图  启动中...    ║
echo ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM 优先使用 PowerShell 启动器（支持自动安装 Python）
echo 🔧 正在启动...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "启动webui.ps1"

if %errorlevel% neq 0 (
    echo.
    echo ⚠️ 启动失败
    echo 请确保已安装 Python 3.10+ 后直接运行: python webui.py
    pause
)
