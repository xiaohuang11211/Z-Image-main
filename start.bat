@echo off
title Z-Image WebUI

echo ========================================
echo   Z-Image WebUI  Starting...
echo ========================================

cd /d "%~dp0"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTHON_CMD=

where python >nul 2>&1
if %errorlevel% equ 0 set PYTHON_CMD=python

if "%PYTHON_CMD%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
)
if "%PYTHON_CMD%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
)
if "%PYTHON_CMD%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
)
if "%PYTHON_CMD%"=="" (
    if exist "C:\Python312\python.exe" set PYTHON_CMD=C:\Python312\python.exe
)
if "%PYTHON_CMD%"=="" (
    if exist "C:\Python313\python.exe" set PYTHON_CMD=C:\Python313\python.exe
)

if "%PYTHON_CMD%"=="" (
    echo [!] Python not found. Auto-installing via start.ps1...
    powershell -NoProfile -ExecutionPolicy Bypass -File "start.ps1"
    if errorlevel 1 (
        echo [!] Auto-install failed.
        pause
        exit /b 1
    )
    exit /b 0
)

echo [OK] Python: %PYTHON_CMD%

"%PYTHON_CMD%" -c "import torch, gradio, transformers" 2>nul
if errorlevel 1 (
    echo [..] Installing dependencies, one-time setup...
    "%PYTHON_CMD%" -m pip install -e . 2>&1
    if errorlevel 1 (
        echo [!] Install failed. Try: pip install -e .
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
)

echo [OK] Starting server...
echo.

"%PYTHON_CMD%" webui.py

pause
