@echo off
title Z-Image WebUI

echo ========================================
echo   Z-Image WebUI  Starting...
echo ========================================

cd /d "%~dp0"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set PYTHON_CMD=

REM Check _venv (created by start.ps1)
if exist "_venv\Scripts\python.exe" set PYTHON_CMD=%CD%\_venv\Scripts\python.exe

REM Check common Python install locations (skip Windows Store stub)
if "%PYTHON_CMD%"=="" (
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "C:\Python312\python.exe"
        "C:\Python313\python.exe"
    ) do if exist %%p set PYTHON_CMD=%%p
)

REM Check system PATH (skip WindowsApps)
if "%PYTHON_CMD%"=="" (
    for /f "tokens=*" %%a in ('where python 2^>nul') do (
        echo %%a | findstr /i "WindowsApps" >nul
        if errorlevel 1 set PYTHON_CMD=%%a
    )
)

REM Auto-download Python if not found
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

REM Check if dependencies are installed
"%PYTHON_CMD%" -c "import torch, gradio, transformers" 2>nul
if errorlevel 1 (
    echo [..] Installing dependencies (one-time setup)...
    "%PYTHON_CMD%" -m pip install -e . 2>&1
    if errorlevel 1 (
        echo [!] Install failed. Try: pip install -e .
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
)

echo [OK] Starting server...
start "Z-Image" "%PYTHON_CMD%" webui.py

echo [WAIT] Waiting for server to be ready...
powershell -Command "while(1){try{$r=Invoke-WebRequest 'http://localhost:7860' -UseBasicParsing;break}catch{Start-Sleep 2}}" >nul

echo [OK] Server ready!
start "" "http://localhost:7860"

echo.
echo ========================================
echo   Server running at http://localhost:7860
echo   Close the "Z-Image" window to stop.
echo ========================================
