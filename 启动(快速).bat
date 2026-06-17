@echo off
title Z-Image WebUI

echo ========================================
echo   Z-Image WebUI  Starting...
echo ========================================

cd /d "%~dp0"

set PYTHON_CMD=

REM Check common Python install locations (skip Windows Store stub)
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python313\python.exe"
    "C:\Users\zzz\AppData\Local\Programs\Python\Python312\python.exe"
) do if exist %%p set PYTHON_CMD=%%p

REM Check system PATH (skip WindowsApps)
if "%PYTHON_CMD%"=="" (
    for /f "tokens=*" %%a in ('where python 2^>nul') do (
        echo %%a | findstr /i "WindowsApps" >nul
        if errorlevel 1 set PYTHON_CMD=%%a
    )
)

if "%PYTHON_CMD%"=="" (
    echo [!] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON_CMD%
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
