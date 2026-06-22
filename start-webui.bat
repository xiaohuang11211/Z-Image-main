@echo off
title Z-Image WebUI

echo ========================================
echo   Z-Image WebUI  Starting...
echo ========================================

cd /d "%~dp0"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Find real Python (skip Windows Store stub)
set PYTHON_CMD=
where python /all >"%TEMP%\py_list.txt" 2>nul
for /f "tokens=*" %%a in ('type "%TEMP%\py_list.txt"') do (
    echo %%a | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set PYTHON_CMD=%%a
        goto :found
    )
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if exist "C:\Python312\python.exe" set PYTHON_CMD=C:\Python312\python.exe
if exist "C:\Python313\python.exe" set PYTHON_CMD=C:\Python313\python.exe

:found
if "%PYTHON_CMD%"=="" (
    echo [!] Python not found. Trying auto-install...
    powershell -NoProfile -ExecutionPolicy Bypass -File "start.ps1"
    if errorlevel 1 (
        echo [!] Auto-install failed. Please install Python 3.10+ from python.org
        pause
        exit /b 1
    )
    exit /b 0
)

echo [OK] Python: %PYTHON_CMD%
echo [OK] Starting server...
start "Z-Image" "%PYTHON_CMD%" webui.py

echo [WAIT] Waiting for server...
powershell -Command "while(1){try{$r=Invoke-WebRequest 'http://localhost:7860' -UseBasicParsing;break}catch{Start-Sleep 2}}" >nul

echo [OK] Server is ready!
start "" "http://localhost:7860"

echo.
echo ========================================
echo   Browser opened at http://localhost:7860
echo   Close the server window to stop.
echo ========================================
