<#
  Z-Image WebUI Smart Launcher (English version)
  Auto-detects/downloads Python + dependencies
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir   = Join-Path $ScriptDir "_venv"
$PyExe     = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "     Z-Image WebUI  Starting...           " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

function Install-PythonAndDeps {
    Write-Host "`n[Download] Python not found, downloading Python 3.12 ..." -ForegroundColor Yellow

    $url = "https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe"
    $installer = Join-Path $env:TEMP "python-installer.exe"

    try {
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    } catch {
        Write-Host "[Error] Download failed: $_" -ForegroundColor Red
        Write-Host "[Hint] Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Yellow
        pause
        exit 1
    }

    Write-Host "[Install] Installing Python (silent)..." -ForegroundColor Yellow
    $installPath = Join-Path $ScriptDir "_python"
    Start-Process -Wait -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 TargetDir=`"$installPath`" PrependPath=0 Include_test=0"
    Remove-Item $installer -Force

    $sysPy = Join-Path $installPath "python.exe"
    if (!(Test-Path $sysPy)) {
        Write-Host "[Error] Python installation failed" -ForegroundColor Red
        pause
        exit 1
    }

    Write-Host "[Venv] Creating virtual environment..." -ForegroundColor Yellow
    & $sysPy -m venv $VenvDir

    Write-Host "[Pip] Installing dependencies (first time may be slow)..." -ForegroundColor Yellow
    $deps = @(
        "torch>=2.5.0", "transformers>=4.51.0",
        "safetensors", "pillow", "accelerate",
        "huggingface_hub>=0.25.0", "gradio>=4.0,<7.0", "psutil",
        "loguru", "tqdm"
    )
    foreach ($dep in $deps) {
        Write-Host "  $dep ..." -NoNewline
        try {
            & $PyExe -m pip install $dep --quiet --no-warn-script-location 2>&1 | Out-Null
            Write-Host " OK" -ForegroundColor Green
        } catch {
            Write-Host " FAILED (continuing)" -ForegroundColor Yellow
        }
    }

    # Install the project itself
    Write-Host "  zimage-native ..." -NoNewline
    try {
        Set-Location $ScriptDir
        & $PyExe -m pip install -e . --quiet --no-warn-script-location 2>&1 | Out-Null
        Write-Host " OK" -ForegroundColor Green
    } catch {
        Write-Host " FAILED" -ForegroundColor Yellow
    }

    Write-Host "`n[OK] Environment ready!" -ForegroundColor Green
}

function Ensure-Python {
    # 1. Prefer virtual environment
    if (Test-Path $PyExe) { return $PyExe }
    # 2. Check system Python (>=3.10)
    try {
        $sysPy = (Get-Command python -ErrorAction Stop).Source
        $ver = & $sysPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ([version]$ver -ge [version]"3.10") { return $sysPy }
    } catch {}
    # 3. Check locally installed Python
    $localPy = Join-Path $ScriptDir "_python\python.exe"
    if (Test-Path $localPy) { return $localPy }
    # 4. Download and install
    Install-PythonAndDeps
    return $PyExe
}

try {
    $python = Ensure-Python
    Write-Host "`n[Python] $python" -ForegroundColor Gray
    Set-Location $ScriptDir

    Write-Host "`n[Launch] Starting WebUI ..." -ForegroundColor Green
    Write-Host "   URL: http://localhost:7860" -ForegroundColor Cyan
    Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Gray

    Start-Sleep 2
    Start-Process "http://localhost:7860"
    & $python webui.py

    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        Write-Host "`n[Error] Abnormal exit ($LASTEXITCODE)" -ForegroundColor Red
        pause
    }
} catch {
    Write-Host "`n[Error] $_" -ForegroundColor Red
    pause
}
