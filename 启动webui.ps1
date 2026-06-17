<#
  Z-Image WebUI 智能启动器
  自动检测/安装 Python + 依赖
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir   = Join-Path $ScriptDir "_venv"
$PyExe     = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║      ⚡ Z-Image 文生图  启动中...    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════╝" -ForegroundColor ""

function Install-PythonAndDeps {
    Write-Host "`n📥 未检测到 Python，正在下载 Python 3.12 ..." -ForegroundColor Yellow

    $url = "https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe"
    $installer = Join-Path $env:TEMP "python-installer.exe"

    try {
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    } catch {
        Write-Host "❌ 下载失败: $_" -ForegroundColor Red
        Write-Host "💡 请从 https://www.python.org/downloads/ 手动安装 Python 3.10+" -ForegroundColor Yellow
        pause
        exit 1
    }

    Write-Host "📦 安装 Python（静默安装）..." -ForegroundColor Yellow
    $installPath = Join-Path $ScriptDir "_python"
    Start-Process -Wait -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 TargetDir=`"$installPath`" PrependPath=0 Include_test=0"
    Remove-Item $installer -Force

    $sysPy = Join-Path $installPath "python.exe"
    if (!(Test-Path $sysPy)) {
        Write-Host "❌ Python 安装失败" -ForegroundColor Red
        pause
        exit 1
    }

    Write-Host "📦 创建虚拟环境..." -ForegroundColor Yellow
    & $sysPy -m venv $VenvDir

    Write-Host "📦 安装依赖（首次较慢）..." -ForegroundColor Yellow
    $deps = @(
        "torch>=2.5.0", "transformers>=4.51.0",
        "safetensors", "pillow", "accelerate",
        "huggingface_hub>=0.25.0", "gradio", "psutil",
        "loguru", "tqdm"
    )
    foreach ($dep in $deps) {
        Write-Host "  ⏳ $dep ..." -NoNewline
        try {
            & $PyExe -m pip install $dep --quiet --no-warn-script-location 2>&1 | Out-Null
            Write-Host " ✅" -ForegroundColor Green
        } catch {
            Write-Host " ⚠️ 失败（继续）" -ForegroundColor Yellow
        }
    }

    # 安装项目自身
    Write-Host "  ⏳ zimage-native ..." -NoNewline
    try {
        Set-Location $ScriptDir
        & $PyExe -m pip install -e . --quiet --no-warn-script-location 2>&1 | Out-Null
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ⚠️ 失败" -ForegroundColor Yellow
    }

    Write-Host "`n✅ 环境准备完成！" -ForegroundColor Green
}

function Ensure-Python {
    # 1. 优先虚拟环境
    if (Test-Path $PyExe) { return $PyExe }
    # 2. 检测系统 Python（≥3.10）
    try {
        $sysPy = (Get-Command python -ErrorAction Stop).Source
        $ver = & $sysPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ([version]$ver -ge [version]"3.10") { return $sysPy }
    } catch {}
    # 3. 检测本地安装的 Python
    $localPy = Join-Path $ScriptDir "_python\python.exe"
    if (Test-Path $localPy) { return $localPy }
    # 4. 下载安装
    Install-PythonAndDeps
    return $PyExe
}

try {
    $python = Ensure-Python
    Write-Host "`n🐍 Python: $python" -ForegroundColor Gray
    Set-Location $ScriptDir

    Write-Host "`n🚀 启动 WebUI ..." -ForegroundColor Green
    Write-Host "   网址: http://localhost:7860" -ForegroundColor Cyan
    Write-Host "   Ctrl+C 停止`n" -ForegroundColor Gray

    Start-Sleep 2
    Start-Process "http://localhost:7860"
    & $python webui.py

    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        Write-Host "`n❌ 异常退出 ($LASTEXITCODE)" -ForegroundColor Red
        pause
    }
} catch {
    Write-Host "`n❌ 错误: $_" -ForegroundColor Red
    pause
}
