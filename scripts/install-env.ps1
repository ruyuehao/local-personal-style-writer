# install-env.ps1 — create the Python venv and install dependencies.
# Called by run.ps1. Uses `uv` when available, falls back to `python -m venv`.
$ErrorActionPreference = 'Stop'

$Root     = Split-Path -Parent $PSScriptRoot
$PSScript = $PSScriptRoot

$infoPath = Join-Path $Root 'info.json'
$info     = Get-Content $infoPath -Raw | ConvertFrom-Json
$venvName = $info.venv_name
$pyVer    = $info.python_version
$venv     = Join-Path $env:USERPROFILE ".openvino\venv\$venvName"
$uv       = Get-Command uv -ErrorAction SilentlyContinue

if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    Write-Output "创建虚拟环境: $venv (Python $pyVer)"
    if ($uv) {
        $ErrorActionPreference = 'Continue'
        & uv venv --python $pyVer $venv 2>&1
        $ErrorActionPreference = 'Stop'
    } else {
        $ErrorActionPreference = 'Continue'
        & python -m venv $venv 2>&1
        $ErrorActionPreference = 'Stop'
    }
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Output "虚拟环境已存在: $venv"
}

$python = Join-Path $venv 'Scripts\python.exe'

Write-Output "安装依赖（可能需要几分钟）..."
$ErrorActionPreference = 'Continue'
if ($uv) {
    & uv pip install --python $python -r (Join-Path $Root 'requirements.txt') -q 2>&1
} else {
    & $python -m pip install --upgrade pip -q 2>&1
    & $python -m pip install -r (Join-Path $Root 'requirements.txt') -q 2>&1
}
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) { exit 1 }

# 运行时目录
New-Item -ItemType Directory -Path (Join-Path $Root 'profiles'),
                                     (Join-Path $Root 'data\rag_index'),
                                     (Join-Path $Root 'logs'),
                                     (Join-Path $Root 'saves') -Force | Out-Null

Write-Output "依赖安装完成"
