# run.ps1 — fixed entry point (never rename)
# 宿主（Qoder / Marvis）唯一调用点： scripts\run.ps1 "帮我写一篇关于 AI PC 的短文"
$ErrorActionPreference = 'Stop'

$Root      = Split-Path -Parent $PSScriptRoot   # skill 根目录
$PSScript  = $PSScriptRoot                      # scripts/

# --- 0. 无参数：仅打印用法（退出码 2），跳过环境/模型准备 --------------------
if ($args.Count -eq 0) {
    $infoPath0 = Join-Path $Root 'info.json'
    $info0     = Get-Content $infoPath0 -Raw | ConvertFrom-Json
    $venv0     = Join-Path $env:USERPROFILE ".openvino\venv\$($info0.venv_name)"
    $py0       = Join-Path $venv0 'Scripts\python.exe'
    if (Test-Path $py0) {
        & $py0 (Join-Path $PSScript 'client.py') @args
    } else {
        & python (Join-Path $PSScript 'client.py') @args
    }
    exit $LASTEXITCODE
}

# --- 0.5 硬件门禁（先于安装，失败快速退出，避免在不支持环境白装依赖）------
# 说明：OpenVINO 本体由 install-env.ps1 安装，此处只做"能否跑"的廉价前置判断。
# 在 Marvis 等宿主里此检测由宿主完成；standalone 运行时由本脚本兜底。
if ($IsLinux -or $IsMacOS) {
    Write-Output "This skill targets Windows AI PC (Intel NPU/GPU). Unsupported OS."
    exit 1
}
$npuGpu = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
    $_.FriendlyName -match "NPU|Neural Processing|Intel.*Graphics|GPU"
}
if (-not $npuGpu) {
    Write-Output "提示: 未检测到 Intel NPU / GPU，将回退 CPU 推理（仍可用，速度较慢）。"
}

# --- 1. 安装 Python 环境（uv 建 venv）---------------------------------------
& (Join-Path $PSScript 'install-env.ps1')
if ($LASTEXITCODE -ne 0) { exit 1 }

# --- 2. 解析 venv python ----------------------------------------------------
$infoPath = Join-Path $Root 'info.json'
$info     = Get-Content $infoPath -Raw | ConvertFrom-Json
$venv     = Join-Path $env:USERPROFILE ".openvino\venv\$($info.venv_name)"
$python   = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Output "Python 环境未就绪: $python"
    exit 1
}

# --- 3. 硬件 / OpenVINO 就绪检测（无 OpenVINO 则退出）-----------------------
$hw = & $python -c 'import openvino; print(''OK'')' 2>$null
if ($hw -ne 'OK') {
    Write-Output 'This skill requires OpenVINO (Intel AI PC with NPU/GPU, or CPU fallback). OpenVINO is not importable.'
    exit 1
}
$hwAcc = & $python -c 'import openvino; print(''ACC'' if (''GPU'' in openvino.Core().available_devices or ''NPU'' in openvino.Core().available_devices) else ''CPU'')' 2>$null
if ($hwAcc.Trim() -ne 'ACC') {
    Write-Output '提示: 当前仅检测到 CPU，推理仍可进行但速度较慢（无 NPU/GPU 加速）。'
}

# --- 4. 确保全部模型就绪（遍历 info.json 的 models[] 逐一校验 required_files；
#        任一缺失则触发下载，下载后复检；仍不齐返回 3，提示重新运行）------------
function Test-AllModelsReady {
    param ([string]$ModelRoot, $Models)
    foreach ($m in $Models) {
        $finalDir = Join-Path $ModelRoot $m.dir_name
        foreach ($rf in $m.required_files) {
            if (-not (Test-Path (Join-Path $finalDir $rf))) { return $false }
        }
    }
    return $true
}

$modelRoot = Join-Path $Root 'models'
if (-not (Test-AllModelsReady -ModelRoot $modelRoot -Models $info.models)) {
    Write-Output '--- 部分模型未就绪，开始下载（首次约 5.6GB，可能耗时较久）---'
    & (Join-Path $Root 'models\download.ps1')
    if (-not (Test-AllModelsReady -ModelRoot $modelRoot -Models $info.models)) {
        Write-Output '模型仍在下载或未完成。请重新运行 scripts\run.ps1 继续下载。'
        exit 3
    }
}

# --- 5. 调用客户端（透传所有参数）------------------------------------------
& $python (Join-Path $PSScript 'client.py') @args
exit $LASTEXITCODE
