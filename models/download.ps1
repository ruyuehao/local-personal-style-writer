$ErrorActionPreference = "Stop"

$ModelDir = $PSScriptRoot
# info.json 位于 models/ 的上一级（skill 根目录），作为模型清单的单一数据源
$InfoPath = Join-Path (Split-Path -Parent $ModelDir) 'info.json'
$info = Get-Content $InfoPath -Raw | ConvertFrom-Json

function Test-ModelComplete {
    param ([string]$Dir, [array]$Required)
    foreach ($rf in $Required) {
        if (-not (Test-Path (Join-Path $Dir $rf))) { return $false }
    }
    return $true
}

$allOk = $true
foreach ($m in $info.models) {
    $dirName    = $m.dir_name
    $modelId    = $m.model_id
    $required   = $m.required_files
    $finalDir   = Join-Path $ModelDir $dirName
    $partialDir = Join-Path $ModelDir "$dirName.partial"

    # 已完成则跳过（含重跑时的幂等）
    if (Test-ModelComplete -Dir $finalDir -Required $required) {
        Write-Host "[skip] $dirName 已就绪，跳过下载"
        continue
    }

    Write-Host "[download] $modelId -> $dirName (先下到 .partial)..."
    # 清掉可能残留的半下载目录，保证干净重试
    if (Test-Path $partialDir) { Remove-Item $partialDir -Recurse -Force }

    modelscope download `
        --model $modelId `
        --local_dir "$partialDir"

    # 校验 required_files 全部就位，杜绝「半下载被误判为完成模型」
    if (-not (Test-ModelComplete -Dir $partialDir -Required $required)) {
        Write-Host "错误: $dirName 下载后缺少必需文件 ($($required -join ', '))，已中止。"
        $allOk = $false
        break
    }

    # 原子重命名：仅在所有文件校验通过后才把 .partial 变为正式目录
    if (Test-Path $finalDir) { Remove-Item $finalDir -Recurse -Force }
    Move-Item $partialDir $finalDir
    Write-Host "[ok] $dirName 就绪"
}

if (-not $allOk) {
    Write-Host "模型下载未完成，请重新运行 scripts\run.ps1 继续。"
    exit 1
}

Write-Host ""
Write-Host "所有模型下载完成"

