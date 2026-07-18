# test.ps1 — end-to-end smoke test for the local-style-writer skill.
# Run from the skill root:  .\tests\test.ps1
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$run  = Join-Path $Root 'scripts\run.ps1'

function Invoke-Skill {
    param([string]$Args)
    $out = & $run @Args 2>&1
    return ,@($LASTEXITCODE, ($out -join "`n"))
}

Write-Host "=== [1/3] 生成测试 ==="
$code, $out = Invoke-Skill @('帮我写一篇关于端侧 AI 的 50 字短文', '--length', '50字')
if ($code -ne 0) {
    Write-Host "✗ 生成失败 (exit=$code):`n$out"
    exit 1
}
if ($out -notmatch '【生成内容】' -or $out -match '错误') {
    Write-Host "✗ 生成输出异常:`n$out"
    exit 1
}
Write-Host "✓ 生成成功"

Write-Host ""
Write-Host "=== [2/3] 风格分析测试 ==="
$code, $out = Invoke-Skill @('--analyze', '这是一段用于风格分析的测试文本。')
if ($code -ne 0) {
    Write-Host "✗ 分析失败 (exit=$code):`n$out"
    exit 1
}
if ($out -notmatch '【风格分析】') {
    Write-Host "✗ 分析输出异常:`n$out"
    exit 1
}
Write-Host "✓ 风格分析成功"

Write-Host ""
Write-Host "=== [3/3] 帮助/参数检查 ==="
$code, $out = Invoke-Skill @()
if ($code -ne 2) {
    Write-Host "✗ 无参数时应返回用法提示 (exit=2)，实际 exit=$code:`n$out"
    exit 1
}
Write-Host "✓ 参数检查通过"

Write-Host ""
Write-Host "ALL TESTS PASSED"
