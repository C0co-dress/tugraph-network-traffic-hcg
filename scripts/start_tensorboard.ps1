param(
  [int]$Port = 6006
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $Root "runs"

Write-Host "TensorBoard logdir: $LogDir"

$env:PYTHONUTF8 = "1"
& "D:\Python313\python.exe" -X utf8 -m tensorboard.main --logdir $LogDir --port $Port
