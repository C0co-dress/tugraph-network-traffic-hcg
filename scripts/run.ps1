param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

# Force UTF-8 mode on Chinese Windows — resolves path issues with CJK usernames.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = if ($env:TUGRAPH3_ROOT) {
  Resolve-Path $env:TUGRAPH3_ROOT
} else {
  Resolve-Path (Join-Path $ScriptDir "..")
}

Write-Host "TUGRAPH3_ROOT = $Root" -ForegroundColor Cyan

& "D:\Python313\python.exe" -X utf8 (Join-Path $ScriptDir "run_experiment.py") @PythonArgs
