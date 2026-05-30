param(
  [string]$Image = "tugraph/tugraph-runtime-centos7:latest",
  [int]$HttpPort = 7070,
  [int]$BoltPort = 7687,
  [string]$ContainerName = "tugraph-traffic-lab"
)

$ErrorActionPreference = "Stop"

# Resolve root the same way run_experiment.py does, respecting TUGRAPH3_ROOT.
if ($env:TUGRAPH3_ROOT) {
  $Root = Resolve-Path $env:TUGRAPH3_ROOT
} else {
  $Root = Resolve-Path (Join-Path $PSScriptRoot "..")
}
$ImportDir = Join-Path $Root "tugraph_import"
$DbDir = Join-Path $Root "tugraph_db"

# Docker on Windows may reject paths with CJK characters in volume mounts
# when Hyper-V / WSL translation is involved.
if ($ImportDir -match '[一-鿿㐀-䶿豈-﫿]') {
  Write-Warning "Import path contains Chinese characters: $ImportDir"
  Write-Warning "Docker volume mount may fail. Set TUGRAPH3_ROOT to an ASCII-only path as a workaround."
}

New-Item -ItemType Directory -Force -Path $DbDir | Out-Null

Write-Host "Starting TuGraph container..."
docker run -d --name $ContainerName `
  -p ${HttpPort}:7070 -p ${BoltPort}:7687 `
  -v "${ImportDir}:/data/import" `
  -v "${DbDir}:/var/lib/lgraph" `
  $Image

Write-Host "TuGraph container started: $ContainerName"
Write-Host "Browser: http://localhost:$HttpPort"
Write-Host "Import CSV files from /data/import. Use tugraph_import/sanity_checks.cypher after import."
