<#
.SYNOPSIS
  Run StemSep Python tests using the correct project virtual environment.

.DESCRIPTION
  Prefers `StemSepApp\.venv` (Windows-first), then repo `.venv`, then system Python.
  Executes `python -m pytest` so module resolution is consistent across environments.
#>

[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-PythonExe {
  param([string]$RepoRoot)
  $candidates = @(
    (Join-Path $RepoRoot "StemSepApp\.venv\Scripts\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\python.exe")
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { return (Resolve-Path $c).Path }
  }

  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return $cmd.Source }
  $cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
  if ($cmd3 -and $cmd3.Source) { return $cmd3.Source }
  throw "Could not find Python interpreter."
}

$repoRoot = Resolve-RepoRoot
Push-Location $repoRoot
try {
  $pythonExe = Resolve-PythonExe -RepoRoot $repoRoot
  $pytestArgs = @()
  if ($Args -and $Args.Count -gt 0) {
    $pytestArgs = $Args
  } else {
    $pytestArgs = @("-q", "StemSepApp/tests")
  }

  Write-Host "Repo root : $repoRoot"
  Write-Host "Python    : $pythonExe"
  Write-Host "Pytest    : $($pytestArgs -join ' ')"

  & $pythonExe -m pytest @pytestArgs
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}

