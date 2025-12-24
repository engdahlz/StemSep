<#
.SYNOPSIS
  Run a Python script in this repo without relying on an auto-activated venv.

.DESCRIPTION
  This helper is meant to avoid issues where your shell profile tries to auto-activate a venv
  that may not exist (or may be different per machine).

  Behavior:
  - If StemSepApp\.venv exists: uses its python.
  - Else if .venv exists at repo root: uses its python.
  - Else: uses system python from PATH.
  - Runs the provided script with any additional args.
  - Keeps working directory at repo root to make relative paths predictable.

.USAGE
  From repo root:
    .\scripts\run_py.ps1 .\validate_model_registry.py
    .\scripts\run_py.ps1 .\scripts\download_configs.py --help
    .\scripts\run_py.ps1 -Venv StemSepApp .\StemSepApp\src\main.py

.PARAMETER Script
  Path to the python script to execute (relative or absolute).

.PARAMETER Venv
  Which venv location to prefer:
    - "StemSepApp" (default): StemSepApp\.venv
    - "Root": .\.venv
    - "None": do not use venv even if present

.PARAMETER Python
  Explicit python executable path to use (overrides venv selection).

.PARAMETER Args
  Any additional args to pass to the script.

.NOTES
  - This script intentionally does NOT try to activate a venv in the current shell.
    It directly executes the python executable, which is more robust.
  - Windows uses python.exe in Scripts\ (not bin/).
#>

[CmdletBinding(PositionalBinding = $true)]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Script,

  [Parameter(Mandatory = $false)]
  [ValidateSet("StemSepApp", "Root", "None")]
  [string]$Venv = "StemSepApp",

  [Parameter(Mandatory = $false)]
  [string]$Python,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  # This file lives at: <repo_root>\scripts\run_py.ps1
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-PythonExe {
  param(
    [string]$RepoRoot,
    [string]$PreferredVenv,
    [string]$ExplicitPython
  )

  if ($ExplicitPython -and $ExplicitPython.Trim().Length -gt 0) {
    $p = $ExplicitPython.Trim()
    if (Test-Path $p) { return (Resolve-Path $p).Path }
    throw "Explicit -Python path not found: $p"
  }

  if ($PreferredVenv -ne "None") {
    $venvPath = $null

    if ($PreferredVenv -eq "StemSepApp") {
      $venvPath = Join-Path $RepoRoot "StemSepApp\.venv"
    } elseif ($PreferredVenv -eq "Root") {
      $venvPath = Join-Path $RepoRoot ".venv"
    }

    if ($venvPath -and (Test-Path $venvPath)) {
      $candidate = Join-Path $venvPath "Scripts\python.exe"
      if (Test-Path $candidate) {
        return (Resolve-Path $candidate).Path
      }
    }
  }

  # Fall back to PATH
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return $cmd.Source }

  $cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
  if ($cmd3 -and $cmd3.Source) { return $cmd3.Source }

  throw "Could not find Python. Install Python or create a venv at StemSepApp\.venv or .\.venv."
}

$repoRoot = Resolve-RepoRoot
Push-Location $repoRoot
try {
  $scriptPath = $Script
  if (-not [System.IO.Path]::IsPathRooted($scriptPath)) {
    $scriptPath = Join-Path $repoRoot $scriptPath
  }

  if (-not (Test-Path $scriptPath)) {
    throw "Script not found: $scriptPath"
  }

  $pythonExe = Resolve-PythonExe -RepoRoot $repoRoot -PreferredVenv $Venv -ExplicitPython $Python

  Write-Host "Repo root : $repoRoot"
  Write-Host "Python    : $pythonExe"
  Write-Host "Script    : $scriptPath"
  if ($Args -and $Args.Count -gt 0) {
    Write-Host "Args      : $($Args -join ' ')"
  }

  & $pythonExe $scriptPath @Args
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
