<#
.SYNOPSIS
  Set up the StemSepApp Python virtual environment and install backend requirements robustly.

.DESCRIPTION
  This script creates/uses `StemSepApp\.venv` and installs `StemSepApp\requirements.txt`
  with extra robustness for common Windows issues:
  - pip cache permission/locking errors
  - partially installed environments
  - missing build tools messages (surfaced clearly in the log)

  It writes a log file to:
    StemSepApp\install.log

  Default behavior:
  - Creates venv if missing
  - Upgrades pip
  - Attempts to purge pip cache (best-effort)
  - Installs requirements with --no-cache-dir
  - Verifies a few critical imports (psutil, yaml, aiohttp)

.USAGE
  From repo root:
    .\scripts\setup_backend_venv.ps1

  Custom python executable (optional):
    .\scripts\setup_backend_venv.ps1 -PythonExe "C:\Python312\python.exe"

  Force rebuild venv:
    .\scripts\setup_backend_venv.ps1 -RecreateVenv

  Skip cache purge:
    .\scripts\setup_backend_venv.ps1 -SkipCachePurge

  Install without --no-cache-dir (not recommended on Windows):
    .\scripts\setup_backend_venv.ps1 -AllowCache

  Python Launcher convenience:
    You can pass a command string like: -PythonExe "py -3.12"
    The script will split it into executable + args.

.NOTES
  - This script does NOT require running as Administrator.
  - If an antivirus is locking files under %LocalAppData%\pip\cache, this script
    avoids cache usage by default and tries to purge the cache.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [string]$PythonExe = "python",

  [Parameter(Mandatory = $false)]
  [switch]$RecreateVenv,

  [Parameter(Mandatory = $false)]
  [switch]$SkipCachePurge,

  [Parameter(Mandatory = $false)]
  [switch]$AllowCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-RepoRoot {
  # This file lives at: <repo_root>\scripts\setup_backend_venv.ps1
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Write-Log {
  param(
    [Parameter(Mandatory = $true)][string]$Message,
    [Parameter(Mandatory = $false)][ValidateSet("INFO","WARN","ERROR")][string]$Level = "INFO"
  )
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "[$ts][$Level] $Message"
  Write-Host $line
  if ($script:LogPath) {
    try {
      Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    } catch {
      # ignore log write failure
    }
  }
}

function Resolve-PythonCommand {
  param(
    [Parameter(Mandatory = $true)][string]$PythonCommand
  )
  # Supports:
  # - "python"
  # - "C:\...\python.exe"
  # - "py -3.12"  (Python Launcher)
  $trimmed = ""
  if ($null -ne $PythonCommand) {
    $trimmed = $PythonCommand.Trim()
  }
  if (-not $trimmed) { throw "PythonExe cannot be empty" }
  # Split on whitespace (simple + sufficient for our usage)
  $parts = $trimmed -split '\s+'
  $exe = $parts[0]
  $args = @()
  if ($parts.Length -gt 1) { $args = $parts[1..($parts.Length-1)] }
  return [pscustomobject]@{ Exe = $exe; Args = $args }
}

function Exec {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $false)][string]$WorkingDirectory
  )

  $argLine = ($Arguments | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
  }) -join " "

  Write-Log "RUN: $FilePath $argLine"

  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $FilePath
  $psi.Arguments = $argLine
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true
  if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  [void]$p.Start()

  $stdout = $p.StandardOutput.ReadToEnd()
  $stderr = $p.StandardError.ReadToEnd()

  $p.WaitForExit()

  if ($stdout) { Write-Log $stdout.TrimEnd() }
  if ($stderr) { Write-Log $stderr.TrimEnd() "WARN" }

  if ($p.ExitCode -ne 0) {
    throw "Command failed (exit $($p.ExitCode)): $FilePath $argLine"
  }
}

function Get-VenvPython {
  param(
    [Parameter(Mandatory = $true)][string]$VenvDir
  )
  $py = Join-Path $VenvDir "Scripts\python.exe"
  if (Test-Path $py) { return (Resolve-Path $py).Path }
  throw "Venv python not found at: $py"
}

function Ensure-Venv {
  param(
    [Parameter(Mandatory = $true)][string]$BackendDir,
    [Parameter(Mandatory = $true)][string]$PyExe,
    [Parameter(Mandatory = $false)][switch]$ForceRecreate
  )

  $venvDir = Join-Path $BackendDir ".venv"

  if ($ForceRecreate -and (Test-Path $venvDir)) {
    Write-Log "RecreateVenv enabled: removing existing $venvDir" "WARN"
    Remove-Item -Recurse -Force -LiteralPath $venvDir
  }

  if (-not (Test-Path $venvDir)) {
    Write-Log "Creating venv at $venvDir"
    $pc = Resolve-PythonCommand -PythonCommand $PyExe
    $args = @()
    if ($pc.Args -and $pc.Args.Count -gt 0) { $args += $pc.Args }
    $args += @("-m","venv",".venv")
    Exec -FilePath $pc.Exe -Arguments $args -WorkingDirectory $BackendDir
  } else {
    Write-Log "Venv already exists at $venvDir"
  }

  return $venvDir
}

function Try-PipCachePurge {
  param(
    [Parameter(Mandatory = $true)][string]$VenvPython
  )

  Write-Log "Attempting pip cache purge (best-effort)..."
  try {
    Exec -FilePath $VenvPython -Arguments @("-m","pip","cache","purge") -WorkingDirectory $script:BackendDir
    Write-Log "pip cache purge completed"
    return
  } catch {
    Write-Log "pip cache purge failed or unsupported: $($_.Exception.Message)" "WARN"
  }

  # Fallback: remove cache directory directly
  $cacheDir = Join-Path $env:LocalAppData "pip\cache"
  if (Test-Path $cacheDir) {
    try {
      Write-Log "Attempting direct delete of cache directory: $cacheDir" "WARN"
      Remove-Item -Recurse -Force -LiteralPath $cacheDir
      Write-Log "Deleted cache directory: $cacheDir"
    } catch {
      Write-Log "Failed to delete cache directory (may be locked by AV): $($_.Exception.Message)" "WARN"
    }
  } else {
    Write-Log "pip cache directory not found at $cacheDir"
  }
}

# -------------------- Main --------------------

$repoRoot = Resolve-RepoRoot
$script:BackendDir = Join-Path $repoRoot "StemSepApp"
$requirementsPath = Join-Path $script:BackendDir "requirements.txt"
$script:LogPath = Join-Path $script:BackendDir "install.log"

# Reset log
try { Remove-Item -Force -LiteralPath $script:LogPath -ErrorAction SilentlyContinue } catch {}

Write-Log "Repo root  : $repoRoot"
Write-Log "Backend dir: $script:BackendDir"
Write-Log "Python exe : $PythonExe"
Write-Log "Log file   : $script:LogPath"
Write-Log "RecreateVenv: $([bool]$RecreateVenv)"
Write-Log "SkipCachePurge: $([bool]$SkipCachePurge)"
Write-Log "AllowCache: $([bool]$AllowCache)"

if (-not (Test-Path $script:BackendDir)) {
  Write-Log "ERROR: StemSepApp directory not found at $script:BackendDir" "ERROR"
  exit 2
}
if (-not (Test-Path $requirementsPath)) {
  Write-Log "ERROR: requirements.txt not found at $requirementsPath" "ERROR"
  exit 2
}

try {
  $venvDir = Ensure-Venv -BackendDir $script:BackendDir -PyExe $PythonExe -ForceRecreate:$RecreateVenv
  $venvPython = Get-VenvPython -VenvDir $venvDir

  # Upgrade pip first
  Write-Log "Upgrading pip..."
  Exec -FilePath $venvPython -Arguments @("-m","pip","install","--upgrade","pip") -WorkingDirectory $script:BackendDir

  if (-not $SkipCachePurge) {
    Try-PipCachePurge -VenvPython $venvPython
  } else {
    Write-Log "Skipping cache purge (SkipCachePurge enabled)" "WARN"
  }

  # Install requirements
  $pipArgs = @("-m","pip","install")
  if (-not $AllowCache) {
    $pipArgs += "--no-cache-dir"
  }
  $pipArgs += @("-r","requirements.txt")

  Write-Log "Installing requirements..."
  Exec -FilePath $venvPython -Arguments $pipArgs -WorkingDirectory $script:BackendDir

  # Quick sanity imports
  Write-Log "Verifying critical imports..."
  Exec -FilePath $venvPython -Arguments @(
    "-c",
    "import psutil, yaml, aiohttp; print('OK imports:', 'psutil', psutil.__version__, 'yaml', getattr(yaml,'__version__','?'), 'aiohttp', aiohttp.__version__)"
  ) -WorkingDirectory $script:BackendDir

  # CUDA readiness checks (best-effort; do not fail setup)
  Write-Log "Checking CUDA readiness (best-effort)..."
  try {
    $torchCheck = "import importlib.util, sys; spec=importlib.util.find_spec('torch'); " +
      "(print('TORCH: not installed') or sys.exit(0)) if spec is None else None; " +
      "import torch; ver=getattr(torch,'__version__','?'); " +
      "has_cuda=bool(getattr(torch,'cuda',None)) and bool(torch.cuda.is_available()); " +
      "dev_count=int(torch.cuda.device_count()) if has_cuda else 0; " +
      "name0=torch.cuda.get_device_name(0) if has_cuda and dev_count>0 else None; " +
      "print('TORCH:', ver, '| cuda_available=', has_cuda, '| device_count=', dev_count, '| gpu0=', name0); " +
      "print('TORCH WARNING: CPU-only build detected (install CUDA wheels).') if sys.platform=='win32' and str(ver).endswith('+cpu') else None"

    $onnxCheck = "import importlib.util, sys; spec=importlib.util.find_spec('onnxruntime'); " +
      "(print('ONNXRUNTIME: not installed') or sys.exit(0)) if spec is None else None; " +
      "import onnxruntime as ort; providers=list(ort.get_available_providers()) if hasattr(ort,'get_available_providers') else []; " +
      "print('ONNXRUNTIME:', getattr(ort,'__version__','?'), '| providers=', providers); " +
      "print('ONNXRUNTIME NOTE: CUDAExecutionProvider not present (install onnxruntime-gpu and ensure it overrides CPU binaries).') if 'CUDAExecutionProvider' not in providers else None"

    Exec -FilePath $venvPython -Arguments @("-c", $torchCheck) -WorkingDirectory $script:BackendDir

    # Optional: run the repo's check_cuda.py for an explicit torch CUDA report
    $checkCudaScript = Join-Path $repoRoot "check_cuda.py"
    if (Test-Path $checkCudaScript) {
      try {
        Exec -FilePath $venvPython -Arguments @($checkCudaScript) -WorkingDirectory $repoRoot
      } catch {
        Write-Log "check_cuda.py failed (continuing): $($_.Exception.Message)" "WARN"
      }
    }

    Exec -FilePath $venvPython -Arguments @("-c", $onnxCheck) -WorkingDirectory $script:BackendDir

    Exec -FilePath $venvPython -Arguments @(
      "-c",
      "print('CUDA readiness check complete.')"
    ) -WorkingDirectory $script:BackendDir
  } catch {
    Write-Log "CUDA readiness check failed (continuing): $($_.Exception.Message)" "WARN"
  }

  Write-Log "SUCCESS: Backend venv is ready."
  Write-Log "Next: start the Electron app from repo root:"
  Write-Log "  cd electron-poc"
  Write-Log "  npm run electron:dev"
  exit 0
}
catch {
  Write-Log "FAILED: $($_.Exception.Message)" "ERROR"
  Write-Log "See log: $script:LogPath" "ERROR"
  Write-Log "Common fixes:" "WARN"
  Write-Log "  - Close running Python/Electron processes that may lock files" "WARN"
  Write-Log "  - Temporarily disable AV real-time scanning for %LocalAppData%\pip\cache" "WARN"
  Write-Log "  - Re-run with: -RecreateVenv" "WARN"
  exit 1
}
