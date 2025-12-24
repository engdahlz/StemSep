<#
.SYNOPSIS
  Detect and optionally fix hardcoded virtualenv auto-activation (especially broken .venv activate.ps1 paths).

.DESCRIPTION
  Some shells / tasks / editor integrations end up running a command like:

    powershell -Command '. "C:\path\to\repo\.venv\Scripts\activate.ps1"'

  If that venv was removed or moved, every invocation prints an error. This utility:
  - Scans common user-level config files for hardcoded activate.ps1 invocations
  - Reports findings with file + line number + matched text
  - Optionally rewrites those invocations into a safe "Test-Path guard"
  - Creates backups before modifying anything

  It focuses on fixing the specific class of issue: a hardcoded dot-source of Activate.ps1
  that isn't guarded by existence checks.

.SAFETY
  - Default mode is READ-ONLY reporting.
  - With -Apply, it modifies files but ALWAYS writes a timestamped backup first.
  - It never deletes files.
  - It does not attempt to "activate" anything; it only edits config text.

.USAGE
  # Report only (recommended first)
  .\scripts\fix_auto_activate.ps1

  # Apply fixes (with backups)
  .\scripts\fix_auto_activate.ps1 -Apply

  # Apply fixes, but only inside the repo root (if your hardcoded path is there)
  .\scripts\fix_auto_activate.ps1 -Apply -RestrictToRepo

  # Apply fixes and also scan VS Code user settings.json (best-effort)
  .\scripts\fix_auto_activate.ps1 -ScanVSCode -Apply

.NOTES
  This script runs on Windows PowerShell 5.1+ and PowerShell 7+.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [switch]$Apply,

  [Parameter(Mandatory = $false)]
  [switch]$RestrictToRepo,

  [Parameter(Mandatory = $false)]
  [switch]$ScanVSCode,

  [Parameter(Mandatory = $false)]
  [switch]$WhatIfOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  try {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  } catch {
    return $null
  }
}

function Get-Timestamp {
  return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function Backup-File {
  param(
    [Parameter(Mandatory = $true)][string]$Path
  )
  $ts = Get-Timestamp
  $bak = "$Path.bak.$ts"
  Copy-Item -LiteralPath $Path -Destination $bak -Force
  return $bak
}

function Read-AllText {
  param([Parameter(Mandatory = $true)][string]$Path)
  return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Write-AllTextUtf8NoBom {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Content
  )
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function Get-CandidateFiles {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $false)][switch]$IncludeVSCode
  )

  $userHome = $env:USERPROFILE
  $docs = Join-Path $userHome "Documents"

  $paths = New-Object System.Collections.Generic.List[string]

  # PowerShell profiles (Windows PowerShell + PowerShell 7)
  $paths.Add((Join-Path $docs "WindowsPowerShell\Microsoft.PowerShell_profile.ps1"))
  $paths.Add((Join-Path $docs "WindowsPowerShell\profile.ps1"))
  $paths.Add((Join-Path $docs "PowerShell\Microsoft.PowerShell_profile.ps1"))
  $paths.Add((Join-Path $docs "PowerShell\profile.ps1"))

  # Git Bash / MSYS bash configs
  $paths.Add((Join-Path $userHome ".bashrc"))
  $paths.Add((Join-Path $userHome ".bash_profile"))
  $paths.Add((Join-Path $userHome ".profile"))

  # Common Windows Terminal settings (optional; JSON)
  $paths.Add((Join-Path $userHome "AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"))
  $paths.Add((Join-Path $userHome "AppData\Local\Microsoft\Windows Terminal\settings.json"))

  if ($IncludeVSCode) {
    # VS Code user settings
    $paths.Add((Join-Path $userHome "AppData\Roaming\Code\User\settings.json"))
    $paths.Add((Join-Path $userHome "AppData\Roaming\Code - Insiders\User\settings.json"))
  }

  # Filter existing files
  $existing = $paths | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
  return $existing
}

function Find-HardcodedActivatePs1 {
  param(
    [Parameter(Mandatory = $true)][string]$Content
  )

  # We target suspicious patterns:
  # - powershell/pwsh running -Command ". '...activate.ps1'" or ". \"...activate.ps1\""
  # - direct dot-sourcing of an activate.ps1 path
  #
  # Regex notes:
  # - Case-insensitive
  # - We capture the quoted path to activate.ps1
  $patterns = @(
    '(?im)\b(pwsh|powershell)\b[^\r\n]*?-Command[^\r\n]*?\.\s*["'']([^"'']*?activate\.ps1)["'']',
    '(?im)(?:^|\s)\.\s*["'']([^"'']*?activate\.ps1)["'']'
  )

  $hits = New-Object System.Collections.Generic.List[object]
  foreach ($pat in $patterns) {
    $m = [regex]::Matches($Content, $pat)
    foreach ($mm in $m) {
      $path = $null
      if ($mm.Groups.Count -ge 3) {
        # pattern 1: group2 is path
        $path = $mm.Groups[2].Value
      } elseif ($mm.Groups.Count -ge 2) {
        # pattern 2: group1 is path
        $path = $mm.Groups[1].Value
      }
      $hits.Add([pscustomobject]@{
        MatchText = $mm.Value
        ActivatePath = $path
        Index = $mm.Index
        Length = $mm.Length
      })
    }
  }
  return $hits
}

function Get-LineInfo {
  param(
    [Parameter(Mandatory = $true)][string]$Content,
    [Parameter(Mandatory = $true)][int]$Index
  )
  # Determine line number and line text at a character index
  $before = $Content.Substring(0, [Math]::Min($Index, $Content.Length))
  $lineNumber = ($before -split "(\r\n|\n)").Count
  $lines = $Content -split "\r\n|\n"
  $ln = [Math]::Max(1, [Math]::Min($lineNumber, $lines.Count))
  $lineText = $lines[$ln - 1]
  return [pscustomobject]@{
    LineNumber = $ln
    LineText = $lineText
  }
}

function Is-GuardedAlready {
  param(
    [Parameter(Mandatory = $true)][string]$LineText
  )
  # Heuristic: if the line already contains Test-Path and Activate.ps1, assume it's guarded.
  $lt = $LineText.ToLowerInvariant()
  return ($lt.Contains("test-path") -and ($lt.Contains("activate.ps1") -or $lt.Contains("activate.ps1".ToLowerInvariant())))
}

function Build-GuardedReplacement {
  param(
    [Parameter(Mandatory = $true)][string]$ActivatePath
  )

  # We will replace with a guarded dot-source. If path is absolute, keep it.
  # The guard uses PowerShell syntax; for bash scripts that call powershell -Command,
  # it still works as long as the replacement is inserted within PowerShell context.
  #
  # We emit a minimal safe form:
  #   if (Test-Path "<path>") { . "<path>" }
  #
  # Escape double-quotes for safe inline usage in most contexts
  $p = $ActivatePath
  if (-not $p) { $p = "" }

  $pEsc = $p.Replace('"', '""')
  return "if (Test-Path `"$pEsc`") { . `"$pEsc`" }"
}

function Restrict-PathToRepo {
  param(
    [Parameter(Mandatory = $true)][string]$ActivatePath,
    [Parameter(Mandatory = $true)][string]$RepoRoot
  )
  if (-not $ActivatePath) { return $false }
  try {
    $rp = $RepoRoot.Replace("/", "\")
    $ap = $ActivatePath.Replace("/", "\")
    return $ap.ToLowerInvariant().Contains($rp.ToLowerInvariant())
  } catch {
    return $false
  }
}

$repoRoot = Get-RepoRoot
if (-not $repoRoot) {
  Write-Host "WARN: Could not resolve repo root from script location."
  $repoRoot = ""
}

Write-Host "Repo root : $repoRoot"
if ($Apply) {
  Write-Host "Mode      : APPLY"
} else {
  Write-Host "Mode      : REPORT"
}
if ($RestrictToRepo) { Write-Host "Restrict  : Only fix activate.ps1 paths that include repo root" }
if ($ScanVSCode) { Write-Host "Scan      : Including VS Code user settings.json" }
if ($WhatIfOnly) { Write-Host "WhatIf    : Enabled (no file writes)" }

$files = @(Get-CandidateFiles -RepoRoot $repoRoot -IncludeVSCode:$ScanVSCode)

if (-not $files -or $files.Length -eq 0) {
  Write-Host "No candidate config files found to scan."
  exit 0
}

$totalHits = 0
$totalFixable = 0
$totalChanged = 0

foreach ($file in $files) {
  Write-Host ""
  Write-Host "== Scanning: $file =="
  $content = Read-AllText -Path $file

  $hits = Find-HardcodedActivatePs1 -Content $content
  if (-not $hits -or $hits.Count -eq 0) {
    Write-Host "  No activate.ps1 patterns found."
    continue
  }

  $totalHits += $hits.Count

  # We'll operate line-by-line for safer patching (avoid rewriting unrelated blocks)
  $lines = $content -split "\r\n|\n"
  $lineEnding = "`r`n"
  if ($content -notmatch "`r`n") {
    $lineEnding = "`n"
  }

  $changedThisFile = $false

  for ($i = 0; $i -lt $hits.Count; $i++) {
    $hit = $hits[$i]
    $li = Get-LineInfo -Content $content -Index $hit.Index

    $ln = [int]$li.LineNumber
    $lineText = [string]$li.LineText

    $activatePath = [string]$hit.ActivatePath

    Write-Host ("  Hit #{0} at line {1}:" -f ($i + 1), $ln)
    Write-Host ("    Match : {0}" -f ($hit.MatchText -replace "\s+", " ").Trim())
    if ($activatePath) {
      Write-Host ("    Path  : {0}" -f $activatePath)
    }

    if (Is-GuardedAlready -LineText $lineText) {
      Write-Host "    Status: Already guarded (skipping)"
      continue
    }

    if ($RestrictToRepo -and -not (Restrict-PathToRepo -ActivatePath $activatePath -RepoRoot $repoRoot)) {
      Write-Host "    Status: Not under repo root (RestrictToRepo enabled; skipping)"
      continue
    }

    $totalFixable++

    if (-not $Apply) {
      Write-Host "    Fix   : (preview) would guard this activation"
      continue
    }

    # Build guarded replacement and patch the specific line.
    # For common patterns:
    # - powershell -Command '. "<path>"'
    # - . "<path>"
    #
    # We'll replace the whole line if it is only the dot-source,
    # otherwise replace the dot-source segment inside the line.
    $guard = Build-GuardedReplacement -ActivatePath $activatePath

    $newLine = $lineText

    # Replace dot-sourcing with guarded expression
    # Handles . "<path>" and . '<path>'
    if ($activatePath) {
      $escapedPath = [regex]::Escape($activatePath)
      $newLine = [regex]::Replace(
        $newLine,
        "(?i)\.\s*['""]$escapedPath['""]",
        $guard
      )
    } else {
      # As fallback, just avoid raw dot-source by inserting a generic guard for ".venv"
      $newLine = $guard
    }

    if ($newLine -eq $lineText) {
      # If we didn't change anything, don't mark as changed
      Write-Host "    Status: Could not rewrite safely (no changes applied)"
      continue
    }

    Write-Host "    Status: Will apply guard"
    $lines[$ln - 1] = $newLine
    $changedThisFile = $true
  }

  if ($Apply -and $changedThisFile) {
    if ($WhatIfOnly) {
      Write-Host "  WhatIf: would write changes (skipped)"
      continue
    }

    $bak = Backup-File -Path $file
    Write-Host "  Backup  : $bak"
    $newContent = ($lines -join $lineEnding)
    Write-AllTextUtf8NoBom -Path $file -Content $newContent
    Write-Host "  Updated : $file"
    $totalChanged++
  }
}

Write-Host ""
Write-Host "Scan summary:"
Write-Host "  Total hits      : $totalHits"
Write-Host "  Fixable hits    : $totalFixable"
Write-Host "  Files updated   : $totalChanged"

if (-not $Apply -and $totalFixable -gt 0) {
  Write-Host ""
  Write-Host "Next step:"
  Write-Host "  Re-run with -Apply to write guarded fixes (backups will be created)."
  Write-Host "  Example: .\scripts\fix_auto_activate.ps1 -Apply"
}

exit 0
