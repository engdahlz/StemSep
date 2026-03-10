param(
    [string]$GuideDocId = "17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c",
    [string]$SnapshotPath = "C:\Users\engdahlz\StemSep\docs\vendor\deton24_guide\latest.txt",
    [string]$DigestPath = "C:\Users\engdahlz\StemSep\StemSepApp\assets\registry\guide_digest.json",
    [string]$CandidatesPath = "C:\Users\engdahlz\StemSep\reports\guide_refresh\candidates.json"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gws -ErrorAction SilentlyContinue)) {
    throw "gws CLI is required to refresh the guide snapshot."
}

$snapshotDir = Split-Path -Parent $SnapshotPath
$candidatesDir = Split-Path -Parent $CandidatesPath
New-Item -ItemType Directory -Force -Path $snapshotDir | Out-Null
New-Item -ItemType Directory -Force -Path $candidatesDir | Out-Null

$tempDir = Join-Path $env:TEMP "stemsep-guide-refresh"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
Push-Location $tempDir
$exportMeta = gws drive files export --params "{\"fileId\":\"$GuideDocId\",\"mimeType\":\"text/plain\"}" | ConvertFrom-Json
Pop-Location

$savedFile = if ($exportMeta.saved_file) {
    Join-Path $tempDir $exportMeta.saved_file
} else {
    Join-Path $tempDir "download.txt"
}
if (-not (Test-Path $savedFile)) {
    throw "Guide export did not produce a text snapshot."
}

$previous = if (Test-Path $SnapshotPath) { Get-Content -Raw $SnapshotPath } else { "" }
$latest = Get-Content -Raw $savedFile
$changed = $previous -ne $latest

Set-Content -Path $SnapshotPath -Value $latest -Encoding utf8

$digest = if (Test-Path $DigestPath) {
    Get-Content -Raw $DigestPath | ConvertFrom-Json
} else {
    [pscustomobject]@{}
}

$report = [pscustomobject]@{
    generated_at = (Get-Date).ToString("o")
    guide_doc_id = $GuideDocId
    changed = $changed
    snapshot_path = $SnapshotPath
    digest_path = $DigestPath
    notes = @(
        "Review the updated guide snapshot and manually curate any new models or workflows before promoting them.",
        "This script only refreshes the source snapshot and candidate report; it does not modify recipes or model promotion status."
    )
    candidate_actions = @(
        "Diff latest snapshot against previous guide revision.",
        "Add or update model_overrides in guide_digest.json.",
        "Run local QA before promoting any workflow to curated simple mode."
    )
    source_priority = $digest.meta.source_priority
}

$report | ConvertTo-Json -Depth 8 | Set-Content -Path $CandidatesPath -Encoding utf8
Remove-Item $savedFile -ErrorAction SilentlyContinue
Remove-Item $tempDir -Force -Recurse -ErrorAction SilentlyContinue

Write-Output "Guide snapshot refreshed."
Write-Output "Snapshot: $SnapshotPath"
Write-Output "Candidate report: $CandidatesPath"
