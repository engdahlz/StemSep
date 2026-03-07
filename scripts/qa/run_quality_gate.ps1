param(
    [switch]$SkipGuideSync,
    [switch]$SkipBackend,
    [switch]$SkipUi
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

if (-not $SkipGuideSync) {
    Invoke-Step "Guide sync report" {
        & powershell -ExecutionPolicy Bypass -File .\scripts\run_py.ps1 .\scripts\registry\sync_guide_knowledge.py --allow-fetch-failure
    }

    Invoke-Step "Catalog sync report" {
        & powershell -ExecutionPolicy Bypass -File .\scripts\run_py.ps1 .\scripts\registry\sync_model_catalog_metrics.py
    }
}

Invoke-Step "Registry validation" {
    & powershell -ExecutionPolicy Bypass -File .\scripts\run_py.ps1 .\scripts\registry\validate_model_registry.py --check-verified-urls
}

if (-not $SkipUi) {
    Invoke-Step "UI typecheck" {
        & npm run ui:typecheck
    }

    Invoke-Step "UI test suite" {
        & npm run ui:test
    }
}

if (-not $SkipBackend) {
    Invoke-Step "Backend test suite" {
        Push-Location .\stemsep-backend
        try {
            & cargo test
        }
        finally {
            Pop-Location
        }
    }
}

Write-Host ""
Write-Host "Quality gate completed." -ForegroundColor Green
