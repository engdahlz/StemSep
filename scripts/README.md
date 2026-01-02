# Repo Scripts (Development / Maintenance Utilities)

This directory contains **developer/maintenance scripts**. They are **not required** to run the main
application, but are useful for model management, debugging, QA, and one-off maintenance tasks.

## Current layout (subfolders)

Scripts are organized into these subfolders:

- `scripts/registry/` — model registry generation/validation/auditing/link repair tooling
- `scripts/dev/` — debugging and investigation tooling
- `scripts/qa/` — QA utilities (A/B comparisons, reports)
- `scripts/download/` — download/bootstrap helpers (VR/ZFTurbo resources)
- `scripts/mdx/` — MDX/ONNX validation utilities
- `scripts/analysis/` — analysis helpers (bands/frequency inspection, MSST experiments)

Recommended invocation pattern (from repo root):

- Windows (PowerShell):
  - `python .\scripts\<subfolder>\<script>.py`
- macOS/Linux:
  - `python ./scripts/<subfolder>/<script>.py`

## Root-level scripts (keep an eye on these)

Some scripts intentionally remain at `scripts/` root because they are **production-critical**, orchestrators,
or are called by other parts of the system:

- `inference.py` — the per-job Python runner spawned by the Rust backend (core separation execution path)
- `generate_model_registry.py` — generates `StemSepApp/assets/models.json.bak` from the v2 registry source
- `verify_backend.py` — backend verification / smoke checks (useful for CI and environment validation)
- `backend_headless_download.py` / `backend_headless_separate.py` / `run_backend_separation.py` — backend orchestration helpers
- `run_py.ps1` / `run_py.sh` — repo helpers for running Python scripts reliably without relying on shell venv activation

If you move or rename any of these, update all call sites (Electron, Rust backend, docs, and CI).

## Important notes

- These scripts may:
  - download model files/configs,
  - modify files under `StemSepApp/assets/`,
  - write output folders/files next to the repo root,
  - require FFmpeg, Python dependencies, and sometimes GPU/CUDA.
- Assume **you should run them from the repo root** unless the script says otherwise.
- Many of them are “one-off” utilities. Treat them as power tools; review before running.

Recommended invocation pattern (from repo root):

- Windows (PowerShell):
  - `python .\scripts\<subfolder>\<script>.py`
- macOS/Linux:
  - `python ./scripts/<subfolder>/<script>.py`

### Backend environment setup (Windows)

If the Electron backend (Python bridge) fails with missing modules like `psutil`, you need a working backend venv at:
- `StemSepApp\.venv`

Use:

- `setup_backend_venv.ps1`
  - Purpose: Create/use `StemSepApp\.venv` and install `StemSepApp\requirements.txt` robustly.
  - Handles common Windows issues:
    - pip cache permission/locking problems
    - partial installs
  - Writes an install log to: `StemSepApp\install.log`
  - Examples:
    - Default (recommended):
      - `.\scripts\setup_backend_venv.ps1`
    - Force rebuild venv:
      - `.\scripts\setup_backend_venv.ps1 -RecreateVenv`
    - Use a specific Python executable:
      - `.\scripts\setup_backend_venv.ps1 -PythonExe "C:\Python312\python.exe"`

### Fixing broken auto-activation of `.venv` (Windows)

If you see PowerShell errors like:

- `. "...\ .venv\Scripts\activate.ps1" is not recognized ...`

…you likely have a hardcoded auto-activate command somewhere in your shell/editor config.
Use:

- `fix_auto_activate.ps1`
  - Purpose: Scan common user config files for hardcoded `activate.ps1` dot-sourcing and optionally rewrite it into a safe `Test-Path` guard.
  - Safe by default: report-only
  - Apply mode: writes changes **with timestamped backups**
  - Examples:
    - Report only:
      - `.\scripts\fix_auto_activate.ps1`
    - Apply fixes:
      - `.\scripts\fix_auto_activate.ps1 -Apply`
    - Include VS Code user settings scan:
      - `.\scripts\fix_auto_activate.ps1 -ScanVSCode -Apply`

## Script index

### `scripts/registry/` — Model registry / config maintenance

- `registry/audit_model_links.py`
  - Purpose: Audit model download links in registries (public vs gated vs broken).
  - Typical use: Verify registry URLs are valid and downloadable.

- `registry/repair_model_registry_links.py`
  - Purpose: Repair missing/broken download links in the aggregate model registry.
  - Typical use: Apply curated link fixes after an audit.

- `registry/suggest_hf_link_fixes.py`
  - Purpose: Suggest fixes for failing Hugging Face URLs based on link audit output.
  - Typical use: Generate candidate filename/link suggestions for HF-hosted artifacts.

- `registry/validate_models.py`
  - Purpose: Bulk validation across the model set (sanity-check catalog consistency).
  - Typical use: Catch missing fields, mismatched stems, inconsistent naming.

- `registry/validate_model_registry.py`
  - Purpose: Validate model registry consistency and expected local filenames.
  - Typical use: Safe validator (no downloads); run after registry edits.

- `download_configs.py`
  - Purpose: Fetch/download YAML config files referenced by model entries.
  - Typical use: Ensure configs exist locally after updating registry links.

- `fix_all_configs.py`
  - Purpose: Attempt to repair/normalize config files in bulk.
  - Typical use: When many configs drift from the expected schema and need normalization.

- `fix_mel_configs.py`
  - Purpose: Focused fixes for Mel-Roformer / mel-band roformer config variants.
  - Typical use: When mel-band configs cause load errors due to schema/field differences.

- `restore_configs.py`
  - Purpose: Restore configs from backups or known-good sources (depends on script behavior).
  - Typical use: Roll back after an experimental mass edit.

### `scripts/download/` — Download helpers

- `download/download_vr_arch.py`
  - Purpose: Fetch VR-Arch model assets/weights/configs as used by the VR pipeline.
  - Typical use: Populate a missing VR model set.

- `download/download_zfturbo.py`
  - Purpose: Download or prepare ZFTurbo-related resources.
  - Typical use: Setting up ZFTurbo models or cloning/bootstrapping their artifacts.

- `download/download_zfturbo_models.py`
  - Purpose: Download specific ZFTurbo model weights/configs.
  - Typical use: Populate the local model cache for ZFTurbo experiments.

- `download/vendor_zfturbo_architectures.py`
  - Purpose: Vendor/update ZFTurbo architecture reference data for this repo.
  - Typical use: Sync architecture snapshots used for cross-checking.

### `scripts/mdx/` — MDX / ONNX checks and utilities

- `mdx/check_mdx_model.py`
  - Purpose: Validate a single MDX/ONNX model file or run a quick sanity check.
  - Typical use: Confirm a downloaded `.onnx` is usable before wiring it into the app.

- `mdx/check_all_mdx.py`
  - Purpose: Iterate and validate multiple MDX/ONNX models.
  - Typical use: Bulk verification after updating the MDX model list.

### `scripts/dev/` — Debugging / experimentation

- `dev/debug_engine.py`
  - Purpose: Debug the separation engine end-to-end (often verbose logging).
  - Typical use: Trace failures across model loading → inference → output writing.

- `dev/debug_separation_flow.py`
  - Purpose: Debug/trace the orchestration flow (progress reporting, pipeline steps, etc.).
  - Typical use: When the UI/bridge reports wrong progress or the flow breaks mid-run.

- `dev/diagnose_model.py`
  - Purpose: Diagnose a specific model setup (config/weights alignment, expected params).
  - Typical use: When a single model fails to load or produces unexpected output.

- `test_bsroformer.py`
  - Purpose: Local test harness for BS-Roformer inference/behavior.
  - Typical use: Quick correctness checks during Roformer-related refactors.

### `scripts/qa/` — QA helpers

- `qa/compare_outputs.py`
  - Purpose: Compare separation outputs (quality regression checks or A/B comparisons).
  - Typical use: Validate refactors or model changes.

- `qa/qa_audio_report.py`
  - Purpose: Generate QA-oriented audio reports (summaries, stats, or checks).
  - Typical use: Investigate regressions and produce shareable diagnostics.

### `scripts/analysis/` — Analysis helpers

- `analysis/calc_bands.py`
  - Purpose: Compute/inspect band layouts (useful for roformer band configs).
  - Typical use: When tuning `freqs_per_bands` or related band definitions.

- `analysis/extract_freqs.py`
  - Purpose: Extract/analyze frequency-related info from audio/configs.
  - Typical use: Investigate banding artifacts or config mismatches.

- `analysis/msst_mel_band_roformer.py`
  - Purpose: Experimental mel-band roformer/MSST-related script (naming suggests research tooling).
  - Typical use: Only if you’re working on mel-band/MSST experiments.

- `validate_models.py`
  - Purpose: Bulk validation across the model set (may overlap with other validators).
  - Typical use: Sanity-check model catalog consistency.

## Relationship to `registry/validate_model_registry.py`

The safe registry validator now lives under:
- `scripts/registry/validate_model_registry.py`

It validates model registry consistency and expected local filenames (no downloads).

This `scripts/` directory also contains heavier tools that may download or mutate files; review before running.

## Adding new scripts

- Put new dev utilities in `scripts/`.
- Prefer names that describe the action: `download_*`, `validate_*`, `debug_*`, `fix_*`.
- If the script is destructive, print a clear warning and support a `--dry-run` flag.
