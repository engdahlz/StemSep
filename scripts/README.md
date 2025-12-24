# Repo Scripts (Development / Maintenance Utilities)

This directory contains **developer/maintenance scripts** that were previously located in the repo root.
They are **not required** to run the main application, but are useful for model management, debugging,
and one-off maintenance tasks.

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
  - `python .\scripts\<script>.py`
- macOS/Linux:
  - `python ./scripts/<script>.py`

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

### Model registry / config maintenance

- `download_configs.py`
  - Purpose: Fetch/download YAML config files referenced by model entries.
  - Typical use: After updating `StemSepApp/assets/models/*.json` links, ensure configs exist locally.

- `fix_all_configs.py`
  - Purpose: Attempt to repair/normalize config files in bulk.
  - Typical use: When many configs drift from the expected schema and need normalization.

- `fix_mel_configs.py`
  - Purpose: Focused fixes for Mel-Roformer / mel-band roformer config variants.
  - Typical use: When mel-band configs cause load errors due to schema/field differences.

- `restore_configs.py`
  - Purpose: Restore configs from backups or known-good sources (depends on script behavior).
  - Typical use: Roll back after an experimental mass edit.

- `diagnose_model.py`
  - Purpose: Diagnose a specific model setup (config/weights alignment, expected params).
  - Typical use: When a single model fails to load or produces unexpected output.

### Download helpers

- `download_vr_arch.py`
  - Purpose: Fetch VR-Arch model assets/weights/configs as used by the VR pipeline.
  - Typical use: Populate a missing VR model set.

- `download_zfturbo.py`
  - Purpose: Download or prepare ZFTurbo-related resources.
  - Typical use: Setting up ZFTurbo models or cloning/bootstrapping their artifacts.

- `download_zfturbo_models.py`
  - Purpose: Download specific ZFTurbo model weights/configs.
  - Typical use: Populate the local model cache for ZFTurbo experiments.

### MDX / ONNX checks and utilities

- `check_mdx_model.py`
  - Purpose: Validate a single MDX/ONNX model file or run a quick sanity check.
  - Typical use: Confirm a downloaded `.onnx` is usable before wiring it into the app.

- `check_all_mdx.py`
  - Purpose: Iterate and validate multiple MDX/ONNX models.
  - Typical use: Bulk verification after updating the MDX model list.

### Debugging / experimentation

- `debug_engine.py`
  - Purpose: Debug the separation engine end-to-end (often verbose logging).
  - Typical use: Trace failures across model loading → inference → output writing.

- `debug_separation_flow.py`
  - Purpose: Debug/trace the orchestration flow (progress reporting, pipeline steps, etc.).
  - Typical use: When the UI/bridge reports wrong progress or the flow breaks mid-run.

- `compare_outputs.py`
  - Purpose: Compare separation outputs (quality regression checks or A/B comparisons).
  - Typical use: Validate refactors or model changes.

- `test_bsroformer.py`
  - Purpose: Local test harness for BS-Roformer inference/behavior.
  - Typical use: Quick correctness checks during Roformer-related refactors.

### Analysis helpers

- `calc_bands.py`
  - Purpose: Compute/inspect band layouts (useful for roformer band configs).
  - Typical use: When tuning `freqs_per_bands` or related band definitions.

- `extract_freqs.py`
  - Purpose: Extract/analyze frequency-related info from audio/configs.
  - Typical use: Investigate banding artifacts or config mismatches.

- `msst_mel_band_roformer.py`
  - Purpose: Experimental mel-band roformer/MSST-related script (naming suggests research tooling).
  - Typical use: Only if you’re working on mel-band/MSST experiments.

- `validate_models.py`
  - Purpose: Bulk validation across the model set (may overlap with other validators).
  - Typical use: Sanity-check model catalog consistency.

## Relationship to `validate_model_registry.py`

At the repo root there is `validate_model_registry.py` which validates model registry consistency
(`StemSepApp/assets/models/*.json`) and expected local filenames. It is a “safe” validator (no downloads).

This `scripts/` directory contains heavier tools that may download or mutate files.

## Adding new scripts

- Put new dev utilities in `scripts/`.
- Prefer names that describe the action: `download_*`, `validate_*`, `debug_*`, `fix_*`.
- If the script is destructive, print a clear warning and support a `--dry-run` flag.
