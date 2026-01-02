# Artifacts (developer notes)

This folder contains **development artifacts** and one-off files that are useful for debugging, reproduction, and historical context, but are **not required** to run StemSep.

## What belongs here

- Crash dumps / stack traces (e.g. `error_traceback_*.txt`)
- Temporary data exports used during investigation (e.g. `spreadsheet_data.txt`)
- Build/debug output captured from the field
- Minimal repro notes that don’t fit in formal docs

## What does NOT belong here

- Secrets, tokens, credentials, or private keys (never store those in the repo)
- Large binaries or datasets (put under `docs/vendor/` if they must be tracked)
- Generated build outputs (should be ignored by version control)
- Anything required by the app at runtime

## Naming conventions

Prefer descriptive filenames:
- `error_traceback_<origin>.txt`
- `<date>_<topic>_<short-description>.txt`

## Notes

Artifacts are allowed to be messy, but should remain **safe to share** (no sensitive data).
If an artifact becomes important long-term, promote it into a proper document under `docs/`.