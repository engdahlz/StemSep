# StemSep

StemSep is a local-first audio separation workspace built around an Electron desktop app, a Rust backend bridge, and a Python inference/runtime stack.

The repo is organized so the desktop product can move quickly without losing access to the heavier Python and model-management tooling that still powers separation.

## Current product focus

- `electron-poc/` is the primary product surface.
- `stemsep-backend/` handles desktop IPC, orchestration, and native capture support.
- `StemSepApp/` contains the Python runtime, registry, and separation engine.
- `scripts/` contains developer tooling, QA helpers, registry maintenance, and backend utilities.

Current desktop import paths include:

- local audio file upload
- YouTube-based import
- Qobuz browser-backed library search and hidden capture setup

## Repository map

| Path | Purpose |
| --- | --- |
| `electron-poc/` | Electron + React desktop UI |
| `stemsep-backend/` | Rust bridge, protocol handling, native helpers |
| `StemSepApp/` | Python runtime, registry assets, separation logic |
| `scripts/` | Developer scripts, QA, registry sync, setup helpers |
| `docs/` | Backend contract notes, guide mappings, vendor references |
| `.github/` | CI and GitHub workflow configuration |

## Quick start

```powershell
git clone https://github.com/engdahlz/StemSep.git
cd StemSep
npm run install:ui
.\scripts\setup_backend_venv.ps1
npm run electron:dev
```

## Common commands

```powershell
# Electron UI
npm run electron:dev
npm run ui:typecheck
npm run ui:test

# Python/runtime
npm run python:test

# Rust backend
cd stemsep-backend
cargo test
```

## Development workflow

Use Electron as the default surface for product work. The Python app remains important for runtime compatibility, registry maintenance, and backend validation, but new user-facing work should land in `electron-poc/` first.

If you are touching multiple layers, the typical path is:

1. Update UI and renderer state in `electron-poc/`.
2. Wire IPC and desktop behavior in `electron-poc/electron/`.
3. Extend native/backend behavior in `stemsep-backend/` or `scripts/inference.py`.
4. Keep registry/runtime definitions in `StemSepApp/assets/` and `StemSepApp/src/` aligned.

## Documentation

- Docs index: [docs/README.md](./docs/README.md)
- Python/runtime notes: [StemSepApp/README.md](./StemSepApp/README.md)
- Script and tooling overview: [scripts/README.md](./scripts/README.md)
- Registry notes: [StemSepApp/assets/registry/README.md](./StemSepApp/assets/registry/README.md)
- Vendor references: [docs/vendor/README.md](./docs/vendor/README.md)

## Repository hygiene

- Generated outputs, smoke-test folders, screenshots, and ad-hoc research artifacts should stay out of git.
- Large vendor references belong under `docs/vendor/`, not at the repo root.
- Local assistant or editor state should remain ignored.

## Notes

- The Tkinter/Python UI is kept for compatibility and debugging, not as the primary product path.
- `StemSepApp/dev_vendor/audio_separator/` remains an intentional vendored dependency.
