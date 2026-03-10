# StemSep

StemSep is a local audio separation platform built around an Electron frontend, a Rust orchestration layer, and a Python inference/runtime stack.

## Active product structure

- `electron-poc/`: primary desktop product UI
- `stemsep-backend/`: Rust IPC/orchestration backend
- `scripts/`: Python bridge and developer tooling
- `StemSepApp/`: Python runtime, registry, vendored inference dependencies
- `docs/`: minimal local docs plus pinned vendor references

## UI policy

- Primary UI: `electron-poc/`
- Internal/legacy UI: `python -m stemsep`

The Tkinter UI is kept for debugging and compatibility only. New product work should target Electron first.

## Quick start

```powershell
git clone https://github.com/engdahlz/StemSep.git
cd StemSep
npm run install:ui
.\scripts\setup_backend_venv.ps1
npm run electron:dev
```

## Developer checks

```powershell
npm run ui:typecheck
npm run ui:test
npm run python:test
cd stemsep-backend; cargo test
```

## Local documentation

- Python/runtime notes: [StemSepApp/README.md](./StemSepApp/README.md)
- Script/tooling overview: [scripts/README.md](./scripts/README.md)
- Troubleshooting: [StemSepApp/docs/TROUBLESHOOTING.md](./StemSepApp/docs/TROUBLESHOOTING.md)
- Vendor references: [docs/vendor/README.md](./docs/vendor/README.md)

## Notes

- Generated outputs, reports, screenshots and sample media do not belong in git.
- `StemSepApp/dev_vendor/audio_separator/` is an intentional vendored runtime dependency.
- Guide- and MSST-derived reference material is kept locally in curated form under `docs/vendor/`.
