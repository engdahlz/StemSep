# StemSepApp

`StemSepApp/` contains the Python runtime used by StemSep.

## What lives here

- `src/stemsep/audio/`: inference and orchestration helpers
- `src/stemsep/models/`: registry/model loading
- `src/stemsep/core/`: config, diagnostics and app services
- `src/stemsep/ui_legacy/`: internal Tkinter UI kept for debugging/compatibility
- `assets/`: model registry, recipes and runtime metadata
- `dev_vendor/audio_separator/`: vendored runtime dependency used by the separator stack

## Entry points

- Preferred package entry: `python -m stemsep`
- Backward-compatible wrapper: `python src/main.py`

The `python -m stemsep` path starts the internal legacy UI. Product-facing work should target the Electron app in the repo root.

## Environment

Windows-first setup:

```powershell
.\scripts\setup_backend_venv.ps1
```

Manual setup:

```powershell
cd StemSepApp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Useful docs

- Troubleshooting: [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
- Registry internals: [assets/registry/README.md](./assets/registry/README.md)

## Notes

- Do not remove `dev_vendor/audio_separator/` unless the runtime is rewritten to stop importing it.
- Treat `ui_legacy/` as internal support code, not the primary product surface.
