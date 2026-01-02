# StemSep V3

> 🎵 **AI-Powered Audio Stem Separation** - Extract vocals, instruments, drums, and bass from any song using state-of-the-art machine learning models.

## Artifacts & repo hygiene (important)

This repo contains a mix of production code and development artifacts. To avoid accidental breakage and bloated history:

- **Do not commit generated outputs** (build artifacts, logs, temp separation outputs).
- **Keep long-term reference materials** (guides, PDFs, spreadsheets, sample media) under `docs/vendor/`.
- **Keep ad-hoc debug artifacts** (stack traces, one-off investigation exports) under `docs/notes/artifacts/`.
- **Never commit secrets** (tokens, private keys, credentials). Use `*.example.json` patterns and environment variables instead.

The `.gitignore` is intentionally strict for common generated files and local caches; when in doubt, treat an output as non-source and keep it out of git unless it is explicitly intended as a tracked reference under `docs/vendor/`.

## Repo housekeeping

- Developer utilities live in `scripts/` (not required to run the app).
- Draft documentation lives in `docs/drafts/`.
- Developer notes/artifacts live in `docs/notes/` (not required to run the app).
  - One-off crash dumps and investigation outputs go in `docs/notes/artifacts/`.
- Vendored upstream reference snapshots and large long-term references live in `docs/vendor/` (reference-only; not required to run the app).

## UI direction (recommended)

- **Primary UI (supported):** `electron-poc/` (Electron + React)
- **Legacy/dev tooling:** `StemSepApp/src/main.py` (Tkinter). This is kept mainly for debugging and internal tooling; feature work should target the Electron UI first.

### Running Python utilities reliably (Windows)

If your PowerShell profile tries to auto-activate a missing venv, use the repo helper:

> Note: If you generate logs or outputs while debugging, keep them out of git (or place long-term references under `docs/vendor/` and short-lived artifacts under `docs/notes/artifacts/`).

```powershell
.\scripts\run_py.ps1 .\scripts\registry\validate_model_registry.py
.\scripts\run_py.ps1 .\scripts\download_configs.py --help
```

### Running Python utilities reliably (macOS/Linux)

Use the shell helper to run scripts without relying on virtualenv activation:

```sh
./scripts/run_py.sh ./scripts/registry/validate_model_registry.py
./scripts/run_py.sh ./scripts/download_configs.py --help
```

[![Electron](https://img.shields.io/badge/Electron-React-47848F?logo=electron&logoColor=white)](https://www.electronjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Features

- 🎤 **Vocal & Instrumental Separation** - Clean extraction with industry-leading quality
- 🥁 **Multi-Stem Extraction** - Drums, bass, vocals, other instruments
- 🤖 **Multiple AI Architectures** - BS-Roformer, Mel-Roformer, MDX23C, MDX-Net, Demucs
- ⚡ **GPU Accelerated** - CUDA support for fast processing
- 🖥️ **Modern Desktop UI** - Electron + React with dark/light themes
- 📦 **Model Manager** - Download and organize 30+ pre-trained models

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Electron + React + TypeScript + TailwindCSS |
| Backend | Rust (IPC, Orchestration) + Python 3.10+ (Inference) |
| Models | BS-Roformer, Mel-Roformer, MDX23C, MDX-Net, Demucs |
| Audio | librosa, soundfile, FFmpeg |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/engdahlz/StemSep.git
cd StemSep

# Install Python dependencies
cd StemSepApp
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Install Electron dependencies
cd ../
npm run install:ui

# Run the app
npm run electron:dev
```

## Project Structure

```
StemSep-V3/
├── electron-poc/         # Electron + React frontend (primary UI)
│   ├── src/              # React components & stores
│   └── electron/         # Main process & preload
│
├── stemsep-backend/      # Rust backend service (IPC & Orchestration)
│
├── scripts/              # Python inference & utility scripts
│   └── inference.py      # Core inference script called by Rust backend
│
├── StemSepApp/           # Core Python logic (imported by inference.py)
│   ├── assets/           # Model registry, presets, app assets
│   ├── src/
│   │   ├── audio/        # Separation engine, roformer models
│   │   ├── core/         # Configuration, GPU detection
│   │   ├── models/       # Model factory, managers
│   │   └── ui/           # Tkinter UI (legacy/dev tooling)
│   └── README.md         # Detailed documentation
│
├── docs/                 # Documentation
│   └── drafts/           # Work-in-progress docs (not required to run)
│
└── models/               # Downloaded model weights (user machine)
```

## Documentation

📚 See [StemSepApp/README.md](./StemSepApp/README.md) for:
- Detailed installation instructions
- Model recommendations
- Usage guide
- Troubleshooting tips

🛠️ Developer utilities:
- See [`scripts/README.md`](./scripts/README.md) for maintenance/debug scripts.

📝 Drafts/WIP docs:
- See [`docs/drafts/`](./docs/drafts/) for in-progress documentation.

🗒️ Notes / artifacts (non-essential):
- See [`docs/notes/`](./docs/notes/) for extracted text, recovered configs, and other dev artifacts.

📦 Vendored upstream references (non-essential):
- See [`docs/vendor/`](./docs/vendor/) for pinned upstream snapshots used for reference/cross-checking.

## Models

Supports 30+ pre-trained models from the community:

| Model | Architecture | Best For |
|-------|-------------|----------|
| BS-Roformer 2025.07 | BS-Roformer | Highest quality vocals |
| Mel-Roformer Kim | Mel-Roformer | Clean instrumentals |
| MDX23C-8KFFT | MDX23C | Fast processing |
| htdemucs | Demucs | Multi-stem (drums, bass, other) |

## License

The project license text is not finalized yet. See [LICENSE-CHOOSE.md](LICENSE-CHOOSE.md).

Model licenses/terms vary by source; downloading model weights does not automatically grant redistribution rights.

## Contributing

- See [CONTRIBUTING.md](CONTRIBUTING.md).
- Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Security reports: [SECURITY.md](SECURITY.md).

## Credits

- [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [Facebook Demucs](https://github.com/facebookresearch/demucs)
