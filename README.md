# StemSep V3

> 🎵 **AI-Powered Audio Stem Separation** - Extract vocals, instruments, drums, and bass from any song using state-of-the-art machine learning models.

## Repo housekeeping

- Developer utilities live in `scripts/` (not required to run the app).
- Draft documentation lives in `docs/drafts/`.
- Developer notes/artifacts live in `docs/notes/` (not required to run the app).
- Vendored upstream reference snapshots live in `docs/vendor/` (reference-only; not required to run the app).

## UI direction (recommended)

- **Primary UI (supported):** `electron-poc/` (Electron + React)
- **Legacy/dev tooling:** `StemSepApp/src/main.py` (Tkinter). This is kept mainly for debugging and internal tooling; feature work should target the Electron UI first.

### Running Python utilities reliably (Windows)

If your PowerShell profile tries to auto-activate a missing venv, use the repo helper:

```powershell
.\scripts\run_py.ps1 .\validate_model_registry.py
.\scripts\run_py.ps1 .\scripts\download_configs.py --help
```

### Running Python utilities reliably (macOS/Linux)

Use the shell helper to run scripts without relying on virtualenv activation:

```sh
./scripts/run_py.sh ./validate_model_registry.py
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
| Backend | Python 3.10+ + PyTorch + CUDA |
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
cd ../electron-poc
npm install

# Run the app
npm run electron:dev
```

## Project Structure

```
StemSep-V3/
├── electron-poc/         # Electron + React frontend (primary UI)
│   ├── src/              # React components & stores
│   ├── electron/         # Main process & preload
│   └── python-bridge.py  # IPC bridge to Python
│
├── StemSepApp/           # Python backend
│   ├── assets/           # Model registry, presets, app assets
│   ├── src/
│   │   ├── audio/        # Separation engine, roformer models
│   │   ├── core/         # Configuration, GPU detection
│   │   ├── models/       # Model factory, managers
│   │   └── ui/           # Tkinter UI (legacy/dev tooling)
│   └── README.md         # Detailed documentation
│
├── scripts/              # Developer/maintenance utilities (optional)
+├── docs/                 # Documentation
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

This project is for educational and personal use. Model licenses vary - check individual model licenses on Hugging Face.

## Credits

- [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [Facebook Demucs](https://github.com/facebookresearch/demucs)
