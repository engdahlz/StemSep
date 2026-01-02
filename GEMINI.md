# StemSep V3 Context

## Project Overview

**StemSep V3** is a desktop application for AI-powered audio stem separation. It allows users to extract vocals, drums, bass, and other instruments from audio files using state-of-the-art machine learning models (BS-Roformer, Mel-Roformer, MDX23C, Demucs).

**Architecture:**
The application follows a hybrid architecture:
*   **Frontend (UI):** Electron + React + TypeScript (located in `electron-poc/`).
*   **Orchestration (Backend):** Rust service (located in `stemsep-backend/`) handling IPC and process management.
*   **Inference Engine:** Python 3.10+ (located in `StemSepApp/`), driven by `scripts/inference.py`.

## Key Directories

*   `electron-poc/`: Main Electron/React application source code.
*   `stemsep-backend/`: Rust backend source code.
*   `StemSepApp/`: Core Python library containing model logic and separation engines.
*   `scripts/`: Python utility scripts for inference, model downloading, and validation.
*   `models/`: Directory where downloaded model weights are stored.
*   `docs/`: Project documentation.

## Build and Run Instructions

### Prerequisites
*   Node.js (v20 LTS recommended)
*   Python 3.10+
*   Rust (latest stable)

### Development Setup

1.  **Python Environment:**
    ```powershell
    cd StemSepApp
    python -m venv .venv
    .\.venv\Scripts\Activate
    pip install -r requirements.txt
    ```

2.  **UI Dependencies:**
    ```powershell
    # From project root
    npm run install:ui
    ```

### Running the Application

*   **Development Mode (Hot Reload):**
    ```powershell
    # From project root
    npm run electron:dev
    ```
    This command concurrently runs the Vite dev server and the Electron app.

*   **Production Build:**
    ```powershell
    # From project root
    npm run electron:build
    ```
    This builds the Rust backend (`cargo build --release`), the React frontend (`vite build`), and packages the Electron application using `electron-builder`.

### Utility Scripts

The `scripts/` directory contains various helpers. Use the provided shell wrappers for reliability:

*   **Windows:** `.\scripts\run_py.ps1 <script_path> [args]`
*   **macOS/Linux:** `./scripts/run_py.sh <script_path> [args]`

Example:
```powershell
.\scripts\run_py.ps1 .\scripts\registry\validate_model_registry.py
```

## Development Conventions

*   **UI/Frontend:**
    *   Uses TailwindCSS for styling.
    *   State management via Zustand.
    *   Typecheck with `npm run ui:typecheck`.
    *   Test with `npm run ui:test`.
*   **Code Style:**
    *   Follow existing formatting (Rustfmt, Prettier, Black/Ruff for Python).
    *   Keep changes small and focused.
*   **Models:**
    *   New models in the registry must have a stable `model_id` and valid download URL.
    *   Check `CONTRIBUTING.md` for detailed guidelines.
