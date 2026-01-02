# StemSepApp Tests (Windows + Python 3.12)

This directory contains the Python test suite for **StemSepApp**.

The tests are designed to run on **Windows** with **Python 3.12**. Some tests require **CUDA-enabled PyTorch** and **onnxruntime** to cover GPU and audio-separator paths.

> Important: Running tests with Python 3.14 may break due to third-party dependency compatibility. Use Python 3.12.

---

## Preferred workflow (recommended)

We keep a pinned dev requirements file to make test runs reproducible across machines:

- `StemSepApp/requirements-dev.txt`

### 1) Install dev/test deps (pinned)
From repo root:

- `py -3.12 -m pip install --upgrade pip`
- `py -3.12 -m pip install -r StemSepApp/requirements-dev.txt`

### 2) Install CUDA torch + torchaudio (PyTorch CUDA index)
PyTorch CUDA wheels are not installed from PyPI. Install them separately:

- `py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu121`
- `py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 3) Run tests
- `py -3.12 -m pytest -q StemSepApp/tests`

---

## Quick Start (manual install)

If you don’t want to use the pinned file, install the required packages manually (less reproducible):

1) Verify Python 3.12 is available:
- `py -3.12 -V`

2) Minimal set to import most modules:
- `py -3.12 -m pip install pytest pyyaml soundfile numpy`

3) Additional deps used by some tests/vendor paths:
- `py -3.12 -m pip install onnxruntime requests tqdm librosa spafe`

4) CUDA torch stack:
- `py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu121`
- `py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu121`

5) Run:
- `py -3.12 -m pytest -q StemSepApp/tests`

---

## Notes

- CUDA torch wheels bundle CUDA runtime components needed by PyTorch; you do **not** need a separate CUDA Toolkit install just to import/use torch.
- You still need a compatible NVIDIA driver installed for CUDA execution.
- If you prefer CPU-only torch for lightweight CI:
  - `py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu`

---

## Project Import Path

`StemSepApp/tests/conftest.py` adds `StemSepApp/src` to `sys.path`, so tests can import application modules directly without requiring an editable install.

If you want editable installs, note that packaging configuration may need to be aligned with the current repo layout.

---

## Troubleshooting

### “No module named pytest”
You installed pytest into a different Python than the one you’re using to run tests. Re-run install using the same interpreter:

- `py -3.12 -m pip install pytest`

### CUDA torch installs but GPU isn’t usable
- Verify NVIDIA driver is installed and up to date.
- Check `torch.cuda.is_available()` in a Python shell using `py -3.12`.

### Import errors in vendored audio-separator code
Install:
- `requests`, `tqdm`, `onnxruntime`

### BandIt test failing due to missing `spafe`
Install:
- `py -3.12 -m pip install spafe`
