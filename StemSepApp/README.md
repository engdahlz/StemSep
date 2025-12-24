# StemSep - Advanced Audio Stem Separation

A Windows application for separating audio files into individual stems (vocals, instruments, drums, bass, etc.) using state-of-the-art AI models running locally.

## Features

### 🎵 Stem Separation
- **Vocal & Instrumental**: Extract clean vocal and instrumental tracks
- **Karaoke Mode**: Separate lead vocals, backing vocals, and instrumental
- **Multi-Stem**: Extract drums, bass, and other individual instruments
- **Specialized Models**: Dedicated models for drums, bass, strings, and more

### 🤖 AI Models
- **BS-Roformer**: Best overall quality with superior SDR scores
- **Mel-Roformer**: Excellent instrumental separation with high fullness
- **MDX23C**: Fast and efficient for quick processing
- **SCNet**: Optimized for busy songs with many instruments
- All models run locally - no internet required after download

### 🎮 GPU Acceleration
- Full CUDA support for NVIDIA GPUs
- Automatic GPU detection and configuration
- Optimized for 6GB+ VRAM (BS-Roformer) and 4GB+ VRAM (MDX23C)
- CPU fallback for systems without GPU

### 💡 User Experience
- **Modern UI**: Clean, professional interface with dark/light theme
- **Drag & Drop**: Easy file upload with visual feedback
- **Pre-configured Presets**: Optimized settings for different use cases
- **Real-time Progress**: Monitor separation progress with ETA
- **Built-in Player**: Preview separated stems before downloading
- **Model Management**: Download, organize, and manage AI models

### 🔧 Advanced Features
- **Custom Settings**: Fine-tune chunk size, overlap, and other parameters
- **Batch Processing**: Queue multiple files for sequential separation
- **Ensemble Support**: Combine multiple models for better results
- **Detailed Logging**: Comprehensive logs for troubleshooting
- **Settings Persistence**: Your preferences are saved automatically

## System Requirements

### Minimum
- Windows 10 or 11
- 4 GB RAM
- 4 GB free disk space
- CPU: Dual-core processor

### Recommended
- Windows 11
- 8+ GB RAM
- NVIDIA GPU with 6-8GB VRAM (RTX 3060 or better)
- 10 GB free disk space
- SSD storage for faster I/O

## Quick Start

### Option 1: Download Portable Version (Easiest)
1. Download `StemSep_Portable.zip` from releases
2. Extract to any folder
3. Run `StemSep.exe`
4. Go to Models tab and download your first model
5. Return to Home to start separating!

### Option 2: Build from Source
1. Install Python 3.8+ (Linux: usually `python3` via your package manager)
2. Clone or download this repository
3. Open a terminal in the `StemSepApp` folder
4. Install dependencies and run the app (Linux and Windows steps below)

#### Linux (Recommended)
Use the helper script that installs system libraries, creates a venv, and installs Python deps.

```
chmod +x install_linux.sh
# CPU-only
./install_linux.sh

# NVIDIA GPU with CUDA 12.1 wheels
./install_linux.sh --cuda cu121

# NVIDIA GPU with CUDA 11.8 wheels
./install_linux.sh --cuda cu118

# Activate and run
source .venv/bin/activate
python src/main.py
```

System packages the script installs (or that you should ensure are installed):
- Tk/Tkinter: `python3-tk` or `tk`
- ffmpeg, libsndfile
- SDL2 + mixer/image/ttf (for pygame)
- ALSA (usually present by default)

If you prefer manual steps on Linux:
```
python3 -m venv .venv
source .venv/bin/activate
# Choose ONE of the following PyTorch installs
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio              # CPU
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio           # CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchaudio           # CUDA 11.8

pip install -r requirements.txt
python src/main.py
```

### Diagnostics

After installation you can quickly verify core components:

```
source .venv/bin/activate
python - <<'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available():
        import torch
        print('CUDA version:', getattr(torch.version, 'cuda', 'unknown'))
except Exception as e:
    print('Torch import failed:', e)

try:
    import customtkinter, tkinter
    print('Tkinter/CustomTkinter: OK')
except Exception as e:
    print('Tkinter failed:', e)

try:
    import librosa, soundfile
    print('Audio libs (librosa/soundfile): OK')
except Exception as e:
    print('Audio import failed:', e)
PY
```

#### Windows
```
pip install -r requirements.txt
python src/main.py
```

### Option 3: Create Executable
```
python build_exe.py
```
This creates a standalone executable in `dist/StemSep_Portable/`

## Usage Guide

### 1. First Time Setup
- Launch StemSep
- Go to **Models** tab
- Download at least one model (BS-Roformer recommended)
- Return to **Home** tab

### 2. Separating Audio
1. **Choose a Preset**:
   - Vocal & Instrumental: Most common use case
   - Karaoke: For lead/back vocal separation
   - Drums & Bass: Extract rhythm section
   - Instruments Only: Full instrumental mix

2. **Load Audio File**:
   - Drag & drop file to the drop zone, or
   - Click the drop zone to browse

3. **Configure Settings** (optional):
   - Click "Advanced Settings" to adjust model and output directory
   - Or stick with the default preset settings

4. **Click SEPARATE**:
   - Watch real-time progress in "Active Separations" tab
   - Estimated time is shown based on file length and GPU
   - Can cancel at any time

5. **Review & Download**:
   - Click "Play" to preview each stem
   - Navigate to output directory to find your files
   - All stems are saved as high-quality WAV files

### 3. Managing Models
- **Download**: Go to Models tab, click "Download" on any model
- **Info**: Each model shows SDR, fullness, bleedless scores
- **Remove**: Click "Remove" to free up disk space
- **Filter**: Use dropdown to filter by architecture

## Model Recommendations

### Best Quality (Requires 6GB+ VRAM)
- **BS-Roformer 2025.07**: Overall best quality, cleanest results
- **Mel-Roformer Kim**: Best bleedless score (46.56), preserves clarity
- **SCNet XL**: Good for complex songs with many instruments

### Balanced (Requires 4-6GB VRAM)
- **MDX23C-8KFFT-InstVoc HQ**: Fast processing, decent quality
- **Mel-Roformer Gabox**: Good instrumental separation

### Karaoke (Requires 6GB VRAM)
- **BS-Roformer Karaoke (becruily & frazer)**: Best for lead/back vocal separation
- **BS Roformer 3-Stem (MVSep)**: Returns 3 separate stems

## Tips & Best Practices

### For Best Results
1. **Use high-quality source audio**: WAV or FLAC preferred over MP3
2. **Ensure centered vocals**: Most karaoke models work best with centered lead vocals
3. **Close other GPU applications**: Free up VRAM for faster processing
4. **Use appropriate models**:
   - BS-Roformer for highest quality
   - MDX23C for quick previews
   - SCNet for busy/production-heavy songs

### GPU Optimization
- **6GB VRAM**: Use Mel-Roformer or MDX23C models
- **8GB+ VRAM**: Can use all models including BS-Roformer
- **Lower VRAM**: Reduce chunk size in Settings to 80000-120000
- **No GPU**: CPU mode works but is significantly slower (10-50x)

### Troubleshooting
- **"ModuleNotFoundError"**: Run `pip install -r requirements.txt`
- **CUDA out of memory**: Lower chunk size in Settings
- **Slow processing**: Check GPU utilization, close other apps
- **Low separation quality**: Try a different model from the same architecture

## Architecture

```
StemSep/
├── src/
│   ├── main.py              # Application entry point
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── logger.py        # Logging setup
│   │   ├── gpu_detector.py  # GPU detection and reporting
│   │   └── separation_manager.py  # Job queue management
│   ├── models/
│   │   └── model_manager.py # Model download and management
│   ├── audio/
│   │   ├── player.py        # Audio playback
│   │   └── separation_engine.py  # Core separation logic
│   └── ui/
│       ├── main_window.py   # Main window
│       ├── landing_page.py  # Home page with presets
│       ├── active_separations_window.py  # Progress tracking
│       ├── models_page.py   # Model management
│       └── settings_dialog.py  # Settings UI
├── models/                  # Downloaded model files
├── temp/                    # Temporary processing files
├── output/                  # Separated audio output
└── logs/                    # Application logs
```

## Model Sources

All models are sourced from:
- [Hugging Face](https://huggingface.co/) - Primary model repository
- [MVSep.com](https://mvsep.com) - Model quality rankings and testing
- [UVR Community](https://github.com/Anjok07/ultimatevocalremover) - Original models

## Supported Formats

### Input
- MP3 (all bitrates)
- WAV (all formats)
- FLAC
- M4A/AAC
- OGG
- And more via librosa

### Output
- WAV (uncompressed, 16-bit/24-bit, original sample rate)

## Performance

### Typical Processing Times (per minute of audio)
- **RTX 4090 + BS-Roformer**: 15-30 seconds
- **RTX 3080 + MDX23C**: 10-20 seconds
- **RTX 3060 + MDX23C**: 20-40 seconds
- **CPU (Ryzen 5)**: 5-10 minutes

## Credits

Based on research and models from:
- BS-Roformer: `lj1995` and community contributors
- Mel-Roformer: Gabox, Kim, and MVSep team
- MDX23C: `domic不为` and Facebook Research
- SCNet: `ZFTurbo` (ZFTurbo)
- Demucs: `Facebook Research`

## License

This project is for educational and personal use only. Model licenses vary - check individual model licenses on Hugging Face.

## Support

For issues, questions, or feature requests:
- Check the README and documentation
- Review the logs in the `logs/` directory
- Ensure all dependencies are installed
- Verify GPU drivers are up to date

## Acknowledgments

Special thanks to:
- The Ultimate Vocal Remover (UVR) community
- MVSep team for model quality testing
- Researchers behind BS-Roformer, Mel-Roformer, MDX23C, and SCNet
- Contributors to the open-source AI music separation ecosystem
