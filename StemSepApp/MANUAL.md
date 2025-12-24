# StemSep - Complete Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [User Interface](#user-interface)
5. [Separation Process](#separation-process)
6. [Model Management](#model-management)
7. [Settings](#settings)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)
11. [FAQ](#faq)

## Introduction

StemSep is a Windows application for separating audio files into individual stems (vocals, instruments, drums, etc.) using state-of-the-art AI models. All processing happens locally on your computer - no internet required after initial model download.

### Key Features
- **Multiple AI Architectures**: BS-Roformer, Mel-Roformer, MDX23C, SCNet
- **GPU Acceleration**: CUDA support for NVIDIA GPUs
- **Easy-to-Use Interface**: Modern UI with drag-and-drop
- **Pre-configured Presets**: Optimized for different use cases
- **Real-time Progress**: Monitor processing with ETA
- **Built-in Playback**: Preview separated stems
- **Model Management**: Download, organize, and manage models

## Installation

### Option 1: Portable Version
1. Download `StemSep_Portable.zip` from releases
2. Extract to desired location
3. Run `StemSep.exe`
4. No installation required!

### Option 2: From Source

#### Prerequisites
- Windows 10 or 11
- Python 3.8+
- 4GB RAM (8GB+ recommended)
- GPU optional but recommended

#### Steps
1. Install Python from [python.org](https://python.org)
   - ✅ Check "Add Python to PATH"
2. Download StemSep source code
3. Run `install_dependencies.bat`
   - Installs all required packages
   - May take 10-20 minutes
4. Run `run.bat` to start

### Building Executable
```batch
python build_exe.py
```
Creates standalone executable in `dist/StemSep_Portable/`

## Getting Started

### First Launch
1. **Start StemSep** - Window opens with Home tab
2. **Check System Info** - Verify GPU detection
3. **Download Model** - Go to Models tab, download BS-Roformer 2025.07
4. **Return to Home** - Ready to separate!

### System Requirements Check
Look at "System Information" on Home tab:
- ✅ **GPU detected with 6GB+ VRAM**: Can use all models
- ⚠️ **GPU detected with 4GB VRAM**: Use MDX23C models
- ❌ **No GPU detected**: CPU mode (very slow)

## User Interface

### Header
- **StemSep** - Application title
- **Settings** - Open settings dialog
- **Toggle Theme** - Switch between dark/light mode

### Navigation Bar
Three main tabs:
- **Home** - Upload files and start separation
- **Active Separations** - Monitor running jobs
- **Models** - Manage and download models

### Home Tab Sections

#### System Information
- GPU model and VRAM
- Recommended model type
- Current system status

#### Preset Selection
Five preset buttons:
1. **Vocal & Instrumental** - Separate vocals from backing track
2. **Karaoke** - Extract lead, back, and instrumental
3. **Drums & Bass** - Extract rhythm section
4. **Instruments Only** - Full instrumental mix
5. **Multi-Stem** - 4+ separate tracks

#### File Drop Zone
- Drag & drop audio files
- Click to browse for files
- Shows selected file name

#### Advanced Settings
- **Model**: Choose specific model
- **Output**: Set output directory

#### Separate Button
- Starts separation process
- Changes to Active Separations tab

### Active Separations Tab

#### Separation Card
For each job, shows:
- **File name** - Original audio file
- **Preset/Model** - Current settings
- **Progress bar** - Real-time progress
- **Percentage** - Completion status
- **Elapsed time** - Time since start
- **Remaining time** - Estimated completion
- **Cancel button** - Stop processing
- **Play button** - Preview when done

### Models Tab

#### Filter Dropdown
Filter models by architecture:
- All
- BS-Roformer
- Mel-Roformer
- MDX23C
- SCNet

#### Model Cards
Each model shows:
- **Name** - Model name
- **Description** - What it's good for
- **Metrics** - SDR, fullness, bleedless scores
- **Info** - Architecture, VRAM, speed
- **Download/Remove button** - Install or uninstall
- **Size** - File size for download

### Settings Dialog

Three tabs:

#### UI Tab
- **Theme**: Dark or light mode

#### Processing Tab
- **Chunk Size**: 80000-240000 (default 160000)
  - Lower = less VRAM, slower
  - Higher = more VRAM, faster
- **Overlap**: 0.5-0.9 (default 0.75)
- **CPU Threads**: Number of parallel threads

#### Paths Tab
- **Models Directory**: Where models are stored
- **Output Directory**: Where results are saved

## Separation Process

### Step-by-Step

1. **Choose Preset**
   - Click preset button on Home tab
   - Preset configures optimal settings

2. **Load Audio**
   - Drag & drop file to drop zone
   - Or click drop zone to browse
   - Supported: MP3, WAV, FLAC, M4A, OGG

3. **Review Settings**
   - Check model in Advanced Settings
   - Verify output directory
   - Adjust if needed

4. **Start Separation**
   - Click SEPARATE button
   - Switches to Active Separations tab

5. **Monitor Progress**
   - Watch real-time progress bar
   - See elapsed and remaining time
   - Cancel if needed

6. **Review Results**
   - Click Play to preview stems
   - Files saved in output directory
   - All stems as WAV files

### Output Files
Separated stems are saved as:
- `vocals.wav`
- `instrumental.wav`
- Or stem-specific names (e.g., `drums.wav`, `bass.wav`)

## Model Management

### Downloading Models
1. Go to **Models** tab
2. Click **Download** on desired model
3. Wait for download (100-500MB)
4. Model installed and ready to use

### Model Information
Each model card shows:

**Metrics**:
- **SDR** (Signal-to-Distortion Ratio) - Higher is better
- **Fullness** - How complete the stem sounds
- **Bleedless** - How clean separation is (less bleed)

**Technical Info**:
- **Architecture** - BS-Roformer, Mel-Roformer, etc.
- **VRAM** - Required GPU memory
- **Speed** - Fast, medium, or slow

### Removing Models
1. Find installed model (shows "✓ Installed")
2. Click **Remove** button
3. Confirm removal
4. Model deleted from disk

### Model Recommendations

**Best Quality** (6GB+ VRAM):
- BS-Roformer 2025.07
- Mel-Roformer Kim

**Fast Processing** (4GB VRAM):
- MDX23C-8KFFT-InstVoc HQ

**Karaoke**:
- BS-Roformer Karaoke (becruily & frazer)

**Complex/Busy Songs**:
- SCNet XL

## Settings

### UI Settings
- **Theme**: Toggle between dark and light
- Automatically saved
- Applied immediately

### Processing Settings
- **Chunk Size**: VRAM management
  - Default: 160000
  - For 4GB VRAM: 80000-120000
  - For 8GB+ VRAM: 200000-240000
- **Overlap**: Separation quality
  - Default: 0.75
  - Lower = faster, slightly lower quality
  - Higher = slower, better quality
- **CPU Threads**: Parallel processing
  - Default: Number of CPU cores
  - Can lower to reduce CPU usage

### Paths
- **Models Directory**: Default `~/.stemsep/models`
- **Output Directory**: Default `~/StemSep_Output`
- Click Browse to change

## Advanced Features

### Ensemble Models
Combine multiple models for better results:
1. Run first model
2. Run second model on same input
3. Mix results in DAW
4. Benefits from each model's strengths

### Custom Presets
Create your own presets:
1. Adjust all settings
2. Note model and parameters
3. Save settings in config.json
4. Or use consistent settings manually

### Batch Processing
Process multiple files:
1. Start first separation
2. When complete, start second
3. Continue sequentially
4. No limit on queue size

### Playback Controls
Preview stems:
- Click **Play** on completed job
- Controls in popup player
- Switch between stems
- Compare to original

## Troubleshooting

### Common Issues

#### "No module named 'torch'"
- Run `install_dependencies.bat`
- Or: `pip install torch`

#### "CUDA out of memory"
- Lower chunk size to 80000-120000
- Close other GPU applications
- Use smaller model (MDX23C)

#### "No GPU detected"
- Update GPU drivers
- Install CUDA Toolkit
- Check GPU in Task Manager

#### "Model download failed"
- Check internet connection
- Verify Hugging Face accessibility
- Try manual download

#### "Audio format not supported"
- Convert to MP3, WAV, FLAC, M4A, or OGG
- Use VLC or online converter

#### "Separation produces poor quality"
- Try different model
- Check input file quality
- Ensure vocals are centered for karaoke

### Getting Help
1. Check logs in `logs/` directory
2. Read `docs/TROUBLESHOOTING.md`
3. Run `test_app.py` to diagnose issues
4. Verify system requirements

### Log Files
Located in `%USERPROFILE%\.stemsep\logs\`
- `stemsep_YYYYMMDD.log` - Daily log files
- Contains all errors and activities
- Useful for debugging

## Performance Optimization

### For Speed
1. **Use MDX23C models** - Fastest
2. **Increase chunk size** (if VRAM allows)
3. **Close other applications** - Free resources
4. **Use SSD storage** - Faster I/O
5. **Update GPU drivers** - Better performance

### For Quality
1. **Use BS-Roformer 2025.07** - Best overall
2. **Use high-quality source** - WAV/FLAC
3. **Increase overlap** - Better separation
4. **Close other GPU apps** - Prevent memory issues
5. **Use de-bleeding models** - Cleaner results

### GPU Optimization

**6GB VRAM**:
- Mel-Roformer or MDX23C
- Chunk size: 160000
- Should work well

**8GB+ VRAM**:
- All models including BS-Roformer
- Chunk size: 200000-240000
- Can run multiple at once

**4GB VRAM**:
- MDX23C only
- Chunk size: 80000-120000
- May need CPU fallback

**No GPU**:
- Works but very slow (10-50x)
- For testing only
- Not recommended for regular use

### Expected Performance

**Per minute of audio**:
- RTX 4090 + BS-Roformer: 15-30 sec
- RTX 3080 + MDX23C: 10-20 sec
- RTX 3060 + MDX23C: 20-40 sec
- GTX 1660 + MDX23C: 40-60 sec
- CPU only: 5-10 minutes

## FAQ

### Q: Is internet required?
**A**: Only for downloading models. After that, fully offline.

### Q: Can I use my own models?
**A**: Limited support. Currently supports pre-configured models from Hugging Face.

### Q: What audio formats are supported?
**A**: Input: MP3, WAV, FLAC, M4A, OGG. Output: WAV only (best quality).

### Q: Can I process multiple files at once?
**A**: Yes, start new separations after previous complete. Queued processing.

### Q: Does it work on Mac/Linux?
**A**: Windows-focused, but code may work on other platforms with modifications.

### Q: Is there a file size limit?
**A**: No hard limit, but very long files may need high RAM. Consider splitting.

### Q: Can I commercial use the separated stems?
**A**: Check local copyright laws. This is a technical tool - you're responsible for usage.

### Q: Why is CPU mode so slow?
**A**: AI models are designed for GPU acceleration. CPU can run them but is much slower.

### Q: What if I don't have enough VRAM?
**A**: Lower chunk size in Settings, or use CPU mode (slow).

### Q: Can I cancel a separation?
**A**: Yes, click Cancel in Active Separations tab.

### Q: Are the models free?
**A**: Yes, all models are open source and free to use (check individual licenses).

### Q: How do I report a bug?
**A**: Check logs, run test_app.py, and provide system info and error messages.

### Q: Can I contribute?
**A**: Yes! This is open source. See GitHub for contribution guidelines.

### Q: Will it work with my GPU?
**A**: NVIDIA GPUs with CUDA support work best. AMD/Intel have limited support.

### Q: How accurate is the separation?
**A**: Depends on model and audio. SDR scores 10-17 for vocals. Not perfect but high quality.

### Q: Can it separate into more than 2 stems?
**A**: Yes, specialized models exist for drums, bass, strings, etc. Multi-stem presets available.

## Appendix

### Keyboard Shortcuts
- `Ctrl+O` - Open file (planned)
- `Escape` - Cancel current operation (planned)
- `Space` - Play/pause (planned)

### File Locations
- Models: `%USERPROFILE%\.stemsep\models\`
- Temp: `%USERPROFILE%\.stemsep\temp\`
- Output: `%USERPROFILE%\StemSep_Output\`
- Logs: `%USERPROFILE%\.stemsep\logs\`
- Config: `%USERPROFILE%\.stemsep\config.json`

### Supported Sample Rates
Input audio can be:
- 44.1 kHz (CD quality)
- 48 kHz (professional)
- 96 kHz (high-resolution)
- And others

All output is saved at original sample rate for best quality.

### Technical Details

**Chunk Size**:
Audio is processed in chunks to manage memory. Smaller chunks use less VRAM but require more processing overhead. Larger chunks are faster but need more memory.

**Overlap**:
Adjacent chunks overlap by this percentage. Higher overlap gives better quality at the cost of speed. 0.75 is a good balance.

**SDR Metric**:
Signal-to-Distortion Ratio measures quality. 10+ is good, 15+ is excellent, 20+ is rare for music separation.

### Resources
- **Documentation**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **Models Guide**: `docs/MODELS.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Changelog**: `CHANGELOG.md`
- **License**: `LICENSE`

---

For more information, visit the project repository or read the full documentation.
