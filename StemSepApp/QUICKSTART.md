# Quick Start Guide

Get up and running with StemSep in 5 minutes!

## 🚀 Option 1: Portable Version (Fastest)

1. **Download** the latest `StemSep_Portable.zip` from releases
2. **Extract** to any folder (e.g., `C:\StemSep`)
3. **Run** `StemSep.exe`
4. **Download** a model (see below)
5. **Start separating!**

That's it! No installation required.

## 🔧 Option 2: From Source

### Prerequisites
- Python 3.8 or higher
- Windows 10/11
- 4GB RAM minimum (8GB+ recommended)
- GPU optional but recommended

### Steps

1. **Install Python** (if not already installed)
   - Download from [python.org](https://python.org)
   - ✅ Check "Add Python to PATH" during installation

2. **Download StemSep**
   - Download this repository as ZIP
   - Extract to `C:\StemSep\`

3. **Install Dependencies**
   ```batch
   # Open Command Prompt or PowerShell in C:\StemSep\
   cd /d C:\StemSep

   # Run the install script
   install_dependencies.bat
   ```
   This installs all required packages. First install may take 10-20 minutes.

4. **Run the Application**
   ```batch
   # Using the run script
   run.bat

   # Or directly
   python src/main.py
   ```

5. **Success!** StemSep window should open

## 📦 First Time Setup

### 1. Download Your First Model

1. Click **Models** tab in StemSep
2. Click **Download** on a model:
   - **BS-Roformer 2025.07** (recommended, best quality)
   - **MDX23C-8KFFT** (faster, requires less VRAM)

Models are 100-500MB each. Download time depends on your internet speed.

### 2. Check GPU (Optional)

**If you have an NVIDIA GPU:**
- Go to Home tab
- Look at "System Information" section
- Should show your GPU and VRAM amount

**If no GPU is detected:**
- You'll see "CPU mode - very slow"
- Consider installing a GPU, or just be patient
- CPU processing is 10-50x slower

## ✂️ Your First Separation

### Step 1: Choose a Preset
On the Home tab, click a preset button:
- **Vocal & Instrumental** - Most common
- **Karaoke** - For lead/back vocal separation
- **Drums & Bass** - Extract rhythm section
- **Instruments Only** - Full instrumental mix

### Step 2: Load Your Audio
- **Drag & drop** an audio file onto the drop zone
- **Or click** the drop zone to browse
- Supported: MP3, WAV, FLAC, M4A, OGG

### Step 3: Separate
- Click **SEPARATE** button
- Switch to **Active Separations** tab
- Watch real-time progress
- Wait for completion (time depends on hardware)

### Step 4: Listen & Download
- Click **Play** to preview each stem
- Files are saved in the output directory (default: `~/StemSep_Output`)
- All stems are saved as high-quality WAV files

## 🎛️ Understanding the Interface

### Home Tab
- **Presets**: Pre-configured separation types
- **Drop Zone**: Where you drag & drop audio files
- **Advanced Settings**: Fine-tune model and output directory
- **SEPARATE Button**: Starts the process

### Active Separations Tab
- **Progress Bar**: Real-time processing status
- **Elapsed/Remaining**: Time tracking
- **Play Button**: Preview when done
- **Cancel Button**: Stop processing

### Models Tab
- **Filter**: By architecture (BS-Roformer, Mel-Roformer, etc.)
- **Model Cards**: Show metrics (SDR, fullness, bleedless)
- **Download Button**: Get new models
- **Remove Button**: Delete installed models

### Settings
- **UI**: Theme (dark/light)
- **Processing**: Chunk size, overlap, CPU threads
- **Paths**: Model location, output directory

## 🎯 Quick Tips

### For Best Results
1. **Use WAV/FLAC** source files when possible
2. **BS-Roformer 2025.07** is the best overall model
3. **Close other GPU apps** (games, video editors) during processing
4. **Ensure vocals are centered** in the mix (important for karaoke)

### For Faster Processing
1. **Use MDX23C** models (faster)
2. **Lower chunk size** if you get "out of memory" errors
3. **Use SSD** for temporary files and output
4. **Close unnecessary programs**

### GPU Recommendations
- **4GB VRAM**: Can use MDX23C models
- **6GB+ VRAM**: Can use all models
- **8GB+ VRAM**: Can use largest models and multiple at once
- **No GPU**: Works but very slow (CPU mode)

## ⚙️ Troubleshooting

### "Python not found"
→ Install Python from python.org and check "Add to PATH"

### "ModuleNotFoundError"
→ Run: `install_dependencies.bat`

### "CUDA out of memory"
→ Lower chunk size in Settings → Processing to 80000-120000

### "No GPU detected"
→ Update GPU drivers from manufacturer website

### Processing is very slow
→ Check Task Manager → Performance → GPU
→ Should show 80-100% utilization

### Want more help?
→ Read: `docs/TROUBLESHOOTING.md`

## 📁 File Locations

After first run, StemSep creates these folders in your user directory:

- **Models**: `%USERPROFILE%\.stemsep\models\`
- **Temp**: `%USERPROFILE%\.stemsep\temp\`
- **Output**: `%USERPROFILE%\StemSep_Output\`
- **Logs**: `%USERPROFILE%\.stemsep\logs\`
- **Config**: `%USERPROFILE%\.stemsep\config.json`

## 🎓 What's Next?

1. **Try different models** - Each has unique characteristics
2. **Experiment with presets** - Find what works for your music
3. **Read the full docs** - `docs/MODELS.md` for model details
4. **Join the community** - Share tips and get help

## 🏗️ Build Executable (Developers)

Want to create your own .exe file?

```batch
# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
python build_exe.py

# Find your .exe in: dist/StemSep_Portable/StemSep.exe
```

## 📚 Additional Resources

- **Full Documentation**: `README.md`
- **Model Guide**: `docs/MODELS.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Changes**: `CHANGELOG.md`

## 🆘 Need Help?

1. **Check the logs** in `logs/` directory
2. **Read troubleshooting guide** in `docs/`
3. **Verify system requirements** are met
4. **Update GPU drivers** if using NVIDIA/AMD/Intel

## ⚡ Quick Reference

| Action | How |
|--------|-----|
| Load audio | Drag & drop to Home tab |
| Start separation | Click SEPARATE button |
| View progress | Go to Active Separations tab |
| Play preview | Click Play button on completed job |
| Download model | Go to Models tab, click Download |
| Change settings | Click Settings in header |
| Toggle theme | Click Toggle Theme in header |
| Cancel job | Click Cancel in Active Separations |
| Clear completed | Remove all finished jobs |
| Check logs | See `logs/` directory |

---

**Happy Separating! 🎵**

For more detailed information, see the full README.md file.
