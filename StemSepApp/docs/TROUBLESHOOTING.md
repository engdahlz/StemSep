# Troubleshooting Guide

This guide helps you solve common issues with StemSep.

## Installation Issues

### Error: "Python not found"
**Solution**:
1. Install Python 3.8+ from [python.org](https://python.org)
2. During installation, check "Add Python to PATH"
3. Restart your computer after installation
4. Run the install script again

### Error: "pip not recognized"
**Solution**:
1. Reinstall Python and check "Add Python to PATH"
2. Or use: `python -m ensurepip --upgrade`
3. Or run: `python -m pip --version` to verify

### Error: "Microsoft Visual C++ 14.0 is required"
**Solution**:
1. Install Microsoft C++ Build Tools
2. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. During installation, select "C++ build tools" workload
4. Restart and try installing again

### Error: "No module named 'torch'"
**Solution**:
1. Update pip: `python -m pip install --upgrade pip`
2. Install PyTorch separately: `python -m pip install torch`
3. Reinstall requirements: `pip install -r requirements.txt`

### Error: "Failed to build wheel for..." while installing dependencies
**Solution**:
1. Update your graphics drivers (NVIDIA/AMD/Intel)
2. Install Microsoft C++ Build Tools (see above)
3. Try installing packages individually: `pip install package_name`
4. Use CPU-only PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

## GPU Issues

### Error: "CUDA out of memory"
**Solutions**:
1. Lower chunk size in Settings → Processing to 80000-120000
2. Close other GPU-intensive applications (games, video editors, browsers with hardware acceleration)
3. Restart StemSep to free GPU memory
4. Use a smaller model (MDX23C instead of BS-Roformer)
5. Reduce batch size in advanced settings

### Error: "No CUDA-capable device is detected"
**Solutions**:
1. Update GPU drivers from manufacturer website
2. Install CUDA Toolkit from NVIDIA
3. Verify GPU supports CUDA (must be NVIDIA, compute capability 6.0+)
4. Check that GPU is not being used by another application
5. Reinstall PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### GPU detected but very slow performance
**Solutions**:
1. Check if GPU is being used: Task Manager → Performance tab
2. Update GPU drivers
3. Check Power Plan: Set to "High Performance"
4. Verify PyTorch CUDA installation: Run `python -c "import torch; print(torch.cuda.is_available())"`
5. Increase chunk size in Settings

### Error: "ModuleNotFoundError: No module named 'torch._dynamo.polyfills.fx'"
**Solutions**:
1. Update PyTorch: `pip install --upgrade torch`
2. Use MSST instead of UVR for affected BS-Roformer models
3. Rollback to stable PyTorch version: `pip install torch==2.0.1`

### Invalid parameter error on AMD/Intel GPUs
**Solutions**:
1. Disable GPU conversion in Settings
2. Use CPU mode (slower but compatible)
3. Use MSST framework instead of UVR
4. Update to latest AMD/Intel drivers

## Model Issues

### Model download fails
**Solutions**:
1. Check internet connection
2. Try downloading again
3. Manually download from HuggingFace and place in models directory
4. Check firewall/antivirus settings
5. Clear temp directory: `del /f /s %TEMP%\stemsep*`
6. Check disk space (models are 100-500MB each)

### Model loads but separation fails
**Solutions**:
1. Verify model file is complete (check file size)
2. Re-download the model
3. Check logs for specific error
4. Try a different model from the same architecture
5. Restart application

### Error: "Model not compatible with this version"
**Solutions**:
1. Update to latest StemSep version
2. Use older model version
3. Check model documentation for requirements
4. Reinstall PyTorch and dependencies

### Model quality is poor
**Solutions**:
1. Try a different model (different architecture)
2. Check input file quality (use WAV/FLAC)
3. Use ensemble of two models
4. Apply de-bleeding model to instrumental stem
5. Check if vocals are centered (important for karaoke)

## Audio Processing Issues

### Error: "Unsupported audio format"
**Solutions**:
1. Convert to supported format: MP3, WAV, FLAC, M4A, OGG
2. Use online converter or VLC Media Player
3. Check file is not corrupted
4. Try playing the file in regular media player

### Error: "Audio file too long"
**Solutions**:
1. Split audio file into smaller segments
2. Increase system memory (RAM)
3. Use lower quality model for long files
4. Close other applications to free memory

### Error: "Failed to save output file"
**Solutions**:
1. Check output directory exists and is writable
2. Ensure you have sufficient disk space
3. Close any programs that might be using the output files
4. Run as administrator
5. Check filename doesn't contain invalid characters

### Separation produces silence
**Solutions**:
1. Input file might be mono or have very quiet audio
2. Model is not appropriate for the audio type
3. Check input file in regular media player
4. Try different model architecture
5. Verify input file isn't just noise

### Output file has artifacts or sounds bad
**Solutions**:
1. This is normal for some models
2. Try a different model
3. Use higher quality source file
4. Apply de-bleeding or de-noising models
5. Manual post-processing in DAW

## Application Issues

### Application won't start
**Solutions**:
1. Run as administrator
2. Check Windows version (requires Windows 10+)
3. Install all dependencies: `pip install -r requirements.txt`
4. Check Windows Defender/antivirus isn't blocking
5. Check logs in `logs/` directory for error details

### UI appears broken or glitchy
**Solutions**:
1. Update graphics drivers
2. Clear application cache: delete `%APPDATA%\StemSep` folder
3. Reset settings to default
4. Reinstall the application
5. Try running on dedicated GPU instead of integrated

### Application crashes
**Solutions**:
1. Check logs in `logs/` directory
2. Update all dependencies
3. Reduce chunk size in Settings
4. Close other applications
5. Run in safe mode (no other programs)
6. Report issue with logs attached

### High memory usage
**Solutions**:
1. Close other applications
2. Reduce chunk size
3. Process one file at a time
4. Use lighter models (MDX23C)
5. Restart application periodically

### Settings not saving
**Solutions**:
1. Run as administrator
2. Check if config.json file exists
3. Verify folder permissions
4. Delete config.json to reset to defaults
5. Check disk space

## Performance Issues

### Processing is very slow
**Possible Causes**:
- Not using GPU (check Task Manager → Performance)
- Using CPU-only mode
- High chunk size with low VRAM
- Too many background processes
- Slow hard drive

**Solutions**:
1. Verify GPU is detected and being used
2. Update GPU drivers
3. Use appropriate model for your GPU
4. Close unnecessary applications
5. Use SSD for temporary and output files
6. Use MDX23C model for faster processing
7. Check Windows Power Plan is "High Performance"

### Long wait times even for short files
**Solutions**:
1. Check if model is downloading/first time loading
2. Verify GPU is being used
3. Check for background updates
4. Restart application
5. Use faster model (MDX23C)

### System becomes unresponsive during processing
**Solutions**:
1. Lower chunk size
2. Close other applications
3. Use only one separation at a time
4. Reduce priority in Task Manager
5. Consider upgrading hardware

## Playback Issues

### Can't preview audio
**Solutions**:
1. Check if output file was created successfully
2. Verify audio device is working
3. Update audio drivers
4. Try different output file
5. Check volume levels

### Audio playback is choppy
**Solutions**:
1. Close other audio applications
2. Update audio drivers
3. Use WASAPI instead of DirectSound
4. Check system resources
5. Try different audio file

## Getting Help

### Check Logs
All errors are logged in the `logs/` directory. Check the most recent log file for details.

### Enable Debug Logging
1. Go to Settings → Advanced
2. Change log level to DEBUG
3. Reproduce the issue
4. Check logs for detailed error messages

### System Information
Run GPU detection to get system info:
```python
python src/core/gpu_detector.py
```

This prints:
- GPU model and VRAM
- CUDA version
- PyTorch installation
- System specs

### Before Reporting an Issue
Please provide:
1. System information (GPU, RAM, Windows version)
2. Log files from `logs/` directory
3. Model being used
4. Input file format and duration
5. Exact error message (if any)
6. Steps to reproduce

### Common Solutions to Try First
1. Restart the application
2. Restart your computer
3. Update GPU drivers
4. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
5. Clear temp directory
6. Run as administrator
7. Check Windows updates
8. Verify antivirus isn't blocking

## Known Issues and Workarounds

### Issue: "Invalid parameter" on AMD/Intel GPUs
**Workaround**: Disable GPU conversion in Settings, use CPU mode

### Issue: BS-Roformer sounds filtered
**Workaround**: Normal behavior due to strong denoising. Try Mel-Roformer for less filtering.

### Issue: Very slow on Apple Silicon (M1/M2)
**Workaround**: Use chunk size over 7, limited MPS support

### Issue: Stereo errors in MSST
**Workaround**: Update MSST with `git pull` or edit inference.py line 59

### Issue: Phase issues in separated stems
**Workaround**: Use BS 2025.07 as reference for phase fixing, use UVR GUI phase fix tool

## Performance Benchmarks

### Reference Hardware Performance
- **RTX 4090 + BS-Roformer**: ~15-30 sec per minute of audio
- **RTX 3080 + MDX23C**: ~10-20 sec per minute
- **RTX 3060 + MDX23C**: ~20-40 sec per minute
- **GTX 1660 + MDX23C**: ~40-60 sec per minute
- **CPU Ryzen 5 5600X**: ~5-10 min per minute (all models)

### Expected Wait Times
- **2-minute song**: 30 seconds to 5 minutes (depending on hardware)
- **5-minute song**: 1-15 minutes
- **Long tracks (10+ min)**: Consider splitting first

### Slow Performance Indicators
If processing takes longer than:
- 2 min per minute of audio on RTX 3060
- 5 min per minute of audio on GTX 1660
- 10 min per minute of audio on CPU

Then you may have:
- Incorrect GPU setup
- Model mismatched to hardware
- Other performance issues

Check GPU utilization in Task Manager - should be 80-100% during processing.
