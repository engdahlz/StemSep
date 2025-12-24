# Changelog

All notable changes to StemSep will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-06

### Added
- Initial release
- Complete stem separation application for Windows
- Support for multiple AI architectures:
  - BS-Roformer (Band-Split Roformer)
  - Mel-Roformer (Melband Roformer)
  - MDX23C
  - SCNet (Sparse Compression Network)
- GPU detection and CUDA acceleration
- Model management system with download capabilities
- Pre-configured presets for different use cases:
  - Vocal & Instrumental separation
  - Karaoke (Lead/Back/Instrumental)
  - Drums & Bass extraction
  - Instruments Only
  - Multi-Stem (4+ tracks)
- Real-time progress tracking with ETA
- Built-in audio player for preview
- Modern UI with customtkinter:
  - Dark/Light theme toggle
  - Professional black-and-white inspired design
  - Responsive layout
- Advanced settings for:
  - Chunk size optimization
  - Overlap configuration
  - CPU thread count
  - Custom output directories
- Comprehensive logging system
- Error handling and user feedback
- Drag-and-drop file upload
- Support for multiple audio formats (MP3, WAV, FLAC, M4A, OGG)
- Model metadata and statistics (SDR, fullness, bleedless scores)
- Portable executable generation with PyInstaller

### Features
- Automatic GPU detection and capability reporting
- VRAM-based model recommendations
- Model download with progress tracking
- Queue-based separation processing
- Cancel and pause support
- Model removal and cleanup
- Settings persistence
- Detailed system information display

### Documentation
- Comprehensive README with usage guide
- Model recommendations and best practices
- System requirements and troubleshooting
- Build instructions for developers
- Changelog tracking

### Under the Hood
- Asynchronous processing with ThreadPoolExecutor
- Async/await pattern for model loading
- Progress callback system
- Thread-safe job management
- Configuration management with JSON
- Logging with rotation
- Audio file validation
- Cross-platform compatibility (Windows focus)

## [0.9.0] - Development

### Added
- Initial project structure
- Core UI framework with customtkinter
- GPU detection system
- Model manager foundation
- Audio player implementation
- Separation engine architecture
- Main window and navigation
- Landing page with drag-and-drop
- Active separations window
- Models page
- Settings dialog
- Build system with PyInstaller

## Roadmap

### [1.1.0] - Planned
- Model ensemble support (combine multiple models)
- Advanced karaoke features (vocal isolation, de-bleeding)
- Batch processing (process multiple files at once)
- Preset customization (save your own presets)
- Audio effects (normalize, trim silence, fade)
- Shortcut keys for common actions
- Model update notifications

### [1.2.0] - Planned
- Plugin system for custom models
- Integration with external tools (DAW plugins)
- Cloud model backup/sync
- Advanced playback controls (AB compare, loop)
- Stem mixing capabilities
- Export to different formats (MP3, FLAC with quality settings)

### [1.3.0] - Planned
- Multi-language support
- Audio restoration models (de-reverb, denoise)
- Quality enhancement (upscaling, noise reduction)
- Streaming audio input (microphone, system audio)
- Real-time preview while adjusting settings
- Comparison mode (before/after A/B)
