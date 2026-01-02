# Models Guide

This document provides detailed information about the AI models available in StemSep.

Note: model availability is defined by the in-app registry (`StemSepApp/assets/models.json.bak`). Some older guide nicknames (e.g. “SCNet XL”) map to different shipped `model_id`s.

## Registry source of truth (v2) + generator workflow

We are migrating to a robust, long-term registry format.

- **Source of truth (edit this):** `StemSepApp/assets/registry/models.v2.source.json`
- **Schema (optional validation):** `StemSepApp/assets/registry/models.v2.schema.json`
- **Generated legacy registry (do not hand-edit):** `StemSepApp/assets/models.json.bak`
- **Generator script:** `scripts/generate_model_registry.py`

### How to update the registry

1) Edit the v2 source registry:
- Add/update models in `StemSepApp/assets/registry/models.v2.source.json`

2) (Optional) validate the v2 source against the schema:
- The generator supports best-effort schema validation (requires a JSON Schema validator library in your Python env).

3) Generate the legacy registry used by the current app:
- Run: `python scripts/generate_model_registry.py --pretty`

### Important rules

- Do **not** hand-edit `StemSepApp/assets/models.json.bak` (it is generated).
- Keep changes additive and backwards compatible.
- When adding new models, include deterministic fields for:
  - runtime routing/compatibility
  - requirements (dependencies/manual steps)
  - phase-fix reference eligibility (when applicable)

## Architecture Overview

StemSep supports several state-of-the-art audio source separation architectures:

### 1. BS-Roformer (Band-Split Roformer)

**Overview**: One of the best performing architectures for vocal/instrumental separation.

**Characteristics**:
- **Quality**: Highest overall quality
- **Speed**: Slower processing time
- **VRAM**: Requires 6GB+ for optimal performance
- **Artifacts**: May sound slightly filtered due to strong denoising

**Best Models**:
- **BS-Roformer Viperx** (Recommended)
  - StemSep `model_id`: `bs-roformer-viperx-1297`
  - Use case: General vocal/instrumental separation

**Karaoke Variants**:
- **BS-Roformer Karaoke (Becruily)**
  - StemSep `model_id`: `bs-roformer-karaoke-becruily`
  - Use case: Karaoke tracks with harmonies

-- **BS Roformer 3-Stem (MVSEP Team)**
  - Not currently shipped as a downloadable StemSep registry model (MVSEP-only / manual download territory)

**Specialized**:
- **BS-Roformer Drums**: Dedicated drum separation
- **BS-Roformer Bass**: Dedicated bass separation
- **BS-Roformer Strings**: String instruments (SDR: 5.41)

### 2. Mel-Roformer (Melband Roformer)

**Overview**: Excellent for instrumental separation with focus on fullness.

**Characteristics**:
- **Quality**: Very good instrumental quality
- **Speed**: Medium processing speed
- **VRAM**: Requires 6GB
- **Strengths**: Preserves instrument clarity

**Best Models**:
- **Mel-Roformer Inst (Gabox)**
  - SDR: 16.62
  - Inst. fullness: 29.96
  - Bleedless: 44.61
  - Use case: Fullness-focused instrumental separation

- **Mel-Roformer Kim**
  - SDR: 17.32
  - Inst. fullness: 27.44
  - Bleedless: 46.56 (best score!)
  - Use case: When cleanest separation is critical

### 3. MDX23C

**Overview**: Fast and efficient architecture with good quality-speed balance.

**Characteristics**:
- **Quality**: Good quality, less clean than Roformers
- **Speed**: Fast processing
- **VRAM**: Can run on 4GB GPUs
- **Settings**: n_fft: 12288, cutoff at 14.7kHz

**Best Model**:
- **MDX23C-8KFFT HQ**
  - StemSep `model_id`: `mdx23c-8kfft-hq`
  - Use case: When speed is more important than perfect quality

### 4. SCNet (Sparse Compression Network)

**Overview**: Alternative architecture optimized for complex, busy songs.

**Characteristics**:
- **Quality**: Between Roformers and MDX
- **Speed**: Medium processing
- **VRAM**: 6GB recommended
- **Strengths**: Better at handling complexity

**Best Models**:
- **SCNet (StemSep)**
  - StemSep `model_id`: `scnet-large`
  - Use case: Busy songs, piano, drums

## Model Selection Guide

### For Vocal/Instrumental Separation
**Recommended**: `bs-roformer-viperx-1297`
- Best overall quality
- Cleanest results
- Works well on most music genres

**Alternative**: Mel-Roformer Kim
- Best bleedless score
- Preserves instrument clarity
- Good for acoustic music

### For Karaoke
**Recommended**: `bs-roformer-karaoke-becruily`
- Superior harmony detection
- Better lead/back vocal differentiation
- Less bleed

**Alternative**: MVSEP 3-stem models (not shipped in registry)
- Returns 3 stems directly
- Good for harmonies
- No bleed when working

### For Fast Processing
**Recommended**: `mdx23c-8kfft-hq`
- Fastest model
- Works on 4GB GPUs
- Decent quality

### For Busy/Complex Songs
**Recommended**: `scnet-large`
- Best fullness score
- Handles complexity well
- Good for piano, drums

**Strategy**: Ensemble SCNet with Roformer
- Combine SCNet (for fullness) with Roformer (for cleanliness)
- Better overall results
- Slower but higher quality

### For Single Instruments
- **Drums**: BS-Roformer SW drums
- **Bass**: BS-Roformer SW bass
- **Strings**: BS-Roformer Strings
- **Piano**: Multiple digital piano models available

## GPU Requirements

### Minimum (4GB VRAM)
- MDX23C models only
- Reduced chunk size recommended (80000-120000)
- May need to disable some features

### Recommended (6GB VRAM)
- All models except heaviest variants
- Mel-Roformer and BS-Roformer
- Full feature set
- Standard chunk size (160000)

### Optimal (8GB+ VRAM)
- All models including ensembles
- Can use largest models (BS-Roformer, `scnet-large`)
- Can run multiple separations simultaneously
- No VRAM limitations

### No GPU (CPU Only)
- Works but very slow (10-50x slower)
- Only for testing or very short files
- Not recommended for regular use

## Performance Tips

### For Best Quality
1. Use `bs-roformer-viperx-1297` as default
2. Close other GPU-intensive applications
3. Ensure GPU drivers are up to date
4. Use WAV/FLAC source files when possible
5. For extra clean results: use `gabox-denoise-debleed` on the instrumental stem

### For Best Speed
1. Use MDX23C models
2. Increase chunk size if you have VRAM (speeds up processing)
3. Close unnecessary applications
4. Use SSD for temporary files

### For Karaoke Tracks
1. Ensure lead vocals are centered
2. Use BS-Roformer Karaoke models
3. If bleed occurs, try "extract vocals first" approach
4. For double vocals, run model twice with different settings

### For Problematic Tracks
1. Try multiple models (each handles different genres differently)
2. Ensemble two models for better results
3. Use De-bleeding models on problematic stems
4. Check MVSEP quality checker for reference scores

## Model Metrics Explained

### SDR (Signal-to-Distortion Ratio)
- **What**: Measures how much signal vs. distortion
- **Higher**: Better
- **Range**: Usually 5-20 for music separation
- **Note**: Not the only quality metric

### Fullness
- **What**: How complete/complete the stem sounds
- **Higher**: More complete
- **Range**: 20-35 typical
- **Note**: Higher is not always better (can mean more bleed)

### Bleedless
- **What**: How clean separation is (less bleed from other stems)
- **Higher**: Better (cleaner)
- **Range**: 30-50 typical
- **Note**: Good balance is important

## Advanced Usage

### Ensemble Models
Combine multiple models for better results:

**Example**: Combine SCNet (fullness) with Mel-Roformer (cleanliness)
1. Run `scnet-large` on original file
2. Run Mel-Roformer on original file
3. Average the results
4. Manual mixing in DAW

**Best Ensemble**: "Mvsep + gabox + frazer/becruily"
- SDR: 10.6
- Combines multiple models
- Requires manual processing

### Custom Workflows
1. **Extract Vocals First**: Get vocals, then instrumental separately
2. **De-bleed Separation**: Run `gabox-denoise-debleed` on instrumental
3. **Phase Fix**: Use phase fixing tools for final polish
4. **Multi-pass**: Run model multiple times with different settings

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify HuggingFace link is active
- Try downloading manually from HuggingFace
- Clear temp directory

### Out of Memory Errors
- Lower chunk size to 80000-120000
- Close other GPU applications
- Try MDX23C model (lower VRAM)
- Restart application

### Poor Separation Quality
- Try different model architecture
- Check if source file is high quality
- Use ensemble approach
- Verify model downloaded completely

### Slow Processing
- Check GPU utilization in Task Manager
- Update GPU drivers
- Increase chunk size
- Use MDX23C for speed over quality

### CUDA Errors
- Update PyTorch: `pip install torch --upgrade`
- Check CUDA compatibility
- Reinstall CUDA toolkit
- Try CPU mode as fallback
