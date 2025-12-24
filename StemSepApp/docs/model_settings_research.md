# Model Recommended Settings Research

This document contains the recommended processing parameters for each model in StemSep-V3, extracted from official YAML configs on HuggingFace/GitHub.

## Key Parameters
- **chunk_size**: Number of audio samples processed at once (higher = more memory, potentially better quality)
- **num_overlap**: Overlap factor between chunks (2 = 50% overlap, typically sufficient)
- **overlap**: Converted from num_overlap → 1 - (1/num_overlap) = 0.5 for num_overlap=2

## Standard Conversions
| num_overlap | overlap (decimal) |
|-------------|-------------------|
| 1 | 0.0 |
| 2 | 0.50 |
| 4 | 0.75 |

---

## Mel-Roformer Models

### Gabox Models (MelBandRoformer)
**Source**: https://huggingface.co/GaboxR67/MelBandRoformers

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| gabox-inst-v7n | 485100 | 2 | 0.5 | Standard Gabox config |
| gabox-inst-v6n | 485100 | 2 | 0.5 | Same config |
| gabox-inst-fv7-plus | 485100 | 2 | 0.5 | |
| gabox-inst-fv7z | 485100 | 2 | 0.5 | |
| gabox-inst-fv8b | 485100 | 2 | 0.5 | |
| gabox-inst-fv4-noise | 485100 | 2 | 0.5 | |
| gabox-inst-experimental | 485100 | 2 | 0.5 | |
| gabox-small-inst | 485100 | 2 | 0.5 | Smaller model, same config |
| gabox-karaoke | 485100 | 2 | 0.5 | |
| gabox-voc-fv3 | 485100 | 2 | 0.5 | |
| gabox-voc-fv4 | 485100 | 2 | 0.5 | |
| gabox-voc-fv5 | 485100 | 2 | 0.5 | |
| gabox-voc-fv6 | 485100 | 2 | 0.5 | |
| gabox-voc-gabox2 | 485100 | 2 | 0.5 | |
| gabox-denoise-debleed | 485100 | 2 | 0.5 | |
| gabox-resurrection | 485100 | 2 | 0.5 | |

### Unwa Models (Mel-Band-Roformer)
**Source**: https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| unwa-inst-v1e | 485100 | 2 | 0.5 | Same as v1e+ |
| unwa-inst-v1e-plus | 485100 | 2 | 0.5 | From config_melbandroformer_inst.yaml |
| unwa-inst-fno | 485100 | 2 | 0.5 | FNO variant, same base config |
| unwa-duality-v2 | 485100 | 2 | 0.5 | |
| unwa-big-beta-5e | 485100 | 2 | 0.5 | |
| unwa-big-beta-6x | 485100 | 2 | 0.5 | |
| unwa-resurrection-inst | 485100 | 2 | 0.5 | |

### Kim Mel-Roformer
**Source**: https://huggingface.co/KimberleyJSN/melbandroformer

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| mel-band-roformer-kim | 352800 | 2 | 0.5 | Standard Kim config |

### Mel-Roformer Karaoke (Aufr33/Becruily)
| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| mel-band-karaoke-aufr33 | 352800 | 2 | 0.5 | |
| mel-band-karaoke-becruily | 352800 | 2 | 0.5 | |

### Other Mel-Roformers
| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| mel-roformer-debleed | 485100 | 2 | 0.5 | Unwa config |
| mel-roformer-decrowd | 485100 | 2 | 0.5 | |
| mel-roformer-dereverb-anvuew-v2 | 485100 | 2 | 0.5 | |
| mel-roformer-drumsep-5stem | 352800 | 2 | 0.5 | MDX23C style |
| mesk-rifforge | 485100 | 2 | 0.5 | Metal-focused |
| amane-full-scratch | 485100 | 2 | 0.5 | |
| amane-inst-fullness | 485100 | 2 | 0.5 | |

---

## BS-Roformer Models

**Source**: https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| bs-roformer-viperx-1296 | 352800 | 2 | 0.5 | Standard BS-Roformer |
| bs-roformer-viperx-1297 | 352800 | 2 | 0.5 | From yaml config |
| bs-roformer-karaoke-becruily | 352800 | 2 | 0.5 | Karaoke variant |
| bs-roformer-resurrection | 352800 | 2 | 0.5 | Unwa retrain |

---

## Anvuew Models

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| anvuew-dereverb | 352800 | 2 | 0.5 | Room dereverb |
| anvuew-karaoke | 352800 | 2 | 0.5 | |
| anvuew-vocals | 352800 | 2 | 0.5 | |

---

## Becruily Models

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| becruily-guitar | 352800 | 2 | 0.5 | |
| becruily-inst | 352800 | 2 | 0.5 | |
| becruily-karaoke | 352800 | 2 | 0.5 | |
| becruily-vocal | 352800 | 2 | 0.5 | |

---

## Sucial Models (VR Architecture)

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| sucial-debreath | 256 | 2 | 0.5 | VR uses different chunk logic |
| sucial-dereverb | 256 | 2 | 0.5 | |
| sucial-male-female | 256 | 2 | 0.5 | |

---

## MDX23C Models

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| mdx23c-8kfft-hq | 352800 | 2 | 0.5 | Standard MDX23C |

---

## HTDemucs Models

HTDemucs uses internal chunking via `split=True` in demucs.apply_model(), not chunk_size/overlap.

| Model | chunk_size | overlap | Notes |
|-------|------------|---------|-------|
| htdemucs | N/A | N/A | Uses internal split logic |
| htdemucs-6s | N/A | N/A | 6-stem version |

---

## SCNet Models

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| scnet-large | 352800 | 2 | 0.5 | |
| scnet-xl | 352800 | 2 | 0.5 | XL variant |

---

## Other/Utility Models

| Model | chunk_size | num_overlap | overlap | Notes |
|-------|------------|-------------|---------|-------|
| aufr33-denoise | 352800 | 2 | 0.5 | Denoise utility |
| reverb-hq | 352800 | 2 | 0.5 | Reverb removal |
| 5_hp-karaoke-uvr | 256 | 2 | 0.5 | VR arch |

---

## Summary by Architecture

| Architecture | Default chunk_size | Default overlap | Notes |
|--------------|-------------------|-----------------|-------|
| **Mel-Roformer (Gabox/Unwa)** | 485100 | 0.5 | Larger chunk for quality |
| **Mel-Roformer (Kim/Aufr33)** | 352800 | 0.5 | Standard |
| **BS-Roformer** | 352800 | 0.5 | All use same config |
| **MDX23C** | 352800 | 0.5 | |
| **SCNet** | 352800 | 0.5 | |
| **HTDemucs** | N/A | N/A | Internal chunking |
| **VR (UVR)** | 256 | 0.5 | Different approach |
