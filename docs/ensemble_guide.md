# 🎵 Ensemble Separation Guide

## What is Ensemble?

Ensemble combines **multiple AI models** to produce better separation results. Each model has strengths and weaknesses - combining them gives you the best of both worlds.

---

## Algorithms

| Algorithm | Best For | Trade-off |
|-----------|----------|-----------|
| **Average** | Highest SDR, balanced quality | May miss some detail |
| **Max Spec** | Maximum detail/fullness | May increase bleed |
| **Min Spec** | Minimum bleed/artifacts | May lose some detail |
| **Frequency Split** | Bass from one, treble from another | Requires 2 models |

---

## Per-Stem Algorithm Presets

Use different algorithms for different stems:

| Preset | Vocals | Instrumental | Use Case |
|--------|--------|--------------|----------|
| **Balanced SDR** | Average | Average | Safest default |
| **Best Vocals** | Max Spec | Average | Full, detailed vocals |
| **Best Instrumental** | Average | Max Spec | Full, detailed instruments |
| **Bleedless Inst** | Max Spec | Min Spec | Clean instrumental (karaoke) |
| **Bleedless Vocals** | Min Spec | Max Spec | Clean acapella |

---

## Recommended Model Combos

### For Vocals
| Combination | Quality |
|-------------|---------|
| BS-Roformer ViperX 1297 + Mel-Band-Roformer Kim | ⭐⭐⭐⭐⭐ |
| BS-Roformer ViperX 1297 + Unwa Resurrection Voc | ⭐⭐⭐⭐ |

### For Instrumentals
| Combination | Quality |
|-------------|---------|
| Unwa Inst v1e + Becruily Inst | ⭐⭐⭐⭐⭐ |
| HyperACE + BS-Roformer ViperX 1297 | ⭐⭐⭐⭐⭐ |
| Unwa Inst FNO + Gabox Inst fv7z | ⭐⭐⭐⭐ |

---

## Quick Start

1. Go to **Advanced Mode** → Enable **Ensemble**
2. Add 2 models (more than 2 increases processing time significantly)
3. Select algorithm:
   - **Average** for balanced quality
   - Open **Advanced Settings** for per-stem algorithms
4. Click **Start Separation**

---

## Tips

- 🎯 **2 models is optimal** - more models = diminishing returns + longer processing
- ⚡ **Max Spec + Min Spec combo** reduces artifacts without losing detail
- 🔊 **Use Frequency Split** when you want bass from one model and highs from another
- 💾 **Model caching** - the app caches models in VRAM for faster repeated runs
