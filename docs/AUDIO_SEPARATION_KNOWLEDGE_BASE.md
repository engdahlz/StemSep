# Audio Separation Knowledge Base & Engineering Guide

**Version:** 1.0
**Source:** Consolidated insights from NotebookLM, ZFTurbo, Anjok, Aufr33, and community Discord logs (2024-2025).

This document serves as the "Bible" for the audio separation engine, documenting model architectures, optimal settings, advanced workflows, and troubleshooting strategies.

---

## 1. Model Architectures & Selection

Understanding the strengths and weaknesses of different neural network architectures is critical for selecting the right tool.

| Architecture | Best For | Pros | Cons |
|--------------|----------|------|------|
| **BS-Roformer** | Vocals, Clean Inst | Extreme bleed reduction (Bleedless). State-of-the-art SDR. | Can sound "thin" or "filtered". Computationally heavy. |
| **Mel-Band Roformer** | Instrumentals | Better "Fullness" than BS-Roformer. Preserves body. | Slightly more bleed than BS-Roformer. |
| **MDX23C** | Speed, Natural Sound | Fast inference. Natural stereo image. Good for "Quick Mode". | More bleed than Roformers. Lower SDR. |
| **SCNet (XL)** | Bass, Low Mids | Incredible definition in low frequencies (<1kHz). Natural decay. | Very heavy VRAM usage. Can have "buzzing" if undertrained. |
| **Demucs (v4/HT)** | Drums, General | Good all-rounder. Excellent drum separation (4-stem). | Outdated for vocals/inst compared to Roformers. |

---

## 2. Model Rankings (Late 2024/2025)

Based on community benchmarks and SDR/Fullness metrics.

### 🎤 Vocals (Lead)
1. **BS-Roformer 2025.07** (MVSEP) - *King of Bleedless (SDR ~11.89)*.
2. **Gabox Voc_Fv4** - *Best Local All-Rounder*. Good balance of fullness/bleedless.
3. **Unwa Big Beta 5e** - *Max Fullness*. Use if result sounds thin, but expect bleed.

### 🎸 Instrumentals
1. **Unwa Inst v1e+** - *King of Fullness*. Preserves transients and body. Needs Phase Fix.
2. **Becruily Inst** - *Cleanest Local Model*. High bleedless score. Good phase reference.
3. **BS-Roformer Viperx 1297** - *Reliable Workhorse*. Aggressive separation.

### 🥁 Drums
1. **MDX23C DrumSep (Jarredou)** - *5-stem split*. Superior kick/snare separation.
2. **MVSep Percussion Ensemble** - *Top Tier (Online)*.

### 🎸 Bass
1. **SCNet XL** - *Top Definition*. Handles sub-bass best.
2. **BS-Roformer Bass** - *Alternative*. Good if SCNet causes OOM.

---

## 3. Optimal Settings & Configuration

### Overlap & Chunk Size
*Critical for VRAM management and quality.*

**Formula for Max Overlap (Roformer):**
$$ \text{Max Overlap} = \frac{\text{dim\_t} - 1}{100} $$
*Example:* `dim_t` 256 -> Max Overlap 2.55.

**VRAM Tier Guidelines:**
| VRAM | Chunk Size (`dim_t`) | Overlap | TTA | Notes |
|------|---------------------|---------|-----|-------|
| **<6GB** | 112455 (2.55s) | 2 | OFF | "Lite" mode. Prevents OOM. |
| **8GB** | 352800 (8s) | 2-4 | OFF | Standard. Good balance. |
| **12GB+**| 485100 (11s) | 4-8 | ON/OFF | "Quality" mode. Max context. |

### TTA (Test-Time Augmentation)
* **Demucs:** Highly recommended (Shifts=2-10). Linearly improves quality.
* **Roformer/MDX:** **Generally NOT recommended.**
    * Adds 3x processing time.
    * Gain is often negligible (<0.05 SDR).
    * Can smear transients (soft snare) or cause phase issues.
    * *Exception:* Use for extreme restoration if standard pass fails.

---

## 4. Advanced Workflows & "Golden Chains"

These are multi-stage pipelines for maximizing quality.

### 🏆 Golden Chain A: Ultimate Vocal (Studio)
*Goal: 0% Instrumental Bleed, 100% Body.*
1. **Isolation:** `Gabox Voc_Fv4` (High Bleedless).
2. **De-Reverb:** `Anvuew De-Reverb` (Mono). Removes room sound.
3. **Karaoke Split:** `Becruily Karaoke`. Separates backing vocals from lead.
4. **Cleanup:** `Mel-Roformer De-Noise` (optional).

### 🏆 Golden Chain B: Ultimate Instrumental (Audiophile)
*Goal: Full spectrum, no "Roformer Noise" (metallic buzzing).*
1. **Separation:** `Unwa Inst v1e+` (Max Fullness, but noisy).
2. **Phase Fix:**
    * **Target:** Unwa Output.
    * **Source:** `Becruily Vocal` (Clean Phase).
    * **Settings:** 500Hz - 5000Hz, Weight 2.0.
    * *Result:* Unwa's body with Becruily's silence/cleanness.
3. **De-Bleed:** `DeBleed-MelBand-Roformer`. Removes final 5% bleed.

### 🏆 Golden Chain C: Extreme Restoration (Live)
*Goal: Rescue noisy/bootleg recordings.*
1. **Pre-Process:** `Mel-Roformer De-Crowd`. **Overlap 2 ONLY**. Removes audience.
2. **Separation:** `BS-Roformer Viperx`. Aggressive model.
3. **De-Reverb:** On vocals only.
4. **De-Noise:** Final polish.

### 🛠️ Hybrid Frequency Split
*Best for Instrumentals.*
* **Low (<1kHz):** `SCNet XL` (Warm bass, natural decay).
* **High (>1kHz):** `Unwa Inst v1e` (Crisp transients).
* **Crossover:** 1000 Hz with "Feathering" (Soft crossfade) to avoid clicks.

---

## 5. Artifacts & Troubleshooting

| Artifact | Sound | Cause | Solution |
|----------|-------|-------|----------|
| **Metallic Buzzing** | "Swishing", robot noise in high mids. | Roformer phase reconstruction failure. | **Phase Fix** (500-5000Hz). |
| **Muddiness** | "Underwater", lack of punch. | Over-filtering, spectral holes. | **Demudder** (High-pass 100Hz). Check Overlap (too high?). |
| **Ghosting** | Pre-echo, double attacks. | TTA on transient-heavy tracks. | Turn **TTA OFF**. |
| **Hollow Bass** | Weak kick drum. | Using Roformer for <100Hz. | Use **SCNet** or Demucs for bass. |
| **Phasing** | "Flanger" effect. | Phase Fix with bad source or alignment. | Verify source (vocal) is clean. |

---

## 6. Hardware Optimization Guide

### NVIDIA (CUDA)
* **Gold Standard.**
* **FP16:** Use `use_amp: true`. Boosts speed 2x with <0.1% quality loss.
* **Batch Size:** Increase if VRAM allows (e.g., Batch 4 on RTX 4090).

### AMD (DirectML / ROCm)
* **DirectML:** Slower. Strict memory limits.
    * *Fix:* Lower `chunk_size` to 112455 on <8GB cards.
* **ROCm (Linux):** Much faster. Use `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

### Intel (OpenVINO)
* **Optimization:** Convert models to OpenVINO format for huge speedup on ARC/iGPU.

---

## 7. Ensemble Strategies

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **Average** | Weighted mean. Safest. | General purpose, SDR maximization. |
| **Max Spec** | Takes max magnitude per bin. | **Vocals** (retains breath/detail). |
| **Min Spec** | Takes min magnitude per bin. | **Instrumentals** (aggressively cuts vocals). |
| **Phase Fix** | Mag from A + Phase from B. | **Roformer cleanup**. |
| **Freq Split**| Lows from A + Highs from B. | **Hybrid Custom** (SCNet+Roformer). |

**Manual Mixing Rule:**
If mixing stems manually in DAW: Decrease volume by **-3dB** per added stem to match "Average" algorithm logic.
