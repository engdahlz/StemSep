"""
Pitch Shift Preprocessing for Audio Separation

Based on the "Soprano" or Pitch Shift trick from UVR/MVSEP community:
- Shift pitch down 2-6 semitones before separation
- This helps AI models detect high-frequency sounds (soprano vocals, cymbals)
- Shift back up after separation

Best for: High-pitched vocals, drums/cymbals, hard-panned mixes, artifact reduction
"""

import numpy as np
import logging
from typing import Tuple, Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


def shift_for_separation(audio: np.ndarray, sr: int, semitones: int = -4) -> Tuple[np.ndarray, int]:
    """
    Shift pitch down to improve high-frequency separation.
    
    Args:
        audio: Audio array (channels, samples) or (samples,)
        sr: Sample rate
        semitones: Semitones to shift (negative = down). Recommended: -2 to -6
        
    Returns:
        Tuple of (shifted_audio, original_semitones) for later restoration
    """
    if not LIBROSA_AVAILABLE:
        logger.warning("Librosa not available, skipping pitch shift")
        return audio, 0
    
    if semitones == 0:
        return audio, 0
    
    # Ensure 2D
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    
    shifted = np.zeros_like(audio)
    
    for ch in range(audio.shape[0]):
        shifted[ch] = librosa.effects.pitch_shift(
            audio[ch], 
            sr=sr, 
            n_steps=semitones,
            res_type='kaiser_fast'  # Faster than default
        )
    
    if was_1d:
        shifted = shifted[0]
    
    logger.info(f"Shifted audio by {semitones} semitones for improved high-freq detection")
    return shifted, -semitones  # Return inverse for restoration


def restore_pitch(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    """
    Restore pitch after separation.
    
    Args:
        audio: Separated audio array
        sr: Sample rate
        semitones: Semitones to shift back (positive = up)
        
    Returns:
        Pitch-restored audio
    """
    if not LIBROSA_AVAILABLE or semitones == 0:
        return audio
    
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    
    restored = np.zeros_like(audio)
    
    for ch in range(audio.shape[0]):
        restored[ch] = librosa.effects.pitch_shift(
            audio[ch],
            sr=sr,
            n_steps=semitones,
            res_type='kaiser_fast'
        )
    
    if was_1d:
        restored = restored[0]
    
    logger.info(f"Restored pitch by {semitones} semitones")
    return restored


def lossless_slowdown(audio: np.ndarray, sr: int, factor: float = 0.75) -> Tuple[np.ndarray, int]:
    """
    Alternative "lossless" method: Change sample rate to slow audio without resampling.
    
    Per NotebookLM: This shifts high frequencies down into model's robust range.
    
    Args:
        audio: Audio array
        sr: Original sample rate  
        factor: Slowdown factor (0.75 = 75% speed)
        
    Returns:
        Tuple of (audio, modified_sample_rate) - note: audio is unchanged,
        but the returned sr should be used for processing
    """
    new_sr = int(sr * factor)
    logger.info(f"Lossless slowdown: treating audio as {new_sr}Hz instead of {sr}Hz (factor={factor})")
    return audio, new_sr


def restore_speed(audio: np.ndarray, original_sr: int, processed_sr: int) -> np.ndarray:
    """
    Restore audio to original speed by resampling.
    
    Args:
        audio: Processed audio at modified sample rate
        original_sr: Original sample rate
        processed_sr: Sample rate used during processing
        
    Returns:
        Resampled audio at original speed
    """
    if not LIBROSA_AVAILABLE or original_sr == processed_sr:
        return audio
    
    was_1d = audio.ndim == 1
    if was_1d:
        audio = audio[np.newaxis, :]
    
    restored = np.zeros((audio.shape[0], int(audio.shape[1] * original_sr / processed_sr)))
    
    for ch in range(audio.shape[0]):
        restored[ch] = librosa.resample(
            audio[ch],
            orig_sr=processed_sr,
            target_sr=original_sr,
            res_type='soxr_hq'
        )
    
    if was_1d:
        restored = restored[0]
    
    logger.info(f"Restored speed from {processed_sr}Hz to {original_sr}Hz")
    return restored


# Recommended presets
PRESET_SOPRANO_VOCALS = {"semitones": -4, "description": "High-pitched vocals"}
PRESET_DRUMS_CYMBALS = {"semitones": -2, "description": "Hi-hats, cymbals, snare highs"}
PRESET_HARD_PANNED = {"semitones": -4, "description": "1970s hard-panned mixes"}
PRESET_ARTIFACT_FIX = {"semitones": [-2, -4], "description": "Multi-pass for 'ah ha hah' artifacts"}
