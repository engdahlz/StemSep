"""
Stem preview functionality for listening to separated audio before saving.

This module provides functionality to load, play, and preview separated stems,
compare them with original audio, and perform quick quality checks.
"""

import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

logger = logging.getLogger(__name__)


class StemPreview:
    """
    Manages preview of separated audio stems.
    
    Provides functionality to load stems, generate preview clips,
    and prepare stems for playback without full loading.
    """
    
    def __init__(self):
        """Initialize StemPreview."""
        logger.debug("StemPreview initialized")
    
    def load_stem(self, file_path: str) -> Dict[str, Any]:
        """
        Load a stem audio file.
        
        Args:
            file_path: Path to stem audio file.
            
        Returns:
            dict: Stem data containing:
                - audio (np.ndarray): Audio data
                - sample_rate (int): Sample rate
                - duration_seconds (float): Duration
                - channels (int): Number of channels
                - file_path (str): Original file path
                
        Raises:
            ValueError: If file_path is empty or file cannot be loaded.
            
        Example:
            >>> preview = StemPreview()
            >>> stem = preview.load_stem("vocals.wav")
            >>> print(f"Loaded {stem['duration_seconds']:.1f}s of audio")
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        logger.debug(f"Loading stem: {file_path}")
        
        try:
            audio, sample_rate = sf.read(file_path, dtype='float32')
            
            # Handle mono vs stereo
            if audio.ndim == 1:
                channels = 1
                duration = len(audio) / sample_rate
            else:
                channels = audio.shape[1]
                duration = audio.shape[0] / sample_rate
            
            stem_data = {
                "audio": audio,
                "sample_rate": sample_rate,
                "duration_seconds": round(duration, 2),
                "channels": channels,
                "file_path": file_path
            }
            
            logger.info(f"Loaded stem: {duration:.1f}s, {sample_rate}Hz, {channels}ch")
            return stem_data
            
        except Exception as e:
            logger.error(f"Failed to load stem: {e}")
            raise ValueError(f"Cannot load stem: {e}")
    
    def generate_preview_clip(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float = 0.0,
        duration: float = 10.0
    ) -> np.ndarray:
        """
        Generate a preview clip from audio data.
        
        Args:
            audio: Audio data array.
            sample_rate: Sample rate in Hz.
            start_time: Start time in seconds.
            duration: Duration of clip in seconds.
            
        Returns:
            np.ndarray: Preview clip audio data.
            
        Raises:
            ValueError: If parameters are invalid.
            
        Example:
            >>> preview = StemPreview()
            >>> stem = preview.load_stem("vocals.wav")
            >>> clip = preview.generate_preview_clip(
            ...     stem['audio'],
            ...     stem['sample_rate'],
            ...     start_time=30.0,
            ...     duration=10.0
            ... )
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if start_time < 0:
            raise ValueError("Start time must be non-negative")
        if duration <= 0:
            raise ValueError("Duration must be positive")
        
        logger.debug(f"Generating preview clip: start={start_time}s, duration={duration}s")
        
        # Calculate sample positions
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + duration) * sample_rate)
        
        # Handle mono vs stereo
        if audio.ndim == 1:
            total_samples = len(audio)
        else:
            total_samples = audio.shape[0]
        
        # Clamp to audio bounds
        start_sample = max(0, min(start_sample, total_samples - 1))
        end_sample = max(start_sample + 1, min(end_sample, total_samples))
        
        # Extract clip
        if audio.ndim == 1:
            clip = audio[start_sample:end_sample]
        else:
            clip = audio[start_sample:end_sample, :]
        
        logger.debug(f"Generated clip: {len(clip)} samples")
        return clip
    
    def save_preview_clip(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str
    ) -> bool:
        """
        Save a preview clip to file.
        
        Args:
            audio: Audio data to save.
            sample_rate: Sample rate in Hz.
            output_path: Where to save the clip.
            
        Returns:
            bool: True if successful, False otherwise.
            
        Example:
            >>> preview = StemPreview()
            >>> clip = preview.generate_preview_clip(audio, sr)
            >>> preview.save_preview_clip(clip, sr, "preview.wav")
        """
        try:
            sf.write(output_path, audio, sample_rate)
            logger.info(f"Saved preview clip: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save preview: {e}")
            return False
    
    def compare_stems(
        self,
        stem_files: Dict[str, str],
        start_time: float = 0.0,
        duration: float = 10.0
    ) -> Dict[str, Any]:
        """
        Compare multiple stems by generating aligned preview clips.
        
        Args:
            stem_files: Dictionary mapping stem names to file paths.
            start_time: Start time for comparison clips.
            duration: Duration of comparison clips.
            
        Returns:
            dict: Comparison data with clips for each stem.
            
        Example:
            >>> preview = StemPreview()
            >>> stems = {
            ...     "vocals": "vocals.wav",
            ...     "instrumental": "instrumental.wav"
            ... }
            >>> comparison = preview.compare_stems(stems, start_time=30.0)
        """
        logger.info(f"Comparing {len(stem_files)} stems")
        
        comparison = {
            "stems": {},
            "sample_rate": None,
            "start_time": start_time,
            "duration": duration
        }
        
        for stem_name, file_path in stem_files.items():
            try:
                # Load stem
                stem_data = self.load_stem(file_path)
                
                # Set sample rate from first stem
                if comparison["sample_rate"] is None:
                    comparison["sample_rate"] = stem_data["sample_rate"]
                
                # Generate preview clip
                clip = self.generate_preview_clip(
                    stem_data["audio"],
                    stem_data["sample_rate"],
                    start_time,
                    duration
                )
                
                comparison["stems"][stem_name] = {
                    "clip": clip,
                    "file_path": file_path,
                    "duration": stem_data["duration_seconds"],
                    "channels": stem_data["channels"]
                }
                
                logger.debug(f"Added {stem_name} to comparison")
                
            except Exception as e:
                logger.error(f"Failed to add {stem_name} to comparison: {e}")
                comparison["stems"][stem_name] = {
                    "error": str(e)
                }
        
        return comparison
    
    def analyze_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Perform basic quality analysis on audio.
        
        Args:
            audio: Audio data to analyze.
            
        Returns:
            dict: Quality metrics containing:
                - rms_level (float): RMS level (0-1)
                - peak_level (float): Peak level (0-1)
                - dynamic_range_db (float): Dynamic range in dB
                - clipping_detected (bool): Whether clipping is present
                
        Example:
            >>> preview = StemPreview()
            >>> stem = preview.load_stem("vocals.wav")
            >>> quality = preview.analyze_quality(stem['audio'])
            >>> if quality['clipping_detected']:
            ...     print("Warning: Clipping detected!")
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        
        logger.debug("Analyzing audio quality")
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio**2))
        
        # Calculate peak level
        peak = np.max(np.abs(audio))
        
        # Calculate dynamic range (in dB)
        if rms > 0:
            dynamic_range = 20 * np.log10(peak / rms)
        else:
            dynamic_range = 0.0
        
        # Check for clipping (samples at or near ±1.0)
        clipping_threshold = 0.99
        clipping_detected = np.any(np.abs(audio) >= clipping_threshold)
        
        metrics = {
            "rms_level": round(float(rms), 4),
            "peak_level": round(float(peak), 4),
            "dynamic_range_db": round(float(dynamic_range), 2),
            "clipping_detected": bool(clipping_detected)
        }
        
        logger.debug(f"Quality metrics: RMS={metrics['rms_level']:.4f}, Peak={metrics['peak_level']:.4f}")
        return metrics
    
    def get_waveform_data(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate downsampled waveform data for visualization.
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate in Hz.
            num_points: Number of points for visualization.
            
        Returns:
            dict: Waveform data containing:
                - time (np.ndarray): Time points
                - amplitude (np.ndarray): Amplitude values
                - sample_rate (int): Original sample rate
                
        Example:
            >>> preview = StemPreview()
            >>> stem = preview.load_stem("vocals.wav")
            >>> waveform = preview.get_waveform_data(stem['audio'], stem['sample_rate'])
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        if num_points <= 0:
            raise ValueError("Number of points must be positive")
        
        logger.debug(f"Generating waveform with {num_points} points")
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
        
        # Downsample for visualization
        total_samples = len(audio_mono)
        if total_samples > num_points:
            # Take every nth sample
            step = total_samples // num_points
            downsampled = audio_mono[::step][:num_points]
        else:
            downsampled = audio_mono
        
        # Generate time axis
        duration = total_samples / sample_rate
        time = np.linspace(0, duration, len(downsampled))
        
        return {
            "time": time,
            "amplitude": downsampled,
            "sample_rate": sample_rate
        }
    
    def mix_stems(
        self,
        stem_data: Dict[str, Dict[str, Any]],
        mix_levels: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Mix multiple stems together with optional level control.
        
        Args:
            stem_data: Dictionary of loaded stem data (from load_stem).
            mix_levels: Optional dictionary of gain levels (0-1) per stem.
            
        Returns:
            np.ndarray: Mixed audio.
            
        Raises:
            ValueError: If stems have incompatible properties.
            
        Example:
            >>> preview = StemPreview()
            >>> vocals = preview.load_stem("vocals.wav")
            >>> inst = preview.load_stem("instrumental.wav")
            >>> mixed = preview.mix_stems({
            ...     "vocals": vocals,
            ...     "instrumental": inst
            ... }, mix_levels={"vocals": 0.8, "instrumental": 1.0})
        """
        if not stem_data:
            raise ValueError("No stems provided for mixing")
        
        logger.info(f"Mixing {len(stem_data)} stems")
        
        # Get reference properties from first stem
        first_stem = next(iter(stem_data.values()))
        ref_sr = first_stem["sample_rate"]
        ref_shape = first_stem["audio"].shape
        
        # Validate all stems have same properties
        for name, stem in stem_data.items():
            if stem["sample_rate"] != ref_sr:
                raise ValueError(f"Stem '{name}' has different sample rate: {stem['sample_rate']} vs {ref_sr}")
            if stem["audio"].shape != ref_shape:
                raise ValueError(f"Stem '{name}' has different shape: {stem['audio'].shape} vs {ref_shape}")
        
        # Initialize mix with zeros
        mixed = np.zeros_like(first_stem["audio"])
        
        # Add each stem with its gain level
        for name, stem in stem_data.items():
            gain = 1.0
            if mix_levels and name in mix_levels:
                gain = max(0.0, min(1.0, mix_levels[name]))  # Clamp to 0-1
            
            mixed += stem["audio"] * gain
            logger.debug(f"Added {name} with gain {gain}")
        
        # Normalize if clipping
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
            logger.warning(f"Mixed audio clipped, normalized by {peak:.2f}")
        
        logger.info("Mixing complete")
        return mixed
    
    def get_stem_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic info about a stem without fully loading it.
        
        Args:
            file_path: Path to stem file.
            
        Returns:
            dict: Stem info containing:
                - duration_seconds (float): Duration
                - sample_rate (int): Sample rate
                - channels (int): Number of channels
                - file_size_mb (float): File size in MB
                - format (str): File format
                
        Example:
            >>> preview = StemPreview()
            >>> info = preview.get_stem_info("vocals.wav")
            >>> print(f"Duration: {info['duration_seconds']}s")
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        try:
            path_obj = Path(file_path)
            
            # Get file size
            file_size_mb = round(path_obj.stat().st_size / (1024**2), 2)
            
            # Get audio info without loading full data
            info = sf.info(file_path)
            
            return {
                "duration_seconds": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size_mb": file_size_mb,
                "format": path_obj.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stem info: {e}")
            raise ValueError(f"Cannot get stem info: {e}")


if __name__ == "__main__":
    # Quick test/demo
    logging.basicConfig(level=logging.INFO)
    
    preview = StemPreview()
    
    print("\n=== Stem Preview Demo ===\n")
    print("This module provides functionality to:")
    print("  - Load and preview separated audio stems")
    print("  - Generate preview clips from specific time ranges")
    print("  - Compare multiple stems side by side")
    print("  - Analyze audio quality (RMS, peak, clipping)")
    print("  - Generate waveform data for visualization")
    print("  - Mix stems together with level control")
    print("\nAll methods include comprehensive error handling and logging.")
