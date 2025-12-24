"""
Audio file validation and property extraction.

This module provides functionality to validate audio files, extract properties,
and estimate processing requirements.
"""

import logging
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class AudioValidator:
    """
    Validates audio files and extracts properties.
    
    Provides methods to check file format, extract audio properties,
    estimate processing time and VRAM requirements.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    
    def __init__(self):
        """Initialize AudioValidator."""
        logger.debug("AudioValidator initialized")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            dict: Validation result containing:
                - valid (bool): Whether file is valid
                - exists (bool): Whether file exists
                - format_supported (bool): Whether format is supported
                - readable (bool): Whether file can be read
                - reason (str): Explanation if not valid
                
        Raises:
            ValueError: If file_path is empty or None.
            
        Example:
            >>> validator = AudioValidator()
            >>> result = validator.validate_file("song.mp3")
            >>> if result['valid']:
            ...     print("File is valid!")
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        logger.debug(f"Validating audio file: {file_path}")
        
        result = {
            "valid": False,
            "exists": False,
            "format_supported": False,
            "readable": False,
            "reason": ""
        }
        
        try:
            path_obj = Path(file_path)
            
            # Check if file exists
            result["exists"] = path_obj.exists()
            if not result["exists"]:
                result["reason"] = "File does not exist"
                logger.warning(f"File not found: {file_path}")
                return result
            
            # Check if it's a file (not directory)
            if not path_obj.is_file():
                result["reason"] = "Path is not a file"
                logger.warning(f"Not a file: {file_path}")
                return result
            
            # Check format
            ext = path_obj.suffix.lower()
            result["format_supported"] = ext in self.SUPPORTED_FORMATS
            if not result["format_supported"]:
                result["reason"] = f"Unsupported format: {ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                logger.warning(result["reason"])
                return result
            
            # Try to read file
            try:
                # Quick check - just load first second
                y, sr = librosa.load(file_path, sr=None, duration=1.0)
                result["readable"] = True
            except Exception as e:
                result["reason"] = f"Cannot read audio file: {e}"
                logger.error(result["reason"])
                return result
            
            # All checks passed
            result["valid"] = True
            result["reason"] = "File is valid"
            logger.info(f"File validated successfully: {file_path}")
            
        except Exception as e:
            result["reason"] = f"Validation error: {e}"
            logger.error(result["reason"])
        
        return result
    
    def get_audio_properties(self, file_path: str) -> Dict[str, Any]:
        """
        Extract audio file properties.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            dict: Audio properties containing:
                - duration_seconds (float): Duration in seconds
                - sample_rate (int): Sample rate in Hz
                - channels (int): Number of audio channels
                - bit_depth (int|None): Bit depth if available
                - file_size_mb (float): File size in MB
                - format (str): File format
                
        Raises:
            ValueError: If file_path is empty or file is invalid.
            
        Example:
            >>> validator = AudioValidator()
            >>> props = validator.get_audio_properties("song.mp3")
            >>> print(f"Duration: {props['duration_seconds']:.1f}s")
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        # Validate file first
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            raise ValueError(f"Invalid audio file: {validation['reason']}")
        
        logger.debug(f"Extracting properties from: {file_path}")
        
        try:
            path_obj = Path(file_path)
            
            # Get file size
            file_size_mb = round(path_obj.stat().st_size / (1024**2), 2)
            
            # Load audio info (don't load the full audio data)
            info = sf.info(file_path)
            
            properties = {
                "duration_seconds": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "bit_depth": info.subtype_info if hasattr(info, 'subtype_info') else None,
                "file_size_mb": file_size_mb,
                "format": path_obj.suffix.lower()
            }
            
            logger.info(f"Extracted properties: {properties['duration_seconds']}s, {properties['sample_rate']}Hz, {properties['channels']}ch")
            return properties
            
        except Exception as e:
            logger.error(f"Error extracting properties: {e}")
            raise ValueError(f"Cannot extract properties: {e}")
    
    def estimate_processing_time(
        self,
        duration_seconds: float,
        model_speed: str = "medium",
        use_gpu: bool = True
    ) -> Dict[str, float]:
        """
        Estimate processing time for audio separation.
        
        Args:
            duration_seconds: Audio duration in seconds.
            model_speed: Model speed category ("fast", "medium", "slow").
            use_gpu: Whether GPU will be used.
            
        Returns:
            dict: Time estimates containing:
                - estimated_seconds (float): Estimated processing time
                - min_seconds (float): Minimum estimate
                - max_seconds (float): Maximum estimate
                
        Raises:
            ValueError: If duration is negative or model_speed is invalid.
            
        Example:
            >>> validator = AudioValidator()
            >>> estimate = validator.estimate_processing_time(180, "medium", True)
            >>> print(f"Estimated: {estimate['estimated_seconds']:.0f} seconds")
        """
        if duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        
        valid_speeds = {"fast", "medium", "slow"}
        if model_speed not in valid_speeds:
            raise ValueError(f"Invalid model_speed. Must be one of: {valid_speeds}")
        
        logger.debug(f"Estimating processing time for {duration_seconds}s audio")
        
        # Base multipliers (processing time / audio duration)
        # These are rough estimates based on typical model performance
        gpu_multipliers = {
            "fast": 0.5,    # Processes faster than real-time
            "medium": 1.0,  # About real-time
            "slow": 2.0     # Twice as long as audio
        }
        
        cpu_multipliers = {
            "fast": 3.0,
            "medium": 5.0,
            "slow": 10.0
        }
        
        multiplier = gpu_multipliers[model_speed] if use_gpu else cpu_multipliers[model_speed]
        
        estimated = duration_seconds * multiplier
        
        # Add variance (±30%)
        variance = 0.3
        min_time = estimated * (1 - variance)
        max_time = estimated * (1 + variance)
        
        result = {
            "estimated_seconds": round(estimated, 1),
            "min_seconds": round(min_time, 1),
            "max_seconds": round(max_time, 1)
        }
        
        logger.debug(f"Estimated processing time: {result['estimated_seconds']}s")
        return result
    
    def estimate_vram_usage(
        self,
        duration_seconds: float,
        sample_rate: int = 44100,
        model_base_vram: float = 2.0
    ) -> Dict[str, float]:
        """
        Estimate VRAM usage for processing.
        
        Args:
            duration_seconds: Audio duration in seconds.
            sample_rate: Audio sample rate in Hz.
            model_base_vram: Base VRAM requirement of model in GB.
            
        Returns:
            dict: VRAM estimates containing:
                - estimated_gb (float): Estimated VRAM usage
                - min_gb (float): Minimum estimate
                - max_gb (float): Maximum estimate
                
        Raises:
            ValueError: If parameters are invalid.
            
        Example:
            >>> validator = AudioValidator()
            >>> vram = validator.estimate_vram_usage(180, 44100, 6.0)
            >>> print(f"Estimated VRAM: {vram['estimated_gb']:.1f} GB")
        """
        if duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if model_base_vram < 0:
            raise ValueError("Model base VRAM must be non-negative")
        
        logger.debug(f"Estimating VRAM for {duration_seconds}s at {sample_rate}Hz")
        
        # Base VRAM from model
        vram = model_base_vram
        
        # Add overhead for audio data (very rough estimate)
        # ~0.1GB per minute of audio at 44.1kHz
        audio_overhead = (duration_seconds / 60.0) * 0.1 * (sample_rate / 44100.0)
        vram += audio_overhead
        
        # Add 20% buffer for processing overhead
        vram *= 1.2
        
        # Variance (±15%)
        variance = 0.15
        min_vram = vram * (1 - variance)
        max_vram = vram * (1 + variance)
        
        result = {
            "estimated_gb": round(vram, 2),
            "min_gb": round(min_vram, 2),
            "max_gb": round(max_vram, 2)
        }
        
        logger.debug(f"Estimated VRAM: {result['estimated_gb']} GB")
        return result
    
    def validate_batch(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Validate a batch of audio files.
        
        Args:
            file_paths: List of file paths to validate.
            
        Returns:
            dict: Batch validation result containing:
                - total_files (int): Total number of files
                - valid_files (int): Number of valid files
                - invalid_files (int): Number of invalid files
                - results (list): Individual validation results
                - total_duration_seconds (float): Total duration of valid files
                - total_size_mb (float): Total size of valid files
                
        Example:
            >>> validator = AudioValidator()
            >>> result = validator.validate_batch(["song1.mp3", "song2.wav"])
            >>> print(f"{result['valid_files']}/{result['total_files']} files valid")
        """
        logger.info(f"Validating batch of {len(file_paths)} files")
        
        results = []
        valid_count = 0
        total_duration = 0.0
        total_size = 0.0
        
        for file_path in file_paths:
            try:
                validation = self.validate_file(file_path)
                
                if validation["valid"]:
                    valid_count += 1
                    
                    # Get properties for valid files
                    try:
                        props = self.get_audio_properties(file_path)
                        total_duration += props["duration_seconds"]
                        total_size += props["file_size_mb"]
                        
                        validation["properties"] = props
                    except Exception as e:
                        logger.warning(f"Could not get properties for {file_path}: {e}")
                
                results.append({
                    "file": file_path,
                    "validation": validation
                })
                
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                results.append({
                    "file": file_path,
                    "validation": {
                        "valid": False,
                        "reason": f"Validation error: {e}"
                    }
                })
        
        batch_result = {
            "total_files": len(file_paths),
            "valid_files": valid_count,
            "invalid_files": len(file_paths) - valid_count,
            "results": results,
            "total_duration_seconds": round(total_duration, 2),
            "total_size_mb": round(total_size, 2)
        }
        
        logger.info(f"Batch validation complete: {valid_count}/{len(file_paths)} valid")
        return batch_result
    
    def check_compatibility(
        self,
        file_path: str,
        required_sample_rate: Optional[int] = None,
        required_channels: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if audio file meets specific requirements.
        
        Args:
            file_path: Path to audio file.
            required_sample_rate: Required sample rate (None = any).
            required_channels: Required number of channels (None = any).
            
        Returns:
            dict: Compatibility result containing:
                - compatible (bool): Whether file meets requirements
                - current_sample_rate (int): Current sample rate
                - current_channels (int): Current channels
                - needs_resampling (bool): Whether resampling is needed
                - needs_channel_conversion (bool): Whether channel conversion is needed
                - message (str): Explanation
                
        Example:
            >>> validator = AudioValidator()
            >>> result = validator.check_compatibility("song.mp3", required_sample_rate=44100)
            >>> if not result['compatible']:
            ...     print(f"Needs conversion: {result['message']}")
        """
        try:
            props = self.get_audio_properties(file_path)
            
            needs_resampling = False
            needs_channel_conversion = False
            messages = []
            
            # Check sample rate
            if required_sample_rate is not None:
                if props["sample_rate"] != required_sample_rate:
                    needs_resampling = True
                    messages.append(
                        f"Sample rate mismatch: {props['sample_rate']}Hz vs required {required_sample_rate}Hz"
                    )
            
            # Check channels
            if required_channels is not None:
                if props["channels"] != required_channels:
                    needs_channel_conversion = True
                    messages.append(
                        f"Channel mismatch: {props['channels']}ch vs required {required_channels}ch"
                    )
            
            compatible = not (needs_resampling or needs_channel_conversion)
            
            result = {
                "compatible": compatible,
                "current_sample_rate": props["sample_rate"],
                "current_channels": props["channels"],
                "needs_resampling": needs_resampling,
                "needs_channel_conversion": needs_channel_conversion,
                "message": "; ".join(messages) if messages else "File is compatible"
            }
            
            logger.info(f"Compatibility check: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            raise ValueError(f"Cannot check compatibility: {e}")


if __name__ == "__main__":
    # Quick test/demo
    import logging
    logging.basicConfig(level=logging.INFO)
    
    validator = AudioValidator()
    
    print("\n=== Audio Validator Demo ===\n")
    print(f"Supported formats: {', '.join(validator.SUPPORTED_FORMATS)}")
    
    # Demo with a hypothetical file
    print("\nValidation demo (will fail if file doesn't exist):")
    result = validator.validate_file("test_audio.mp3")
    print(f"Valid: {result['valid']}")
    print(f"Reason: {result['reason']}")
    
    # Demo time estimation
    print("\nProcessing time estimates for 3-minute song:")
    for speed in ["fast", "medium", "slow"]:
        gpu_time = validator.estimate_processing_time(180, speed, use_gpu=True)
        cpu_time = validator.estimate_processing_time(180, speed, use_gpu=False)
        print(f"  {speed.capitalize()}:")
        print(f"    GPU: ~{gpu_time['estimated_seconds']:.0f}s")
        print(f"    CPU: ~{cpu_time['estimated_seconds']:.0f}s")
    
    # Demo VRAM estimation
    print("\nVRAM estimates for 3-minute song:")
    for model_vram in [6.0, 8.0, 10.0]:
        vram = validator.estimate_vram_usage(180, 44100, model_vram)
        print(f"  {model_vram}GB model: ~{vram['estimated_gb']:.1f}GB needed")
