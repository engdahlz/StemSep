"""
Audio stem separation engine
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional
import yaml

# Add tuple constructor for safe_load to handle !!python/tuple tags in configs
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

import numpy as np

try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

try:
    import torch
    import torchaudio

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from demucs.api import load_track, separate

    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

from models.architectures.demucs_wrapper import DemucsModel
from models.architectures.mdxnet import MDXNetModel

# Pitch preprocessing for improved high-frequency separation
try:
    from audio.pitch_preprocessor import (
        PRESET_DRUMS_CYMBALS,
        PRESET_SOPRANO_VOCALS,
        restore_pitch,
        shift_for_separation,
    )

    PITCH_PREPROCESS_AVAILABLE = True
except ImportError:
    PITCH_PREPROCESS_AVAILABLE = False

# Simple separator using audio-separator library (proven to work, like UVR5-UI)
from audio.simple_separator import SimpleSeparator


def _get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with linear fade-in at beginning and fade-out at end.

    This is the EXACT implementation from ZFTurbo's Music-Source-Separation-Training.
    Returns a torch.Tensor for proper GPU acceleration and .clone() support.

    Args:
        window_size: Total size of the window (chunk size)
        fade_size: Size of fade-in and fade-out regions (typically chunk_size // 10)

    Returns:
        torch.Tensor of shape (window_size,) with values [0, 1] for fade regions
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


# Overlap Settings (per NotebookLM recommendations)
# Higher overlap = better quality but slower; too high causes muddiness/misalignment
OVERLAP_FAST = 2  # Safest for low VRAM (4GB), fastest
OVERLAP_BALANCED = 4  # Good SDR, reasonable speed
OVERLAP_QUALITY = 8  # Sweet spot, requires dim_t >= 801
OVERLAP_MAX = 16  # Diminishing returns, may cause muddiness


def validate_overlap(overlap: int, dim_t: int, logger=None) -> int:
    """
    Validate overlap setting against dim_t to prevent stem misalignment.

    Per NotebookLM: Max_Overlap = (dim_t - 1) / 100
    Example: dim_t 256 -> max overlap 2
             dim_t 801 -> max overlap 8
             dim_t 1101 -> max overlap 11

    Args:
        overlap: Requested overlap value (2-16)
        dim_t: Segment size / chunk size
        logger: Optional logger for warnings

    Returns:
        Safe overlap value (may be reduced)
    """
    if dim_t <= 0:
        return OVERLAP_BALANCED

    max_safe = (dim_t - 1) // 100
    if max_safe < 1:
        max_safe = 1

    if overlap > max_safe:
        if logger:
            logger.warning(
                f"Overlap {overlap} exceeds max safe {max_safe} for dim_t={dim_t}. "
                f"Reducing to prevent stem misalignment."
            )
        return max_safe

    return overlap


def apply_phase_swap(
    fullness_audio: np.ndarray,
    bleedless_audio: np.ndarray,
    sr: int = 44100,
    low_hz: int = 500,
    high_hz: int = 5000,
    high_freq_weight: float = 0.8,
) -> np.ndarray:
    """
    Apply phase swapping to clean noise from Fullness model output using Bleedless model as reference.

    Uses frequency-based phase transfer matching the UVR/MVSEP Phase Fixer algorithm:
    - Below low_hz: Keep original fullness phase
    - Above high_hz: Use bleedless phase (scaled by high_freq_weight)
    - Between: Blend based on frequency position

    Args:
        fullness_audio: Audio from Fullness model (shape: channels, samples)
        bleedless_audio: Audio from Bleedless model (shape: channels, samples)
        sr: Sample rate
        low_hz: Low frequency cutoff (default 500, aggressive: 100-200)
        high_hz: High frequency cutoff (default 5000, aggressive: 100-500)
        high_freq_weight: Weight for high frequency phase transfer (default 0.8, Becruily benefits from 2.0)

    Returns:
        Cleaned audio array
    """
    if fullness_audio.shape != bleedless_audio.shape:
        # Pad/trim to match lengths
        min_len = min(fullness_audio.shape[-1], bleedless_audio.shape[-1])
        fullness_audio = fullness_audio[..., :min_len]
        bleedless_audio = bleedless_audio[..., :min_len]

    # Ensure 2D (channels, samples)
    if fullness_audio.ndim == 1:
        fullness_audio = fullness_audio[np.newaxis, :]
    if bleedless_audio.ndim == 1:
        bleedless_audio = bleedless_audio[np.newaxis, :]

    result = np.zeros_like(fullness_audio)
    n_fft = 2048
    hop_length = 512

    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    for ch in range(fullness_audio.shape[0]):
        # Compute spectrograms
        full_stft = librosa.stft(fullness_audio[ch], n_fft=n_fft, hop_length=hop_length)
        bleed_stft = librosa.stft(
            bleedless_audio[ch], n_fft=n_fft, hop_length=hop_length
        )

        # Get magnitudes and phases
        full_mag = np.abs(full_stft)
        full_phase = np.angle(full_stft)
        bleed_phase = np.angle(bleed_stft)

        # Create frequency-based blending mask
        # Below low_hz: 0 (use fullness phase)
        # Above high_hz: 1 (use bleedless phase)
        # Between: linear interpolation
        blend_mask = np.zeros(len(freqs))
        for i, freq in enumerate(freqs):
            if freq <= low_hz:
                blend_mask[i] = 0.0
            elif freq >= high_hz:
                blend_mask[i] = high_freq_weight
            else:
                # Linear interpolation
                t = (freq - low_hz) / (high_hz - low_hz)
                blend_mask[i] = t * high_freq_weight

        # Expand mask to match STFT shape (frequencies, time frames)
        blend_mask = blend_mask[:, np.newaxis]

        # Blend phases
        blended_phase = full_phase * (1 - blend_mask) + bleed_phase * blend_mask

        # Reconstruct with fullness magnitude and blended phase
        cleaned_stft = full_mag * np.exp(1j * blended_phase)
        cleaned_audio = librosa.istft(
            cleaned_stft, hop_length=hop_length, length=fullness_audio.shape[-1]
        )

        result[ch] = cleaned_audio

    return result


class SeparationEngine:
    """Engine for performing audio stem separation"""

    # Architectures that use ZFTurbo inference engine (Roformer family + SCNet)
    # Note: MDX23C uses SimpleSeparator's built-in handler, not zfturbo_wrapper
    ZFTURBO_ARCHITECTURES = ["Mel-Roformer", "BS-Roformer", "SCNet"]

    def __init__(self, model_manager=None, gpu_enabled: bool = True):
        self.logger = logging.getLogger("StemSep")
        self.model_manager = model_manager
        self.gpu_enabled = gpu_enabled and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_enabled else "cpu")
        self.progress_callbacks: List[Callable] = []
        
        # Model cache for faster repeated separations
        # Key: model_id, Value: loaded model/separator instance
        self._model_cache: Dict[str, any] = {}
        self._cache_max_size: int = 2  # Keep max 2 models in VRAM
        self._cache_order: List[str] = []  # LRU tracking

        # Initialize SimpleSeparator (uses audio-separator library like UVR5-UI)
        # Use model_manager's models_dir if available, otherwise default
        if model_manager and hasattr(model_manager, "models_dir"):
            models_dir = model_manager.models_dir
        else:
            models_dir = Path.home() / ".stemsep" / "models"
        self.simple_separator = SimpleSeparator(models_dir=str(models_dir))

        self.logger.info(f"SeparationEngine initialized on {self.device}")

    def add_progress_callback(self, callback: Callable):
        """Add a progress callback"""
        self.progress_callbacks.append(callback)

    def _notify_progress(self, progress: float, message: str = ""):
        """Notify progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    # ==================== Model Cache Management ====================
    
    def _get_cached_model(self, model_id: str) -> Optional[any]:
        """Get model from cache if available, updating LRU order."""
        if model_id in self._model_cache:
            # Move to end of LRU list (most recently used)
            if model_id in self._cache_order:
                self._cache_order.remove(model_id)
            self._cache_order.append(model_id)
            self.logger.info(f"Model cache hit: {model_id}")
            return self._model_cache[model_id]
        return None
    
    def _cache_model(self, model_id: str, model: any):
        """Cache model, evicting oldest if at capacity."""
        # Check if already cached
        if model_id in self._model_cache:
            return
        
        # Evict oldest if at capacity
        while len(self._model_cache) >= self._cache_max_size and self._cache_order:
            oldest_id = self._cache_order.pop(0)
            if oldest_id in self._model_cache:
                self.logger.info(f"Evicting model from cache: {oldest_id}")
                del self._model_cache[oldest_id]
                self._cleanup_memory()
        
        # Add to cache
        self._model_cache[model_id] = model
        self._cache_order.append(model_id)
        self.logger.info(f"Cached model: {model_id} (cache size: {len(self._model_cache)})")
    
    def clear_model_cache(self):
        """Clear all cached models and free memory."""
        self._model_cache.clear()
        self._cache_order.clear()
        self._cleanup_memory()
        self.logger.info("Model cache cleared")
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()


    async def separate(
        self,
        file_path: str,
        model_id: str,
        output_dir: str,
        stems: Optional[List[str]] = None,
        device: Optional[str] = None,
        overlap: float = 0.25,
        segment_size: int = 352800,
        tta: bool = False,
        shifts: int = 1,
        progress_callback: Optional[Callable] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> tuple[Dict[str, str], str]:
        """
        Separate audio into stems
        Returns:
            Tuple containing:
            - Dictionary of stem names to output file paths
            - String indicating device used ('CPU' or 'GPU')
        """

        def notify(prog, msg):
            if progress_callback:
                progress_callback(prog, msg)
            self._notify_progress(prog, msg)

        try:
            if check_cancelled and check_cancelled():
                raise asyncio.CancelledError("Job cancelled")

            if not AUDIO_LIBS_AVAILABLE:
                raise ImportError(
                    "Required audio libraries not installed (librosa, soundfile)"
                )

            # Validate input
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")

            # Set device for this separation
            if device:
                separation_device = (
                    torch.device(device) if TORCH_AVAILABLE else self.device
                )
            else:
                separation_device = self.device

            device_name = "GPU" if "cuda" in str(separation_device) else "CPU"
            self.logger.info(
                f"SeparationEngine.separate called for {file_path} with model {model_id} on {device_name}"
            )
            notify(0.0, f"Loading audio file (using {device_name})...")

            # Load audio
            self.logger.info("Loading audio file...")

            if check_cancelled and check_cancelled():
                raise asyncio.CancelledError("Job cancelled")

            # Ensure ffmpeg is available via static_ffmpeg if installed
            try:
                import static_ffmpeg

                static_ffmpeg.add_paths()
            except ImportError:
                pass

            # === AUDIO LOADING - ZFTurbo Official Pattern ===
            # Use librosa.load with target SR directly - this produces float32
            # which is critical for correct demix operation.
            # See: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/inference.py
            TARGET_SR = 44100

            try:
                # Load audio exactly like ZFTurbo's inference.py
                audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=False)
                self.logger.info(
                    f"Loaded audio with librosa: shape={audio.shape}, dtype={audio.dtype}, sr={sr}"
                )
            except Exception as e:
                self.logger.warning(
                    f"librosa.load failed: {e}. Trying with sr=None and resampling."
                )
                # Fallback: load at native SR then resample
                audio, sr = librosa.load(file_path, sr=None, mono=False)
                if sr != TARGET_SR:
                    self.logger.info(f"Resampling audio from {sr} Hz to {TARGET_SR} Hz")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                    sr = TARGET_SR

            # Handle mono -> stereo (ZFTurbo pattern)
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=0)
            if audio.shape[0] == 1:
                # Convert mono to stereo by duplicating channel
                audio = np.concatenate([audio, audio], axis=0)
                self.logger.info(f"Converted mono to stereo: shape={audio.shape}")

            # Ensure float32 dtype (critical for demix!)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                self.logger.info(f"Converted audio to float32")

            notify(10.0, "Audio loaded")

            if check_cancelled and check_cancelled():
                raise asyncio.CancelledError("Job cancelled")

            # Select model based on ID
            model_info = self._get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Unknown model: {model_id}")

            # Apply VRAM Guard
            requested_segment_size = segment_size

            segment_size = self._apply_vram_guard(
                model_info, segment_size, separation_device
            )

            notify(15.0, f"Checking architecture for {model_info['architecture']}...")
            # Determine architecture type FIRST before loading model
            arch = model_info.get("architecture", "").lower()
            # Architectures that use SimpleSeparator (audio-separator library):
            # - Roformer family: uses audio-separator's proven RoformerLoader
            # - MDX23C: uses audio-separator's MDX23C handler (PyTorch-based, not ONNX)
            # - VR: uses audio-separator's VR handler (UVR models)
            # NOTE: Demucs/HTDemucs NOT included - uses legacy path with DemucsModel wrapper
            # NOTE: SCNet NOT included - audio-separator doesn't support it, uses legacy path with our SCNet wrapper
            zfturbo_archs = [
                "mel-roformer",
                "melroformer",
                "bs-roformer",
                "bsroformer",
                "mdx23c",  # MDX23C uses audio-separator's PyTorch loader (not ModelFactory's ONNX loader)
                "roformer",
                "vr",  # VR models (UVR) use audio-separator's VR handler
            ]
            use_zfturbo_engine = any(z in arch for z in zfturbo_archs)
            
            # IMPORTANT: HyperACE needs our custom BSRoformerHyperACE implementation (with SegmModel),
            # NOT audio-separator's BS-Roformer which lacks the HyperACE modules.
            # Force legacy path for HyperACE to use ModelFactory->BSRoformerHyperACE.
            if "hyperace" in arch:
                use_zfturbo_engine = False
                self.logger.info(f"HyperACE detected - using legacy path with ModelFactory")
            print(
                f"*** Architecture: {arch}, use_zfturbo_engine: {use_zfturbo_engine} ***"
            )

            # Only load model for legacy path (ZFTurbo loads its own model internally)
            model = None
            if not use_zfturbo_engine:
                notify(20.0, f"Loading {model_info['architecture']} model...")
                model = await self._load_model(model_id, device=separation_device)
            notify(25.0, "Ready")

            if check_cancelled and check_cancelled():
                raise asyncio.CancelledError("Job cancelled")

            # Read target_instrument from YAML config (to correctly map model output)
            target_instrument = None
            if self.model_manager:
                config_path = self.model_manager.models_dir / f"{model_id}.yaml"
                if config_path.exists():
                    try:
                        import yaml

                        with open(config_path, "r", encoding="utf-8") as f:
                            # Use safe_load for security (per Gemini Code Assist review)
                            yaml_config = yaml.safe_load(f)
                            # target_instrument is under 'training' key
                            if "training" in yaml_config:
                                target_instrument = yaml_config["training"].get(
                                    "target_instrument"
                                )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not read target_instrument from YAML: {e}"
                        )

            # Fallback: Infer target from model name if not specified in YAML
            if not target_instrument:
                model_id_lower = model_id.lower()
                if "inst" in model_id_lower or "instrumental" in model_id_lower:
                    target_instrument = "instrumental"
                elif "voc" in model_id_lower or "vocal" in model_id_lower:
                    target_instrument = "vocals"
                elif "karaoke" in model_id_lower:
                    target_instrument = (
                        "vocals"  # Karaoke models typically output vocals
                    )
                elif "drum" in model_id_lower:
                    target_instrument = "drums"
                elif "bass" in model_id_lower:
                    target_instrument = "bass"
                elif "guitar" in model_id_lower:
                    target_instrument = "guitar"
                else:
                    # Default to vocals for unrecognized models
                    target_instrument = "vocals"
                self.logger.info(
                    f"Inferred target_instrument from model name: {target_instrument}"
                )

            self.logger.info(f"Model {model_id} target_instrument: {target_instrument}")

            # Perform separation
            notify(30.0, "Starting separation...")

            # arch and use_zfturbo_engine already determined above
            self.logger.info(
                f"Architecture: {arch}, using ZFTurbo engine: {use_zfturbo_engine}"
            )

            if use_zfturbo_engine:
                # === USE SIMPLE SEPARATOR (audio-separator library) ===
                # This is the proven implementation used by UVR5-UI
                print(f"*** ENTERING SIMPLE SEPARATOR PATH for {model_id} ***")
                self.logger.info(
                    f"Using SimpleSeparator (audio-separator) for {model_id}"
                )
                notify(35.0, "Running audio-separator...")

                # SimpleSeparator wraps audio-separator library
                # Run in thread pool since audio-separator is synchronous
                loop = asyncio.get_event_loop()

                audio_separator_segment_size = 256
                segment_seconds = max(0.001, float(requested_segment_size) / 44100.0)

                if "roformer" in arch:
                    if overlap is None:
                        overlap_divisor = 8
                        audio_separator_overlap = max(0.05, segment_seconds / float(overlap_divisor))
                    elif overlap < 1:
                        audio_separator_overlap = max(
                            0.05,
                            segment_seconds * (1.0 - float(overlap)),
                        )
                    else:
                        if 2 <= int(overlap) <= 50 and float(overlap).is_integer():
                            audio_separator_overlap = max(
                                0.05, segment_seconds / float(int(overlap))
                            )
                        else:
                            audio_separator_overlap = max(0.05, float(overlap))
                else:
                    if overlap is None:
                        audio_separator_overlap = 8
                    elif overlap < 1:
                        denom = max(1e-6, 1.0 - float(overlap))
                        audio_separator_overlap = int(round(1.0 / denom))
                    else:
                        audio_separator_overlap = int(overlap)

                    audio_separator_overlap = max(2, min(50, audio_separator_overlap))

                result_paths = await loop.run_in_executor(
                    None,
                    lambda: self.simple_separator.separate(
                        audio_path=str(file_path),
                        model_id=model_id,
                        output_dir=str(output_dir),
                        output_format="wav",
                        segment_size=audio_separator_segment_size,
                        overlap=audio_separator_overlap,
                        batch_size=1,
                        progress_callback=lambda pct, msg: notify(35 + pct * 0.45, msg),
                    ),
                )

                device_used = "GPU" if self.gpu_enabled else "CPU"
                notify(100.0, "Complete")
                return result_paths, device_used

            # === LEGACY SEPARATION PATH ===
            # Used for Demucs, MDXNet, and fallback

            # OOM Retry Loop
            max_retries = 2
            separated = None

            for attempt in range(max_retries + 1):
                try:
                    # Use legacy separation
                    separated = await self._perform_separation(
                        model,
                        audio,
                        sr,
                        model_id,
                        stems or model_info["stems"],
                        overlap=overlap,
                        segment_size=segment_size,
                        tta=tta,
                        shifts=shifts,
                        progress_callback=progress_callback,
                        check_cancelled=check_cancelled,
                        target_instrument=target_instrument,
                    )
                    break  # Success!
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and attempt < max_retries:
                        self.logger.warning(
                            f"OOM detected. Retrying with smaller segment size... ({segment_size} -> {segment_size // 2})"
                        )
                        segment_size //= 2
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            import gc

                            gc.collect()
                        notify(
                            30.0,
                            f"OOM Retry {attempt + 1}/{max_retries} (Size: {segment_size})",
                        )
                        continue
                    else:
                        raise e  # Re-raise other errors or if retries exhausted

            notify(80.0, "Separation complete")

            # Save stems
            notify(85.0, "Saving output files...")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if separated is None:
                raise RuntimeError("Separation produced no outputs (internal error)")

            output_files: Dict[str, str] = {}
            for stem_name, stem_audio in separated.items():
                if check_cancelled and check_cancelled():
                    raise asyncio.CancelledError("Job cancelled")

                output_file = output_path / f"{stem_name}.wav"
                sf.write(
                    output_file,
                    stem_audio.T if getattr(stem_audio, "ndim", 1) > 1 else stem_audio,
                    sr,
                )
                output_files[stem_name] = str(output_file)

            notify(100.0, "Complete")
            self.logger.info(f"Separation completed. Output files: {output_files}")

            return output_files, device_name

        except asyncio.CancelledError:
            self.logger.info("Separation cancelled by user")
            notify(0.0, "Cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Separation error: {e}")
            notify(0.0, f"Error: {e}")
            raise

    async def _load_model(self, model_id: str, device=None):
        """Load a model by ID"""
        try:
            # Get model info
            model_info = self._get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")

            # Determine device
            if device is None:
                device = self.device

            self.logger.info(
                f"Loading model {model_id} ({model_info.get('architecture', 'Unknown')}) on {device}"
            )

            # Ensure model is downloaded
            # NOTE: Demucs/HTDemucs models load via torch.hub cache, not our models directory,
            # so skip the installation check for them.
            arch = model_info.get("architecture", "").lower()
            is_demucs = "demucs" in arch or "htdemucs" in arch
            
            if self.model_manager and not is_demucs:
                if not self.model_manager.is_model_installed(model_id):
                    # IMPORTANT: Do not silently substitute a different model.
                    # If the selected model is missing, we attempt exactly one explicit download
                    # (consistent with the UI "Download" action). If it still isn't installed,
                    # we fail fast with a clear message.
                    self.logger.info(
                        f"Model {model_id} not found locally. Downloading..."
                    )
                    self._notify_progress(20.0, f"Downloading {model_info['name']}...")

                    success = await self.model_manager.download_model(model_id)

                    if not success or not self.model_manager.is_model_installed(
                        model_id
                    ):
                        raise FileNotFoundError(
                            f"Model '{model_id}' could not be downloaded/installed. "
                            f"No fallback model will be used. Please install/download the selected model and try again."
                        )

                    self.logger.info(f"Download complete for {model_id}")

                # Get path from installed models or search for it (to handle different extensions like .onnx)
                if model_id in self.model_manager.installed_models:
                    checkpoint_path = self.model_manager.installed_models[
                        model_id
                    ].file_path
                else:
                    # Fallback search
                    candidates = list(
                        self.model_manager.models_dir.glob(f"{model_id}.*")
                    )
                    if candidates:
                        checkpoint_path = candidates[0]
                    else:
                        # Default fallback
                        checkpoint_path = (
                            self.model_manager.models_dir / f"{model_id}.ckpt"
                        )
            else:
                # Fallback if no manager (shouldn't happen in normal app usage)
                checkpoint_path = Path(f"models/{model_id}.ckpt")

            # Get configuration
            # Priority: 1. YAML file, 2. Factory Defaults based on ID/Architecture
            config = {}
            from models.model_factory import ModelFactory

            if self.model_manager:
                config_path = self.model_manager.models_dir / f"{model_id}.yaml"

                # If config missing but URL exists, try to download
                if not config_path.exists() and model_info.get("config_url"):
                    self.logger.info(f"Config missing for {model_id}, downloading...")
                    try:
                        import aiohttp

                        async with aiohttp.ClientSession() as session:
                            async with session.get(model_info["config_url"]) as resp:
                                if resp.status == 200:
                                    text = await resp.text()
                                    config_path.write_text(text, encoding="utf-8")
                                    self.logger.info(
                                        f"Config downloaded to {config_path}"
                                    )
                                else:
                                    self.logger.warning(
                                        f"Config download failed with status {resp.status}"
                                    )
                    except Exception as e:
                        self.logger.warning(f"Failed to download config: {e}")

                if config_path.exists():
                    try:
                        import yaml

                        with open(config_path, "r", encoding="utf-8") as f:
                            # ZFTurbo/pcunwa configs use !!python/tuple which safe_load doesn't support.
                            # Use FullLoader but note: only load configs from trusted sources (our models dir).
                            try:
                                loaded_config = yaml.load(f, Loader=yaml.FullLoader)
                            except Exception:
                                f.seek(0)  # Reset file position
                                loaded_config = yaml.safe_load(f)

                            # Handle config structure (sometimes under 'model' key)
                            if "model" in loaded_config:
                                config = loaded_config["model"]
                            else:
                                config = loaded_config
                            self.logger.info(f"Loaded config from {config_path}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load config from {config_path}: {e}"
                        )

            # If no config found in YAML, try to extract from checkpoint
            if not config:
                self.logger.info(
                    f"No YAML config found, analyzing checkpoint for {model_id}"
                )
                try:
                    from models.checkpoint_analyzer import analyze_checkpoint

                    config = analyze_checkpoint(checkpoint_path)

                    # Auto-save extracted config for future use
                    if config and self.model_manager:
                        import yaml

                        config_to_save = {"model": config.copy()}
                        # Remove non-serializable items
                        if "freqs_per_bands" in config_to_save["model"]:
                            config_to_save["model"]["freqs_per_bands"] = list(
                                config_to_save["model"]["freqs_per_bands"]
                            )
                        config_path.write_text(
                            yaml.dump(config_to_save, default_flow_style=False)
                        )
                        self.logger.info(
                            f"Auto-saved extracted config to {config_path}"
                        )

                    # Override architecture from detected if JSON didn't specify
                    if config.get("architecture") and model_info.get(
                        "architecture"
                    ) in ["Unknown", None, ""]:
                        model_info["architecture"] = config["architecture"]
                except Exception as e:
                    self.logger.warning(f"Failed to analyze checkpoint: {e}")
                    config = ModelFactory.get_model_config(
                        model_id, architecture=model_info.get("architecture")
                    )

            # Inject model path into config for models that load weights during init (like MDX/ONNX)
            config["model_path"] = str(checkpoint_path)

            # Instantiate model
            model = ModelFactory.create_model(model_info["architecture"], config)

            # For PyTorch models, load state dict and move to device
            # MDX-Net (ONNX) and Demucs models load weights in constructor and handle device internally
            if not isinstance(model, (MDXNetModel, DemucsModel)):
                # Load state dict
                # Map location is important to avoid OOM if loading directly to GPU
                state_dict = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )

                # Handle potential state dict mismatch (e.g. 'state_dict' key wrapper)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                # Remap checkpoint keys if using older format
                # NOTE: Skip for HyperACE - its checkpoint already has matching keys and remapping breaks it
                arch_lower = model_info.get("architecture", "").lower()
                skip_remap = "hyperace" in arch_lower
                
                if not skip_remap:
                    try:
                        from models.checkpoint_analyzer import remap_checkpoint_keys

                        state_dict = remap_checkpoint_keys(state_dict, model.state_dict())
                    except Exception as e:
                        self.logger.warning(f"Key remapping failed: {e}")
                else:
                    self.logger.info(f"Skipping key remapping for {arch_lower} (already matching)")

                # Load into model
                try:
                    model.load_state_dict(state_dict, strict=False)
                except RuntimeError as e:
                    # Log short warning instead of full traceback to avoid IPC freeze
                    self.logger.warning(
                        f"State dict mismatch (expected for hybrid configs)."
                    )

                model.to(device)
                model.eval()

            return model
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def _perform_separation(
        self,
        model,
        audio,
        sr,
        model_id: str,
        stems: List[str],
        overlap: float = 0.25,
        segment_size: int = 352800,
        tta: bool = False,
        shifts: int = 1,
        progress_callback: Optional[Callable] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
        target_instrument: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform the actual separation loop.

        Args:
            target_instrument: What the model actually outputs (e.g. 'vocals', 'other', 'instrumental').
                             Used to correctly map model output to stem names.
        """

        def notify(prog, msg):
            if progress_callback:
                progress_callback(prog, msg)
            self._notify_progress(prog, msg)

        if check_cancelled and check_cancelled():
            raise asyncio.CancelledError("Job cancelled")

        # Ensure at least 1 shift pass
        if shifts < 1:
            shifts = 1

        # Log advanced parameters
        self.logger.info(
            f"Separating with params: overlap={overlap}, segment_size={segment_size}, tta={tta}, shifts={shifts}"
        )

        # Ensure audio is (channels, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        channels, length = audio.shape
        original_length = length  # Store for later cropping

        # ZFTurbo demix() parameters
        chunk_size = segment_size

        # Calculate num_overlap from overlap fraction (inverse of ZFTurbo's approach)
        # ZFTurbo uses num_overlap as chunk_size // step, typically 2-8
        # Our overlap param is 0.25 = 75% overlap = num_overlap of 4
        num_overlap = max(2, int(1 / (1 - overlap))) if overlap < 1 else 4

        fade_size = chunk_size // 10  # Linear fade region (10% of chunk)
        step = chunk_size // num_overlap  # Step between chunks
        border = chunk_size - step  # Amount of overlap on each side

        self.logger.info(
            f"ZFTurbo demix params: chunk_size={chunk_size}, num_overlap={num_overlap}, step={step}, fade_size={fade_size}, border={border}"
        )

        # Create linear fade window (not Hanning!)
        windowing_array = _get_windowing_array(chunk_size, fade_size)

        notify(35.0, "Starting inference loop (ZFTurbo demix)...")

        final_result_sum = None

        try:
            with torch.no_grad():
                for shift_idx in range(shifts):
                    if check_cancelled and check_cancelled():
                        raise asyncio.CancelledError("Job cancelled")

                    shift_amount = 0
                    if shift_idx > 0:
                        import random

                        shift_amount = random.randint(0, chunk_size)
                        self.logger.info(
                            f"Applying shift {shift_idx}/{shifts}: {shift_amount} samples"
                        )

                    # Apply shift
                    shifted_audio = np.roll(audio, shift_amount, axis=-1)

                    # ZFTurbo: Add REFLECT padding on BOTH ends to handle edge artifacts
                    if length > 2 * border and border > 0:
                        # Pad with reflection at both start and end
                        padded_audio = np.pad(
                            shifted_audio, ((0, 0), (border, border)), mode="reflect"
                        )
                        added_border = True
                    else:
                        # Audio too short for reflect padding, use constant
                        padded_audio = shifted_audio
                        added_border = False

                    padded_length = padded_audio.shape[1]

                    # Initialize result and weight accumulators (per ZFTurbo)
                    # We'll determine num_stems from first model output
                    current_shift_outputs = None
                    weight_accumulator = None

                    # Process chunks
                    i = 0  # Current position in padded audio
                    chunk_idx = 0
                    total_chunks = (padded_length - chunk_size) // step + 1

                    while i < padded_length:
                        if check_cancelled and check_cancelled():
                            raise asyncio.CancelledError("Job cancelled")

                        # Extract chunk at position i
                        chunk = padded_audio[:, i : i + chunk_size]
                        chunk_len = chunk.shape[-1]

                        # Pad chunk if it's shorter than chunk_size (at the end)
                        if chunk_len < chunk_size:
                            if chunk_len > chunk_size // 2:
                                pad_mode = "reflect"
                            else:
                                pad_mode = "constant"
                            chunk = np.pad(
                                chunk,
                                ((0, 0), (0, chunk_size - chunk_len)),
                                mode=pad_mode,
                            )

                        # Convert to tensor
                        chunk_tensor = (
                            torch.from_numpy(chunk.copy()).float().to(self.device)
                        )
                        chunk_tensor = chunk_tensor.unsqueeze(0)  # Batch dim

                        # Run model
                        outputs = model(chunk_tensor)

                        if outputs.ndim == 3 and outputs.shape[0] == 1:
                            outputs = outputs.unsqueeze(1)

                        if tta:
                            # 3-Pass TTA (ZFTurbo method):
                            # 1. Original (already computed above)
                            # 2. Stereo-inverted (L↔R swap)
                            # 3. Phase-inverted (polarity flip)

                            # Pass 2: Stereo swap (L↔R)
                            if chunk_tensor.shape[1] >= 2:
                                chunk_stereo_swap = torch.stack(
                                    [chunk_tensor[:, 1, :], chunk_tensor[:, 0, :]],
                                    dim=1,
                                )
                            else:
                                chunk_stereo_swap = chunk_tensor
                            outputs_stereo = model(chunk_stereo_swap)
                            if outputs_stereo.ndim == 3:
                                outputs_stereo = outputs_stereo.unsqueeze(1)
                            if outputs_stereo.shape[2] >= 2:
                                outputs_stereo = torch.stack(
                                    [
                                        outputs_stereo[:, :, 1, :],
                                        outputs_stereo[:, :, 0, :],
                                    ],
                                    dim=2,
                                )

                            # Pass 3: Phase invert (polarity flip)
                            chunk_tensor_inv = -chunk_tensor
                            outputs_inv = model(chunk_tensor_inv)
                            if outputs_inv.ndim == 3:
                                outputs_inv = outputs_inv.unsqueeze(1)
                            outputs_inv = -outputs_inv

                            # Average all 3 passes
                            outputs = (outputs + outputs_stereo + outputs_inv) / 3.0

                        outputs = (
                            outputs.squeeze(0).cpu().numpy()
                        )  # (stems, channels, samples)

                        # Initialize buffers if first chunk
                        if current_shift_outputs is None:
                            num_stems = outputs.shape[0]
                            current_shift_outputs = np.zeros(
                                (num_stems, channels, padded_length)
                            )
                            weight_accumulator = np.zeros(
                                (num_stems, channels, padded_length)
                            )

                        # ZFTurbo windowing: use .clone() to avoid in-place modification issues
                        window = windowing_array.clone()
                        if i == 0:
                            # First chunk: No fade-in (set fade region to 1)
                            window[:fade_size] = 1.0
                        if i + step >= padded_length:
                            # Last chunk: No fade-out (set fade region to 1)
                            window[-fade_size:] = 1.0

                        # Convert window to numpy for accumulation
                        win_np = window.numpy()

                        # Accumulate with window weighting (ZFTurbo method)
                        for s in range(outputs.shape[0]):
                            for c in range(channels):
                                current_shift_outputs[s, c, i : i + chunk_len] += (
                                    outputs[s, c, :chunk_len] * win_np[:chunk_len]
                                )
                                weight_accumulator[s, c, i : i + chunk_len] += win_np[
                                    :chunk_len
                                ]

                        # Update progress
                        total_progress_range = 55.0
                        shift_progress_chunk = total_progress_range / shifts
                        base_shift_progress = 35.0 + (shift_idx * shift_progress_chunk)

                        prog = (
                            base_shift_progress
                            + (chunk_idx / max(1, total_chunks)) * shift_progress_chunk
                        )

                        if chunk_idx % 10 == 0:
                            notify(
                                prog,
                                f"Processing chunk {chunk_idx + 1}/{total_chunks} (Shift {shift_idx + 1}/{shifts})",
                            )

                        # Move to next chunk position
                        i += step
                        chunk_idx += 1

                    # Normalize by accumulated weights (ZFTurbo: result / counter)
                    # Avoid division by zero
                    weight_accumulator = np.maximum(weight_accumulator, 1e-8)
                    current_shift_outputs = current_shift_outputs / weight_accumulator

                    # Handle NaN values that might occur
                    np.nan_to_num(current_shift_outputs, copy=False, nan=0.0)

                    # ZFTurbo: Remove the reflect padding we added at the start
                    if added_border and border > 0:
                        current_shift_outputs = current_shift_outputs[
                            :, :, border:-border
                        ]

                    # Crop back to original length (in case of rounding)
                    current_shift_outputs = current_shift_outputs[
                        :, :, :original_length
                    ]

                    # Unshift
                    current_shift_outputs = np.roll(
                        current_shift_outputs, -shift_amount, axis=-1
                    )

                    # Accumulate to final result
                    if final_result_sum is None:
                        final_result_sum = current_shift_outputs
                    else:
                        final_result_sum += current_shift_outputs

                # Average over shifts
                final_outputs = final_result_sum / shifts

                notify(90.0, "Inference complete")

                # Map to dict
                separated = {}

                if final_outputs.shape[0] == 1:
                    # Single stem model - need to determine what it outputs and compute residual
                    model_output = final_outputs[0]  # Shape: (channels, samples)

                    # Normalize target_instrument to lowercase for comparison
                    target = (target_instrument or "").lower()

                    # Determine which stem the model actually outputs
                    # "other" and "instrumental" are equivalent
                    if target in ["vocals", "vocal"]:
                        # Model outputs vocals
                        output_stem = "vocals"
                        residual_stem = "instrumental"
                    elif target in ["other", "instrumental", "instrument", "inst", "karaoke"]:
                        # Model outputs instrumental/other
                        output_stem = "instrumental"
                        residual_stem = "vocals"
                    else:
                        # Fallback: use first stem as output
                        output_stem = stems[0] if stems else "vocals"
                        residual_stem = stems[1] if len(stems) > 1 else None
                        self.logger.warning(
                            f"Unknown target_instrument '{target}', using stems order"
                        )

                    self.logger.info(
                        f"Stem mapping: model outputs '{output_stem}', residual is '{residual_stem}'"
                    )

                    # Assign model output
                    separated[output_stem] = model_output

                    # Compute residual if needed
                    if residual_stem and len(stems) > 1:
                        residual = audio - model_output
                        separated[residual_stem] = residual
                else:
                    for i, stem_name in enumerate(stems):
                        if i < final_outputs.shape[0]:
                            separated[stem_name] = final_outputs[i]

                return separated

        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback

            with open("error_traceback.txt", "w") as f:
                f.write(traceback.format_exc())
            self.logger.error(f"Inference failed: {e}")
            raise

    def _get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information"""
        if self.model_manager:
            model = self.model_manager.get_model(model_id)
            if model:
                return model.to_dict()

        # Fallback for testing/legacy
        models = {
            "bs_roformer_2025_07": {
                "architecture": "BS-Roformer",
                "stems": ["vocals", "instrumental"],
            },
            "mdx23c_8kfft_hq": {
                "architecture": "MDX23C",
                "stems": ["vocals", "instrumental"],
            },
            "scnet_xl": {"architecture": "SCNet", "stems": ["vocals", "instrumental"]},
        }
        return models.get(model_id)

    def _apply_vram_guard(self, model_info, segment_size, device):
        """Check VRAM and adjust segment size using safe presets"""
        if not model_info or not str(device).startswith("cuda") or not TORCH_AVAILABLE:
            return segment_size

        try:
            # Get device index
            idx = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(idx)
            total_mem_gb = props.total_memory / (1024**3)

            # Defined safe chunk sizes from documentation
            CHUNK_11S = 485100  # Best quality for Roformers
            CHUNK_8S = 352800  # Standard balance
            CHUNK_2_5S = 112455  # Low VRAM fallback

            # Determine safe limit based on VRAM tiers
            if total_mem_gb >= 10.0:
                # Ultra High VRAM: Can handle 11s+ chunks
                safe_limit = CHUNK_11S
                tier = "Ultra (>=10GB)"
            elif total_mem_gb >= 8.0:
                # High VRAM: 11s chunks
                safe_limit = CHUNK_11S
                tier = "High (8-10GB)"
            elif total_mem_gb >= 5.0:
                # Medium VRAM: Standard 8s
                safe_limit = CHUNK_8S
                tier = "Medium (5-8GB)"
            elif total_mem_gb >= 4.0:
                # 4GB VRAM: Use 2.55s chunks (max for 4GB AMD/Intel)
                safe_limit = CHUNK_2_5S
                tier = "Low (4-5GB)"
            else:
                # Very Low VRAM (<4GB): Force smallest chunks
                safe_limit = CHUNK_2_5S // 2  # ~1.3s chunks
                tier = "Very Low (<4GB)"

            # Check if current segment_size exceeds the safe limit for this hardware
            if segment_size > safe_limit:
                self.logger.warning(
                    f"VRAM Guard ({tier}): Reducing segment_size from {segment_size} to {safe_limit} to prevent OOM."
                )
                return safe_limit

            # If we have extra VRAM and using default small chunk, maybe boost it?
            # (Only if user didn't explicitly set a small chunk, but hard to know here.
            # Let's assume if they passed 352800/485100 they know what they are doing,
            # but if it's some random small number we might leave it.)

            return segment_size

        except Exception as e:
            self.logger.warning(f"VRAM Guard check failed: {e}")
            return segment_size

    def validate_audio_file(self, file_path: str) -> Dict:
        """Validate and get info about an audio file"""
        try:
            if not Path(file_path).exists():
                return {"valid": False, "error": "File not found"}

            if not AUDIO_LIBS_AVAILABLE:
                return {"valid": False, "error": "Audio libraries not installed"}

            # Use librosa's audio info methods without loading the full file
            # This is much faster for large files

            # Try soundfile first (fastest method)
            if sf is not None:
                try:
                    self.logger.info(
                        f"Attempting validation with soundfile for {file_path}"
                    )
                    info = sf.info(file_path)
                    self.logger.info("Soundfile validation successful")
                    return {
                        "valid": True,
                        "sample_rate": info.samplerate,
                        "duration": info.duration,
                        "channels": info.channels,
                    }
                except Exception as e:
                    self.logger.warning(
                        f"Soundfile validation failed: {e}. Falling back to librosa."
                    )
                    # If soundfile fails, fall through to librosa method
                    pass

            # Fallback to librosa (slower but more compatible)
            # Note: We still load a small sample to verify the file can be read
            self.logger.info(f"Attempting validation with librosa for {file_path}")
            y, sr = librosa.load(
                file_path, sr=None, duration=1.0
            )  # Only load 1 second for validation
            self.logger.info("Librosa load successful")
            # Use path parameter (preferred in current librosa, filename is deprecated)
            # This may still load the file internally depending on format, but is more efficient than passing y=
            try:
                duration = librosa.get_duration(path=file_path)
            except TypeError:
                # Fallback for older librosa versions that only support filename
                duration = librosa.get_duration(filename=file_path)

            self.logger.info("Librosa duration check successful")
            return {
                "valid": True,
                "sample_rate": sr,
                "duration": duration,
                "channels": 1 if y.ndim == 1 else y.shape[0],
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


class SeparationJob:
    """Represents a separation job"""

    def __init__(
        self,
        job_id: str,
        file_path: str,
        model_id: str,
        output_dir: str,
        stems: List[str] = None,
        progress_callback: Callable = None,
        device: str = None,
        invert: bool = False,
        normalize: bool = False,
        bit_depth: str = "16",
        on_complete: Optional[Callable] = None,
    ):
        self.id = job_id
        self.file_path = file_path
        self.model_id = model_id

        # Temp & Commit Logic
        self.final_output_dir = output_dir
        # Create temp dir
        self.temp_dir = tempfile.mkdtemp(prefix=f"stemsep_{self.id}_")
        # Override output_dir to use temp for processing
        self.output_dir = self.temp_dir

        self.stems = stems
        self.progress_callback = progress_callback
        self.on_complete = on_complete
        self.device = device
        self.invert = invert
        self.normalize = normalize
        self.bit_depth = bit_depth
        self.status = "pending"
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.output_files = {}
        self.error = None
        self.actual_device = None
        self.shifts = 1
        self.bitrate = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "file_name": Path(self.file_path).name,
            "file_path": self.file_path,
            "model": self.model_id,
            "status": self.status,
            "progress": self.progress,
            "output_files": list(self.output_files.values()),
            "device": self.actual_device,
        }
