"""
Thin wrapper using audio-separator's proven Roformer loading approach.

Based on nomadkaraoke/python-audio-separator's RoformerLoader implementation.
Uses local roformer modules (audio-separator's implementation with skip_final_norm support).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ============================================================================
# LOCAL ROFORMER MODULES (with skip_final_norm support)
# ============================================================================
import torch
import torch.nn as nn

from stemsep.audio.roformer import BSRoformer, MelBandRoformer


class RoformerSeparator:
    """
    Roformer separator using audio-separator's proven loading approach.

    Key insight from audio-separator:
    - Models can be loaded directly from the 'model:' config section
    - BSRoformer if 'freqs_per_bands' present, MelBandRoformer if 'num_bands'
    - Config just needs model params, not full training/inference sections
    """

    def __init__(self, models_dir: Path, device: str = None):
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(__name__)

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"RoformerSeparator initialized with device: {self.device}")

        # Model cache
        self._model_cache = {}

    def _load_config(self, model_id: str) -> Dict[str, Any]:
        """Load model config from YAML file."""
        config_path = self.models_dir / f"{model_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            # Use FullLoader to support !!python/tuple tag in YAML
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def _detect_model_type(self, model_cfg: Dict[str, Any]) -> str:
        """
        Detect model type from config.

        Based on audio-separator's logic:
        - freqs_per_bands -> bs_roformer
        - num_bands -> mel_band_roformer
        """
        if "freqs_per_bands" in model_cfg:
            return "bs_roformer"
        elif "num_bands" in model_cfg:
            return "mel_band_roformer"
        else:
            raise ValueError(
                "Cannot detect model type: missing freqs_per_bands or num_bands"
            )

    def _create_bs_roformer(self, model_cfg: Dict[str, Any]):
        """Create BSRoformer model using local implementation with skip_final_norm support."""
        # Build args based on audio-separator's approach
        model_args = {
            "dim": model_cfg["dim"],
            "depth": model_cfg["depth"],
            "stereo": model_cfg.get("stereo", True),
            "num_stems": model_cfg.get("num_stems", 1),
            "time_transformer_depth": model_cfg.get("time_transformer_depth", 1),
            "freq_transformer_depth": model_cfg.get("freq_transformer_depth", 1),
            "freqs_per_bands": tuple(model_cfg["freqs_per_bands"]),  # Must be tuple
            "dim_head": model_cfg.get("dim_head", 64),
            "heads": model_cfg.get("heads", 8),
            "attn_dropout": model_cfg.get("attn_dropout", 0.0),
            "ff_dropout": model_cfg.get("ff_dropout", 0.0),
            "flash_attn": model_cfg.get("flash_attn", True),
        }

        # Optional params including skip_final_norm (now supported!)
        for key in [
            "stft_n_fft",
            "stft_hop_length",
            "stft_win_length",
            "mask_estimator_depth",
            "dim_freqs_in",
            "stft_normalized",
            "skip_final_norm",
            "skip_connection",
            "mlp_expansion_factor",
            "sage_attention",
            "zero_dc",
            "use_torch_checkpoint",
        ]:
            if key in model_cfg:
                model_args[key] = model_cfg[key]

        self.logger.info(
            f"Creating BSRoformer with skip_final_norm={model_args.get('skip_final_norm', False)}"
        )
        return BSRoformer(**model_args)

    def _create_mel_band_roformer(self, model_cfg: Dict[str, Any]):
        """Create MelBandRoformer model using local implementation with skip_final_norm support."""

        model_args = {
            "dim": model_cfg["dim"],
            "depth": model_cfg["depth"],
            "stereo": model_cfg.get("stereo", True),
            "num_stems": model_cfg.get("num_stems", 1),
            "time_transformer_depth": model_cfg.get("time_transformer_depth", 1),
            "freq_transformer_depth": model_cfg.get("freq_transformer_depth", 1),
            "num_bands": model_cfg["num_bands"],
            "dim_head": model_cfg.get("dim_head", 64),
            "heads": model_cfg.get("heads", 8),
            "attn_dropout": model_cfg.get("attn_dropout", 0.0),
            "ff_dropout": model_cfg.get("ff_dropout", 0.0),
            "flash_attn": model_cfg.get("flash_attn", True),
        }

        # Optional params including skip_final_norm (now supported!)
        for key in [
            "sample_rate",
            "stft_n_fft",
            "stft_hop_length",
            "stft_win_length",
            "stft_normalized",
            "mask_estimator_depth",
            "skip_final_norm",
            "skip_connection",
            "mlp_expansion_factor",
            "sage_attention",
            "zero_dc",
            "use_torch_checkpoint",
        ]:
            if key in model_cfg:
                model_args[key] = model_cfg[key]

        self.logger.info(
            f"Creating MelBandRoformer with skip_final_norm={model_args.get('skip_final_norm', False)}"
        )
        return MelBandRoformer(**model_args)

    def load_model(self, model_id: str) -> Tuple[nn.Module, Dict, str]:
        """
        Load model using audio-separator's proven approach.

        Returns:
            Tuple of (model, config, model_type)
        """
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        config = self._load_config(model_id)

        # Get model section (audio-separator's approach)
        model_cfg = config.get("model", config)

        # Detect model type
        model_type = self._detect_model_type(model_cfg)
        self.logger.info(f"Detected model type: {model_type} for {model_id}")

        # Create model
        if model_type == "bs_roformer":
            model = self._create_bs_roformer(model_cfg)
        else:
            model = self._create_mel_band_roformer(model_cfg)

        # Load checkpoint
        ckpt_path = self.models_dir / f"{model_id}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Handle different checkpoint formats (from audio-separator)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Load state dict with strict=False to handle weight mismatches
        # (some model configs like skip_final_norm aren't supported by ZFTurbo's official code)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            self.logger.warning(
                f"Missing keys when loading weights: {len(result.missing_keys)} keys"
            )
            self.logger.debug(f"Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            self.logger.warning(
                f"Unexpected keys in checkpoint: {len(result.unexpected_keys)} keys"
            )
            self.logger.info(
                f"Unexpected keys (first 10): {result.unexpected_keys[:10]}"
            )
        model.to(self.device)
        model.eval()

        self.logger.info(f"Model {model_id} loaded successfully on {self.device}")

        # Cache
        self._model_cache[model_id] = (model, config, model_type)

        return model, config, model_type

    def separate(
        self,
        audio_path: str,
        model_id: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, str]:
        """
        Separate audio file into stems.

        Args:
            audio_path: Path to input audio file
            model_id: Model ID to use
            output_dir: Directory to save output files
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping stem name to output file path
        """
        import librosa
        import soundfile as sf

        def notify(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        notify(0, "Loading model...")
        model, config, model_type = self.load_model(model_id)

        notify(15, "Loading audio...")

        # Get sample rate from config or default
        model_cfg = config.get("model", config)
        audio_cfg = config.get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", model_cfg.get("sample_rate", 44100))

        # Load audio
        mix, sr = librosa.load(audio_path, sr=sample_rate, mono=False)

        # Normalize channel shape to (C, T)
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)

        expects_stereo = bool(model_cfg.get("stereo", True))

        # Mono -> stereo (if model expects stereo)
        if expects_stereo and mix.shape[0] == 1:
            mix = np.concatenate([mix, mix], axis=0)

        # Stereo/multichannel -> mono (if model expects mono)
        if not expects_stereo and mix.shape[0] > 1:
            mix = np.mean(mix, axis=0, keepdims=True)

        mix_orig = mix.copy()

        notify(25, "Running separation...")

        # Get inference params
        inference_cfg = config.get("inference", {})
        chunk_size = audio_cfg.get(
            "chunk_size", inference_cfg.get("chunk_size", 485100)
        )
        num_overlap = inference_cfg.get("num_overlap", 4)
        batch_size = inference_cfg.get("batch_size", 1)

        # Run demix
        waveforms = self._demix(model, mix, chunk_size, num_overlap, batch_size)

        notify(70, "Computing residual...")

        # Process outputs
        output_waveforms = {}

        # The model outputs a tensor - first stem is the target
        if isinstance(waveforms, np.ndarray):
            if waveforms.ndim == 3:
                # Multiple stems
                output_waveforms["target"] = waveforms[0]
            else:
                output_waveforms["target"] = waveforms

        # Infer stem name from model_id
        model_id_lower = model_id.lower()
        if "inst" in model_id_lower or "instrumental" in model_id_lower:
            stem_name = "instrumental"
            residual_name = "vocals"
        elif "voc" in model_id_lower or "vocal" in model_id_lower:
            stem_name = "vocals"
            residual_name = "instrumental"
        else:
            stem_name = "target"
            residual_name = "residual"

        # Rename and compute residual
        if "target" in output_waveforms:
            output_waveforms[stem_name] = output_waveforms.pop("target")
            output_waveforms[residual_name] = mix_orig - output_waveforms[stem_name]

        notify(85, "Saving outputs...")

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        output_paths = {}

        for name, audio in output_waveforms.items():
            output_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(output_path, audio.T, sample_rate)
            output_paths[name] = output_path
            self.logger.info(f"Saved {name} to {output_path}")

        notify(100, "Complete")

        return output_paths

    def _demix(
        self, model, mix: np.ndarray, chunk_size: int, num_overlap: int, batch_size: int
    ) -> np.ndarray:
        """
        Run separation on audio mix.

        Based on ZFTurbo's demix implementation.
        """
        mix_tensor = torch.tensor(mix, dtype=torch.float32)

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix_tensor.shape[-1]

        # Windowing array
        window = self._get_windowing_array(chunk_size, fade_size)

        # Add padding
        if length_init > 2 * border and border > 0:
            mix_tensor = nn.functional.pad(mix_tensor, (border, border), mode="reflect")

        with torch.cuda.amp.autocast(enabled=True):
            with torch.inference_mode():
                # Initialize result
                num_stems = 1  # Most models output 1 stem
                req_shape = (num_stems,) + mix_tensor.shape
                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)

                i = 0
                batch_data = []
                batch_locations = []

                while i < mix_tensor.shape[1]:
                    # Extract chunk
                    part = mix_tensor[:, i : i + chunk_size].to(self.device)
                    chunk_len = part.shape[-1]

                    # Pad if needed
                    if chunk_len > chunk_size // 2:
                        pad_mode = "reflect"
                    else:
                        pad_mode = "constant"
                    part = nn.functional.pad(
                        part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
                    )

                    batch_data.append(part)
                    batch_locations.append((i, chunk_len))
                    i += step

                    # Process batch
                    if len(batch_data) >= batch_size or i >= mix_tensor.shape[1]:
                        arr = torch.stack(batch_data, dim=0)
                        x = model(arr)

                        # Apply windowing
                        win = window.clone()
                        if i - step == 0:
                            win[:fade_size] = 1
                        elif i >= mix_tensor.shape[1]:
                            win[-fade_size:] = 1

                        for j, (start, seg_len) in enumerate(batch_locations):
                            result[..., start : start + seg_len] += (
                                x[j, ..., :seg_len].cpu() * win[..., :seg_len]
                            )
                            counter[..., start : start + seg_len] += win[..., :seg_len]

                        batch_data.clear()
                        batch_locations.clear()

                # Normalize
                estimated = result / counter
                estimated = estimated.cpu().numpy()
                np.nan_to_num(estimated, copy=False, nan=0.0)

                # Remove padding
                if length_init > 2 * border and border > 0:
                    estimated = estimated[..., border:-border]

        return estimated[0]  # Return first (only) stem

    def _get_windowing_array(self, window_size: int, fade_size: int) -> torch.Tensor:
        """Generate windowing array with linear fade."""
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)

        window = torch.ones(window_size)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window
