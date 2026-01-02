"""
Simple audio separator using audio-separator library.
Based on UVR5-UI implementation - proven to work.

Usage:
    separator = SimpleSeparator(models_dir="D:/StemSep Models")
    output_files = separator.separate(
        audio_path="input.wav",
        model_filename="melband_roformer_inst_v1e_plus.ckpt",
        output_dir="./outputs"
    )
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import torch
import yaml

# Always prefer the vendored audio_separator library (StemSepApp/dev_vendor/audio_separator),
# so we can patch/extend behavior (e.g. detailed progress callbacks) without relying on the
# user's globally installed package version.
#
# NOTE: After moving the code under the `stemsep` namespace, this module now lives at:
#   StemSepApp/src/stemsep/audio/simple_separator.py
# So the repo root is 4 parents up from this file.
_repo_root = Path(__file__).resolve().parents[4]
_vendor_root = _repo_root / "StemSepApp" / "dev_vendor"
_vendored_audio_separator = _vendor_root / "audio_separator"

if _vendored_audio_separator.exists():
    vendor_root_str = str(_vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

from audio_separator.separator import Separator


class SimpleSeparator:
    """
    Simple separation wrapper using audio-separator library.
    Based on UVR5-UI's proven implementation.
    """

    # Model filename mappings (our model_id -> audio-separator filename)
    MODEL_FILENAMES = {
        # Unwa Roformer models
        "unwa-inst-v1e-plus": "melband_roformer_inst_v1e_plus.ckpt",
        "unwa-inst-v1e": "melband_roformer_inst_v1e.ckpt",
        "unwa-inst-v1": "melband_roformer_inst_v1.ckpt",
        "unwa-inst-v1-plus": "melband_roformer_inst_v1_plus.ckpt",
        "unwa-inst-v2": "melband_roformer_inst_v2.ckpt",
        "unwa-instvoc-duality-v1": "melband_roformer_instvoc_duality_v1.ckpt",
        "unwa-instvoc-duality-v2": "melband_roformer_instvox_duality_v2.ckpt",
        "unwa-kim-ft": "mel_band_roformer_kim_ft_unwa.ckpt",
        "unwa-kim-ft2": "mel_band_roformer_kim_ft2_unwa.ckpt",
        "unwa-kim-ft2-bleedless": "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
        "unwa-kim-ft3": "mel_band_roformer_kim_ft3_unwa.ckpt",
        # BS Roformer Viperx models
        "bs-roformer-viperx-1297": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "bs-roformer-viperx-1296": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "bs-roformer-viperx-1053": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
        # Mel Roformer Viperx
        "mel-roformer-viperx-1143": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        # Kim / Standard Mel-Band Roformer
        "mel-band-roformer-kim": "mel_band_roformer_kim.ckpt",
        # Karaoke models
        "mel-roformer-karaoke-aufr33-viperx": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "mel-roformer-karaoke-gabox": "mel_band_roformer_karaoke_gabox.ckpt",
        "mel-roformer-karaoke-gabox-v2": "mel_band_roformer_karaoke_gabox_v2.ckpt",
        # Denoise models
        "mel-roformer-denoise-aufr33": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "mel-roformer-denoise-aufr33-aggr": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
        # De-reverb models
        "bs-roformer-dereverb": "deverb_bs_roformer_8_384dim_10depth.ckpt",
        "mel-roformer-dereverb-anvuew": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        # Gabox instrumental/vocal models
        "mel-roformer-vocals-gabox": "mel_band_roformer_vocals_gabox.ckpt",
        "mel-roformer-vocals-gabox-v2": "mel_band_roformer_vocals_v2_gabox.ckpt",
        "mel-roformer-instrumental-gabox": "mel_band_roformer_instrumental_gabox.ckpt",
        # Becruily models
        "mel-roformer-vocals-becruily": "mel_band_roformer_vocals_becruily.ckpt",
        "mel-roformer-instrumental-becruily": "mel_band_roformer_instrumental_becruily.ckpt",
        # Becruily (assets/models IDs) -> actual checkpoint filenames
        # These IDs exist in StemSep's model registry and must resolve to real files for audio-separator.
        "becruily-vocal": "mel_band_roformer_vocals_becruily.ckpt",
        "becruily-inst": "mel_band_roformer_instrumental_becruily.ckpt",
        # Note: if you also have a karaoke model ID in assets, add it here once confirmed.
        # MDX23C models
        "mdx23c-d1581": "MDX23C_D1581.ckpt",
        "mdx23c-instvoc-hq": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "mdx23c-instvoc-hq-2": "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
        # VR Arch models
        "uvr-hp-uvr": "1_HP-UVR.pth",
        "uvr-hp2-uvr": "7_HP2-UVR.pth",
        # Demucs models
        "htdemucs": "htdemucs.yaml",
        "htdemucs-ft": "htdemucs_ft.yaml",
        "htdemucs-6s": "htdemucs_6s.yaml",
    }

    def __init__(self, models_dir: str, log_level: int = logging.INFO):
        """
        Initialize the separator.

        Args:
            models_dir: Directory containing model files
            log_level: Logging level
        """
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger("StemSep.SimpleSeparator")
        self.logger.setLevel(log_level)

        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_autocast = self.device == "cuda"

        self.logger.info(f"SimpleSeparator initialized on {self.device}")

        # Helpful diagnostics: confirm whether we are using vendored audio-separator.
        try:  # pragma: no cover
            import audio_separator

            self.logger.info(
                f"audio_separator package: {Path(audio_separator.__file__).resolve()}"
            )
            self.logger.info(
                f"Separator implementation: {Path(inspect.getfile(Separator)).resolve()}"
            )
        except Exception:
            pass

    def get_model_filename(self, model_id: str) -> str:
        """
        Resolve a model_id to an actual filename inside models_dir.

        Resolution order:
        1) If a local YAML+weights exists for model_id, keep using model_id directly
           (direct-load path uses model_id.{yaml,ckpt/...} and bypasses audio-separator validation).
        2) If a file exists locally matching the registry URL basename, use it.
        3) If a file exists locally matching static MODEL_FILENAMES, use it (legacy compatibility).
        4) If a file exists locally matching model_id + common extensions, use it.
        5) Fall back to returning model_id and let audio-separator handle it.
        """
        # 1) Custom/community models: model_id.yaml + model_id.(ckpt/pth/...) are the "source of truth"
        if self._has_local_yaml(model_id):
            return model_id

        # 2) Registry-derived expected local filename (URL basename)
        expected_from_registry = self._get_expected_local_filename_from_registry(
            model_id
        )
        if (
            expected_from_registry
            and (self.models_dir / expected_from_registry).exists()
        ):
            return expected_from_registry

        # 3) Legacy static mapping (kept as fallback for older installs / renamed files)
        if (
            model_id in self.MODEL_FILENAMES
            and (self.models_dir / self.MODEL_FILENAMES[model_id]).exists()
        ):
            return self.MODEL_FILENAMES[model_id]

        normalized = model_id.lower().replace(" ", "-").replace("_", "-")
        if (
            normalized in self.MODEL_FILENAMES
            and (self.models_dir / self.MODEL_FILENAMES[normalized]).exists()
        ):
            return self.MODEL_FILENAMES[normalized]

        # 4) Direct file existence checks
        for ext in [
            ".ckpt",
            ".chpt",
            ".pth",
            ".pt",
            ".safetensors",
            ".onnx",
            ".yaml",
            "",
        ]:
            if (self.models_dir / f"{model_id}{ext}").exists():
                return f"{model_id}{ext}"

        # 5) Return as-is and let audio-separator handle it
        self.logger.warning(
            f"Model '{model_id}' not found locally via registry/mapping; using directly"
        )
        return model_id

    def _has_local_yaml(self, model_id: str) -> bool:
        """Check if a model has a local YAML config file.

        Note: MDX23C and VR models are excluded from direct loading path because they require
        audio-separator's specialized handlers.
        """
        # These models should use audio-separator's built-in handling, not our direct-load path
        model_lower = model_id.lower()
        # HyperACE checkpoints require a custom model definition (SegmModel/HyperACE modules)
        # not provided by audio-separator's RoformerLoader.
        if "hyperace" in model_lower:
            return False

        if any(x in model_lower for x in ["mdx23c", "vr", "uvr", "_hp"]):
            return False

        # Check for model_id.yaml and any known weight extension
        yaml_path = self.models_dir / f"{model_id}.yaml"
        if not yaml_path.exists():
            return False
        weight_path = self._find_local_weight_file(model_id)
        return weight_path is not None

    def _get_expected_local_filename_from_registry(
        self, model_id: str
    ) -> Optional[str]:
        """
        Derive the expected local filename from the model registry (assets/models/*.json via ModelManager).

        We treat the checkpoint URL basename as the authoritative filename, because:
        - it preserves the correct extension (.ckpt/.pth/.onnx/.yaml)
        - it matches how downloads are typically stored on disk

        Returns:
            The expected filename (e.g. "MelBandRoformer.ckpt") or None if unavailable.
        """
        try:
            from stemsep.models.model_manager import (
                ModelManager,  # local import to avoid heavy imports at module load
            )

            mm = ModelManager(models_dir=self.models_dir)

            # Prefer centralized logic (handles architectures that don't use checkpoint URLs)
            expected = mm.get_expected_local_filename(model_id)
            if expected:
                return expected

            # Backward fallback: derive from registry links if present
            info = mm.get_model(model_id)
            if not info or not getattr(info, "links", None):
                return None

            url = None
            try:
                if isinstance(info.links, dict):
                    url = info.links.get("checkpoint") or info.links.get("config")
            except Exception:
                url = None

            if not url:
                return None

            parsed = urlparse(str(url))
            basename = Path(parsed.path).name
            if not basename:
                return None
            return basename
        except Exception as e:
            # Non-fatal: just fall back to legacy mapping/heuristics
            self.logger.debug(
                f"Registry filename resolver failed for '{model_id}': {e}"
            )
            return None

    def _find_local_weight_file(self, model_id: str) -> Optional[Path]:
        """
        Find a local weight file for a model_id in models_dir.

        We prefer common Roformer checkpoint extensions. This allows community
        models to work even if they are not in audio-separator's internal list.
        """
        for ext in [".ckpt", ".pth", ".pt", ".safetensors", ".onnx"]:
            p = self.models_dir / f"{model_id}{ext}"
            if p.exists():
                return p
        return None

    def _normalize_model_data(self, model_data: Any) -> Any:
        """
        Normalize YAML-loaded model data so it is compatible with downstream loaders.

        Goals:
        - Preserve dict structure expected by audio-separator (top-level 'model' dict if present)
        - Apply common community-model key aliases (e.g. num_subbands -> num_bands)
        - Keep config sequences in a safe, expected shape (tuples where needed)
        """

        def normalize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in obj.items()}

            # Keep lists as lists by default; only convert to tuples when explicitly needed.
            # Some downstream configs expect tuples, but converting every list can break others.
            if isinstance(obj, list):
                return [normalize(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(normalize(v) for v in obj)

            return obj

        normalized = normalize(model_data)

        # If we have a model section, apply model-specific normalizations there too.
        if isinstance(normalized, dict) and isinstance(normalized.get("model"), dict):
            model_cfg = normalized["model"]

            # audio-separator / roformer loaders often want this as tuple.
            if "multi_stft_resolutions_window_sizes" in model_cfg and isinstance(
                model_cfg["multi_stft_resolutions_window_sizes"], list
            ):
                model_cfg["multi_stft_resolutions_window_sizes"] = tuple(
                    model_cfg["multi_stft_resolutions_window_sizes"]
                )

        return normalized

    def _build_audio_separator_model_data(
        self, model_data: Any, *, is_roformer: bool = False
    ) -> Any:
        """
        Build a model_data object in the shape expected by audio-separator CommonSeparator/MDXCSeparator.

        audio-separator expects `model_data` to contain at least:
        - model: model config (for RoformerLoader / constructors)
        - training: instruments + target_instrument (for stem naming + residual logic)

        Community YAMLs typically have {audio, model, training, inference}. We preserve that wrapper.
        """
        if not isinstance(model_data, dict):
            return model_data

        # Prefer preserving wrapper keys if present
        out: Dict[str, Any] = {}
        for key in ("audio", "model", "training", "inference"):
            if key in model_data:
                out[key] = model_data[key]

        # If YAML is already flat (no nested model section), keep as-is
        if not out:
            out = dict(model_data)

        # Some community configs (and ZFTurbo-derived YAMLs) use legacy key `num_subbands`.
        # For Roformer-based models, audio-separator constructors are strict and expect `num_bands`
        # instead, so keep `num_bands` and strip `num_subbands` everywhere.
        inferred_roformer = False
        if isinstance(out.get("model"), dict):
            model_cfg = out["model"]
            inferred_roformer = (
                "num_bands" in model_cfg
                or "freqs_per_bands" in model_cfg
                or "flash_attn" in model_cfg
                # Heuristics for older community Roformer YAMLs that still use `num_subbands`
                # and may omit newer flags.
                or (
                    "dim" in model_cfg
                    and "depth" in model_cfg
                    and (
                        "time_transformer_depth" in model_cfg
                        or "freq_transformer_depth" in model_cfg
                        or "attn_dropout" in model_cfg
                    )
                )
            )

        if is_roformer or inferred_roformer:

            def _strip_num_subbands(obj: Any) -> Any:
                if isinstance(obj, dict):
                    if "num_subbands" in obj and "num_bands" not in obj:
                        obj["num_bands"] = obj["num_subbands"]
                    obj.pop("num_subbands", None)
                    return {k: _strip_num_subbands(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_strip_num_subbands(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_strip_num_subbands(v) for v in obj)
                return obj

            out = _strip_num_subbands(out)

            # Help audio-separator's RoformerLoader detect model type from config even when
            # the checkpoint filename is an alias like `<model_id>.ckpt` (no "mel/roformer" hint).
            # RoformerLoader.detect_model_type() runs BEFORE its internal structure normalization,
            # so it needs key hints at the top-level.
            if isinstance(out.get("model"), dict):
                for k, v in out["model"].items():
                    out.setdefault(k, v)

        # Ensure training exists (audio-separator may index into it)
        if "training" not in out or out["training"] is None:
            out["training"] = {
                "instruments": ["Vocals", "Instrumental"],
                "target_instrument": "",
            }

        return out

    def _load_model_direct(self, separator: Separator, model_id: str) -> bool:
        """
        Load a model directly using its local YAML config, bypassing audio-separator's validation.

        Implementation strategy:
        - Use audio-separator's MDXCSeparator (official path) but inject local YAML+weights
          to avoid supported-model whitelist.
        - Ensure the config shape matches what audio-separator expects (CommonSeparator wants `training`).
        - Rely on audio-separator's internal RoformerLoader (via CommonSeparator) for best compatibility/quality.
        """
        yaml_path = self.models_dir / f"{model_id}.yaml"
        model_path = self._find_local_weight_file(model_id)

        if model_path is None:
            raise FileNotFoundError(
                f"No local weight file found for model '{model_id}' in {self.models_dir}"
            )

        self.logger.info(
            f"Direct loading model: {model_id} (bypassing audio-separator validation)"
        )

        # Load model data from YAML (use FullLoader to handle !!python/tuple tags)
        with open(yaml_path, "r", encoding="utf-8") as f:
            model_data = yaml.load(f, Loader=yaml.FullLoader)

        model_data = self._normalize_model_data(model_data)

        # Heuristic: most community YAML+weights loaded this way are Roformer-derived.
        # We use this to apply strict-key compatibility fixes (e.g. strip num_subbands).
        # Detect Roformer primarily from the YAML contents (filenames are unreliable because
        # this repo often installs alias files like `<model_id>.ckpt` / `<model_id>.yaml`).
        model_section = None
        if isinstance(model_data, dict):
            model_section = model_data.get("model")

        roformer_by_config = False
        if isinstance(model_section, dict):
            roformer_by_config = (
                "num_bands" in model_section
                or "freqs_per_bands" in model_section
                or (
                    "dim" in model_section
                    and "depth" in model_section
                    and (
                        "time_transformer_depth" in model_section
                        or "freq_transformer_depth" in model_section
                        or "attn_dropout" in model_section
                    )
                )
            )

        model_filename_hint = ""
        try:
            model_filename_hint = str(self.get_model_filename(model_id)).lower()
        except Exception:
            model_filename_hint = ""

        is_roformer = (
            roformer_by_config
            or (isinstance(model_data, dict) and bool(model_data.get("is_roformer")))
            or "roformer" in str(yaml_path).lower()
            or "roformer" in model_id.lower()
            or "roformer" in model_filename_hint
            or "melband" in model_filename_hint
            or "roformer" in model_path.name.lower()
        )

        model_data_for_separator = self._build_audio_separator_model_data(
            model_data, is_roformer=is_roformer
        )

        # Belt-and-suspenders: RoformerLoader's config validation is strict about unknown keys.
        # Ensure the legacy key is fully removed even if a community YAML includes it in multiple places.
        if is_roformer and isinstance(model_data_for_separator, dict):

            def _strip_num_subbands(obj: Any) -> Any:
                if isinstance(obj, dict):
                    if "num_subbands" in obj and "num_bands" not in obj:
                        obj["num_bands"] = obj["num_subbands"]
                    obj.pop("num_subbands", None)
                    return {k: _strip_num_subbands(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_strip_num_subbands(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_strip_num_subbands(v) for v in obj)
                return obj

            model_data_for_separator = _strip_num_subbands(model_data_for_separator)

        # Mark as roformer if applicable (helps audio-separator pick loaders)
        if is_roformer and isinstance(model_data_for_separator, dict):
            model_data_for_separator["is_roformer"] = True

        model_name = model_id

        # Common parameters for the separator instance
        common_params = {
            "logger": separator.logger,
            "log_level": separator.log_level,
            "torch_device": separator.torch_device,
            "torch_device_cpu": separator.torch_device_cpu,
            "torch_device_mps": separator.torch_device_mps,
            "onnx_execution_provider": separator.onnx_execution_provider,
            "model_name": model_name,
            "model_path": str(model_path),
            "model_data": model_data_for_separator,
            "progress_callback": getattr(separator, "progress_callback", None),
            "output_format": separator.output_format,
            "output_bitrate": separator.output_bitrate,
            "output_dir": separator.output_dir,
            "normalization_threshold": separator.normalization_threshold,
            "amplification_threshold": separator.amplification_threshold,
            "output_single_stem": separator.output_single_stem,
            "invert_using_spec": separator.invert_using_spec,
            "sample_rate": separator.sample_rate,
            "use_soundfile": separator.use_soundfile,
        }

        # Import and instantiate MDXCSeparator (handles Roformer models via RoformerLoader)
        module = importlib.import_module(
            "audio_separator.separator.architectures.mdxc_separator"
        )
        separator_class = getattr(module, "MDXCSeparator")

        # Get arch-specific params for MDXC
        arch_config = separator.arch_specific_params.get("MDXC", {})

        separator.model_instance = separator_class(
            common_config=common_params, arch_config=arch_config
        )

        self.logger.info(
            f"Model {model_id} loaded directly via audio-separator MDXCSeparator"
        )
        return True

    def separate(
        self,
        audio_path: str,
        model_id: str,
        output_dir: str,
        output_format: str = "wav",
        segment_size: int = 256,
        overlap: float = 8,
        batch_size: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        """
        Separate audio into stems.

        Args:
            audio_path: Path to input audio file
            model_id: Model ID (will be mapped to filename)
            output_dir: Directory to save output files
            output_format: Output format (wav, flac, mp3)
            segment_size: Segment size for processing
            overlap: Overlap between segments
            batch_size: Batch size for GPU processing
            progress_callback: Optional callback(percent, message)

        Returns:
            Dict mapping stem names to output file paths
        """

        def notify(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        # audio-separator internal progress is naturally 0..100 for the separation phase.
        # Map it onto our overall 0..100 scale so progress stays monotonic with the
        # stage markers below (30=run, 90=postprocess).
        def notify_internal(pct: float, msg: str):
            try:
                pct_f = float(pct)
            except Exception:
                pct_f = 0.0
            notify(30.0 + max(0.0, min(100.0, pct_f)) * 0.6, msg)

        notify(0, "Initializing separator...")

        # Preflight via ModelManager:
        # - on-demand safe filename repair (copy legacy "<model_id>.<ext>" -> expected URL-basename)
        # - clear error if model is missing (no silent fallback to a different model)
        #
        # NOTE: This does not download automatically; it only validates and repairs naming.
        try:
            from stemsep.models.model_manager import ModelManager
        except Exception as e:
            raise RuntimeError(
                f"Unable to import ModelManager for preflight. Underlying error: {e}"
            )

        mm = ModelManager(models_dir=self.models_dir)
        preflight_raw = mm.ensure_model_ready(
            model_id,
            auto_repair_filenames=True,
            copy_instead_of_rename=True,
        )

        # Tighten typing: ensure we only call .get(...) on a dict
        preflight: Dict[str, Any] = (
            preflight_raw if isinstance(preflight_raw, dict) else {}
        )

        if not bool(preflight.get("ok", False)) and not self._has_local_yaml(model_id):
            details_any = preflight.get("details", {})
            details: Dict[str, Any] = (
                details_any if isinstance(details_any, dict) else {}
            )

            expected = preflight.get("expected_filename")
            arch = preflight.get("architecture")
            download_url = details.get("download_url")

            msg = (
                f"Model '{model_id}' is not ready/installed.\n"
                f"Architecture: {arch}\n"
                f"Expected filename: {expected}\n"
                f"Models dir: {self.models_dir}\n"
            )
            if download_url:
                msg += f"Download URL: {download_url}\n"
            msg += "No fallback model will be used. Please install/download the selected model."

            raise FileNotFoundError(msg)

        # Get actual model filename (now that naming is repaired/validated)
        model_filename = self.get_model_filename(model_id)
        self.logger.info(f"Separating with model: {model_filename}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        notify(10, f"Loading model {model_filename}...")

        # audio-separator expects overlap as a ratio in [0, 1].
        # Our UI/recipes often use ZFTurbo-style integer overlaps (e.g. 2/4/8),
        # where step = chunk_size / overlap => overlap_ratio = (overlap - 1) / overlap.
        overlap_ratio: float
        try:
            overlap_val = float(overlap)
        except Exception:
            overlap_val = 0.25

        if overlap_val <= 0:
            overlap_ratio = 0.25
        elif overlap_val <= 1.0:
            overlap_ratio = overlap_val
        else:
            overlap_ratio = (overlap_val - 1.0) / overlap_val
            # Keep inside the valid range (and avoid edge-case 1.0)
            if overlap_ratio < 0.0:
                overlap_ratio = 0.0
            if overlap_ratio >= 1.0:
                overlap_ratio = 0.999

            self.logger.info(
                f"Normalizing overlap {overlap_val} -> {overlap_ratio:.4f} for audio-separator"
            )

        # Create separator instance (exactly like UVR5-UI)
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=str(self.models_dir),
            output_dir=str(output_path),
            output_format=output_format,
            use_autocast=self.use_autocast,
            mdxc_params={
                "segment_size": segment_size,
                "overlap": overlap_ratio,
                "batch_size": batch_size,
            },
        )

        # Enable chunk-level progress reporting from vendored audio-separator.
        separator.progress_callback = notify_internal

        # Load model - try direct loading for models with local YAML config
        # This bypasses audio-separator's validation for community models.
        #
        # If the official path (audio-separator MDXCSeparator + RoformerLoader) fails, fall back
        # to our internal RoformerSeparator implementation (still based on audio-separator logic),
        # so community YAML+weights remain functional even if audio-separator validation changes.
        if self._has_local_yaml(model_id):
            try:
                self._load_model_direct(separator, model_id)
            except Exception as e:
                self.logger.warning(
                    f"Direct YAML-based loading failed for '{model_id}' via audio-separator path. "
                    f"Falling back to internal RoformerSeparator. Underlying error: {e}"
                )

                # Fallback: internal roformer implementation (zfturbo_wrapper.RoformerSeparator)
                from stemsep.audio.zfturbo_wrapper import RoformerSeparator

                fallback = RoformerSeparator(
                    models_dir=self.models_dir, device=self.device
                )
                return fallback.separate(
                    audio_path=audio_path,
                    model_id=model_id,
                    output_dir=output_dir,
                    progress_callback=progress_callback,
                )
        else:
            # Use standard audio-separator loading
            separator.load_model(model_filename=model_filename)

        notify(30, "Model loaded. Running separation...")

        # Separate
        output_files = separator.separate(audio_path)

        notify(90, "Separation complete. Processing outputs...")

        # Build result dict with stem names
        result = {}
        for output_file in output_files:
            full_path = str(output_path / output_file)
            # Extract stem name from filename
            stem_name = self._extract_stem_name(output_file)
            result[stem_name] = full_path
            self.logger.info(f"Output: {stem_name} -> {full_path}")

        notify(100, "Complete")

        return result

    def _extract_stem_name(self, filename: str) -> str:
        """
        Extract stem name from audio-separator output filename.

        audio-separator commonly emits filenames like:
          good_morning_(vocals)_mel_band_roformer_....wav
          good_morning_(other)_mel_band_roformer_....wav

        Rule: ALWAYS prefer explicit stem tokens "(_stem_)" in the filename over
        any heuristic that may match model-name substrings (e.g. "..._vocals_becruily...").
        """
        lower = filename.lower()

        # 1) Prefer explicit stem token "(stem)" if present
        # Most common pattern: "..._(stem)_..."
        if "_(" in lower and ")_" in lower:
            try:
                token = lower.split("_(", 1)[1].split(")_", 1)[0].strip()
                token_map = {
                    "vocals": "vocals",
                    "vocal": "vocals",
                    "instrumental": "instrumental",
                    "inst": "instrumental",
                    "drums": "drums",
                    "drum": "drums",
                    "bass": "bass",
                    "other": "other",
                    "guitar": "guitar",
                    "piano": "piano",
                }
                if token in token_map:
                    return token_map[token]
            except Exception:
                pass

        # Also accept a slightly different pattern some tools produce: "..._(stem).wav" or "..._(stem)__..."
        if "_(" in lower and ")" in lower:
            try:
                token = lower.split("_(", 1)[1].split(")", 1)[0].strip()
                token_map = {
                    "vocals": "vocals",
                    "vocal": "vocals",
                    "instrumental": "instrumental",
                    "inst": "instrumental",
                    "drums": "drums",
                    "drum": "drums",
                    "bass": "bass",
                    "other": "other",
                    "guitar": "guitar",
                    "piano": "piano",
                }
                if token in token_map:
                    return token_map[token]
            except Exception:
                pass

        # 2) Fallback heuristic: ONLY inspect the base part before any explicit stem token.
        # This reduces false positives from model identifiers.
        base = lower.split("_(", 1)[0] if "_(" in lower else lower

        if "vocal" in base:
            return "vocals"
        if "instrument" in base or "inst" in base:
            return "instrumental"
        if "drum" in base:
            return "drums"
        if "bass" in base:
            return "bass"
        if "other" in base:
            return "other"
        if "guitar" in base:
            return "guitar"
        if "piano" in base:
            return "piano"

        # 3) Final fallback: use filename stem
        return Path(filename).stem
