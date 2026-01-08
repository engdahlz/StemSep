import asyncio
import gc
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from stemsep.audio.ensemble_blender import EnsembleBlender
from stemsep.audio.separation_engine import SeparationEngine, SeparationJob
from stemsep.models.model_manager import ModelManager

from .recipe_manager import RecipeManager

# Import pydub and static_ffmpeg for format conversion
try:
    import static_ffmpeg
    from pydub import AudioSegment

    static_ffmpeg.add_paths()  # Ensure ffmpeg is in path
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logging.getLogger("StemSep").warning(
        "pydub or static-ffmpeg not found. MP3 export will be disabled."
    )

# Import mutagen for metadata
try:
    import mutagen
    from mutagen.easyid3 import EasyID3
    from mutagen.flac import FLAC
    from mutagen.wave import WAVE

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    logging.getLogger("StemSep").warning(
        "mutagen not found. Metadata preservation will be disabled."
    )


class SeparationManager:
    """Manages multiple separation jobs"""

    def __init__(
        self,
        model_manager: ModelManager,
        gpu_enabled: bool = True,
        max_workers: int = 2,
    ):
        self.logger = logging.getLogger("StemSep")
        self.model_manager = model_manager
        self.recipe_manager = RecipeManager(model_manager.assets_dir)
        self.engine = SeparationEngine(
            model_manager=self.model_manager, gpu_enabled=gpu_enabled
        )
        self.jobs: Dict[str, SeparationJob] = {}
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.job_lock = threading.Lock()
        self.paused = False
        self.queue = []

    def _get_model_recommended_settings(self, model_id: str) -> Dict[str, Any]:
        minfo = self.model_manager.get_model(model_id) if model_id else None
        rec = getattr(minfo, "recommended_settings", None) if minfo else None
        return rec if isinstance(rec, dict) else {}

    def _apply_recommended_defaults(
        self,
        model_id: str,
        overlap: Optional[float],
        segment_size: Optional[int],
        batch_size: Optional[int],
        tta: Optional[bool],
    ) -> tuple[float, int, int, bool]:
        """Apply model registry recommended_settings when caller didn't override.

        Heuristics:
        - overlap: treat legacy default 0.25 as "unset"
        - segment_size: treat 0/None as "auto"
        - batch_size: treat 1/None as "unset" (only auto-increase on GPU)
        - tta: treat False/None as "unset"
        """
        minfo = self.model_manager.get_model(model_id) if model_id else None
        rec = self._get_model_recommended_settings(model_id)

        out_overlap = overlap if overlap is not None else 0.25
        if out_overlap == 0.25:
            r = rec.get("overlap") if isinstance(rec, dict) else None
            if r is None and minfo is not None:
                r = getattr(minfo, "recommended_overlap", None)
            if r is not None:
                if isinstance(r, (int, float)):
                    out_overlap = r
                else:
                    try:
                        out_overlap = float(r)
                    except Exception:
                        out_overlap = r

        out_segment_size = int(segment_size or 0)
        if out_segment_size <= 0:
            r = rec.get("segment_size") if isinstance(rec, dict) else None
            if r is not None:
                try:
                    out_segment_size = int(r)
                except Exception:
                    out_segment_size = 0
        if out_segment_size <= 0:
            out_segment_size = 352800

        out_tta = bool(tta) if tta is not None else False
        if out_tta is False:
            r = rec.get("tta") if isinstance(rec, dict) else None
            if isinstance(r, bool):
                out_tta = r

        out_batch_size = int(batch_size or 1)
        if out_batch_size <= 0:
            out_batch_size = 1
        # Only auto-increase batch size on GPU to avoid CPU regressions.
        if out_batch_size == 1 and getattr(self.engine, "gpu_enabled", False):
            r = rec.get("batch_size") if isinstance(rec, dict) else None
            if r is not None:
                try:
                    rb = int(r)
                    if rb > 0:
                        out_batch_size = rb
                except Exception:
                    pass

        return out_overlap, out_segment_size, out_batch_size, out_tta

        # Cleanup old temp dirs on startup
        self._cleanup_old_temp_dirs()

    def _cleanup_old_temp_dirs(self):
        """Delete temp directories older than 24 hours"""
        try:
            temp_base = Path(tempfile.gettempdir())
            # Look for stemsep_* folders
            for p in temp_base.glob("stemsep_*"):
                if p.is_dir():
                    try:
                        # Check modification time
                        mtime = p.stat().st_mtime
                        if time.time() - mtime > 86400:  # 24 hours
                            self.logger.info(f"Cleaning up old temp dir: {p}")
                            shutil.rmtree(p)
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {p}: {e}")

            # Also cleanup old checkpoint files (older than 24 hours)
            for p in temp_base.glob("stemsep_checkpoint_*.json"):
                try:
                    mtime = p.stat().st_mtime
                    if time.time() - mtime > 86400:
                        self.logger.info(f"Cleaning up old checkpoint: {p}")
                        p.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup checkpoint {p}: {e}")

        except Exception as e:
            self.logger.warning(f"Error during temp cleanup: {e}")

    def _save_job_checkpoint(self, job_id: str, stage: str, data: dict = None):
        """Save job progress checkpoint for crash recovery"""
        import json

        checkpoint_file = (
            Path(tempfile.gettempdir()) / f"stemsep_checkpoint_{job_id}.json"
        )
        checkpoint = {
            "job_id": job_id,
            "stage": stage,
            "timestamp": time.time(),
            "data": data or {},
        }
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            self.logger.debug(f"Saved checkpoint for job {job_id} at stage '{stage}'")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint for job {job_id}: {e}")

    def _load_job_checkpoint(self, job_id: str) -> Optional[dict]:
        """Load job checkpoint if exists"""
        import json

        checkpoint_file = (
            Path(tempfile.gettempdir()) / f"stemsep_checkpoint_{job_id}.json"
        )
        try:
            if checkpoint_file.exists():
                with open(checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                self.logger.info(
                    f"Loaded checkpoint for job {job_id}: stage='{checkpoint.get('stage')}'"
                )
                return checkpoint
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint for job {job_id}: {e}")
        return None

    def _clear_job_checkpoint(self, job_id: str):
        """Clear checkpoint after successful completion"""
        checkpoint_file = (
            Path(tempfile.gettempdir()) / f"stemsep_checkpoint_{job_id}.json"
        )
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.debug(f"Cleared checkpoint for job {job_id}")
        except Exception as e:
            self.logger.warning(f"Failed to clear checkpoint for job {job_id}: {e}")

    def _estimate_memory_requirement(self, file_path: str, model_id: str) -> dict:
        """Estimate memory requirements for processing a file with a model.

        Returns:
            dict with 'estimated_gb', 'available_gb', 'sufficient', 'warning'
        """
        import psutil

        result = {
            "estimated_gb": 0.0,
            "available_gb": 0.0,
            "sufficient": True,
            "warning": None,
        }

        try:
            # Get file duration in seconds
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            # Rough estimate: ~10MB per minute for typical audio
            estimated_duration_min = file_size_mb / 10

            # Base memory per minute of audio (in GB)
            # This is a conservative estimate based on typical model requirements
            memory_per_minute_gb = 0.5  # 500MB per minute as baseline

            # Adjust based on model complexity (if we know it)
            model_info = self.model_manager.get_model_info(model_id)
            if model_info:
                # BS-Roformer models need more memory
                if "roformer" in model_id.lower():
                    memory_per_minute_gb = 0.8
                # MDX models are more efficient
                elif "mdx" in model_id.lower():
                    memory_per_minute_gb = 0.4

            result["estimated_gb"] = estimated_duration_min * memory_per_minute_gb

            # Get available system memory
            mem = psutil.virtual_memory()
            result["available_gb"] = mem.available / (1024**3)

            # Check if we have enough
            # Use 80% of available as safe threshold
            safe_available = result["available_gb"] * 0.8
            result["sufficient"] = result["estimated_gb"] < safe_available

            if not result["sufficient"]:
                result["warning"] = (
                    f"File may require ~{result['estimated_gb']:.1f}GB but only "
                    f"{result['available_gb']:.1f}GB available. Processing may fail or be slow."
                )
                self.logger.warning(result["warning"])
            elif result["estimated_gb"] > result["available_gb"] * 0.5:
                result["warning"] = (
                    f"Large file detected (~{result['estimated_gb']:.1f}GB estimated). "
                    "Close other applications to ensure smooth processing."
                )
                self.logger.info(result["warning"])

        except Exception as e:
            self.logger.warning(f"Could not estimate memory requirements: {e}")

        return result

    def check_memory_for_job(self, file_path: str, model_id: str) -> dict:
        """Public method to check if memory is sufficient for a job.

        Returns dict with 'can_proceed', 'warning', 'estimated_gb', 'available_gb'
        """
        estimate = self._estimate_memory_requirement(file_path, model_id)
        return {
            "can_proceed": estimate["sufficient"],
            "warning": estimate["warning"],
            "estimated_gb": round(estimate["estimated_gb"], 2),
            "available_gb": round(estimate["available_gb"], 2),
        }

    def create_job(
        self,
        file_path: str,
        model_id: str,
        output_dir: str,
        stems: List[str] = None,
        device: str = None,
        overlap: float = 0.25,
        segment_size: int = 0,
        batch_size: int = 1,
        tta: bool = False,
        output_format: str = "wav",
        export_mixes: List[str] = None,
        shifts: int = 1,
        bitrate: str = "320k",
        ensemble_config: List[Dict] = None,
        ensemble_algorithm: str = "average",
        invert: bool = False,
        split_freq: int = 750,
        normalize: bool = False,
        bit_depth: str = "16",
        vc_enabled: bool = False,
        vc_stage: str = "export",
        vc_db_per_extra_model: float = 3.0,
        phase_params: Dict = None,
        phase_fix_enabled: bool = False,
        stem_algorithms: Dict[str, str] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """Create a new separation job

        Args:
            phase_params: Optional override for phase swap settings from UI
                          { 'enabled': True, 'lowHz': 500, 'highHz': 5000, 'highFreqWeight': 0.8 }
        """

        # Handle explicit ensemble config passed from API
        if ensemble_config:
            return self.create_ensemble_job(
                file_path=file_path,
                ensemble_config=ensemble_config,
                output_dir=output_dir,
                stems=stems,
                device=device,
                overlap=overlap,
                segment_size=segment_size,
                batch_size=batch_size,
                tta=tta,
                output_format=output_format,
                export_mixes=export_mixes,
                shifts=shifts,
                bitrate=bitrate,
                algorithm=ensemble_algorithm,
                stem_algorithms=stem_algorithms,
                phase_params=phase_params,
                phase_fix_enabled=phase_fix_enabled,
                invert=invert,
                split_freq=split_freq,
                normalize=normalize,
                bit_depth=bit_depth,
                vc_enabled=vc_enabled,
                vc_stage=vc_stage,
                vc_db_per_extra_model=vc_db_per_extra_model,
                on_complete=on_complete,
            )

        # Check if model_id is a Recipe
        recipe_def = self.recipe_manager.get_recipe(model_id)
        if recipe_def:
            self.logger.info(
                f"Resolved model_id '{model_id}' to Recipe (Type: {recipe_def.get('type', 'ensemble')})"
            )

            # Use recipe defaults if user didn't override
            r_defaults = recipe_def.get("defaults", {})
            # NOTE: Legacy default overlap=0.25 is not the intended "auto" overlap in this project.
            # Recipes/model registry commonly use overlap as an integer divisor (e.g. 4, 8).
            # Treat 0.25 as "unset" so recipe defaults apply.
            final_overlap = (
                overlap
                if (overlap is not None and overlap != 0.25)
                else r_defaults.get("overlap", overlap)
            )
            final_shifts = shifts if shifts is not None else r_defaults.get("shifts", 1)
            final_invert = invert or r_defaults.get("invert", False)
            final_split_freq = (
                split_freq if split_freq != 750 else r_defaults.get("split_freq", 750)
            )
            # Use recipe segment size if provided, otherwise keep passed value (0 = auto)
            # NOTE: recipes.json historically used `chunk_size` for this field.
            recipe_segment_size = (
                r_defaults.get("segment_size")
                if isinstance(r_defaults, dict) and r_defaults.get("segment_size") is not None
                else (r_defaults.get("chunk_size") if isinstance(r_defaults, dict) else None)
            )
            final_segment_size = (
                segment_size if segment_size != 0 else (recipe_segment_size or 0)
            )

            if recipe_def.get("type") in ["pipeline", "chained"]:
                # Normalize recipe step schema to pipeline schema expected by _run_pipeline_job
                # recipes.json typically uses: {name, model_id, input_from, output}
                recipe_target = recipe_def.get("target")
                pipeline_steps = []
                for idx, step in enumerate(recipe_def.get("steps", []) or []):
                    step_name = (
                        step.get("step_name") or step.get("name") or f"step_{idx}"
                    )
                    action = step.get("action") or "separate"
                    normalized: Dict[str, Any] = {
                        "step_name": step_name,
                        "action": action,
                    }

                    # Carry recipe-level intent into each step so the pipeline runner can
                    # default post-processing to the correct stem (e.g. golden_ultimate_vocal
                    # should post-process vocals; golden_ultimate_inst should post-process instrumental).
                    if recipe_target is not None:
                        normalized["recipe_target"] = recipe_target

                    # Primary model for a separation-like step
                    if step.get("model_id") is not None:
                        normalized["model_id"] = step.get("model_id")

                    # Phase fix steps use a reference/source model instead of model_id
                    if step.get("source_model") is not None:
                        normalized["source_model"] = step.get("source_model")
                    if step.get("apply_to") is not None:
                        normalized["apply_to"] = step.get("apply_to")

                    # Preserve common Phase Fix params (accept recipe-style keys)
                    for k in (
                        "lowHz",
                        "highHz",
                        "weight",
                        "highFreqWeight",
                        "low_hz",
                        "high_hz",
                        "high_freq_weight",
                    ):
                        if step.get(k) is not None:
                            normalized[k] = step.get(k)
                    if step.get("input_source"):
                        normalized["input_source"] = step.get("input_source")
                    elif step.get("input_from"):
                        normalized["input_source"] = step.get("input_from")
                    if step.get("output") is not None:
                        normalized["output"] = step.get("output")
                    pipeline_steps.append(normalized)

                return self.create_pipeline_job(
                    file_path=file_path,
                    pipeline_config=pipeline_steps,
                    output_dir=output_dir,
                    stems=stems,
                    device=device,
                    overlap=final_overlap,
                    segment_size=final_segment_size,
                    tta=tta,
                    output_format=output_format,
                    export_mixes=export_mixes,
                    shifts=final_shifts,
                    bitrate=bitrate,
                    normalize=normalize,
                    bit_depth=bit_depth,
                    on_complete=on_complete,
                )

            # Standard Ensemble Recipe
            recipe_cfg = self.recipe_manager.recipe_to_ensemble_config(model_id)
            return self.create_ensemble_job(
                file_path=file_path,
                ensemble_config=recipe_cfg["ensemble_config"],
                output_dir=output_dir,
                stems=stems,
                device=device,
                algorithm=recipe_cfg["algorithm"],
                overlap=final_overlap,
                segment_size=final_segment_size,
                batch_size=batch_size,
                tta=tta,
                output_format=output_format,
                export_mixes=export_mixes,
                shifts=final_shifts,
                bitrate=bitrate,
                invert=final_invert,
                split_freq=final_split_freq,
                normalize=normalize,
                bit_depth=bit_depth,
                post_processing=recipe_def.get("post_processing"),
                on_complete=on_complete,
            )

        # Check if model_id is a preset
        preset = self.model_manager.get_preset(model_id)
        if preset:
            self.logger.info(f"Resolved preset {model_id}")

            # Check for 'models' list (V3 format)
            if "models" in preset and preset["models"]:
                models = preset["models"]
                if len(models) > 1:
                    # Treat as ensemble
                    self.logger.info(
                        f"Preset {model_id} is an ensemble with {len(models)} models."
                    )
                    ensemble_cfg = []
                    for m in models:
                        if isinstance(m, str):
                            ensemble_cfg.append({"model_id": m, "weight": 1.0})
                        else:
                            ensemble_cfg.append(m)

                    # Determine algorithm from settings or default
                    alg = preset.get("settings", {}).get("algorithm", "average")
                    p_seg_size = preset.get("settings", {}).get("segment_size", 0)
                    final_seg = segment_size if segment_size != 0 else p_seg_size

                    return self.create_ensemble_job(
                        file_path=file_path,
                        ensemble_config=ensemble_cfg,
                        output_dir=output_dir,
                        device=device,
                        algorithm=alg,
                        overlap=overlap,
                        segment_size=final_seg,
                        batch_size=batch_size,
                        tta=tta,
                        output_format=output_format,
                        shifts=shifts,
                        bitrate=bitrate,
                        normalize=normalize,
                        bit_depth=bit_depth,
                        on_complete=on_complete,
                    )
                else:
                    # Single model
                    m = models[0]
                    model_id = m if isinstance(m, str) else m["model_id"]
                    self.logger.info(f"Using single model {model_id} from preset.")
                    # Use preset segment size if not overridden
                    p_seg_size = preset.get("settings", {}).get("segment_size", 0)
                    if segment_size == 0 and p_seg_size > 0:
                        segment_size = p_seg_size

            elif "model_id" in preset:
                # Legacy format
                model_id = preset["model_id"]

        # Check for ensemble configuration
        ensemble_config = self._get_ensemble_config(model_id)
        if ensemble_config:
            self.logger.info(f"Model ID {model_id} identified as ensemble preset.")
            return self.create_ensemble_job(
                file_path=file_path,
                ensemble_config=ensemble_config["models"],
                output_dir=output_dir,
                device=device,
                algorithm=ensemble_config.get("algorithm", "average"),
                overlap=overlap,
                segment_size=segment_size,
                batch_size=batch_size,
                tta=tta,
                output_format=output_format,
                shifts=shifts,
                bitrate=bitrate,
                normalize=normalize,
                bit_depth=bit_depth,
                vc_enabled=vc_enabled,
                vc_stage=vc_stage,
                vc_db_per_extra_model=vc_db_per_extra_model,
                on_complete=on_complete,
            )

        # Apply architecture-aware defaults for single-model runs when caller didn't override.
        overlap, segment_size, batch_size, tta = self._apply_recommended_defaults(
            model_id=model_id,
            overlap=overlap,
            segment_size=segment_size,
            batch_size=batch_size,
            tta=tta,
        )

        job_id = str(uuid.uuid4())

        # Create job object
        job = SeparationJob(
            job_id=job_id,
            file_path=file_path,
            model_id=model_id,
            output_dir=output_dir,  # Pass the FINAL desired output dir
            stems=stems,
            device=device,
            normalize=normalize,
            bit_depth=bit_depth,
            vc_enabled=vc_enabled,
            vc_stage=vc_stage,
            vc_db_per_extra_model=vc_db_per_extra_model,
            on_complete=on_complete,
        )

        # Store parameters
        job.overlap = overlap
        job.segment_size = segment_size
        job.batch_size = batch_size
        job.tta = tta
        job.output_format = output_format
        job.export_mixes = export_mixes or []
        job.shifts = shifts
        job.bitrate = bitrate

        with self.job_lock:
            self.jobs[job_id] = job

        self.logger.info(
            f"Created job {job_id} for {file_path} with model {model_id} (Temp: {job.temp_dir})"
        )
        return job_id

    def _get_ensemble_config(self, model_id: str) -> Optional[Dict]:
        """Helper to check if a model_id is an ensemble preset"""
        # We need access to presets. ModelManager has them.
        presets_data = self.model_manager.load_presets()
        if not presets_data:
            return None

        # Check 'ensembles' key
        ensembles = presets_data.get("ensembles", [])
        for ens in ensembles:
            if ens["id"] == model_id:
                return ens
        return None

    def create_ensemble_job(
        self,
        file_path: str,
        ensemble_config: List[Dict[str, Any]],
        output_dir: str,
        stems: List[str] = None,
        device: str = None,
        algorithm: str = "average",
        stem_algorithms: Dict[str, str] = None,
        phase_params: Dict[str, Any] = None,
        phase_fix_enabled: bool = False,
        overlap: float = 0.25,
        segment_size: int = 0,
        batch_size: int = 1,
        tta: bool = False,
        output_format: str = "wav",
        shifts: int = 1,
        bitrate: str = "320k",
        export_mixes: List[str] = None,
        invert: bool = False,
        split_freq: int = 750,
        normalize: bool = False,
        bit_depth: str = "16",
        vc_enabled: bool = False,
        vc_stage: str = "export",
        vc_db_per_extra_model: float = 3.0,
        post_processing: Optional[str] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """
        Create a new ensemble separation job.
        """

        # Pre-flight check for model availability
        valid_config = []
        missing_models = []

        for cfg in ensemble_config:
            mid = cfg["model_id"]
            if self.model_manager.is_model_installed(mid):
                valid_config.append(cfg)
            else:
                missing_models.append(mid)

        if missing_models:
            self.logger.warning(
                f"Ensemble job requested with missing models: {missing_models}"
            )

            # Graceful degradation for Phase Fix
            if phase_fix_enabled:
                if len(valid_config) >= 1:
                    self.logger.warning(
                        "Phase Fix reference or target missing. Falling back to single model separation with: "
                        + valid_config[0]["model_id"]
                    )
                    phase_fix_enabled = False  # Disable since can't do proper phase fix
                    ensemble_config = [valid_config[0]]
                else:
                    raise RuntimeError(
                        f"All models for Phase Fix are missing: {missing_models}"
                    )
            else:
                raise RuntimeError(
                    f"Cannot start ensemble. Missing models: {missing_models}. Please download them first."
                )

        job_id = str(uuid.uuid4())

        # Create job object
        job = SeparationJob(
            job_id=job_id,
            file_path=file_path,
            model_id="ensemble",
            output_dir=output_dir,
            stems=stems,
            device=device,
            invert=invert,
            normalize=normalize,
            bit_depth=bit_depth,
            vc_enabled=vc_enabled,
            vc_stage=vc_stage,
            vc_db_per_extra_model=vc_db_per_extra_model,
            on_complete=on_complete,
        )

        # Attach extra properties
        job.ensemble_config = ensemble_config
        job.ensemble_algorithm = algorithm
        job.overlap = overlap
        job.segment_size = segment_size
        job.batch_size = batch_size
        job.tta = tta
        job.output_format = output_format
        job.export_mixes = export_mixes or []
        job.shifts = shifts
        job.bitrate = bitrate
        job.split_freq = split_freq
        job.stem_algorithms = stem_algorithms  # Per-stem algorithm selection
        job.phase_fix_params = phase_params  # Phase Fixer frequency controls
        job.phase_fix_enabled = phase_fix_enabled  # Phase Fix checkbox state
        job.post_processing = post_processing
        job.vc_enabled = bool(vc_enabled)
        job.vc_stage = vc_stage
        job.vc_db_per_extra_model = vc_db_per_extra_model

        with self.job_lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.logger.info(
                f"Created ensemble job {job_id} with {len(ensemble_config)} models. Invert: {invert}, SplitFreq: {split_freq}, Normalize: {normalize}"
            )

        # Trigger processing
        self._process_queue()

        return job_id

    def _volume_compensation_gain(
        self, num_sources: int, db_per_extra_model: float = 3.0
    ) -> float:
        """Compute a simple headroom gain reduction for multi-source ensembles."""
        try:
            n = int(num_sources or 0)
        except Exception:
            n = 0
        if n <= 1:
            return 1.0
        try:
            db = -abs(float(db_per_extra_model)) * float(n - 1)
            return float(10 ** (db / 20.0))
        except Exception:
            return 1.0

    def _perform_spectral_inversion(self, job: SeparationJob):
        """Perform spectral inversion (Original - Stem = Complement)"""
        if not job.invert:
            return

        self.logger.info(f"Performing spectral inversion for job {job.id}")
        try:
            original, sr = sf.read(job.file_path)
            new_files = {}

            for stem_name, stem_path in list(job.output_files.items()):
                if stem_name == "vocals":
                    target_name = "instrumental"
                elif stem_name == "instrumental":
                    target_name = "vocals"
                else:
                    target_name = f"inverted_remainder_of_{stem_name}"

                if target_name in job.output_files:
                    continue

                stem_audio, _ = sf.read(stem_path)
                min_len = min(len(original), len(stem_audio))

                if original.ndim != stem_audio.ndim:
                    continue

                inverted = original[:min_len] - stem_audio[:min_len]

                ext = Path(stem_path).suffix.lstrip(".")
                out_path = Path(job.output_dir) / f"{target_name}.{ext}"

                sf.write(str(out_path), inverted, sr)
                new_files[target_name] = str(out_path)
                self.logger.info(f"Created inverted stem: {target_name}")

            job.output_files.update(new_files)
        except Exception as e:
            self.logger.error(f"Inversion failed: {e}")

    def start_job(self, job_id: str) -> bool:
        """Start a created job"""
        with self.job_lock:
            if job_id not in self.jobs:
                self.logger.error(f"Job {job_id} not found")
                return False
            job = self.jobs[job_id]
            # Allow restarting 'queued' jobs if resuming
            if job.status not in ["pending", "queued"]:
                self.logger.warning(
                    f"Job {job_id} is not pending (status: {job.status})"
                )
                return False

            job.status = "queued"

            if self.paused:
                self.logger.info(f"Job {job_id} queued (paused)")
                return True

        # Submit to executor
        try:
            self.executor.submit(self._run_job_sync, job_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to submit job {job_id}: {e}")
            job.status = "failed"
            job.error = str(e)
            return False

    def pause_queue(self):
        """Pause queue processing (does not stop running jobs)"""
        self.paused = True
        self.logger.info("Queue paused")

    def resume_queue(self):
        """Resume queue processing"""
        self.paused = False
        self.logger.info("Queue resumed")
        self._process_pending()

    def reorder_queue(self, job_ids: List[str]):
        """Reorder the queue"""
        with self.job_lock:
            valid_ids = [jid for jid in job_ids if jid in self.jobs]
            missing = [jid for jid in self.queue if jid not in valid_ids]
            self.queue = valid_ids + missing
            self.logger.info("Queue reordered")

    def _process_pending(self):
        """Start pending/queued jobs"""
        if self.paused:
            return

        # Snapshot queue to avoid lock issues if start_job locks
        with self.job_lock:
            candidates = [
                jid for jid in self.queue if self.jobs[jid].status == "queued"
            ]

        for jid in candidates:
            # start_job handles the locking and submission
            # It will check self.paused again, so if we paused mid-loop it stops
            self.start_job(jid)

    def _run_job_sync(self, job_id: str):
        """Run job synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            job = self.jobs.get(job_id)
            if job:
                if job.model_id == "ensemble":
                    loop.run_until_complete(self._run_ensemble_job(job))
                elif job.model_id == "pipeline":
                    loop.run_until_complete(self._run_pipeline_job(job))
                else:
                    loop.run_until_complete(self._run_job(job))
        finally:
            loop.close()

    def create_pipeline_job(
        self,
        file_path: str,
        pipeline_config: List[Dict],
        output_dir: str,
        stems: List[str] = None,
        device: str = None,
        overlap: float = 0.25,
        segment_size: int = 0,
        tta: bool = False,
        output_format: str = "wav",
        shifts: int = 1,
        bitrate: str = "320k",
        export_mixes: List[str] = None,
        normalize: bool = False,
        bit_depth: str = "16",
        on_complete: Optional[Callable] = None,
    ) -> str:
        """Create a new pipeline separation job"""
        job_id = str(uuid.uuid4())
        job = SeparationJob(
            job_id=job_id,
            file_path=file_path,
            model_id="pipeline",
            output_dir=output_dir,
            stems=stems,
            device=device,
            normalize=normalize,
            bit_depth=bit_depth,
            on_complete=on_complete,
        )
        job.pipeline_config = pipeline_config
        job.overlap = overlap
        job.segment_size = segment_size
        job.batch_size = 1  # Pipeline steps usually single batch for safety, or pass through? Let's default to 1 for now.
        job.tta = tta
        job.output_format = output_format
        job.shifts = shifts
        job.bitrate = bitrate
        job.export_mixes = export_mixes or []

        with self.job_lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.logger.info(
                f"Created pipeline job {job_id} with {len(pipeline_config)} steps"
            )

        self._process_queue()
        return job_id

    @staticmethod
    def _canonicalize_stem_keys(outputs: Any) -> Any:
        """Normalize common stem key aliases from underlying separators.

        audio-separator Roformer models frequently return `other` + `vocals` for 2-stem separation.
        Internally, StemSep expects the canonical pair `instrumental` + `vocals`.
        """
        if not isinstance(outputs, dict):
            return outputs

        out: Dict[str, Any] = dict(outputs)

        if "vocals" not in out and "vocal" in out:
            out["vocals"] = out.pop("vocal")

        if "instrumental" not in out:
            if "other" in out:
                out["instrumental"] = out.pop("other")
            elif "accompaniment" in out:
                out["instrumental"] = out.pop("accompaniment")

        return out

    async def _run_pipeline_job(self, job: SeparationJob):
        """Run a sequential pipeline job"""
        try:
            job.status = "running"
            job.start_time = asyncio.get_event_loop().time()

            def check_cancelled():
                return job.status == "cancelled"

            # Validate audio
            if check_cancelled():
                raise asyncio.CancelledError()
            validation = self.engine.validate_audio_file(job.file_path)
            if not validation["valid"]:
                raise Exception(f"Invalid audio file: {validation.get('error')}")

            metadata = self._extract_metadata(job.file_path)

            # Temp dir for intermediate steps
            pipeline_temp = Path(job.output_dir) / f"pipeline_{job.id}"
            pipeline_temp.mkdir(exist_ok=True)

            step_outputs = {}  # "stepname" -> {"stem": "path"}

            for i, step in enumerate(job.pipeline_config):
                if check_cancelled():
                    raise asyncio.CancelledError()

                steps_total = max(1, len(job.pipeline_config))
                step_base = 10 + (i / steps_total * 80)
                step_span = 80 / steps_total

                step_name = step.get("step_name", f"step_{i}")
                action = (step.get("action") or "separate").lower()
                model_id = step.get("model_id")
                source_model = step.get("source_model")

                label = model_id or source_model or action
                self._update_job_progress(
                    job.id,
                    step_base,
                    f"Running step {i + 1}: {step_name} ({label})",
                )

                # Determine input file
                current_input = job.file_path

                # By default, pipeline steps should chain: later steps should operate on outputs
                # from earlier steps. Many recipes omit explicit input_source, so we infer
                # the intended stem using apply_to / recipe_target.
                if i > 0 and "input_source" not in step:
                    prev_step = job.pipeline_config[i - 1]
                    prev_step_name = prev_step.get("step_name", f"step_{i - 1}")
                    prev_outputs = step_outputs.get(prev_step_name) or {}

                    if isinstance(prev_outputs, dict) and prev_outputs:
                        desired_stem = step.get("apply_to") or step.get("recipe_target")
                        if isinstance(desired_stem, str):
                            desired_stem = desired_stem.strip().lower()
                        else:
                            desired_stem = None

                        # Sensible fallback ordering.
                        candidates = []
                        if desired_stem:
                            candidates.append(desired_stem)
                        candidates.extend(["instrumental", "vocals", "target"])

                        picked = None
                        for k in candidates:
                            if k in prev_outputs and prev_outputs.get(k):
                                picked = prev_outputs.get(k)
                                break
                        if not picked:
                            picked = list(prev_outputs.values())[0]
                        current_input = picked

                if "input_source" in step:
                    parts = step["input_source"].split(".")
                    src_step = parts[0]
                    src_stem = parts[1] if len(parts) > 1 else None

                    if src_step in step_outputs:
                        outs = step_outputs[src_step]
                        if src_stem and src_stem in outs:
                            current_input = outs[src_stem]
                        elif outs:
                            current_input = list(outs.values())[0]
                        else:
                            raise ValueError(f"No output from step {src_step}")
                    else:
                        self.logger.warning(
                            f"Input source {src_step} not found, using original."
                        )

                # Run separation
                step_out_dir = pipeline_temp / step_name
                step_out_dir.mkdir(exist_ok=True)

                # Special step: Phase Fix
                if action == "phase_fix":
                    if not source_model:
                        raise ValueError(
                            f"Pipeline step '{step_name}' action=phase_fix requires source_model"
                        )
                    if i == 0:
                        raise ValueError(
                            f"Pipeline step '{step_name}' action=phase_fix requires a previous step output"
                        )

                    prev_step = job.pipeline_config[i - 1]
                    prev_step_name = prev_step.get("step_name", f"step_{i - 1}")
                    prev_outputs = step_outputs.get(prev_step_name) or {}

                    apply_to = (step.get("apply_to") or "instrumental").lower()
                    target_path = prev_outputs.get(apply_to)
                    if not target_path and prev_outputs:
                        # Best-effort fallback (first output file)
                        target_path = list(prev_outputs.values())[0]
                    if not target_path:
                        raise ValueError(
                            f"Pipeline step '{step_name}' action=phase_fix could not find previous outputs to apply to"
                        )

                    # Run reference separation (source model) on the original input for clean phase
                    ref_dir = step_out_dir / "_phase_fix_ref"
                    ref_dir.mkdir(exist_ok=True)

                    ref_overlap, ref_seg_size, ref_batch_size, ref_tta = (
                        self._apply_recommended_defaults(
                            model_id=source_model,
                            overlap=job.overlap,
                            segment_size=job.segment_size,
                            batch_size=getattr(job, "batch_size", 1),
                            tta=job.tta,
                        )
                    )

                    ref_outputs, _ = await self.engine.separate(
                        job.file_path,
                        source_model,
                        str(ref_dir),
                        stems=["vocals", "instrumental"],
                        device=job.device,
                        overlap=ref_overlap,
                        segment_size=ref_seg_size,
                        batch_size=ref_batch_size,
                        tta=ref_tta,
                        shifts=job.shifts,
                        progress_callback=lambda p, m: self._update_job_progress(
                            job.id,
                            min(step_base + step_span * 0.7, step_base + (step_span * 0.7) * (float(p) / 100.0)),
                            f"{step_name} (ref): {m}",
                        ),
                        check_cancelled=check_cancelled,
                    )

                    ref_outputs = self._canonicalize_stem_keys(ref_outputs) or {}

                    def _pick(
                        outputs: Dict[str, str], candidates: List[str]
                    ) -> Optional[str]:
                        for k in candidates:
                            if k in outputs and outputs[k]:
                                return outputs[k]
                        return None

                    ref_vocal_path = _pick(ref_outputs, ["vocals", "vocal"])
                    ref_inst_path = _pick(
                        ref_outputs,
                        ["instrumental", "inst", "karaoke", "other", "accompaniment"],
                    )
                    if not ref_vocal_path or not ref_inst_path:
                        raise ValueError(
                            f"Pipeline step '{step_name}' phase_fix requires vocals+instrumental from source_model '{source_model}', got {list(ref_outputs.keys())}"
                        )

                    # Load waveforms
                    target_audio, target_sr = sf.read(target_path)
                    ref_vocal_audio, ref_vocal_sr = sf.read(ref_vocal_path)
                    ref_inst_audio, ref_inst_sr = sf.read(ref_inst_path)
                    if not (target_sr == ref_vocal_sr == ref_inst_sr):
                        raise ValueError(
                            f"Pipeline step '{step_name}' phase_fix sample rates mismatch: target={target_sr}, vocal={ref_vocal_sr}, inst={ref_inst_sr}"
                        )

                    if not TORCH_AVAILABLE:
                        raise ValueError(
                            "Phase Fix requires torch; torch not available"
                        )

                    blender = EnsembleBlender()
                    target_cf = blender._ensure_channels_first(target_audio)
                    ref_vocal_cf = blender._ensure_channels_first(ref_vocal_audio)
                    ref_inst_cf = blender._ensure_channels_first(ref_inst_audio)

                    low_hz = int(step.get("lowHz") or step.get("low_hz") or 500)
                    high_hz = int(step.get("highHz") or step.get("high_hz") or 5000)
                    high_freq_weight = float(
                        step.get("weight")
                        or step.get("highFreqWeight")
                        or step.get("high_freq_weight")
                        or 2.0
                    )

                    self.logger.info(
                        f"Pipeline Phase Fix: apply_to={apply_to} source_model={source_model} {low_hz}-{high_hz}Hz weight={high_freq_weight}"
                    )

                    phase_fixed_only = blender._phase_fix_blend(
                        target_cf,
                        ref_vocal_cf,
                        low_hz=low_hz,
                        high_hz=high_hz,
                        high_freq_weight=high_freq_weight,
                        sample_rate=target_sr,
                    )
                    fixed_cf = blender._max_spec_blend([phase_fixed_only, ref_inst_cf])

                    # Convert back to (samples, channels) for saving
                    fixed = fixed_cf
                    if isinstance(fixed, np.ndarray) and fixed.ndim == 2:
                        fixed = fixed.T
                    elif isinstance(fixed, np.ndarray) and fixed.ndim == 1:
                        fixed = fixed
                    else:
                        fixed = np.asarray(fixed)
                        if fixed.ndim == 2:
                            fixed = fixed.T

                    out_path = step_out_dir / f"{apply_to}.wav"
                    self._save_audio(
                        fixed,
                        target_sr,
                        str(out_path),
                        "wav",
                        bitrate=getattr(job, "bitrate", "320k"),
                        normalize=job.normalize,
                        bit_depth=getattr(job, "bit_depth", "16"),
                    )
                    self._write_metadata(str(out_path), metadata, apply_to)

                    self._update_job_progress(
                        job.id,
                        step_base + step_span,
                        f"{step_name}: phase fix complete",
                    )

                    # Carry forward previous outputs, but replace the applied stem
                    new_outputs = dict(prev_outputs)
                    new_outputs[apply_to] = str(out_path)

                    desired = step.get("output")
                    if desired is not None:
                        desired_stems = (
                            desired if isinstance(desired, list) else [desired]
                        )
                        filtered = {
                            stem: path
                            for stem, path in new_outputs.items()
                            if stem in desired_stems
                        }
                        if filtered:
                            new_outputs = filtered
                        else:
                            self.logger.warning(
                                f"Step '{step_name}' requested output {desired_stems} but produced stems {list(new_outputs.keys())}. "
                                "Keeping all outputs."
                            )

                    step_outputs[step_name] = new_outputs
                    # Continue chaining on the updated stem
                    current_input = new_outputs.get(apply_to) or current_input
                    self._cleanup_memory()
                    continue

                if not model_id:
                    raise ValueError(
                        f"Pipeline step '{step_name}' is missing model_id (action={action}). "
                        "If this is a phase_fix step, it must include source_model."
                    )

                step_overlap, step_seg_size, step_batch_size, step_tta = (
                    self._apply_recommended_defaults(
                        model_id=model_id,
                        overlap=job.overlap,
                        segment_size=job.segment_size,
                        batch_size=getattr(job, "batch_size", 1),
                        tta=job.tta,
                    )
                )

                outputs, _ = await self.engine.separate(
                    current_input,
                    model_id,
                    str(step_out_dir),
                    device=job.device,
                    overlap=step_overlap,
                    segment_size=step_seg_size,
                    batch_size=step_batch_size,
                    tta=step_tta,
                    shifts=job.shifts,
                    progress_callback=lambda p, m: self._update_job_progress(
                        job.id,
                        step_base + step_span * (float(p) / 100.0),
                        f"{step_name}: {m}",
                    ),
                    check_cancelled=check_cancelled,
                )

                outputs = self._canonicalize_stem_keys(outputs) or {}

                desired = step.get("output")
                if desired is not None:
                    desired_stems = desired if isinstance(desired, list) else [desired]
                    filtered = {
                        stem: path
                        for stem, path in outputs.items()
                        if stem in desired_stems
                    }
                    if filtered:
                        outputs = filtered
                    else:
                        self.logger.warning(
                            f"Step '{step_name}' requested output {desired_stems} but produced stems {list(outputs.keys())}. "
                            "Keeping all outputs."
                        )

                step_outputs[step_name] = outputs
                # Advance the chain for the next step, preferring the intended target stem.
                if isinstance(outputs, dict) and outputs:
                    next_stem = step.get("recipe_target")
                    if isinstance(next_stem, str):
                        next_stem = next_stem.strip().lower()
                    else:
                        next_stem = None
                    if next_stem and outputs.get(next_stem):
                        current_input = outputs.get(next_stem)
                    elif outputs.get("instrumental"):
                        current_input = outputs.get("instrumental")
                    elif outputs.get("vocals"):
                        current_input = outputs.get("vocals")
                    elif outputs.get("target"):
                        current_input = outputs.get("target")
                    else:
                        current_input = list(outputs.values())[0]
                self._cleanup_memory()

            if check_cancelled():
                raise asyncio.CancelledError()

            # Finalize: Copy last step outputs to output_dir
            self._update_job_progress(job.id, 95.0, "Finalizing pipeline...")
            final_outputs = {}

            last_step = job.pipeline_config[-1]
            last_step_name = last_step.get(
                "step_name", f"step_{len(job.pipeline_config) - 1}"
            )
            last_outputs = step_outputs.get(last_step_name, {})

            last_outputs = self._map_pipeline_final_outputs(
                last_outputs=last_outputs,
                step_outputs=step_outputs,
                pipeline_config=job.pipeline_config,
            )

            for stem, path in last_outputs.items():
                if check_cancelled():
                    raise asyncio.CancelledError()

                output_format = getattr(job, "output_format", "wav").lower()
                out_path = Path(job.output_dir) / f"{stem}.{output_format}"

                # Convert/Copy
                data, sr = sf.read(path)
                self._save_audio(
                    data,
                    sr,
                    str(out_path),
                    output_format,
                    bitrate=getattr(job, "bitrate", "320k"),
                    normalize=job.normalize,
                    bit_depth=getattr(job, "bit_depth", "16"),
                )
                self._write_metadata(str(out_path), metadata, stem)
                final_outputs[stem] = str(out_path)

            # Cleanup
            import shutil

            try:
                shutil.rmtree(pipeline_temp)
            except Exception as e:
                self.logger.warning(f"Failed to clean temp: {e}")

            job.output_files = final_outputs
            job.status = "completed"
            job.progress = 100.0
            if job.on_complete:
                try:
                    job.on_complete(final_outputs)
                except Exception as e:
                    self.logger.error(f"Completion callback failed: {e}")

        except asyncio.CancelledError:
            job.status = "cancelled"
            self.logger.info(f"Pipeline Job {job.id} cancelled")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            job.end_time = asyncio.get_event_loop().time()
            self._cleanup_memory()

    @staticmethod
    def _map_pipeline_final_outputs(
        last_outputs: dict, step_outputs: dict, pipeline_config: list
    ) -> dict:
        """Map pipeline-final outputs into canonical stems when the last step uses generic keys.

        Some post-processing models emit keys like {target,residual}. For UI/exports we
        prefer canonical stems (instrumental/vocals) when possible.
        """
        mapped_outputs = (
            dict(last_outputs) if isinstance(last_outputs, dict) else last_outputs
        )
        if not isinstance(mapped_outputs, dict):
            return mapped_outputs

        # target -> instrumental
        if "target" in mapped_outputs and "instrumental" not in mapped_outputs:
            mapped_outputs["instrumental"] = mapped_outputs["target"]
            mapped_outputs.pop("target", None)

        # Preserve vocals from earlier steps if we have them; otherwise, fall back to residual.
        if "vocals" not in mapped_outputs:
            vocals_candidate = None
            if isinstance(pipeline_config, list):
                for prev_step in reversed(pipeline_config):
                    prev_name = (prev_step or {}).get("step_name")
                    if not prev_name:
                        continue
                    prev_outs = (step_outputs or {}).get(prev_name) or {}
                    if isinstance(prev_outs, dict) and prev_outs.get("vocals"):
                        vocals_candidate = prev_outs.get("vocals")
                        break

            if vocals_candidate:
                mapped_outputs["vocals"] = vocals_candidate
                mapped_outputs.pop("residual", None)
            elif "residual" in mapped_outputs:
                mapped_outputs["vocals"] = mapped_outputs["residual"]
                mapped_outputs.pop("residual", None)

        return mapped_outputs

    def _cleanup_memory(self):
        """Force garbage collection and clear GPU cache"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            aggressive = os.getenv("STEMSEP_AGGRESSIVE_GPU_CLEANUP", "").lower() in (
                "1",
                "true",
                "yes",
            )
            should_clear = aggressive
            if not should_clear:
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    # Only clear if we're under memory pressure.
                    should_clear = bool(total_b) and (free_b / float(total_b)) < 0.15
                except Exception:
                    should_clear = False

            if should_clear:
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        gc.collect()
        self.logger.debug("Memory cleanup performed")

    async def _run_job(self, job: SeparationJob):
        """Run a standard separation job"""
        try:
            self.logger.info(f"Starting job {job.id} execution")
            job.status = "running"
            job.start_time = asyncio.get_event_loop().time()

            def check_cancelled():
                return job.status == "cancelled"

            # Validate audio file
            job.status = "validating"
            self.logger.info(f"Validating audio file: {job.file_path}")
            if check_cancelled():
                raise asyncio.CancelledError()

            validation = self.engine.validate_audio_file(job.file_path)
            if not validation["valid"]:
                raise Exception(f"Invalid audio file: {validation.get('error')}")
            self.logger.info("Audio validation successful")

            # Save checkpoint after validation
            self._save_job_checkpoint(job.id, "validated", {"file_path": job.file_path})

            # Extract metadata
            metadata = self._extract_metadata(job.file_path)

            # Perform separation
            job.status = "processing"
            self._update_job_progress(job.id, 5.0, "Starting separation...")
            self._save_job_checkpoint(job.id, "processing", {"model_id": job.model_id})
            self.logger.info(f"Calling engine.separate for model {job.model_id}")

            if check_cancelled():
                raise asyncio.CancelledError()

            output_files, used_device = await self.engine.separate(
                job.file_path,
                job.model_id,
                job.output_dir,
                stems=job.stems,
                device=job.device,
                overlap=job.overlap,
                segment_size=job.segment_size,
                batch_size=getattr(job, "batch_size", 1),
                tta=job.tta,
                shifts=getattr(job, "shifts", 1),
                progress_callback=lambda p, m, d=None: self._update_job_progress(
                    job.id, p, m, device=d
                ),
                check_cancelled=check_cancelled,
            )
            job.actual_device = used_device
            self.logger.info(
                f"Engine separation completed on {used_device}. Output files: {output_files}"
            )

            if check_cancelled():
                raise asyncio.CancelledError()

            # Convert output files if needed
            final_output_files = {}
            output_format = getattr(job, "output_format", "wav").lower()

            if output_format != "wav":
                self._update_job_progress(
                    job.id, 95.0, f"Converting to {output_format}..."
                )

                for stem, path in output_files.items():
                    if check_cancelled():
                        raise asyncio.CancelledError()

                    # Load the separated WAV file
                    data, sr = sf.read(path)

                    # Save in new format
                    new_path = str(Path(job.output_dir) / f"{stem}.{output_format}")
                    self._save_audio(
                        data,
                        sr,
                        new_path,
                        output_format,
                        bitrate=getattr(job, "bitrate", "320k"),
                        normalize=job.normalize,
                        bit_depth=getattr(job, "bit_depth", "16"),
                    )

                    # Write metadata
                    self._write_metadata(new_path, metadata, stem)

                    # Remove original WAV if different
                    if new_path != path:
                        try:
                            os.remove(path)
                        except:
                            pass

                    final_output_files[stem] = new_path
                job.output_files = final_output_files
            else:
                # For WAV, just write metadata to existing files
                for stem, path in output_files.items():
                    self._write_metadata(path, metadata, stem)
                job.output_files = output_files

            # Handle mixes (Instrumental/Karaoke)
            export_mixes = getattr(job, "export_mixes", [])
            if export_mixes:
                self._process_mixes(
                    job.output_files,
                    export_mixes,
                    job.output_dir,
                    output_format,
                    metadata,
                    bitrate=getattr(job, "bitrate", "320k"),
                    normalize=job.normalize,
                    bit_depth=getattr(job, "bit_depth", "16"),
                )

            # Perform spectral inversion if requested
            if getattr(job, "invert", False):
                self._perform_spectral_inversion(job)

            job.status = "completed"
            job.progress = 100.0
            self._clear_job_checkpoint(job.id)  # Clear checkpoint on success
            if job.on_complete:
                try:
                    job.on_complete(job.output_files)
                except Exception as e:
                    self.logger.error(f"Completion callback failed: {e}")

        except asyncio.CancelledError:
            job.status = "cancelled"
            self.logger.info(f"Job {job.id} cancelled by user")
            self._clear_job_checkpoint(job.id)  # Clear checkpoint on cancel
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Job {job.id} failed: {e}")
            # Keep checkpoint on failure for potential retry/debugging
        finally:
            job.end_time = asyncio.get_event_loop().time()
            self._cleanup_memory()

    async def _run_ensemble_job(self, job: SeparationJob):
        """Run an ensemble separation job"""
        try:
            job.status = "running"
            job.start_time = asyncio.get_event_loop().time()

            def check_cancelled():
                return job.status == "cancelled"

            # Validate audio
            job.status = "validating"
            if check_cancelled():
                raise asyncio.CancelledError()

            validation = self.engine.validate_audio_file(job.file_path)
            if not validation["valid"]:
                raise Exception(f"Invalid audio file: {validation.get('error')}")

            # Extract metadata
            metadata = self._extract_metadata(job.file_path)

            # Prepare for ensemble
            config = job.ensemble_config
            total_models = len(config)
            stem_map: Dict[str, List[np.ndarray]] = {}
            weights = []

            # Create temporary directory for intermediate outputs
            temp_dir = Path(job.output_dir) / f"temp_{job.id}"
            temp_dir.mkdir(exist_ok=True)

            # Run each model sequentially (to save VRAM)
            # In future, could run in parallel if VRAM allows
            for i, model_cfg in enumerate(config):
                if check_cancelled():
                    raise asyncio.CancelledError()

                model_id = model_cfg["model_id"]
                weight = model_cfg.get("weight", 1.0)
                weights.append(weight)

                progress_start = 10 + (i / total_models) * 80
                progress_span = 80 / max(1, total_models)
                self._update_job_progress(
                    job.id,
                    progress_start,
                    f"Running model {i + 1}/{total_models}: {model_id}",
                )

                # Determine segment size for this model
                current_overlap, current_seg_size, current_batch_size, current_tta = (
                    self._apply_recommended_defaults(
                        model_id=model_id,
                        overlap=getattr(job, "overlap", None),
                        segment_size=getattr(job, "segment_size", 0),
                        batch_size=getattr(job, "batch_size", 1),
                        tta=getattr(job, "tta", False),
                    )
                )

                # Run separation for this model
                model_output_dir = temp_dir / f"model_{i}"
                model_output_dir.mkdir(exist_ok=True)

                # We assume default params for now, or could extract from config
                output_files, used_device = await self.engine.separate(
                    job.file_path,
                    model_id,
                    str(model_output_dir),
                    device=job.device,
                    overlap=current_overlap,
                    segment_size=current_seg_size,
                    batch_size=current_batch_size,
                    tta=current_tta,
                    shifts=getattr(job, "shifts", 1),
                    progress_callback=lambda p, m: self._update_job_progress(
                        job.id,
                        progress_start + progress_span * (float(p) / 100.0),
                        f"{model_id}: {m}",
                    ),
                    check_cancelled=check_cancelled,
                )
                job.actual_device = used_device  # Update with latest used device

                if check_cancelled():
                    raise asyncio.CancelledError()

                # Load results into memory
                for stem, path in output_files.items():
                    data, sr = sf.read(path)

                    # Normalize stem names for consistent ensemble blending
                    # Different models may call the same stem differently
                    normalized_stem = stem
                    if stem == "other":
                        # 'other' is typically the non-vocal part (instrumental)
                        # Check if we already have 'instrumental' from another model
                        if "instrumental" in stem_map or i > 0:
                            normalized_stem = "instrumental"
                    elif stem == "instrumental":
                        # Keep as is, but log if we also have 'other'
                        if "other" in stem_map:
                            # Move existing 'other' data to 'instrumental'
                            if "instrumental" not in stem_map:
                                stem_map["instrumental"] = []
                            stem_map["instrumental"].extend(stem_map.pop("other"))

                    if normalized_stem not in stem_map:
                        stem_map[normalized_stem] = []
                    stem_map[normalized_stem].append(data)

                # Cleanup memory after each model to prevent VRAM accumulation
                self._cleanup_memory()

            if check_cancelled():
                raise asyncio.CancelledError()

            # Blending
            self._update_job_progress(job.id, 90.0, "Blending stems...")

            # DEBUG: Log stem_map details before blending
            self.logger.info(f"stem_map keys: {list(stem_map.keys())}")
            for stem_name, arrays in stem_map.items():
                shapes = [arr.shape for arr in arrays]
                self.logger.info(
                    f"  {stem_name}: {len(arrays)} arrays, shapes: {shapes}"
                )

            blender = EnsembleBlender()

            # Use per-stem algorithms if provided, otherwise use global algorithm
            stem_algorithms = getattr(job, "stem_algorithms", None)

            phase_fix_enabled = getattr(job, "phase_fix_enabled", False)
            pf_params = getattr(job, "phase_fix_params", None) or {}

            def _save_phase_fix_intermediate(phase_fixed_audio: np.ndarray):
                try:
                    phase_fix_preview = (
                        Path(job.output_dir) / "instrumental_phase_fixed.wav"
                    )
                    preview_data = phase_fixed_audio
                    if (
                        preview_data.ndim == 2
                        and preview_data.shape[0] < preview_data.shape[1]
                    ):
                        preview_data = preview_data.T
                    self.logger.info(
                        f"Saving Phase Fix intermediate instrumental to {phase_fix_preview}"
                    )
                    self._save_audio(
                        preview_data,
                        44100,
                        str(phase_fix_preview),
                        "wav",
                        bitrate=getattr(job, "bitrate", "320k"),
                        normalize=False,
                        bit_depth=getattr(job, "bit_depth", "16"),
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to save Phase Fix intermediate file: {e}"
                    )

            def _log_phase_fix_verification(
                original_audio: np.ndarray,
                phase_fixed_audio: np.ndarray,
                reference_vocal: np.ndarray,
                low_hz: int,
                high_hz: int,
            ):
                try:
                    a0 = blender._ensure_channels_first(original_audio)
                    a1 = blender._ensure_channels_first(phase_fixed_audio)
                    rv = blender._ensure_channels_first(reference_vocal)

                    analysis_len = min(
                        a0.shape[-1], a1.shape[-1], rv.shape[-1], 44100 * 10
                    )
                    if analysis_len <= 0:
                        return

                    x0 = a0[:, :analysis_len].astype(np.float64, copy=False)
                    x1 = a1[:, :analysis_len].astype(np.float64, copy=False)
                    diff = x1 - x0
                    rms_in = float(np.sqrt(np.mean(x0 * x0)))
                    rms_diff = float(np.sqrt(np.mean(diff * diff)))
                    rel = rms_diff / max(rms_in, 1e-12)
                    self.logger.info(
                        f"Phase Fix verification (time-domain, first {analysis_len} samples): rms_in={rms_in:.6f}, rms_diff={rms_diff:.6f}, rel_diff={rel:.6f}"
                    )

                    if TORCH_AVAILABLE:
                        n_fft = 4096
                        hop_length = 1024
                        if analysis_len < n_fft * 2:
                            return

                        device = torch.device("cpu")
                        window = torch.hann_window(n_fft, device=device)
                        freq_per_bin = 44100 / n_fft
                        low_bin = max(0, int(low_hz / freq_per_bin))
                        high_bin = min(n_fft // 2 + 1, int(high_hz / freq_per_bin))
                        if high_bin <= low_bin:
                            return

                        xin = torch.from_numpy(x0[0].astype(np.float32, copy=False)).to(
                            device
                        )
                        xout = torch.from_numpy(
                            x1[0].astype(np.float32, copy=False)
                        ).to(device)
                        xref = torch.from_numpy(
                            rv[0, :analysis_len].astype(np.float32, copy=False)
                        ).to(device)

                        spec_in = torch.stft(
                            xin,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            return_complex=True,
                        )
                        spec_out = torch.stft(
                            xout,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            return_complex=True,
                        )
                        spec_ref = torch.stft(
                            xref,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            return_complex=True,
                        )

                        ang_in = torch.angle(spec_in)[low_bin:high_bin]
                        ang_out = torch.angle(spec_out)[low_bin:high_bin]
                        ang_ref = torch.angle(spec_ref)[low_bin:high_bin]

                        def _wrap(d: torch.Tensor) -> torch.Tensor:
                            return torch.atan2(torch.sin(d), torch.cos(d))

                        mad_before = float(
                            torch.mean(torch.abs(_wrap(ang_in - ang_ref))).cpu().item()
                        )
                        mad_after = float(
                            torch.mean(torch.abs(_wrap(ang_out - ang_ref))).cpu().item()
                        )
                        self.logger.info(
                            f"Phase Fix verification (phase alignment, {low_hz}-{high_hz}Hz): mean_abs_diff_before={mad_before:.6f} rad, after={mad_after:.6f} rad"
                        )
                except Exception as e:
                    self.logger.warning(f"Phase Fix verification failed: {e}")

            self.logger.info(
                f"Phase Fix enabled: {phase_fix_enabled}, algorithm: {job.ensemble_algorithm}"
            )

            # Step 1: Apply Phase Fix if enabled (as a PRE-STEP)
            phase_fixed_stem_map = stem_map
            if phase_fix_enabled:
                self.logger.info(
                    "Phase Fix enabled - applying phase correction BEFORE ensemble..."
                )
                self._update_job_progress(
                    job.id, 91.0, "Phase Fix: applying correction..."
                )

                inst_list = stem_map.get("instrumental", [])
                voc_list = stem_map.get("vocals", [])

                low_hz = int(pf_params.get("lowHz", 500))
                high_hz = int(pf_params.get("highHz", 5000))
                high_freq_weight = float(pf_params.get("highFreqWeight", 2.0))

                if len(inst_list) >= 2 and len(voc_list) >= 1:
                    fullness_inst = inst_list[0]
                    ref_inst = inst_list[1]
                    ref_vocal = voc_list[-1]

                    fullness_inst_cf = blender._ensure_channels_first(fullness_inst)
                    ref_inst_cf = blender._ensure_channels_first(ref_inst)
                    ref_vocal_cf = blender._ensure_channels_first(ref_vocal)
                    min_len = min(
                        fullness_inst_cf.shape[-1],
                        ref_inst_cf.shape[-1],
                        ref_vocal_cf.shape[-1],
                    )
                    fullness_inst_cf = fullness_inst_cf[..., :min_len]
                    ref_inst_cf = ref_inst_cf[..., :min_len]
                    ref_vocal_cf = ref_vocal_cf[..., :min_len]

                    phase_fixed_only = blender._phase_fix_blend(
                        fullness_inst_cf,
                        ref_vocal_cf,
                        low_hz=low_hz,
                        high_hz=high_hz,
                        high_freq_weight=high_freq_weight,
                    )

                    _log_phase_fix_verification(
                        fullness_inst_cf,
                        phase_fixed_only,
                        ref_vocal_cf,
                        low_hz,
                        high_hz,
                    )
                    _save_phase_fix_intermediate(phase_fixed_only)

                    fixed_inst = blender._max_spec_blend(
                        [phase_fixed_only, ref_inst_cf]
                    )
                    phase_fixed_stem_map = dict(stem_map)
                    phase_fixed_stem_map["instrumental"] = [fixed_inst]
                    self.logger.info(
                        "Phase Fix complete - preparing ensemble with corrected instrumental..."
                    )
                    self._update_job_progress(
                        job.id, 92.0, "Phase Fix: correction complete"
                    )

                elif len(inst_list) >= 1 and len(voc_list) >= 1:
                    self.logger.warning(
                        "Phase Fix: Only 1 instrumental, applying basic phase correction"
                    )

                    inst_cf = blender._ensure_channels_first(inst_list[0])
                    voc_cf = blender._ensure_channels_first(voc_list[-1])
                    min_len = min(inst_cf.shape[-1], voc_cf.shape[-1])
                    inst_cf = inst_cf[..., :min_len]
                    voc_cf = voc_cf[..., :min_len]

                    phase_fixed_only = blender._phase_fix_blend(
                        inst_cf,
                        voc_cf,
                        low_hz=low_hz,
                        high_hz=high_hz,
                        high_freq_weight=high_freq_weight,
                    )

                    _log_phase_fix_verification(
                        inst_cf, phase_fixed_only, voc_cf, low_hz, high_hz
                    )
                    _save_phase_fix_intermediate(phase_fixed_only)

                    phase_fixed_stem_map = dict(stem_map)
                    phase_fixed_stem_map["instrumental"] = [phase_fixed_only]

                else:
                    self.logger.warning(
                        "Phase Fix: Insufficient stems, skipping phase correction"
                    )
                    self._update_job_progress(
                        job.id, 91.0, "Phase Fix: skipped (insufficient stems)"
                    )

            # Step 2: Apply user's ensemble algorithm
            vc_enabled = bool(getattr(job, "vc_enabled", False))
            vc_stage = (
                str(getattr(job, "vc_stage", "export") or "export").strip().lower()
            )
            vc_db = getattr(job, "vc_db_per_extra_model", 3.0)
            stem_source_counts = {
                k: len(v)
                for k, v in phase_fixed_stem_map.items()
                if isinstance(v, list)
            }

            if vc_enabled and vc_stage in ("blend", "both"):
                self.logger.info(
                    f"Volume Compensation enabled (stage={vc_stage}, db_per_extra_model={vc_db}) - applying before blend"
                )
                vc_map = {}
                for stem_name, audio_list in phase_fixed_stem_map.items():
                    if not isinstance(audio_list, list) or not audio_list:
                        continue
                    gain = self._volume_compensation_gain(len(audio_list), vc_db)
                    if gain == 1.0:
                        vc_map[stem_name] = audio_list
                        continue
                    vc_map[stem_name] = [
                        np.asarray(a, dtype=np.float32) * gain for a in audio_list
                    ]
                phase_fixed_stem_map = vc_map

            if stem_algorithms:
                self.logger.info(f"Using per-stem algorithms: {stem_algorithms}")
                blended_stems = blender.blend_stems_mixed(
                    phase_fixed_stem_map,
                    algorithms=stem_algorithms,
                    weights=weights if job.ensemble_algorithm == "average" else None,
                    split_freq=getattr(job, "split_freq", 750),
                )
            else:
                # Use per-stem blending wrapper so weights are only applied when
                # the stem has a matching number of sources (e.g. Phase Fix may
                # collapse instrumental to a single corrected source).
                algorithms = {
                    stem_name: job.ensemble_algorithm
                    for stem_name in phase_fixed_stem_map.keys()
                }
                blended_stems = blender.blend_stems_mixed(
                    phase_fixed_stem_map,
                    algorithms=algorithms,
                    weights=weights if job.ensemble_algorithm == "average" else None,
                    split_freq=getattr(job, "split_freq", 750),
                    low_hz=pf_params.get("lowHz", 500),
                    high_hz=pf_params.get("highHz", 5000),
                    high_freq_weight=pf_params.get("highFreqWeight", 2.0),
                )

            # Optional post-processing (recipe-driven)
            post_processing = getattr(job, "post_processing", None)
            if post_processing:
                self.logger.info(f"Post-processing requested: {post_processing}")

                async def _infer_from_audio_cf(
                    audio_cf: np.ndarray,
                    model_id: str,
                    step_dir: Path,
                    prefix: str,
                ) -> Dict[str, str]:
                    step_dir.mkdir(exist_ok=True)
                    in_path = step_dir / f"{prefix}.wav"

                    to_write = audio_cf
                    if to_write.ndim == 2 and to_write.shape[0] < to_write.shape[1]:
                        to_write = to_write.T

                    try:
                        sf.write(str(in_path), to_write, 44100, subtype="FLOAT")
                    except Exception:
                        sf.write(str(in_path), to_write, 44100)

                    demudder_seg_size = getattr(job, "segment_size", 0)
                    if not demudder_seg_size:
                        minfo = self.model_manager.get_model(model_id)
                        rec = (
                            getattr(minfo, "recommended_settings", None)
                            if minfo
                            else None
                        )
                        rec_seg = (
                            rec.get("segment_size") if isinstance(rec, dict) else None
                        )
                        demudder_seg_size = int(rec_seg) if rec_seg else 352800

                    out_files, used_device = await self.engine.separate(
                        str(in_path),
                        model_id,
                        str(step_dir),
                        device=job.device,
                        overlap=job.overlap,
                        segment_size=demudder_seg_size,
                        batch_size=getattr(job, "batch_size", 1),
                        tta=job.tta,
                        shifts=getattr(job, "shifts", 1),
                        progress_callback=lambda p, m: self._update_job_progress(
                            job.id,
                            93.0 + 1.5 * (float(p) / 100.0),
                            f"Post-processing ({model_id}): {m}",
                        ),
                        check_cancelled=check_cancelled,
                    )
                    job.actual_device = used_device
                    return out_files

                if post_processing == "demudder_phase_rotate":
                    # Demudder requires a second inference pass on a phase-rotated mixture.
                    # We implement this in async-safe form (no run_until_complete).
                    try:
                        import librosa
                        from scipy.signal import hilbert
                    except Exception as e:
                        self.logger.warning(
                            f"Demudder unavailable (missing dependency): {e}"
                        )
                    else:
                        # Choose a model for the demudder pass. For recipe_demudder_metal this is a single-model ensemble.
                        demudder_model_id = None
                        try:
                            if getattr(job, "ensemble_config", None):
                                demudder_model_id = job.ensemble_config[0].get(
                                    "model_id"
                                )
                        except Exception:
                            demudder_model_id = None

                        if not demudder_model_id:
                            self.logger.warning(
                                "Demudder: unable to determine model_id; skipping"
                            )
                        else:
                            if check_cancelled():
                                raise asyncio.CancelledError()

                            self._update_job_progress(
                                job.id,
                                93.0,
                                f"Post-processing: demudder ({demudder_model_id})...",
                            )

                            # Load original mixture in the same style as inference (float32, 44.1k)
                            mix, _sr = librosa.load(job.file_path, sr=44100, mono=False)
                            mix_cf = blender._ensure_channels_first(mix)

                            # Swap L/R and apply 90° phase shift via Hilbert transform
                            if mix_cf.shape[0] >= 2:
                                swapped = np.stack([mix_cf[1], mix_cf[0]], axis=0)
                            else:
                                swapped = mix_cf

                            analytic = hilbert(swapped, axis=-1)
                            phase_shifted = np.imag(analytic).astype(np.float32)

                            inverted = mix_cf - phase_shifted

                            pp_dir = Path(job.output_dir) / f"post_{job.id}"
                            out_files = await _infer_from_audio_cf(
                                inverted, demudder_model_id, pp_dir, "demudder_inverted"
                            )

                            inst_path = out_files.get("instrumental")
                            if not inst_path and out_files:
                                inst_path = list(out_files.values())[0]

                            if inst_path:
                                inst_audio, _ = sf.read(inst_path)
                                inst_cf = blender._ensure_channels_first(inst_audio)
                                blended_stems["instrumental"] = inst_cf
                                self.logger.info(
                                    "Demudder applied: replaced instrumental stem"
                                )
                            else:
                                self.logger.warning(
                                    "Demudder: no instrumental output produced; skipping replacement"
                                )

                            self._cleanup_memory()

                elif post_processing == "demudder_phase_remix":
                    # Uses blended vocals to re-separate a constructed mixture and compute a cleaner instrumental.
                    try:
                        import librosa
                    except Exception as e:
                        self.logger.warning(
                            f"Demudder unavailable (missing dependency): {e}"
                        )
                    else:
                        demudder_model_id = None
                        try:
                            if getattr(job, "ensemble_config", None):
                                demudder_model_id = job.ensemble_config[0].get(
                                    "model_id"
                                )
                        except Exception:
                            demudder_model_id = None

                        vocals = blended_stems.get("vocals")
                        instrumental = blended_stems.get("instrumental")
                        if (
                            not demudder_model_id
                            or vocals is None
                            or instrumental is None
                        ):
                            self.logger.warning(
                                "Demudder remix: missing model_id or required stems (vocals/instrumental); skipping"
                            )
                        else:
                            if check_cancelled():
                                raise asyncio.CancelledError()
                            self._update_job_progress(
                                job.id,
                                93.0,
                                f"Post-processing: demudder remix ({demudder_model_id})...",
                            )

                            original, _ = librosa.load(
                                job.file_path, sr=44100, mono=False
                            )
                            original_cf = blender._ensure_channels_first(original)

                            min_len = min(
                                vocals.shape[-1],
                                instrumental.shape[-1],
                                original_cf.shape[-1],
                            )
                            v = vocals[..., :min_len]
                            inst = instrumental[..., :min_len]
                            orig = original_cf[..., :min_len]

                            mixture = inst + (-v)
                            pp_dir = Path(job.output_dir) / f"post_{job.id}"
                            out_files = await _infer_from_audio_cf(
                                mixture, demudder_model_id, pp_dir, "demudder_remix"
                            )

                            voc_path = out_files.get("vocals")
                            if voc_path:
                                extracted_vocals, _ = sf.read(voc_path)
                                ev_cf = blender._ensure_channels_first(extracted_vocals)
                                demudded = orig - ev_cf[..., :min_len]
                                blended_stems["instrumental"] = demudded.astype(
                                    np.float32, copy=False
                                )
                                self.logger.info(
                                    "Demudder remix applied: replaced instrumental stem"
                                )
                            else:
                                self.logger.warning(
                                    "Demudder remix: no vocals output produced; skipping replacement"
                                )

                            self._cleanup_memory()

                elif post_processing == "demudder_combine":
                    # Runs rotate + remix variants and combines them with the original instrumental.
                    try:
                        import librosa
                        from scipy.signal import hilbert
                    except Exception as e:
                        self.logger.warning(
                            f"Demudder unavailable (missing dependency): {e}"
                        )
                    else:
                        demudder_model_id = None
                        try:
                            if getattr(job, "ensemble_config", None):
                                demudder_model_id = job.ensemble_config[0].get(
                                    "model_id"
                                )
                        except Exception:
                            demudder_model_id = None

                        vocals = blended_stems.get("vocals")
                        instrumental = blended_stems.get("instrumental")
                        if not demudder_model_id or instrumental is None:
                            self.logger.warning(
                                "Demudder combine: missing model_id or instrumental; skipping"
                            )
                        else:
                            self._update_job_progress(
                                job.id,
                                93.0,
                                f"Post-processing: demudder combine ({demudder_model_id})...",
                            )

                            mix, _ = librosa.load(job.file_path, sr=44100, mono=False)
                            mix_cf = blender._ensure_channels_first(mix)

                            # Phase rotate path
                            if mix_cf.shape[0] >= 2:
                                swapped = np.stack([mix_cf[1], mix_cf[0]], axis=0)
                            else:
                                swapped = mix_cf
                            analytic = hilbert(swapped, axis=-1)
                            phase_shifted = np.imag(analytic).astype(np.float32)
                            inverted = mix_cf - phase_shifted

                            pp_dir = Path(job.output_dir) / f"post_{job.id}"
                            rotate_out = await _infer_from_audio_cf(
                                inverted, demudder_model_id, pp_dir, "demudder_rotate"
                            )
                            rotate_inst_path = rotate_out.get("instrumental")
                            rotate_inst = None
                            if rotate_inst_path:
                                audio, _ = sf.read(rotate_inst_path)
                                rotate_inst = blender._ensure_channels_first(audio)

                            # Phase remix path (requires vocals)
                            remix_inst = None
                            if vocals is not None:
                                original_cf = mix_cf
                                min_len = min(
                                    vocals.shape[-1],
                                    instrumental.shape[-1],
                                    original_cf.shape[-1],
                                )
                                mixture = instrumental[..., :min_len] + (
                                    -vocals[..., :min_len]
                                )
                                remix_out = await _infer_from_audio_cf(
                                    mixture, demudder_model_id, pp_dir, "demudder_remix"
                                )
                                voc_path = remix_out.get("vocals")
                                if voc_path:
                                    extracted_vocals, _ = sf.read(voc_path)
                                    ev_cf = blender._ensure_channels_first(
                                        extracted_vocals
                                    )
                                    remix_inst = (
                                        original_cf[..., :min_len]
                                        - ev_cf[..., :min_len]
                                    )

                            # Combine available results
                            orig_inst = instrumental
                            if rotate_inst is None and remix_inst is None:
                                self.logger.warning(
                                    "Demudder combine: no rotate/remix result available; skipping"
                                )
                            else:
                                # Default weights: original 0.4, rotate 0.3, remix 0.3
                                w_orig, w_rot, w_rem = 0.4, 0.3, 0.3
                                parts = []
                                weights_local = []
                                parts.append(orig_inst)
                                weights_local.append(w_orig)
                                if rotate_inst is not None:
                                    parts.append(rotate_inst)
                                    weights_local.append(w_rot)
                                if remix_inst is not None:
                                    parts.append(remix_inst)
                                    weights_local.append(w_rem)

                                min_len = min(p.shape[-1] for p in parts)
                                out = np.zeros_like(
                                    parts[0][..., :min_len], dtype=np.float32
                                )
                                w_sum = float(sum(weights_local))
                                for p, w in zip(parts, weights_local):
                                    out += p[..., :min_len].astype(
                                        np.float32, copy=False
                                    ) * (float(w) / max(1e-12, w_sum))
                                blended_stems["instrumental"] = out
                                self.logger.info(
                                    "Demudder combine applied: replaced instrumental stem"
                                )

                            self._cleanup_memory()
                else:
                    self.logger.warning(
                        f"Unknown post_processing '{post_processing}'. Ignoring."
                    )

            # Save blended outputs
            self._update_job_progress(job.id, 95.0, "Saving ensemble output...")
            final_output_files = {}

            # Filter to only requested stems if specified
            requested_stems = getattr(job, "stems", None)
            for stem, data in blended_stems.items():
                # Skip stems not in the requested list
                if requested_stems and stem not in requested_stems:
                    continue

                if check_cancelled():
                    raise asyncio.CancelledError()

                # Transpose back to (samples, channels) for sf.write if needed
                # EnsembleBlender returns (channels, samples)
                if data.ndim == 2 and data.shape[0] < data.shape[1]:
                    data = data.T

                if vc_enabled and vc_stage in ("export", "both"):
                    gain = self._volume_compensation_gain(
                        stem_source_counts.get(stem, 1), vc_db
                    )
                    if gain != 1.0:
                        data = np.asarray(data, dtype=np.float32) * gain

                output_format = getattr(job, "output_format", "wav").lower()
                out_path = Path(job.output_dir) / f"{stem}.{output_format}"

                self.logger.info(f"Saving {stem} to {out_path}")
                self._save_audio(
                    data,
                    44100,
                    str(out_path),
                    output_format,
                    bitrate=getattr(job, "bitrate", "320k"),
                    normalize=job.normalize,
                    bit_depth=getattr(job, "bit_depth", "16"),
                )

                # Write metadata
                self.logger.info(f"Writing metadata for {stem}")
                self._write_metadata(str(out_path), metadata, stem)

                final_output_files[stem] = str(out_path)

            # Cleanup temp
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")

            job.output_files = final_output_files

            if getattr(job, "invert", False):
                self._perform_spectral_inversion(job)

            job.status = "completed"
            job.progress = 100.0
            if job.on_complete:
                try:
                    job.on_complete(final_output_files)
                except Exception as e:
                    self.logger.error(f"Completion callback failed: {e}")
        except asyncio.CancelledError:
            job.status = "cancelled"
            self.logger.info(f"Ensemble Job {job.id} cancelled by user")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Ensemble Job {job.id} failed: {e}")
            import traceback

            traceback.print_exc()
            with open("error_traceback.txt", "w") as f:
                f.write(traceback.format_exc())
        finally:
            job.end_time = asyncio.get_event_loop().time()
            self._cleanup_memory()

    def export_job_output(
        self, job_id: str, export_path: str, format: str = "mp3", bitrate: str = "320k"
    ) -> bool:
        """Export job output to a specific path/format"""
        with self.job_lock:
            job = self.jobs.get(job_id)

        if not job:
            self.logger.error(f"Job {job_id} not found for export")
            return False

        try:
            dest_dir = Path(export_path)
            dest_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Exporting job {job_id} to {dest_dir} as {format}")

            # Get metadata if possible (from one of the source files)
            metadata = {}
            source_files = list(job.output_files.values())
            if source_files:
                metadata = self._extract_metadata(source_files[0])

            for stem_name, src_path in job.output_files.items():
                src = Path(src_path)
                if not src.exists():
                    self.logger.warning(f"Source file missing for export: {src}")
                    continue

                # Read source
                data, sr = sf.read(str(src))

                # Determine output path
                dst_filename = f"{stem_name}.{format}"
                dst_path = dest_dir / dst_filename

                # Save/Convert
                self._save_audio(
                    data, sr, str(dst_path), format, bitrate=bitrate, normalize=False
                )

                # Write metadata
                self._write_metadata(str(dst_path), metadata, stem_name)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export job {job_id}: {e}", exc_info=True)
            return False

    def export_from_paths(
        self,
        source_files: Dict[str, str],
        export_path: str,
        format: str = "mp3",
        bitrate: str = "320k",
    ) -> bool:
        """Export files directly from paths without job lookup.

        This is used for exporting historical results after app restart,
        when the original job data is no longer in memory.

        Args:
            source_files: Dict mapping stem name to file path
            export_path: Directory to export to
            format: Output format (mp3, wav, flac)
            bitrate: Bitrate for lossy formats
        """
        try:
            dest_dir = Path(export_path)
            dest_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Exporting {len(source_files)} files to {dest_dir} as {format}"
            )

            # Get metadata from first available file
            metadata = {}
            for src_path in source_files.values():
                if Path(src_path).exists():
                    metadata = self._extract_metadata(src_path)
                    break

            exported_count = 0
            for stem_name, src_path in source_files.items():
                src = Path(src_path)
                if not src.exists():
                    self.logger.warning(f"Source file missing for export: {src}")
                    continue

                # Read source
                data, sr = sf.read(str(src))

                # Determine output path
                dst_filename = f"{stem_name}.{format}"
                dst_path = dest_dir / dst_filename

                # Save/Convert
                self._save_audio(
                    data, sr, str(dst_path), format, bitrate=bitrate, normalize=False
                )

                # Write metadata
                self._write_metadata(str(dst_path), metadata, stem_name)
                exported_count += 1

            self.logger.info(
                f"Exported {exported_count}/{len(source_files)} files successfully"
            )
            return exported_count > 0

        except Exception as e:
            self.logger.error(f"Failed to export from paths: {e}", exc_info=True)
            return False

    def save_job_output(self, job_id: str) -> bool:
        """Moves the job output from temp dir to final output dir"""
        with self.job_lock:
            job = self.jobs.get(job_id)

        if not job:
            self.logger.error(f"Job {job_id} not found")
            return False

        if job.status != "completed":
            self.logger.error(f"Job {job_id} is not completed (status: {job.status})")
            return False

        try:
            dest_dir = Path(job.final_output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Saving job {job_id} output to {dest_dir}")

            saved_files = {}

            for stem_name, temp_path in job.output_files.items():
                src = Path(temp_path)
                if src.exists():
                    dst = dest_dir / src.name
                    # Handle duplicates
                    if dst.exists():
                        base = dst.stem
                        ext = dst.suffix
                        count = 1
                        while dst.exists():
                            dst = dest_dir / f"{base}_{count}{ext}"
                            count += 1

                    shutil.move(str(src), str(dst))
                    saved_files[stem_name] = str(dst)

            # Update job output files to point to new location
            job.output_files = saved_files

            # Cleanup temp dir
            try:
                shutil.rmtree(job.temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove temp dir {job.temp_dir}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save job output: {e}", exc_info=True)
            return False

    def discard_job_output(self, job_id: str) -> bool:
        """Deletes the job output (temp dir)"""
        with self.job_lock:
            job = self.jobs.get(job_id)

        if not job:
            return False

        try:
            if os.path.exists(job.temp_dir):
                shutil.rmtree(job.temp_dir)

            # Remove from jobs list
            del self.jobs[job_id]
            return True
        except Exception as e:
            self.logger.error(f"Failed to discard job {job_id}: {e}")
            return False

    def _update_job_progress(
        self, job_id: str, progress: float, message: str, device: str = None
    ):
        """Update job progress"""
        with self.job_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.progress = progress
                if device:
                    job.actual_device = device
                if job.progress_callback:
                    try:
                        if device:
                            job.progress_callback(progress, message, device=device)
                        else:
                            job.progress_callback(progress, message)
                    except Exception as e:
                        self.logger.error(f"Progress callback error: {e}")

    def _on_progress_update(self, progress: float, message: str):
        """Handle engine progress updates"""
        # This is a default handler
        pass

    def get_job(self, job_id: str) -> Optional[SeparationJob]:
        """Get a job by ID"""
        with self.job_lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[SeparationJob]:
        """Get all jobs"""
        with self.job_lock:
            return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        with self.job_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == "running":
                    # In a real implementation, you'd cancel the async task
                    job.status = "cancelled"
                    return True
        return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a completed/failed job"""
        with self.job_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status in ["completed", "failed", "cancelled"]:
                    del self.jobs[job_id]
                    return True
        return False

    def clear_completed(self):
        """Clear all completed jobs"""
        with self.job_lock:
            to_remove = [
                jid
                for jid, job in self.jobs.items()
                if job.status in ["completed", "failed", "cancelled"]
            ]
            for jid in to_remove:
                del self.jobs[jid]

    def get_stats(self) -> Dict:
        """Get separation statistics"""
        with self.job_lock:
            total = len(self.jobs)
            running = sum(1 for j in self.jobs.values() if j.status == "running")
            completed = sum(1 for j in self.jobs.values() if j.status == "completed")
            failed = sum(1 for j in self.jobs.values() if j.status == "failed")
            pending = sum(1 for j in self.jobs.values() if j.status == "pending")

            return {
                "total": total,
                "running": running,
                "completed": completed,
                "failed": failed,
                "pending": pending,
            }

    def shutdown(self):
        """Shutdown the separation manager"""
        self.logger.info("Shutting down SeparationManager...")
        self.executor.shutdown(wait=True)

    def _save_audio(
        self,
        data: np.ndarray,
        sr: int,
        path: str,
        format: str,
        bitrate: str = "320k",
        normalize: bool = False,
        bit_depth: str = "16",
    ):
        """Helper to save audio in requested format"""

        # Sanitize data
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

        if normalize:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                # Peak normalize to -0.1 dB
                data = data / max_val * 0.988

        # Map bit depth to soundfile subtype.
        # Note: StemSep treats "32" as float32 WAV (guide recommendation: preserve headroom).
        bd = "16"
        try:
            bd = str(bit_depth).strip().lower()
        except Exception:
            bd = "16"

        subtype = "PCM_16"
        if bd in ("24", "pcm24", "pcm_24"):
            subtype = "PCM_24"
        elif bd in ("32", "float", "float32", "f32"):
            subtype = "FLOAT"
        else:
            subtype = "PCM_16"

        if format == "wav":
            sf.write(path, data, sr, subtype=subtype)
        elif format == "flac":
            # FLAC doesn't support float32, fallback to 24-bit
            if subtype == "FLOAT":
                self.logger.warning(
                    "float32 requested for FLAC; falling back to 24-bit PCM"
                )
                subtype = "PCM_24"
            sf.write(path, data, sr, format="FLAC", subtype=subtype)
        elif format == "mp3":
            if not HAS_PYDUB:
                self.logger.warning(
                    "MP3 export requested but pydub not available. Falling back to WAV."
                )
                sf.write(path.replace(".mp3", ".wav"), data, sr, subtype=subtype)
                return

            # Convert numpy array to AudioSegment
            # Ensure data is float32/float64, convert to int16 for pydub mp3 export
            # Note: pydub works with int16 mostly.
            if data.dtype in [np.float32, np.float64]:
                audio_data = (data * 32767).astype(np.int16)
            else:
                audio_data = data

            # Handle stereo/mono
            channels = 1
            if len(data.shape) > 1:
                channels = data.shape[1]

            audio_segment = AudioSegment(
                audio_data.tobytes(), frame_rate=sr, sample_width=2, channels=channels
            )

            audio_segment.export(path, format="mp3", bitrate=bitrate)
        else:
            # Fallback to WAV
            self.logger.warning(f"Unknown format {format}, falling back to WAV")
            sf.write(path, data, sr, subtype=subtype)

    def _extract_metadata(self, file_path: str) -> Dict[str, str]:
        """Extract metadata from source file"""
        if not HAS_MUTAGEN:
            return {}

        try:
            meta = {}
            # mutagen.File handles various formats automatically
            f = mutagen.File(file_path, easy=True)

            if f:
                # Map common tags
                for tag in ["title", "artist", "album", "date", "genre"]:
                    if tag in f:
                        meta[tag] = f[tag][0]

                # Handle year/date specifically if needed
                if "date" in meta:
                    meta["year"] = meta["date"]

            return meta
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return {}

    def _write_metadata(self, file_path: str, metadata: Dict[str, str], stem_name: str):
        """Write metadata to output file"""
        if not HAS_MUTAGEN or not metadata:
            return

        try:
            # Determine format from extension
            ext = Path(file_path).suffix.lower()

            if ext == ".mp3":
                audio = EasyID3(file_path)
            elif ext == ".flac":
                audio = FLAC(file_path)
            elif ext == ".wav":
                # WAV metadata is tricky, mutagen.wave.WAVE might not work for all players
                # But we'll try standard RIFF INFO chunks
                # EasyID3 doesn't work for WAV, need specific handling or use mutagen.File
                try:
                    audio = WAVE(file_path)
                except:
                    return  # Skip if WAV structure not supported

                # WAVE in mutagen doesn't support EasyID3-like dict interface directly for all tags
                # We might need to skip WAV for now or implement specific RIFF chunk writing
                # For simplicity/safety, let's skip WAV metadata for now or use basic tags if possible
                # Actually, let's try to use the generic File handler if possible, but writing is format specific
                return
            else:
                return

            # Apply tags
            for key, value in metadata.items():
                if key in ["title", "artist", "album", "date", "genre", "year"]:
                    # For title, append stem name
                    if key == "title":
                        audio["title"] = f"{value} ({stem_name})"
                    else:
                        audio[key] = value

            audio.save()

        except Exception as e:
            self.logger.warning(f"Failed to write metadata to {file_path}: {e}")

    def _process_queue(self):
        """Process the next job in queue (Placeholder)"""
        pass

    def _process_mixes(
        self,
        stems_paths: Dict[str, str],
        mixes: List[str],
        output_dir: str,
        output_format: str,
        metadata: Dict[str, str],
        bitrate: str = "320k",
        normalize: bool = False,
        bit_depth: str = "16",
    ):
        """Create requested mixes from separated stems"""
        if "instrumental" in mixes:
            try:
                # Identify non-vocal stems
                mix_stems = [
                    path
                    for stem, path in stems_paths.items()
                    if stem.lower() != "vocals"
                ]

                if not mix_stems:
                    return

                # Load and sum
                combined_data = None
                sr = 44100

                for path in mix_stems:
                    data, file_sr = sf.read(path)
                    sr = file_sr
                    if combined_data is None:
                        combined_data = data
                    else:
                        # Ensure shapes match
                        if len(data) != len(combined_data):
                            # Truncate to shorter
                            min_len = min(len(data), len(combined_data))
                            data = data[:min_len]
                            combined_data = combined_data[:min_len]
                        combined_data += data

                # Save mix
                mix_path = str(Path(output_dir) / f"Instrumental.{output_format}")
                self._save_audio(
                    combined_data,
                    sr,
                    mix_path,
                    output_format,
                    bitrate=bitrate,
                    normalize=normalize,
                    bit_depth=bit_depth,
                )

                # Write metadata
                mix_meta = metadata.copy()
                if "title" in mix_meta:
                    mix_meta["title"] = f"{mix_meta['title']} (Instrumental)"
                self._write_metadata(mix_path, mix_meta, "Instrumental")

            except Exception as e:
                self.logger.error(f"Failed to create instrumental mix: {e}")
