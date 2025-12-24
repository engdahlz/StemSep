#!/usr/bin/env python3
"""
Python bridge for Electron IPC - handles model operations.
Reads commands from stdin and writes JSON responses to stdout.

V3 IPC additions:
- model_preflight: validate that a selected model is ready (installed / resolvable),
  return expected filename, architecture, and download URL (if missing), without
  silently falling back to other models.
"""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

import re
import tempfile
import time

# Force UTF-8 for stdio to handle special characters in file paths
if sys.platform == "win32":
    if hasattr(sys.stdin, "buffer"):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# CRITICAL: Redirect all stdout to stderr to prevent library noise (ffmpeg, etc.)
# from corrupting the JSON IPC stream.
# We keep a reference to the original stdout to send our actual JSON responses.
_IPC_STREAM = sys.stdout
sys.stdout = sys.stderr

# Ensure ffmpeg is available via imageio-ffmpeg
try:
    import shutil

    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)

    # Check if we need to alias it to ffmpeg.exe for libraries that expect that name
    if Path(ffmpeg_exe).name != "ffmpeg.exe":
        target_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(target_exe):
            try:
                shutil.copy2(ffmpeg_exe, target_exe)
                sys.stderr.write(f"Created ffmpeg.exe alias at {target_exe}\n")
            except Exception as e:
                sys.stderr.write(f"WARNING: Failed to create ffmpeg.exe alias: {e}\n")

    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    # Also set for pydub/others if they check specific env vars
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe
except ImportError:
    sys.stderr.write("WARNING: imageio-ffmpeg not found. FFmpeg might be missing.\n")
except Exception as e:
    sys.stderr.write(f"WARNING: Failed to setup imageio-ffmpeg: {e}\n")

# Add StemSepApp to path.
# Dev layout:
#   <repo_root>/electron-poc/python-bridge.py
#   <repo_root>/StemSepApp/src
# Packaged layout (electron-builder extraResources):
#   <resources>/python-bridge.py
#   <resources>/StemSepApp/src
_bridge_dir = Path(__file__).resolve().parent
_candidates = [
    _bridge_dir / "StemSepApp" / "src",
    _bridge_dir.parent / "StemSepApp" / "src",
]
_added = False
for _p in _candidates:
    try:
        if _p.exists():
            sys.path.insert(0, str(_p))
            _added = True
            break
    except Exception:
        pass

if not _added:
    sys.stderr.write(
        f"WARNING: Could not locate StemSepApp/src from bridge location: {__file__}\n"
    )
    # Keep the historical dev fallback to aid debugging
    sys.path.insert(0, str(_bridge_dir.parent / "StemSepApp" / "src"))

# Graceful shutdown handling
import signal
from typing import Optional, Set

from core.gpu_detector import GPUDetector
from core.logger import setup_logging
from core.separation_manager import SeparationManager
from models.model_manager import ModelManager

_shutdown_requested = False


def get_available_devices():
    """Get available devices using GPUDetector"""
    detector = GPUDetector()
    info = detector.get_gpu_info()
    return info.get("gpus", [])


# Setup logger
log = setup_logging()


def handle_shutdown(signum, frame):
    """Handle graceful shutdown"""
    global _shutdown_requested
    log.info(f"Received shutdown signal {signum}, cleaning up...")
    _shutdown_requested = True

    # Give jobs a chance to save state
    if separation_manager:
        for job_id, job in separation_manager.jobs.items():
            if job.status == "running":
                log.info(f"Saving checkpoint for running job {job_id}")
                separation_manager._save_job_checkpoint(job_id, "interrupted")

    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
if hasattr(signal, "SIGBREAK"):  # Windows
    signal.signal(signal.SIGBREAK, handle_shutdown)

# Create global managers (initialized in main())
manager: Optional[ModelManager] = None
separation_manager: Optional[SeparationManager] = None
CURRENT_CMD_ID = None
background_tasks: Set[asyncio.Task] = set()


def send_ipc_message(msg: dict):
    """Send a raw dictionary as a JSON line to the IPC stream"""
    try:
        json_str = json.dumps(msg)
        _IPC_STREAM.write(json_str + "\n")
        _IPC_STREAM.flush()
    except Exception as e:
        log.error(f"Failed to send IPC message: {e}", exc_info=True)


def send_response(success: bool, data=None, error=None):
    """Send JSON response to Electron via the dedicated IPC stream"""
    response = {"success": success, "data": data, "error": error}
    if CURRENT_CMD_ID is not None:
        response["id"] = CURRENT_CMD_ID

    log.info(f"Sending response: success={success}")
    send_ipc_message(response)


def _is_youtube_url(value: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    v = value.strip()
    return bool(
        re.search(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/", v, re.IGNORECASE)
    )


def _cleanup_old_youtube_temp_dirs(base: Path, max_age_seconds: int = 24 * 3600):
    try:
        now = time.time()
        if not base.exists():
            return
        for p in base.glob("yt_*"):
            try:
                if not p.is_dir():
                    continue
                if now - p.stat().st_mtime > max_age_seconds:
                    import shutil

                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def _download_youtube_audio_to_wav(url: str) -> dict:
    try:
        import yt_dlp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "yt-dlp is not installed in the Python environment. Install 'yt-dlp' and try again."
        ) from e

    base_dir = Path(tempfile.gettempdir()) / "stemsep_youtube"
    base_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_youtube_temp_dirs(base_dir)

    job_dir = base_dir / f"yt_{int(time.time())}_{os.getpid()}"
    job_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = os.environ.get("FFMPEG_BINARY")
    ffmpeg_dir = None
    if ffmpeg_bin:
        try:
            ffmpeg_dir = str(Path(ffmpeg_bin).parent)
        except Exception:
            ffmpeg_dir = None

    last_progress = {"pct": None}

    def _progress_hook(d: dict):
        try:
            status = d.get("status")
            if status == "downloading":
                pct = d.get("_percent_str")
                if pct is not None and pct == last_progress["pct"]:
                    return
                last_progress["pct"] = pct
                send_ipc_message(
                    {
                        "type": "youtube_progress",
                        "status": status,
                        "percent": pct,
                        "speed": d.get("_speed_str"),
                        "eta": d.get("_eta_str"),
                    }
                )
            elif status in ("finished", "error"):
                send_ipc_message({"type": "youtube_progress", "status": status})
        except Exception:
            pass

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(job_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not info:
        raise RuntimeError("Failed to extract YouTube info")

    title = info.get("title")
    vid = info.get("id")
    if not vid:
        raise RuntimeError("Could not determine video id")

    wav_path = job_dir / f"{vid}.wav"
    if not wav_path.exists():
        wav_candidates = list(job_dir.glob("*.wav"))
        if wav_candidates:
            wav_path = wav_candidates[0]
        else:
            raise FileNotFoundError("yt-dlp did not produce a WAV file")

    return {
        "file_path": str(wav_path),
        "title": title or wav_path.name,
        "source_url": info.get("webpage_url") or url,
    }


def global_progress_callback(event, mid, value):
    """Global callback to route ModelManager events to IPC"""
    if event == "progress":
        progress_msg = {"type": "progress", "model_id": mid}
        if isinstance(value, dict):
            progress_msg.update(value)
        else:
            progress_msg["progress"] = value
        # Log occasionally to avoid spam
        if isinstance(value, (int, float)) and int(value) % 10 == 0:
            log.info(f">>> PROGRESS callback for {mid}: {value}%")
        send_ipc_message(progress_msg)
    elif event == "complete":
        complete_msg = {"type": "complete", "model_id": mid, "path": str(value)}
        log.info(f">>> COMPLETE callback for {mid}")
        send_ipc_message(complete_msg)
    elif event == "error":
        error_msg = {"type": "error", "model_id": mid, "error": str(value)}
        log.error(f"Download error for {mid}: {value}")
        send_ipc_message(error_msg)
    elif event == "paused":
        paused_msg = {"type": "paused", "model_id": mid}
        log.info(f">>> PAUSED callback for {mid}")
        send_ipc_message(paused_msg)
    elif event == "start":
        log.info(f">>> START callback for {mid}")


async def _perform_download_task(model_id: str):
    """Actual download logic running in background task"""
    try:
        if manager is None:
            send_ipc_message(
                {
                    "type": "error",
                    "model_id": model_id,
                    "error": "ModelManager is not initialized (bridge startup not complete).",
                }
            )
            return

        log.info(f"Starting download task for model: {model_id}")
        success = await manager.download_model(model_id)
        log.info(f"Download task finished for model: {model_id}, success: {success}")
    except Exception as e:
        log.error(
            f"Exception during background download of {model_id}: {e}", exc_info=True
        )
        # Manager usually emits 'error', but just in case:
        send_ipc_message({"type": "error", "model_id": model_id, "error": str(e)})


async def handle_download(model_id: str):
    """Initiate download in background and return immediately"""
    task = asyncio.create_task(_perform_download_task(model_id))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    log.info(f"Scheduled download task for {model_id}")
    # We return True so the main loop sends the generic success response
    # The actual completion event will be sent by the background task
    return True


def handle_get_models():
    """Get all available models"""
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        log.info("Fetching available models")
        models = manager.get_available_models()
        model_data = [m.to_dict() for m in models]
        log.info(f"Found {len(model_data)} models")
        send_response(True, data=model_data)
    except Exception as e:
        log.error(f"Error getting models: {e}", exc_info=True)
        send_response(False, error=str(e))


async def handle_resolve_youtube(cmd_data):
    try:
        url = cmd_data.get("url")
        if not url or not isinstance(url, str):
            send_response(False, error="url is required")
            return
        if not _is_youtube_url(url):
            send_response(False, error="Provided url does not look like a YouTube URL")
            return

        send_ipc_message({"type": "youtube_progress", "status": "starting"})
        result = await asyncio.to_thread(_download_youtube_audio_to_wav, url)
        send_ipc_message({"type": "youtube_progress", "status": "completed"})
        send_response(True, data=result)
    except Exception as e:
        log.error(f"Error resolving YouTube url: {e}", exc_info=True)
        send_ipc_message({"type": "youtube_progress", "status": "error", "error": str(e)})
        send_response(False, error=str(e))


def handle_model_preflight(cmd_data):
    """
    Preflight a model selection before starting separation.

    Returns a structured readiness report so the UI can:
    - show an explicit install/download prompt if the model is missing
    - display a transparent run summary (architecture, expected filename, etc.)
    - avoid any silent fallback to a different model
    """
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        model_id = cmd_data.get("model_id")
        if not model_id:
            send_response(False, error="model_id is required")
            return

        # Ensure registry + installs are in sync
        try:
            manager._refresh_install_flags()
        except Exception:
            pass

        # Use centralized preflight (includes safe filename repair copy + clear missing report)
        report = manager.ensure_model_ready(
            model_id,
            auto_repair_filenames=True,
            copy_instead_of_rename=True,
        )

        # Provide extra context useful for UI summaries
        model_info = manager.get_model(model_id)
        if model_info:
            report["model_name"] = model_info.name
            report["stems"] = model_info.stems

        send_response(True, data=report)
    except Exception as e:
        log.error(f"Error in model_preflight: {e}", exc_info=True)
        send_response(False, error=str(e))


# NOTE: Duplicate definition removed. Keep the earlier `handle_model_preflight` implementation above.


def handle_ping():
    """Health check - responds with pong to verify bridge is alive"""
    import time

    send_response(True, data={"status": "ok", "timestamp": time.time()})


def handle_check_preset_models(cmd_data):
    """Check for preset models and return availability map."""
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        preset_mappings = cmd_data.get("preset_mappings", {})
        availability = {}

        log.info(f"Checking availability for {len(preset_mappings)} presets...")

        for preset_name, model_id in preset_mappings.items():
            availability[preset_name] = manager.is_model_installed(model_id)

        log.info("Preset models checked.")
        send_response(True, data=availability)
    except Exception as e:
        log.error(f"Error checking preset models: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_get_gpu_devices():
    """Get all available GPU devices with recommended profile"""
    try:
        log.info("Fetching available GPU devices")
        # Use detector directly to get full info
        detector = GPUDetector()
        info = detector.get_gpu_info()
        log.info(f"Found {len(info.get('gpus', []))} devices")

        # Add recommended GPU profile based on detected hardware
        try:
            from core.system_diagnostics import SystemDiagnostics

            diag = SystemDiagnostics()
            gpu_profile = diag.get_recommended_gpu_profile()
            info["recommended_profile"] = {
                "profile_name": gpu_profile.get("profile_name", "cpu"),
                "settings": gpu_profile.get("settings", {}),
                "vram_gb": gpu_profile.get("vram_gb", 0),
            }
            log.info(f"Recommended profile: {gpu_profile.get('profile_name')}")
        except Exception as e:
            log.warning(f"Could not get GPU profile recommendation: {e}")
            info["recommended_profile"] = None

        send_response(True, data=info)
    except Exception as e:
        log.error(f"Error getting GPU devices: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_get_workflows():
    """Get available workflow types (Live vs Studio)"""
    try:
        from models.model_config import WORKFLOW_TYPES

        send_response(True, data={"workflows": WORKFLOW_TYPES})
    except Exception as e:
        log.error(f"Error getting workflows: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_check_memory(cmd_data):
    """Check memory requirements for a file/model combination"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        file_path = cmd_data.get("file_path")
        model_id = cmd_data.get("model_id")

        if not file_path or not model_id:
            send_response(False, error="file_path and model_id are required")
            return

        result = separation_manager.check_memory_for_job(file_path, model_id)
        send_response(True, data=result)
    except Exception as e:
        log.error(f"Error checking memory: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_separation_preflight(cmd_data):
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        file_path = cmd_data.get("file_path")
        model_id = cmd_data.get("model_id")
        stems = cmd_data.get("stems")
        device = cmd_data.get("device")
        overlap = cmd_data.get("overlap")
        segment_size = cmd_data.get("segment_size")
        tta = cmd_data.get("tta", False)
        output_format = cmd_data.get("output_format", "wav")
        shifts = cmd_data.get("shifts")
        bitrate = cmd_data.get("bitrate")

        raw_ensemble_config = cmd_data.get("ensemble_config")
        stem_algorithms = None
        if raw_ensemble_config and isinstance(raw_ensemble_config, dict):
            ensemble_config = raw_ensemble_config.get("models", [])
            ensemble_algorithm = raw_ensemble_config.get("algorithm", "average")
            stem_algorithms = raw_ensemble_config.get("stemAlgorithms")
        else:
            ensemble_config = raw_ensemble_config
            ensemble_algorithm = cmd_data.get("ensemble_algorithm", "average")

        phase_fix_params = None
        phase_fix_enabled = False
        if raw_ensemble_config and isinstance(raw_ensemble_config, dict):
            phase_fix_params = raw_ensemble_config.get("phaseFixParams")
            phase_fix_enabled = raw_ensemble_config.get("phaseFixEnabled", False)

        if not phase_fix_params:
            phase_params = cmd_data.get("phase_params")
            if phase_params and phase_params.get("enabled"):
                phase_fix_params = phase_params
                phase_fix_enabled = True

        report = {
            "can_proceed": True,
            "errors": [],
            "warnings": [],
            "audio": None,
            "missing_models": [],
            "torch_available": False,
            "resolved": {
                "model_id": model_id,
                "stems": stems,
                "device": device,
                "overlap": overlap,
                "segment_size": segment_size,
                "tta": bool(tta),
                "shifts": shifts,
                "bitrate": bitrate,
                "output_format": output_format,
                "ensemble_algorithm": ensemble_algorithm,
                "phase_fix_enabled": bool(phase_fix_enabled),
                "phase_fix_params": phase_fix_params,
            },
            "memory": None,
        }

        if not file_path:
            report["errors"].append("Missing required parameter: file_path")
        else:
            validation = separation_manager.engine.validate_audio_file(file_path)
            if not validation.get("valid"):
                report["errors"].append(
                    f"Invalid audio file: {validation.get('error') or 'Unknown error'}"
                )
            else:
                report["audio"] = validation

        try:
            import torch  # noqa: F401

            report["torch_available"] = True
        except Exception:
            report["torch_available"] = False

        model_ids = []
        if ensemble_config and isinstance(ensemble_config, list) and len(ensemble_config) > 0:
            for cfg in ensemble_config:
                if isinstance(cfg, dict) and cfg.get("model_id"):
                    model_ids.append(cfg["model_id"])
        else:
            if model_id and model_id != "ensemble":
                model_ids = [model_id]
            elif model_id == "ensemble":
                report["errors"].append(
                    "Ensemble selected but no ensemble_config provided"
                )
            else:
                report["errors"].append("Missing model_id or ensemble_config")

        for mid in model_ids:
            if not manager.is_model_installed(mid):
                report["missing_models"].append(mid)

        if report["missing_models"]:
            report["errors"].append(
                f"Missing models: {', '.join(sorted(set(report['missing_models'])))}"
            )

        if overlap is None:
            if model_id and model_id != "ensemble":
                minfo = manager.get_model(model_id)
                if minfo:
                    if getattr(minfo, "recommended_overlap", None):
                        overlap = minfo.recommended_overlap
                    elif (
                        getattr(minfo, "recommended_settings", None)
                        and "overlap" in minfo.recommended_settings
                    ):
                        overlap = minfo.recommended_settings["overlap"]
            if overlap is None:
                overlap = 0.25

        if segment_size is None:
            if model_id and model_id != "ensemble":
                minfo = manager.get_model(model_id)
                if minfo:
                    if getattr(minfo, "recommended_chunk_size", None):
                        segment_size = minfo.recommended_chunk_size
                    elif (
                        getattr(minfo, "recommended_settings", None)
                        and "chunk_size" in minfo.recommended_settings
                    ):
                        segment_size = minfo.recommended_settings["chunk_size"]
            if segment_size is None:
                segment_size = 352800

        if segment_size == 256:
            segment_size = 352800

        report["resolved"]["overlap"] = overlap
        report["resolved"]["segment_size"] = segment_size

        if isinstance(overlap, (int, float)) and float(overlap) > 0.95:
            report["warnings"].append(
                f"Very high overlap ({overlap}). This may be slow and can cause artifacts on some models."
            )

        needs_torch_for_blend = False
        torch_blend_algs = {"max_spec", "min_spec", "phase_fix", "frequency_split"}
        if ensemble_algorithm in torch_blend_algs:
            needs_torch_for_blend = True
        if stem_algorithms and isinstance(stem_algorithms, dict):
            for _, alg in stem_algorithms.items():
                if alg in torch_blend_algs:
                    needs_torch_for_blend = True
                    break

        if phase_fix_enabled:
            if not report["torch_available"]:
                report["errors"].append(
                    "Phase Fix requires torch, but torch is not available in the backend environment."
                )

            low_hz = None
            high_hz = None
            if isinstance(phase_fix_params, dict):
                low_hz = phase_fix_params.get("lowHz")
                high_hz = phase_fix_params.get("highHz")
            if low_hz is not None and high_hz is not None:
                try:
                    low_hz_i = int(low_hz)
                    high_hz_i = int(high_hz)
                    if low_hz_i < 0 or high_hz_i <= 0 or low_hz_i >= high_hz_i:
                        report["errors"].append(
                            f"Invalid Phase Fix frequency range: lowHz={low_hz_i}, highHz={high_hz_i}"
                        )
                except Exception:
                    report["errors"].append(
                        "Invalid Phase Fix frequency params (lowHz/highHz must be integers)"
                    )

        if needs_torch_for_blend and not report["torch_available"]:
            report["warnings"].append(
                "Torch is not available. Max/Min Spec and Frequency Split will fall back to Average blending."
            )

        try:
            memory_checks = []
            if file_path and model_ids:
                for mid in model_ids:
                    memory_checks.append(
                        separation_manager.check_memory_for_job(file_path, mid)
                    )
            if memory_checks:
                max_estimated = max(
                    (mc.get("estimated_gb", 0.0) for mc in memory_checks),
                    default=0.0,
                )
                available = max(
                    (mc.get("available_gb", 0.0) for mc in memory_checks),
                    default=0.0,
                )
                can_proceed = all(mc.get("can_proceed", True) for mc in memory_checks)
                warnings = [mc.get("warning") for mc in memory_checks if mc.get("warning")]
                report["memory"] = {
                    "estimated_gb": round(float(max_estimated), 2),
                    "available_gb": round(float(available), 2),
                    "can_proceed": bool(can_proceed),
                    "warnings": warnings,
                }
                if warnings:
                    report["warnings"].append(warnings[0])
        except Exception as e:
            report["warnings"].append(f"Memory preflight failed: {e}")

        if report["errors"]:
            report["can_proceed"] = False

        send_response(True, data=report)
    except Exception as e:
        log.error(f"Error in separation_preflight: {e}", exc_info=True)
        send_response(False, error=str(e))


def run_post_processing_pipeline(
    output_files, steps, output_dir, device, progress_callback=None
):
    """
    Run post-processing pipeline steps on separated stems.

    Args:
        output_files: Dict of stem -> file path from main separation
        steps: List of { type, modelId, description, targetStem }
        output_dir: Output directory
        device: Device to use for processing
        progress_callback: Optional callback(progress, message)

    Returns:
        Updated output_files dict with processed files
    """
    # Post-processing is optional and should never crash the bridge.
    # Also, avoid accessing globals if the bridge hasn't finished startup.
    if manager is None or separation_manager is None:
        log.warning("Post-processing skipped: backend managers not initialized")
        return output_files

    from pathlib import Path

    result_files = output_files.copy()
    total_steps = len(steps)

    for i, step in enumerate(steps):
        step_type = step.get("type")
        model_id = step.get("modelId")
        description = step.get("description", step_type)
        target_stem = step.get("targetStem", "instrumental")

        if progress_callback:
            progress_callback(i / total_steps, description)

        log.info(
            f"Post-processing step {i + 1}/{total_steps}: {description} ({step_type} with {model_id})"
        )

        # Find the target file to process
        target_file = None
        for stem_name, file_path in result_files.items():
            if target_stem in stem_name.lower() or target_stem == "all":
                target_file = file_path
                break

        if not target_file:
            log.warning(
                f"No {target_stem} file found for post-processing step: {description}"
            )
            continue

        if not Path(target_file).exists():
            log.warning(f"Target file does not exist: {target_file}")
            continue

        try:
            # Get the model for this step
            model = manager.get_model(model_id)
            if not model:
                log.warning(f"Model {model_id} not found for post-processing")
                continue

            if not model.is_installed:
                log.warning(f"Model {model_id} not installed, skipping step")
                continue

            # Create a temp output for this step (for future implementation)
            output_path = Path(output_dir)
            stem_name = Path(target_file).stem
            _step_output = output_path / f"{stem_name}_{step_type}.wav"

            # Run separation with this model on the stem file
            if step_type == "phase_fix":
                # Phase fix uses ensemble with the reference model
                # For now, we'll just log that we would do this
                log.info(
                    f"Phase fix would run with reference model {model_id} on {target_file}"
                )
                # In full implementation: run ensemble with phase_fix algorithm

            elif step_type in ["de_reverb", "de_bleed", "de_noise", "de_breath"]:
                # These are utility models that output clean/dry stem
                log.info(f"Running {step_type} model {model_id} on {target_file}")

                # Use separation manager to run this model
                temp_job_id = separation_manager.create_job(
                    file_path=target_file,
                    model_id=model_id,
                    output_dir=str(output_path),
                    device=device,
                    stems=None,  # Use model's default stems
                    output_format="wav",
                )

                if temp_job_id:
                    separation_manager.start_job(temp_job_id)
                    # Wait for completion (blocking for pipeline)
                    job = separation_manager.get_job(temp_job_id)
                    while job and job.status not in [
                        "completed",
                        "failed",
                        "cancelled",
                    ]:
                        import time

                        time.sleep(0.5)
                        job = separation_manager.get_job(temp_job_id)

                    if job and job.status == "completed" and job.output_files:
                        # Find the "clean" or "dry" output (depends on model)
                        for k, v in job.output_files.items():
                            if (
                                "clean" in k.lower()
                                or "dry" in k.lower()
                                or "no_" not in k.lower()
                            ):
                                result_files[target_stem] = v
                                log.info(
                                    f"Updated {target_stem} with processed file: {v}"
                                )
                                break

        except Exception as e:
            log.warning(f"Failed to run post-processing step {description}: {e}")
            continue

    if progress_callback:
        progress_callback(1.0, "Post-processing complete")

    return result_files


def handle_separate_audio(cmd_data):
    """Handle audio separation request"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        file_path = cmd_data.get("file_path")
        model_id = cmd_data.get("model_id")
        output_dir = cmd_data.get("output_dir")
        stems = cmd_data.get("stems")
        device = cmd_data.get("device")

        # Advanced parameters - default to None to allow model-specific defaults
        overlap = cmd_data.get("overlap")
        segment_size = cmd_data.get("segment_size")
        tta = cmd_data.get("tta", False)
        output_format = cmd_data.get("output_format", "wav")
        export_mixes = cmd_data.get("export_mixes")
        shifts = cmd_data.get("shifts")
        bitrate = cmd_data.get("bitrate")
        invert = cmd_data.get("invert", False)
        normalize = cmd_data.get("normalize", False)
        bit_depth = cmd_data.get("bit_depth", "16")

        # Ensemble parameters
        # Frontend sends ensembleConfig as { models: [...], algorithm: '...', stemAlgorithms: {...} }
        # We need to extract the models list, algorithm, and stemAlgorithms separately
        raw_ensemble_config = cmd_data.get("ensemble_config")
        stem_algorithms = None
        if raw_ensemble_config and isinstance(raw_ensemble_config, dict):
            ensemble_config = raw_ensemble_config.get("models", [])
            ensemble_algorithm = raw_ensemble_config.get("algorithm", "average")
            stem_algorithms = raw_ensemble_config.get("stemAlgorithms")  # { vocals: 'max_spec', instrumental: 'min_spec' }

        else:
            ensemble_config = raw_ensemble_config  # In case it's already a list
            ensemble_algorithm = cmd_data.get("ensemble_algorithm", "average")

        split_freq = cmd_data.get("split_freq")

        # Phase Fix params - from ensembleConfig or legacy phase_params
        phase_fix_params = None
        phase_fix_enabled = False
        if raw_ensemble_config and isinstance(raw_ensemble_config, dict):
            phase_fix_params = raw_ensemble_config.get("phaseFixParams")  # { lowHz, highHz, highFreqWeight }
            phase_fix_enabled = raw_ensemble_config.get("phaseFixEnabled", False)
        
        # Legacy Phase Swap override from UI
        if not phase_fix_params:
            phase_params = cmd_data.get("phase_params")  # { enabled, lowHz, highHz, highFreqWeight }
            if phase_params and phase_params.get("enabled"):
                phase_fix_params = phase_params
                phase_fix_enabled = True

        # Post-processing pipeline steps (from preset)
        post_processing_steps = cmd_data.get(
            "post_processing_steps"
        )  # [{ type, modelId, description, targetStem }]

        if not file_path:
            send_response(
                False, error="Missing required parameter: file_path"
            )
            return

        # Use temp directory if output_dir is empty - user exports later from Results Studio
        if not output_dir:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix="stemsep_preview_")
            log.info(f"Using temp directory for separation preview: {output_dir}")

        if not model_id and not ensemble_config:
            send_response(False, error="Missing model_id or ensemble_config")
            return

        # Get model info to check for recommended settings
        model_info = manager.get_model(model_id) if model_id else None

        # Determine VRAM-safe max segment_size
        vram_limit_msg = ""
        max_safe_segment_size = float("inf")

        if device and device.startswith("cuda"):
            try:
                # Parse device index (e.g., "cuda:0" -> 0)
                device_idx = 0
                if ":" in device:
                    device_idx = int(device.split(":")[1])

                # Get GPU info
                detector = GPUDetector()
                gpu_info = detector.get_gpu_info()

                target_gpu = next(
                    (g for g in gpu_info["gpus"] if g["index"] == device_idx), None
                )

                if target_gpu:
                    vram_gb = target_gpu.get("memory_gb", 0)
                    log.info(f"Target GPU VRAM: {vram_gb} GB")

                    # Apply limits based on Google Doc findings
                    if vram_gb < 5:
                        max_safe_segment_size = 112455  # ~2.55s
                        vram_limit_msg = (
                            f"Limited to {max_safe_segment_size} (Low VRAM < 5GB)"
                        )
                    elif vram_gb < 8:
                        max_safe_segment_size = 352800  # ~8.00s
                        vram_limit_msg = (
                            f"Limited to {max_safe_segment_size} (Medium VRAM < 8GB)"
                        )
            except Exception as e:
                log.warning(f"Failed to check VRAM limits: {e}")

        if model_info:
            # Apply recommended overlap if not specified
            if overlap is None:
                if model_info.recommended_overlap:
                    overlap = model_info.recommended_overlap
                    log.info(f"Using recommended overlap {overlap} for {model_id}")
                elif (
                    model_info.recommended_settings
                    and "overlap" in model_info.recommended_settings
                ):
                    overlap = model_info.recommended_settings["overlap"]
                    log.info(
                        f"Using recommended overlap {overlap} from settings for {model_id}"
                    )
                else:
                    overlap = 0.25  # Default

            # Apply recommended chunk size (segment_size) if not specified
            if segment_size is None:
                if model_info.recommended_chunk_size:
                    segment_size = model_info.recommended_chunk_size
                    log.info(
                        f"Using recommended chunk size {segment_size} for {model_id}"
                    )
                elif (
                    model_info.recommended_settings
                    and "chunk_size" in model_info.recommended_settings
                ):
                    segment_size = model_info.recommended_settings["chunk_size"]
                    log.info(
                        f"Using recommended chunk size {segment_size} from settings for {model_id}"
                    )
                else:
                    segment_size = 256  # Default (likely too small, but keeps existing logic if no recommendation)
        else:
            # Fallbacks if model not found
            if overlap is None:
                overlap = 0.25
            if segment_size is None:
                segment_size = 352800

        # Enforce VRAM safety limits
        if segment_size is not None and segment_size > max_safe_segment_size:
            log.warning(
                f"Reducing segment_size from {segment_size} to {max_safe_segment_size}. Reason: {vram_limit_msg}"
            )
            segment_size = int(max_safe_segment_size)

        # Final safety check for the "256" default which is often too small for Roformers
        if segment_size == 256:
            log.info("segment_size is 256, bumping to safe default 352800")
            segment_size = 352800

        if shifts is None:
            shifts = 1
        if bitrate is None:
            bitrate = "320k"

        display_model = model_id or "Ensemble"
        log.info(
            f"Starting separation for {file_path} with {display_model} (format: {output_format}, segment_size: {segment_size})"
        )

        job_id = separation_manager.create_job(
            file_path=file_path,
            model_id=model_id or "ensemble",
            output_dir=output_dir,
            stems=stems,
            device=device,
            overlap=overlap,
            segment_size=segment_size,
            tta=tta,
            output_format=output_format,
            export_mixes=export_mixes,
            shifts=shifts,
            bitrate=bitrate,
            ensemble_config=ensemble_config,
            ensemble_algorithm=ensemble_algorithm,
            invert=invert,
            normalize=normalize,
            bit_depth=bit_depth,
            split_freq=split_freq,
            phase_params=phase_fix_params,
            phase_fix_enabled=phase_fix_enabled,  # Phase Fix as separate checkbox
            stem_algorithms=stem_algorithms,  # Per-stem algorithm selection
        )

        # Define callbacks closing over job_id
        def separation_progress_callback(progress, message, device=None):
            msg = {
                "type": "separation_progress",
                "job_id": job_id,
                "progress": progress,
                "message": message,
            }
            if device:
                msg["device"] = device
            send_ipc_message(msg)

        def separation_complete_callback(output_files):
            # Check if we need to run post-processing steps
            if post_processing_steps and len(post_processing_steps) > 0:
                log.info(
                    f"Running {len(post_processing_steps)} post-processing step(s)..."
                )

                try:
                    processed_files = run_post_processing_pipeline(
                        output_files=output_files,
                        steps=post_processing_steps,
                        output_dir=output_dir,
                        device=device,
                        progress_callback=lambda prog, msg: send_ipc_message(
                            {
                                "type": "separation_progress",
                                "job_id": job_id,
                                "progress": 0.9
                                + prog * 0.1,  # Last 10% is post-processing
                                "message": f"Post-processing: {msg}",
                            }
                        ),
                    )
                    output_files = processed_files if processed_files else output_files
                except Exception as e:
                    log.warning(f"Post-processing failed, using original files: {e}")

            msg = {
                "type": "separation_complete",
                "job_id": job_id,
                "output_files": output_files,
            }
            send_ipc_message(msg)

        # Attach callbacks
        job = separation_manager.get_job(job_id)
        if job:
            job.progress_callback = separation_progress_callback
            job.on_complete = separation_complete_callback

        if separation_manager.start_job(job_id):
            send_response(True, data={"job_id": job_id, "status": "started"})
        else:
            send_response(False, error="Failed to start separation job")

    except Exception as e:
        log.error(f"Error starting separation: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_cancel_job(cmd_data):
    """Cancel a running separation job"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        job_id = cmd_data.get("job_id")
        if not job_id:
            send_response(False, error="job_id is required")
            return

        log.info(f"Cancelling job {job_id}")
        if separation_manager.cancel_job(job_id):
            send_response(True, data={"job_id": job_id, "status": "cancelled"})
        else:
            send_response(
                False, error=f"Could not cancel job {job_id} (not found or not running)"
            )
    except Exception as e:
        log.error(f"Error cancelling job: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_pause_queue():
    """Pause queue processing"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        separation_manager.pause_queue()
        send_response(True)
    except Exception as e:
        send_response(False, error=str(e))


def handle_resume_queue():
    """Resume queue processing"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        separation_manager.resume_queue()
        send_response(True)
    except Exception as e:
        send_response(False, error=str(e))


def handle_reorder_queue(cmd_data):
    """Reorder the queue"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        job_ids = cmd_data.get("job_ids", [])
        separation_manager.reorder_queue(job_ids)
        send_response(True)
    except Exception as e:
        send_response(False, error=str(e))


def handle_save_output(cmd_data):
    """Save job output from temp to final location"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        job_id = cmd_data.get("job_id")
        if not job_id:
            send_response(False, error="job_id is required")
            return

        if separation_manager.save_job_output(job_id):
            # Return the new paths
            job = separation_manager.get_job(job_id)
            send_response(
                True, data={"job_id": job_id, "output_files": job.output_files}
            )
        else:
            send_response(False, error="Failed to save output")
    except Exception as e:
        log.error(f"Error saving output: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_export_output(cmd_data):
    """Export job output to specific format"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        job_id = cmd_data.get("job_id")
        export_path = cmd_data.get("export_path")
        format = cmd_data.get("format", "mp3")
        bitrate = cmd_data.get("bitrate", "320k")

        if not job_id or not export_path:
            send_response(False, error="job_id and export_path are required")
            return

        if separation_manager.export_job_output(job_id, export_path, format, bitrate):
            send_response(True, data={"job_id": job_id, "status": "exported"})
        else:
            send_response(False, error="Failed to export output")
    except Exception as e:
        log.error(f"Error exporting output: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_export_files(cmd_data):
    """Export files directly from paths (no job lookup required)"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        source_files = cmd_data.get(
            "source_files"
        )  # {"vocals": "C:/path/to/vocals.wav", ...}
        export_path = cmd_data.get("export_path")
        format = cmd_data.get("format", "mp3")
        bitrate = cmd_data.get("bitrate", "320k")

        if not source_files or not export_path:
            send_response(False, error="source_files and export_path are required")
            return

        log.info(f"Exporting {len(source_files)} files to {export_path} as {format}")

        if separation_manager.export_from_paths(
            source_files, export_path, format, bitrate
        ):
            send_response(True, data={"status": "exported", "path": export_path})
        else:
            send_response(False, error="Failed to export files")
    except Exception as e:
        log.error(f"Error exporting files: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_discard_output(cmd_data):
    """Discard job output and delete temp files"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        job_id = cmd_data.get("job_id")
        if not job_id:
            send_response(False, error="job_id is required")
            return

        if separation_manager.discard_job_output(job_id):
            send_response(True, data={"job_id": job_id, "status": "discarded"})
        else:
            send_response(False, error="Failed to discard output")
    except Exception as e:
        log.error(f"Error discarding output: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_pause_download(model_id: str):
    """Pause a running download"""
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        log.info(f"Pausing download for {model_id}")
        if manager.pause_download(model_id):
            send_response(True, data={"model_id": model_id, "status": "paused"})
        else:
            send_response(
                False, error=f"Could not pause download for {model_id} (not active?)"
            )
    except Exception as e:
        log.error(f"Error pausing download: {e}", exc_info=True)
        send_response(False, error=str(e))


async def handle_resume_download(model_id: str):
    """Resume a paused download"""
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        log.info(f"Resuming download for {model_id}")
        success = await manager.resume_download(model_id)
        if success:
            send_response(True, data={"model_id": model_id, "status": "resumed"})
        else:
            send_response(False, error=f"Could not resume download for {model_id}")
    except Exception as e:
        log.error(f"Error resuming download: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_remove_model(model_id: str):
    """Remove an installed model"""
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        log.info(f"Removing model {model_id}")
        if manager.remove_model(model_id):
            send_response(True, data={"model_id": model_id, "status": "removed"})
        else:
            send_response(False, error=f"Could not remove model {model_id}")
    except Exception as e:
        log.error(f"Error removing model: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_import_custom_model(cmd_data):
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        file_path = cmd_data.get("file_path")
        model_name = cmd_data.get("model_name")
        architecture = cmd_data.get("architecture") or "Custom"

        if not file_path or not isinstance(file_path, str):
            send_response(False, error="file_path is required")
            return
        if not model_name or not isinstance(model_name, str):
            send_response(False, error="model_name is required")
            return

        model_id = manager.import_custom_model(
            source_path=Path(file_path),
            model_name=model_name,
            architecture=str(architecture),
        )
        send_response(True, data={"model_id": model_id})

    except Exception as e:
        log.error(f"Error importing custom model: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_get_model_tech(cmd_data):
    try:
        if manager is None:
            send_response(
                False,
                error="ModelManager is not initialized (bridge startup not complete).",
            )
            return

        model_id = cmd_data.get("model_id")
        if not model_id or not isinstance(model_id, str):
            send_response(False, error="model_id is required")
            return

        result = manager.get_model_tech(model_id)
        if isinstance(result, dict) and result.get("ok") is True:
            send_response(True, data=result)
        else:
            err = None
            if isinstance(result, dict):
                err = result.get("error")
            send_response(False, error=str(err or "Unable to fetch model tech"))
    except Exception as e:
        log.error(f"Error fetching model tech: {e}", exc_info=True)
        send_response(False, error=str(e))


def handle_get_recipes():
    """Get all available recipes"""
    try:
        if separation_manager is None:
            send_response(
                False,
                error="SeparationManager is not initialized (bridge startup not complete).",
            )
            return

        recipes = separation_manager.recipe_manager.get_all_recipes()
        send_response(True, data=recipes)
    except Exception as e:
        log.error(f"Error getting recipes: {e}", exc_info=True)
        send_response(False, error=str(e))


async def main():
    """Main loop - read commands from stdin"""
    import argparse

    parser = argparse.ArgumentParser(description="StemSep Python Bridge")
    parser.add_argument("--models-dir", type=str, help="Custom directory for AI models")
    args = parser.parse_args()

    global manager, separation_manager, CURRENT_CMD_ID

    # Initialize ModelManager with custom path if provided
    if args.models_dir:
        log.info(f"Using custom models directory: {args.models_dir}")
        manager = ModelManager(models_dir=Path(args.models_dir))
    else:
        manager = ModelManager()

    # Register global callback once
    assert manager is not None, "ModelManager failed to initialize"
    manager.add_download_callback(global_progress_callback)

    separation_manager = SeparationManager(manager)

    log.info("Python bridge started, ModelManager initialized")

    # Emit bridge_ready event to signal frontend that all managers are ready
    send_ipc_message({
        "type": "bridge_ready",
        "capabilities": ["models", "separation", "recipes", "gpu"],
        "models_count": len(manager.get_available_models()),
        "recipes_count": len(separation_manager.recipe_manager.get_all_recipes())
    })

    handlers = {
        "get_models": handle_get_models,
        "get_recipes": handle_get_recipes,
        "check-preset-models": handle_check_preset_models,
        "get-gpu-devices": handle_get_gpu_devices,
        "get_workflows": handle_get_workflows,
        "download_model": handle_download,
        "pause_download": handle_pause_download,
        "resume_download": handle_resume_download,
        "remove_model": handle_remove_model,
        "import_custom_model": handle_import_custom_model,
        "get_model_tech": handle_get_model_tech,
        "separate_audio": handle_separate_audio,
        "cancel_job": handle_cancel_job,
        "pause_queue": handle_pause_queue,
        "resume_queue": handle_resume_queue,
        "reorder_queue": handle_reorder_queue,
        "model_preflight": handle_model_preflight,
        "separation_preflight": handle_separation_preflight,
        "resolve_youtube": handle_resolve_youtube,
        "ping": handle_ping,
        "check_memory": handle_check_memory,
        "export_output": handle_export_output,
        "export_files": handle_export_files,
        "discard_output": handle_discard_output,
    }

    while True:
        try:
            # Use asyncio.to_thread to make stdin reading non-blocking
            # This allows background tasks (like downloads) to run while waiting for input
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                log.info("Stdin closed, exiting.")
                break

            log.debug(f"Received command: {line.strip()}")
            cmd = json.loads(line.strip())
            command = cmd.get("command")
            CURRENT_CMD_ID = cmd.get("id")

            handler = handlers.get(command)

            if handler:
                if command in [
                    "separate_audio",
                    "check-preset-models",
                    "cancel_job",
                    "reorder_queue",
                    "save_output",
                    "discard_output",
                    "export_output",
                    "export_files",
                    "check_memory",
                    "model_preflight",
                    "separation_preflight",
                    "resolve_youtube",
                    "import_custom_model",
                    "get_model_tech",
                ]:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(cmd)
                    else:
                        handler(cmd)
                else:
                    import inspect

                    sig = inspect.signature(handler)
                    if "model_id" in sig.parameters:
                        model_id = cmd.get("model_id")
                        if not model_id:
                            send_response(
                                False, error="model_id is required for this command"
                            )
                            continue
                        if asyncio.iscoroutinefunction(handler):
                            success = await handler(model_id)
                            send_response(success, data={"model_id": model_id})
                        else:
                            handler(model_id)
                    else:
                        if asyncio.iscoroutinefunction(handler):
                            await handler()
                        else:
                            handler()  # Fixed: Don't pass model_id if not expected
            else:
                log.warning(f"Unknown command received: {command}")
                send_response(False, error=f"Unknown command: {command}")

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON received: {e}", exc_info=True)
            send_response(False, error=f"Invalid JSON: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}", exc_info=True)
            send_response(False, error=f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bridge interrupted by user.")
    except Exception as e:
        log.critical(f"Bridge crashed with unhandled exception: {e}", exc_info=True)
