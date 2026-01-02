#!/usr/bin/env python3
"""
Single-shot inference script for StemSep Rust Backend.
Accepts a JSON configuration string as the first argument.
Outputs newline-delimited JSON events to stdout.
"""

import asyncio
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

try:
    import soundfile as sf
except Exception:
    sf = None


def _parse_structured_device_error(error: Any) -> Optional[dict]:
    """Parse structured device/CUDA errors produced by StemSepApp.

    Expected format:
      "STEMSEP_DEVICE_ERROR {json-payload}"

    Returns a dict payload or None.
    """
    if not error:
        return None
    if not isinstance(error, str):
        error = str(error)

    prefix = "STEMSEP_DEVICE_ERROR "
    if not error.startswith(prefix):
        return None

    raw = error[len(prefix) :].strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "code" in payload and "message" in payload:
            return payload
    except Exception:
        return None

    return None


# Force UTF-8 for stdio
if sys.platform == "win32":
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Redirect stdout to stderr so we can keep stdout clean for IPC
_IPC_STREAM = sys.stdout
sys.stdout = sys.stderr


def send_ipc(msg):
    """Write JSON message to the original stdout"""
    try:
        json_str = json.dumps(msg)
        _IPC_STREAM.write(json_str + "\n")
        _IPC_STREAM.flush()
    except Exception as e:
        sys.stderr.write(f"IPC Error: {e}\n")


# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Try to locate StemSepApp/src
# Priority:
# 1. ../StemSepApp/src (Dev environment relative to scripts/)
# 2. ./StemSepApp/src (Packaged environment)
candidates = [
    REPO_ROOT / "StemSepApp" / "src",
    SCRIPT_DIR / "StemSepApp" / "src",
]

added = False
for p in candidates:
    if p.exists():
        sys.path.insert(0, str(p))
        added = True
        break

if not added:
    send_ipc({"type": "error", "error": "Could not locate StemSepApp source code"})
    sys.exit(1)

# Import core modules
try:
    from stemsep.core.logger import setup_logging
    from stemsep.core.separation_manager import SeparationManager
    from stemsep.models.model_manager import ModelManager
except ImportError as e:
    send_ipc({"type": "error", "error": f"Failed to import core modules: {e}"})
    sys.exit(1)

# Setup graceful shutdown
shutdown_event = asyncio.Event()


def signal_handler(sig, frame):
    sys.stderr.write(f"Received signal {sig}, shutting down...\n")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _normalize_config(config: dict) -> dict:
    """Normalize UI/backend config shapes for the Python SeparationManager.

    Electron UI sends some objects in a richer shape (e.g. ensemble_config as an object
    with {models, algorithm, phaseFixEnabled,...}). The core SeparationManager expects:
      - ensemble_config: List[{model_id, weight?}, ...]
      - ensemble_algorithm: str
      - phase_fix_enabled: bool
      - phase_params: {lowHz, highHz, highFreqWeight, enabled?}

    Additionally, UI may send post-processing steps (auto pipeline). For now we
    implement the most common one: phase_fix for instrumental.
    """
    if not isinstance(config, dict):
        return config

    # 0) Normalize common camelCase keys from the Electron UI into snake_case.
    # The Rust backend often forwards snake_case already, but the UI can send
    # camelCase in some flows (tests, dev harnesses).
    camel_to_snake = {
        "filePath": "file_path",
        "modelId": "model_id",
        "outputDir": "output_dir",
        "outputFormat": "output_format",
        "bitDepth": "bit_depth",
        "exportMixes": "export_mixes",
        "postProcessingSteps": "post_processing_steps",
        "volumeCompensation": "volume_compensation",
    }
    for src_key, dst_key in camel_to_snake.items():
        if dst_key not in config and src_key in config:
            config[dst_key] = config[src_key]

    # 1) Normalize ensemble_config object -> list
    ensemble_cfg = config.get("ensemble_config")
    if isinstance(ensemble_cfg, dict):
        # algorithm
        alg = ensemble_cfg.get("algorithm")
        if isinstance(alg, str) and alg.strip():
            config["ensemble_algorithm"] = alg

        # stemAlgorithms
        stem_alg = ensemble_cfg.get("stemAlgorithms")
        if isinstance(stem_alg, dict) and stem_alg:
            config["stem_algorithms"] = stem_alg

        # phase fix settings
        if ensemble_cfg.get("phaseFixEnabled") is True:
            config["phase_fix_enabled"] = True
            phase_params = ensemble_cfg.get("phaseFixParams")
            if isinstance(phase_params, dict) and phase_params:
                # Ensure expected key casing (lowHz/highHz/highFreqWeight)
                config["phase_params"] = phase_params

        # volume compensation settings
        vc = ensemble_cfg.get("volumeCompensation") or ensemble_cfg.get(
            "volume_compensation"
        )
        if isinstance(vc, dict):
            config["volume_compensation"] = vc

        # models list
        models = ensemble_cfg.get("models")
        if isinstance(models, list):
            config["ensemble_config"] = models
        else:
            # Invalid shape; leave it unset to avoid crashing create_job
            config.pop("ensemble_config", None)

    # 2) Auto post-processing: phase_fix => run a 2-model ensemble with Phase Fix enabled
    post_steps = config.get("post_processing_steps")
    if post_steps is None:
        post_steps = config.get("postProcessingSteps")

    if (
        isinstance(post_steps, list)
        and post_steps
        and not config.get("ensemble_config")
        and isinstance(config.get("model_id"), str)
        and config.get("model_id")
        and config.get("model_id") != "ensemble"
    ):
        for step in post_steps:
            if not isinstance(step, dict):
                continue
            if step.get("type") != "phase_fix":
                continue

            ref_model = step.get("modelId") or step.get("model_id")
            if not isinstance(ref_model, str) or not ref_model.strip():
                continue

            base_model = config.get("model_id")
            if not isinstance(base_model, str) or not base_model.strip():
                continue

            # Convert to an ensemble job; SeparationManager already implements the
            # Phase Fix workflow when phase_fix_enabled is true.
            config["ensemble_config"] = [
                {"model_id": base_model, "weight": 1.0},
                {"model_id": ref_model, "weight": 1.0},
            ]
            config["model_id"] = "ensemble"
            config.setdefault("ensemble_algorithm", "average")
            config["phase_fix_enabled"] = True

            # If no explicit params were provided, use the guide-recommended defaults.
            if not isinstance(config.get("phase_params"), dict) or not config.get(
                "phase_params"
            ):
                config["phase_params"] = {
                    "enabled": True,
                    "lowHz": 500,
                    "highHz": 5000,
                    "highFreqWeight": 2.0,
                }
            break

    return config


def _normalize_volume_compensation(config: dict) -> dict:
    """Normalize volume compensation (VC) settings.

    Accepts either:
      - volume_compensation: { enabled: bool, stage: 'export'|'blend'|'both', dbPerExtraModel?: number }
      - vc_enabled/vc_stage/vc_db_per_extra_model already set
    """
    if not isinstance(config, dict):
        return config

    if (
        "vc_enabled" in config
        or "vc_stage" in config
        or "vc_db_per_extra_model" in config
    ):
        return config

    vc = config.get("volume_compensation")
    if vc is None:
        vc = config.get("volumeCompensation")

    if not isinstance(vc, dict):
        return config

    enabled = vc.get("enabled")
    if isinstance(enabled, bool):
        config["vc_enabled"] = enabled

    stage = vc.get("stage") or vc.get("applyStage") or vc.get("apply_stage")
    if isinstance(stage, str) and stage.strip():
        st = stage.strip().lower()
        if st in ("export", "blend", "both"):
            config["vc_stage"] = st

    db = vc.get("dbPerExtraModel")
    if db is None:
        db = vc.get("db_per_extra_model")
    if db is None:
        db = vc.get("dbPerModel")
    if db is None:
        db = vc.get("db_per_model")

    try:
        if db is not None:
            config["vc_db_per_extra_model"] = float(db)
    except Exception:
        pass

    return config


def _normalize_bit_depth(value) -> str:
    if value is None:
        return "16"
    s = str(value).strip().lower()
    if s in ("", "default", "auto"):
        return "16"
    if s in ("16", "pcm16", "pcm_16"):
        return "16"
    if s in ("24", "pcm24", "pcm_24"):
        return "24"
    # Treat "32" as float32 in StemSep (guide recommendation: float preserves headroom).
    if s in ("32", "float", "float32", "f32"):
        return "float32"
    return s


def _normalize_stems(stems):
    if stems is None:
        return None
    if isinstance(stems, str):
        stems = [stems]
    if not isinstance(stems, list):
        return stems
    out = []
    for stem in stems:
        if not isinstance(stem, str):
            continue
        k = stem.strip().lower().replace(" ", "_")
        k = k.replace("-", "_")
        if k in ("vocals", "vocal"):
            out.append("vocals")
        elif k in ("instrumental", "inst", "no_vocals", "karaoke"):
            out.append("instrumental")
        else:
            out.append(k)
    # Deduplicate while keeping order
    seen = set()
    deduped = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        deduped.append(k)
    return deduped


def _validate_output_lengths(input_path: str, output_files: dict) -> tuple[bool, str]:
    """Return (ok, reason).

    Uses soundfile.info for fast metadata.

    Important: some pipelines/models legitimately change sample rate (e.g. 48k -> 44.1k).
    In that case, frame counts will differ even when the duration matches. Validate by
    duration with a small tolerance instead of strict frame equality.
    """
    if sf is None:
        return True, "soundfile not available"
    if not input_path or not isinstance(output_files, dict) or not output_files:
        return True, "no outputs"

    try:
        in_info = sf.info(input_path)
        in_frames = int(getattr(in_info, "frames", 0) or 0)
        in_sr = int(getattr(in_info, "samplerate", 0) or 0)
        if in_frames <= 0:
            return True, "input frames unknown"
        if in_sr <= 0:
            return True, "input sample rate unknown"
    except Exception as e:
        return True, f"could not inspect input: {e}"

    in_dur = in_frames / float(in_sr)

    mismatches = []
    for stem, path in output_files.items():
        try:
            out_info = sf.info(path)
            out_frames = int(getattr(out_info, "frames", 0) or 0)
            out_sr = int(getattr(out_info, "samplerate", 0) or 0)
            if out_frames <= 0 or out_sr <= 0:
                mismatches.append((stem, {"frames": out_frames, "samplerate": out_sr}))
                continue

            out_dur = out_frames / float(out_sr)

            # Tolerance: allow up to ~5ms plus 1 frame at either rate.
            tol_s = 0.005 + max(1.0 / float(in_sr), 1.0 / float(out_sr))
            if abs(out_dur - in_dur) > tol_s:
                mismatches.append(
                    (
                        stem,
                        {
                            "duration_s": round(out_dur, 6),
                            "frames": out_frames,
                            "samplerate": out_sr,
                        },
                    )
                )
        except Exception as e:
            mismatches.append((stem, f"error:{e}"))

    if mismatches:
        return (
            False,
            f"output length mismatch (input={in_frames}f@{in_sr}Hz={in_dur:.6f}s): {mismatches}",
        )
    return True, "ok"


def _normalize_output_files(output_files: dict) -> dict:
    if not isinstance(output_files, dict):
        return output_files
    normalized = {}
    for stem, path in output_files.items():
        if not isinstance(stem, str):
            continue
        key = _normalize_stems([stem])
        key = key[0] if isinstance(key, list) and key else stem.strip().lower()
        if key in normalized:
            # Avoid clobbering; keep first and suffix later duplicates.
            i = 2
            new_key = f"{key}_{i}"
            while new_key in normalized:
                i += 1
                new_key = f"{key}_{i}"
            key = new_key
        normalized[key] = path
    return normalized


async def main():
    if len(sys.argv) < 2:
        send_ipc({"type": "error", "error": "Usage: inference.py <json_config>"})
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        send_ipc({"type": "error", "error": f"Invalid JSON config: {e}"})
        sys.exit(1)

    config = _normalize_config(config)
    config = _normalize_volume_compensation(config)

    # Initialize logging (logs go to stderr/file due to redirect above)
    setup_logging()

    model_manager = None
    separation_manager = None
    last_job = None

    try:
        # Initialize Managers
        models_dir = config.get("models_dir")
        if models_dir:
            model_manager = ModelManager(models_dir=Path(models_dir))
        else:
            model_manager = ModelManager()

        # Determine device settings
        device_str = config.get("device", "cpu")
        gpu_enabled = "cuda" in device_str.lower()

        separation_manager = SeparationManager(model_manager, gpu_enabled=gpu_enabled)

        # Parse job parameters
        file_path = config.get("file_path")
        model_id = config.get("model_id")
        output_dir = config.get("output_dir")

        if not file_path or not output_dir:
            raise ValueError("file_path and output_dir are required")

        # Prepare job kwargs
        stems = _normalize_stems(config.get("stems"))
        bit_depth = _normalize_bit_depth(config.get("bit_depth", "16"))
        job_kwargs = {
            "file_path": file_path,
            "model_id": model_id,
            "output_dir": output_dir,
            "device": device_str,
            "stems": stems,
            "overlap": config.get("overlap"),
            "segment_size": config.get("segment_size"),
            "batch_size": config.get("batch_size"),
            "tta": config.get("tta", False),
            "output_format": config.get("output_format", "wav"),
            "shifts": config.get("shifts", 1),
            "bitrate": config.get("bitrate"),
            "invert": config.get("invert", False),
            "normalize": config.get("normalize", False),
            "bit_depth": bit_depth,
            "ensemble_config": config.get("ensemble_config"),
            "ensemble_algorithm": config.get("ensemble_algorithm", "average"),
            "stem_algorithms": config.get("stem_algorithms"),
            "phase_params": config.get("phase_params"),
            "phase_fix_enabled": config.get("phase_fix_enabled", False),
            "split_freq": config.get("split_freq"),
            "vc_enabled": config.get("vc_enabled", False),
            "vc_stage": config.get("vc_stage"),
            "vc_db_per_extra_model": config.get("vc_db_per_extra_model"),
        }

        # Filter None values to allow defaults
        job_kwargs = {k: v for k, v in job_kwargs.items() if v is not None}

        async def _run_once(attempt: int):
            # Create Job
            job_id = separation_manager.create_job(**job_kwargs)

            # Setup callbacks
            job = separation_manager.get_job(job_id)

            def progress_callback(progress, message, device=None):
                evt = {
                    "type": "separation_progress",
                    "job_id": job_id,
                    "progress": progress,
                    "message": message,
                    "meta": {"attempt": attempt},
                }
                if device:
                    evt["device"] = device
                send_ipc(evt)

            def complete_callback(output_files):
                # Rust backend handles the final response; keep stdout clean.
                pass

            job.progress_callback = progress_callback
            job.on_complete = complete_callback

            # Start Job
            send_ipc(
                {
                    "type": "separation_started",
                    "job_id": job_id,
                    "meta": {"attempt": attempt},
                }
            )

            success = separation_manager.start_job(job_id)
            if not success:
                raise RuntimeError("Failed to start separation job")

            # Monitor loop
            while job.status in [
                "pending",
                "queued",
                "running",
                "validating",
                "processing",
            ]:
                if shutdown_event.is_set():
                    separation_manager.cancel_job(job_id)
                    send_ipc({"type": "separation_cancelled", "job_id": job_id})
                    break
                await asyncio.sleep(0.1)

            return job

        for attempt in (1, 2):
            job = await _run_once(attempt)
            last_job = job

            if job.status == "completed":
                ok, reason = _validate_output_lengths(file_path, job.output_files)
                if ok or attempt == 2:
                    normalized_outputs = _normalize_output_files(job.output_files)

                    # Ensure UI sees an explicit terminal progress update before completion.
                    send_ipc(
                        {
                            "type": "separation_progress",
                            "job_id": job.job_id,
                            "progress": 100.0,
                            "message": "Complete",
                            "meta": {"attempt": attempt},
                        }
                    )
                    send_ipc(
                        {
                            "type": "separation_complete",
                            "job_id": job.job_id,
                            "output_files": normalized_outputs,
                            "meta": {"attempt": attempt, "output_validation": reason},
                        }
                    )
                    break

                send_ipc(
                    {
                        "type": "separation_progress",
                        "job_id": job.job_id,
                        "progress": 99.0,
                        "message": f"Output validation failed; retrying once: {reason}",
                        "meta": {"attempt": attempt, "output_validation": reason},
                    }
                )
                continue

            if job.status == "failed":
                send_ipc(
                    {
                        "type": "separation_error",
                        "job_id": job.job_id,
                        "error": job.error or "Unknown error",
                        "meta": {"attempt": attempt},
                    }
                )
                break

            if job.status == "cancelled":
                send_ipc(
                    {
                        "type": "separation_cancelled",
                        "job_id": job.job_id,
                        "meta": {"attempt": attempt},
                    }
                )
                break

    except Exception as e:
        # trace = traceback.format_exc()
        job_id = getattr(last_job, "job_id", None) if last_job else None

        err_str = str(e)
        device_payload = _parse_structured_device_error(err_str)

        evt = {"type": "separation_error", "error": err_str}

        if device_payload:
            # Provide stable, actionable fields for the backend/UI (no silent fallback).
            evt["error_code"] = device_payload.get("code")
            evt["error_message"] = device_payload.get("message")
            evt["error_device"] = device_payload.get("device")
            evt["error_details"] = device_payload.get("details")

        if job_id:
            evt["job_id"] = job_id

        send_ipc(evt)
        sys.exit(1)
    finally:
        if separation_manager:
            try:
                # Cleanup/Shutdown logic if needed
                pass
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
