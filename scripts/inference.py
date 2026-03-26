#!/usr/bin/env python3
"""
Single-shot inference script for StemSep Rust Backend.
Accepts either a JSON configuration string or ``--config-file <path>``.
Outputs newline-delimited JSON events to stdout.
"""

import asyncio
import json
import os
import signal
import sys
import traceback
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WorkerProtocol:
    kind: str = "selection_step"
    version: int = 1

    @classmethod
    def from_config(cls, value: Any) -> "WorkerProtocol":
        if not isinstance(value, dict):
            return cls()
        kind = str(value.get("kind") or cls.kind).strip() or cls.kind
        version_raw = value.get("version", cls.version)
        try:
            version = int(version_raw)
        except Exception:
            version = cls.version
        return cls(kind=kind, version=version)


def _first_execution_step(execution_plan: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the first concrete execution step from a resolved plan."""
    if not isinstance(execution_plan, dict):
        return None

    resolved = execution_plan.get("resolved_step")
    if isinstance(resolved, dict):
        return resolved

    steps = execution_plan.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                return step
    return None


@dataclass(frozen=True)
class SeparationWorkerRequest:
    protocol: WorkerProtocol
    raw_config: dict[str, Any]
    job_id: Optional[str]
    file_path: str
    output_dir: str
    models_dir: Optional[str]
    device: str
    gpu_enabled: bool
    job_kwargs: dict[str, Any]
    selection_type: Optional[str]
    selection_id: Optional[str]
    execution_plan: Optional[dict[str, Any]]
    resolved_bundle: Optional[dict[str, Any]]
    resolved_step: Optional[dict[str, Any]]

    @classmethod
    def from_json_arg(cls, raw_json: str) -> "SeparationWorkerRequest":
        try:
            raw_config = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config: {exc}") from exc
        return cls.from_config(raw_config)

    @classmethod
    def from_config_file(cls, path: str) -> "SeparationWorkerRequest":
        config_path = Path(path).expanduser()
        try:
            raw_json = config_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(f"Config file not found: {config_path}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read config file {config_path}: {exc}") from exc
        return cls.from_json_arg(raw_json)

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> "SeparationWorkerRequest":
        protocol = WorkerProtocol.from_config(
            raw_config.get("worker_protocol") if isinstance(raw_config, dict) else None
        )
        strict_worker = protocol.kind == "selection_step"
        config = _normalize_volume_compensation(
            _normalize_config(raw_config, strict_worker=strict_worker)
        )
        protocol = WorkerProtocol.from_config(config.get("worker_protocol"))

        file_path = str(config.get("file_path") or "").strip()
        output_dir = str(config.get("output_dir") or "").strip()
        if not file_path or not output_dir:
            raise ValueError("file_path and output_dir are required")
        job_id = str(config.get("job_id") or "").strip() or None

        device = str(config.get("device") or "cpu").strip() or "cpu"
        workflow = (
            config.get("workflow") if isinstance(config.get("workflow"), dict) else None
        )
        pipeline_config = (
            config.get("pipeline_config")
            if isinstance(config.get("pipeline_config"), list)
            else None
        )
        execution_plan = (
            config.get("execution_plan")
            if isinstance(config.get("execution_plan"), dict)
            else None
        )
        resolved_bundle = (
            config.get("resolved_bundle")
            if isinstance(config.get("resolved_bundle"), dict)
            else None
        )

        resolved_step = _first_execution_step(execution_plan)

        stems = _normalize_stems(config.get("stems"))
        bit_depth = _normalize_bit_depth(config.get("bit_depth", "16"))
        selection_type = (
            str(config.get("selection_type")).strip().lower()
            if config.get("selection_type")
            else None
        )
        selection_id = (
            str(config.get("selection_id")).strip()
            if config.get("selection_id")
            else None
        )
        if not strict_worker and isinstance(execution_plan, dict):
            if not selection_type:
                raw = execution_plan.get("selection_type") or execution_plan.get(
                    "selectionType"
                )
                if isinstance(raw, str) and raw.strip():
                    selection_type = raw.strip().lower()

            if not selection_id:
                raw = execution_plan.get("selection_id") or execution_plan.get("selectionId")
                if isinstance(raw, str) and raw.strip():
                    selection_id = raw.strip()

        resolved_model_id = str(config.get("model_id") or "").strip() or None
        if not resolved_model_id and selection_type == "model" and selection_id:
            resolved_model_id = selection_id
        if not resolved_model_id and isinstance(execution_plan, dict):
            candidate = execution_plan.get("model_id") or execution_plan.get(
                "effective_model_id"
            )
            if isinstance(candidate, str) and candidate.strip():
                resolved_model_id = candidate.strip()
        if not resolved_model_id and isinstance(resolved_step, dict):
            candidate = resolved_step.get("model_id")
            if isinstance(candidate, str) and candidate.strip():
                resolved_model_id = candidate.strip()
        if not resolved_model_id and pipeline_config:
            resolved_model_id = "pipeline"
        if not resolved_model_id and config.get("ensemble_config") is not None:
            resolved_model_id = "ensemble"

        job_kwargs = {
            "file_path": file_path,
            "model_id": resolved_model_id,
            "output_dir": output_dir,
            "device": device,
            "stems": stems,
            "pipeline_config": pipeline_config,
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
            "selection_type": config.get("selection_type"),
            "selection_id": config.get("selection_id"),
            "execution_plan": execution_plan,
            "resolved_bundle": resolved_bundle,
        }

        return cls(
            protocol=protocol,
            raw_config=config,
            job_id=job_id,
            file_path=file_path,
            output_dir=output_dir,
            models_dir=str(config.get("models_dir")).strip()
            if config.get("models_dir")
            else None,
            device=device,
            gpu_enabled="cuda" in device.lower(),
            job_kwargs={k: v for k, v in job_kwargs.items() if v is not None},
            selection_type=selection_type,
            selection_id=selection_id,
            execution_plan=execution_plan,
            resolved_bundle=resolved_bundle,
            resolved_step=resolved_step,
        )


class SeparationWorkerRuntime:
    """Single-job inference worker owned by the Rust control plane."""

    def __init__(self, request: SeparationWorkerRequest):
        self.request = request
        self.model_manager: Optional[ModelManager] = None
        self.separation_manager: Optional[SeparationManager] = None
        self.last_job = None

    def initialize(self) -> None:
        if self.request.models_dir:
            self.model_manager = ModelManager(models_dir=Path(self.request.models_dir))
        else:
            self.model_manager = ModelManager()
        self.separation_manager = SeparationManager(
            self.model_manager,
            gpu_enabled=self.request.gpu_enabled,
            max_workers=1,
        )

    async def run(self) -> None:
        assert self.separation_manager is not None

        for attempt in (1, 2):
            job = await self._run_once(attempt)
            self.last_job = job

            if job.status == "completed":
                ok, reason = _validate_output_lengths(
                    self.request.file_path, job.output_files
                )
                if ok or attempt == 2:
                    await self._emit_completion(job, attempt, reason)
                    return

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
                return

            if job.status == "cancelled":
                send_ipc(
                    {
                        "type": "separation_cancelled",
                        "job_id": job.job_id,
                        "meta": {"attempt": attempt},
                    }
                )
                return

    async def _run_once(self, attempt: int):
        assert self.separation_manager is not None
        event_job_id = self.request.job_id

        def progress_callback(progress, message, device=None):
            evt = {
                "type": "separation_progress",
                "job_id": event_job_id,
                "progress": progress,
                "message": message,
                "meta": {
                    "attempt": attempt,
                    "worker_protocol": {
                        "kind": self.request.protocol.kind,
                        "version": self.request.protocol.version,
                    },
                    "selection_type": self.request.selection_type,
                    "selection_id": self.request.selection_id,
                },
            }
            if device:
                evt["device"] = device
            send_ipc(evt)

        send_ipc(
            {
                "type": "separation_started",
                "job_id": event_job_id,
                "meta": {
                    "attempt": attempt,
                    "worker_protocol": {
                        "kind": self.request.protocol.kind,
                        "version": self.request.protocol.version,
                    },
                    "selection_type": self.request.selection_type,
                    "selection_id": self.request.selection_id,
                    "resolved_step": self.request.resolved_step,
                },
            }
        )

        job = await self.separation_manager.run_worker_job(
            external_job_id=event_job_id,
            **{
                **self.request.job_kwargs,
                "progress_callback": progress_callback,
                "on_complete": lambda output_files: None,
            },
        )

        if shutdown_event.is_set() and job.status in ["pending", "queued", "running", "validating", "processing"]:
            # Local worker shutdown only: the Rust control plane already owns
            # the lifecycle and this worker just mirrors the cancellation state.
            job.status = "cancelled"
            send_ipc({"type": "separation_cancelled", "job_id": event_job_id or job.job_id})

        return job

    async def _emit_completion(self, job, attempt: int, validation_reason: str) -> None:
        event_job_id = self.request.job_id or job.job_id
        normalized_outputs = _normalize_output_files(job.output_files)
        sanity_report, sanity_reason = _audio_sanity_check(
            self.request.file_path, normalized_outputs
        )

        send_ipc(
            {
                "type": "separation_progress",
                "job_id": event_job_id,
                "progress": 99.5,
                "message": f"Sanity check: {sanity_reason}",
                "meta": {"attempt": attempt, "sanity_check": sanity_report},
            }
        )
        send_ipc(
            {
                "type": "separation_progress",
                "job_id": event_job_id,
                "progress": 100.0,
                "message": "Complete",
                "meta": {"attempt": attempt},
            }
        )
        send_ipc(
            {
                "type": "separation_complete",
                "job_id": event_job_id,
                "output_files": normalized_outputs,
                "meta": {
                    "attempt": attempt,
                    "output_validation": validation_reason,
                    "sanity_check": sanity_report,
                    "worker_protocol": {
                        "kind": self.request.protocol.kind,
                        "version": self.request.protocol.version,
                    },
                },
            }
        )


def _normalize_config(config: dict, strict_worker: bool = False) -> dict:
    """Normalize UI/backend config shapes for the Python worker runtime.

    Electron UI sends some objects in a richer shape (e.g. ensemble_config as an object
    with {models, algorithm, phaseFixEnabled,...}). The core SeparationManager expects:
      - ensemble_config: List[{model_id, weight?}, ...]
      - ensemble_algorithm: str
      - phase_fix_enabled: bool
      - phase_params: {lowHz, highHz, highFreqWeight, enabled?}

    Additionally, the UI may now send an explicit pipeline_config derived from the
    resolved workflow plan. When present, that is authoritative and we avoid
    rebuilding the executable pipeline from workflow metadata. In strict worker
    mode (the Rust control plane already resolved the selection), workflow
    metadata is treated as informational only and is not expanded back into
    executable pipeline / ensemble decisions.
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
        "selectionType": "selection_type",
        "selectionId": "selection_id",
        "executionPlan": "execution_plan",
        "resolvedBundle": "resolved_bundle",
        "pipelineConfig": "pipeline_config",
        "postProcessingSteps": "post_processing_steps",
        "volumeCompensation": "volume_compensation",
    }
    for src_key, dst_key in camel_to_snake.items():
        if dst_key not in config and src_key in config:
            config[dst_key] = config[src_key]

    execution_plan = config.get("execution_plan")
    if not isinstance(execution_plan, dict):
        execution_plan = None

    resolved_bundle = config.get("resolved_bundle")
    if not isinstance(resolved_bundle, dict):
        resolved_bundle = None

    if isinstance(execution_plan, dict):
        if not config.get("selection_type"):
            raw = execution_plan.get("selection_type") or execution_plan.get(
                "selectionType"
            )
            if isinstance(raw, str) and raw.strip():
                config["selection_type"] = raw.strip().lower()

        if not config.get("selection_id"):
            raw = execution_plan.get("selection_id") or execution_plan.get("selectionId")
            if isinstance(raw, str) and raw.strip():
                config["selection_id"] = raw.strip()

        if resolved_bundle is None:
            candidate = execution_plan.get("resolved_bundle") or execution_plan.get(
                "resolvedBundle"
            )
            if isinstance(candidate, dict):
                resolved_bundle = candidate
                config["resolved_bundle"] = candidate

        plan_overrides = {
            "model_id": execution_plan.get("effective_model_id")
            or execution_plan.get("model_id"),
            "workflow": None if strict_worker else execution_plan.get("workflow"),
            "pipeline_config": execution_plan.get("pipeline_config"),
            "ensemble_config": execution_plan.get("ensemble_config"),
            "ensemble_algorithm": execution_plan.get("ensemble_algorithm"),
            "stems": execution_plan.get("stems"),
            "device": execution_plan.get("device"),
            "overlap": execution_plan.get("overlap"),
            "segment_size": execution_plan.get("segment_size"),
            "batch_size": execution_plan.get("batch_size"),
            "tta": execution_plan.get("tta"),
            "output_format": execution_plan.get("output_format"),
            "shifts": execution_plan.get("shifts"),
            "bitrate": execution_plan.get("bitrate"),
            "invert": execution_plan.get("invert"),
            "split_freq": execution_plan.get("split_freq"),
            "normalize": execution_plan.get("normalize"),
            "bit_depth": execution_plan.get("bit_depth"),
            "vc_enabled": execution_plan.get("vc_enabled"),
            "vc_stage": execution_plan.get("vc_stage"),
            "vc_db_per_extra_model": execution_plan.get("vc_db_per_extra_model"),
            "phase_params": execution_plan.get("phase_params"),
            "phase_fix_enabled": execution_plan.get("phase_fix_enabled"),
            "stem_algorithms": execution_plan.get("stem_algorithms"),
        }
        for key, value in plan_overrides.items():
            if value is None:
                continue
            config[key] = value

        if strict_worker:
            # Do not let legacy workflow metadata leak back into the worker path.
            # Rust already resolved the effective execution plan.
            config.pop("workflow", None)

    workflow = config.get("workflow")
    if isinstance(workflow, dict) and not strict_worker:
        if "runtime_policy" not in config and isinstance(
            workflow.get("runtimePolicy"), dict
        ):
            config["runtime_policy"] = workflow.get("runtimePolicy")
        if "export_policy" not in config and isinstance(
            workflow.get("exportPolicy"), dict
        ):
            config["export_policy"] = workflow.get("exportPolicy")
        if not config.get("stems") and isinstance(workflow.get("stems"), list):
            config["stems"] = workflow.get("stems")
        if not strict_worker and not config.get("post_processing_steps") and isinstance(
            workflow.get("postprocess"), list
        ):
            config["post_processing_steps"] = workflow.get("postprocess")

        kind = str(workflow.get("kind") or "").strip().lower()
        models = workflow.get("models")
        steps = workflow.get("steps")
        blend = workflow.get("blend") if isinstance(workflow.get("blend"), dict) else {}

        if kind == "ensemble" and not config.get("ensemble_config") and isinstance(models, list):
            ensemble_models = []
            for model in models:
                if not isinstance(model, dict):
                    continue
                model_id = model.get("model_id")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                entry = {"model_id": model_id}
                weight = model.get("weight")
                if isinstance(weight, (int, float)):
                    entry["weight"] = float(weight)
                ensemble_models.append(entry)
            if ensemble_models:
                config["ensemble_config"] = ensemble_models
                config["model_id"] = "ensemble"
                if isinstance(blend.get("algorithm"), str) and blend.get("algorithm").strip():
                    config["ensemble_algorithm"] = blend.get("algorithm")
                if isinstance(blend.get("stemAlgorithms"), dict):
                    config["stem_algorithms"] = blend.get("stemAlgorithms")
                if blend.get("phaseFixEnabled") is True:
                    config["phase_fix_enabled"] = True
                if isinstance(blend.get("phaseFixParams"), dict):
                    config["phase_params"] = blend.get("phaseFixParams")
                if config.get("split_freq") is None and blend.get("splitFreq") is not None:
                    config["split_freq"] = blend.get("splitFreq")

        if kind == "single" and not config.get("model_id") and isinstance(models, list):
            for model in models:
                if isinstance(model, dict) and isinstance(model.get("model_id"), str):
                    config["model_id"] = model.get("model_id")
                    break

        if kind == "pipeline" and not config.get("pipeline_config") and isinstance(steps, list):
            normalized_steps = []
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                normalized = {
                    "step_name": step.get("id")
                    or step.get("name")
                    or f"step_{index + 1}",
                    "action": step.get("action") or "separate",
                }
                for key in (
                    "model_id",
                    "source_model",
                    "input_source",
                    "output",
                    "apply_to",
                    "weight",
                    "optional",
                ):
                    if key in step and step.get(key) is not None:
                        normalized[key] = step.get(key)

                params = step.get("params")
                if isinstance(params, dict):
                    normalized.update(params)

                normalized_steps.append(normalized)

            if normalized_steps:
                config["pipeline_config"] = normalized_steps
                if not isinstance(config.get("model_id"), str) or not config.get("model_id"):
                    config["model_id"] = workflow.get("id") or "pipeline"

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
        not strict_worker
        and isinstance(post_steps, list)
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


def _audio_sanity_check(input_path: str, output_files: dict) -> tuple[dict, str]:
    if sf is None:
        return ({"ok": True, "skipped": True}, "soundfile not available")
    if not isinstance(output_files, dict) or not output_files:
        return ({"ok": True, "skipped": True}, "no outputs")

    try:
        in_info = sf.info(input_path)
        in_sr = int(getattr(in_info, "samplerate", 0) or 0)
        in_frames = int(getattr(in_info, "frames", 0) or 0)
        if in_sr <= 0 or in_frames <= 0:
            return ({"ok": True, "skipped": True}, "input metadata unavailable")
        in_dur_s = in_frames / float(in_sr)
    except Exception as e:
        return ({"ok": True, "skipped": True}, f"could not inspect input: {e}")

    report: dict[str, Any] = {
        "ok": True,
        "input": {
            "sr": in_sr,
            "frames": in_frames,
            "duration_s": round(in_dur_s, 6),
        },
        "stems": {},
    }

    # Lightweight reads (no resampling). This is best-effort and should never fail the run.
    # We intentionally cap the number of frames read for sanity metrics.
    max_frames = min(in_frames, in_sr * 30)  # up to first 30 seconds
    try:
        in_audio, in_audio_sr = sf.read(input_path, always_2d=True, frames=max_frames)
        if int(in_audio_sr) != in_sr:
            # Keep sr from info but note mismatch
            report["input"]["sr_read"] = int(in_audio_sr)
        in_audio = in_audio.astype("float32", copy=False)
    except Exception as e:
        report["ok"] = True
        report["skipped"] = True
        return (report, f"could not read input audio: {e}")

    def _rms(a):
        return float((a * a).mean() ** 0.5) if a.size else 0.0

    def _peak(a):
        return float(abs(a).max()) if a.size else 0.0

    # Per-stem metrics
    for stem, path in output_files.items():
        if not isinstance(path, str) or not path:
            continue
        try:
            info = sf.info(path)
            sr = int(getattr(info, "samplerate", 0) or 0)
            frames = int(getattr(info, "frames", 0) or 0)
            dur_s = round(frames / float(sr), 6) if sr > 0 and frames > 0 else None

            audio, sr_read = sf.read(path, always_2d=True, frames=max_frames)
            audio = audio.astype("float32", copy=False)
            # Use first N frames (matching input slice length for comparison)
            n = min(len(in_audio), len(audio))
            audio_n = audio[:n]

            rms = _rms(audio_n)
            peak = _peak(audio_n)
            is_silent = rms < 1e-4
            is_clipping = peak > 0.999

            report["stems"][stem] = {
                "sr": sr,
                "frames": frames,
                "duration_s": dur_s,
                "rms": round(rms, 8),
                "peak": round(peak, 6),
                "silent": bool(is_silent),
                "clipping": bool(is_clipping),
                "sr_read": int(sr_read),
            }

            if is_silent:
                report["ok"] = False
                report.setdefault("warnings", []).append(f"{stem}: near-silent")
            if is_clipping:
                report["ok"] = False
                report.setdefault("warnings", []).append(f"{stem}: possible clipping")
        except Exception as e:
            report["ok"] = False
            report.setdefault("warnings", []).append(f"{stem}: read failed: {e}")

    # Simple reconstruction check when we have vocals + instrumental.
    try:
        if "vocals" in output_files and "instrumental" in output_files:
            v, _ = sf.read(output_files["vocals"], always_2d=True, frames=max_frames)
            i, _ = sf.read(output_files["instrumental"], always_2d=True, frames=max_frames)
            v = v.astype("float32", copy=False)
            i = i.astype("float32", copy=False)
            n = min(len(in_audio), len(v), len(i))
            if n > 0:
                mix = v[:n] + i[:n]
                ref = in_audio[:n]
                diff = mix - ref
                rms_in = _rms(ref)
                rms_diff = _rms(diff)
                rel = (rms_diff / rms_in) if rms_in > 1e-12 else None
                report["reconstruction"] = {
                    "window_s": round(n / float(in_sr), 6),
                    "rms_in": round(rms_in, 8),
                    "rms_diff": round(rms_diff, 8),
                    "rel_diff": round(float(rel), 6) if rel is not None else None,
                }
    except Exception:
        # Best-effort only
        pass

    return (report, "ok" if report.get("ok") else "warnings")


async def main():
    usage = "Usage: inference.py <json_config> | --config-file <path> | @<path>"
    if len(sys.argv) < 2:
        send_ipc({"type": "error", "error": usage})
        sys.exit(1)

    try:
        arg = sys.argv[1]
        if arg == "--config-file":
            if len(sys.argv) < 3:
                raise ValueError(usage)
            request = SeparationWorkerRequest.from_config_file(sys.argv[2])
        elif arg.startswith("@") and len(arg) > 1:
            request = SeparationWorkerRequest.from_config_file(arg[1:])
        else:
            request = SeparationWorkerRequest.from_json_arg(arg)
    except ValueError as e:
        send_ipc({"type": "error", "error": str(e)})
        sys.exit(1)

    # Initialize logging (logs go to stderr/file due to redirect above)
    setup_logging()

    runtime = SeparationWorkerRuntime(request)

    try:
        runtime.initialize()
        await runtime.run()

    except Exception as e:
        # trace = traceback.format_exc()
        job_id = request.job_id or (
            getattr(runtime.last_job, "job_id", None) if runtime.last_job else None
        )

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
        if runtime.separation_manager:
            try:
                # Cleanup/Shutdown logic if needed
                pass
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
