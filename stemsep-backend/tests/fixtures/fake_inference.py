"""Deterministic fake inference script for backend protocol tests.

Contract:
- Accepts a single JSON config argument (ignored).
- Writes newline-delimited JSON events to stdout.
- Does not require any ML/audio dependencies.
"""

from __future__ import annotations

import json
import sys
import time
import os


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    raw_config = sys.argv[1] if len(sys.argv) > 1 else "{}"
    try:
        parsed_config = json.loads(raw_config)
    except Exception:
        parsed_config = {"_error": "failed_to_parse_config", "raw": raw_config}

    # Intentionally emit a conflicting job_id to ensure the Rust backend
    # rewrites/normalizes job_id for renderer-side filtering.
    python_job_id = "python-job-123"

    emit({"type": "debug_config", "job_id": python_job_id, "config": parsed_config})

    emit({"type": "separation_started", "job_id": python_job_id})
    time.sleep(0.01)

    # Pipeline-style progress messages (Rust parses these into meta.phase/meta.step)
    emit({
        "type": "separation_progress",
        "job_id": python_job_id,
        "progress": 10.0,
        "message": "Running step 1: demix (model-a)"
    })
    time.sleep(0.01)

    emit({
        "type": "separation_progress",
        "job_id": python_job_id,
        "progress": 50.0,
        "message": "Running step 2: demud (model-b)"
    })
    time.sleep(0.01)

    emit({
        "type": "separation_progress",
        "job_id": python_job_id,
        "progress": 95.0,
        "message": "Finalizing pipeline..."
    })
    time.sleep(0.01)

    if os.environ.get("STEMSEP_FAKE_EMIT_RETRY") == "1":
        emit({
            "type": "separation_progress",
            "job_id": python_job_id,
            "progress": 99.0,
            "message": "Output validation failed; retrying once: output length mismatch",
            "meta": {"attempt": 1, "output_validation": "output length mismatch"}
        })
        time.sleep(0.01)

    emit({
        "type": "separation_complete",
        "job_id": python_job_id,
        "output_files": {"vocals": "C:/tmp/vocals.wav", "instrumental": "C:/tmp/instrumental.wav"}
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
