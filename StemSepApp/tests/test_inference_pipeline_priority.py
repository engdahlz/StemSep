import subprocess
import sys
from pathlib import Path


def _run_python(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )


def test_normalize_config_preserves_explicit_pipeline_config():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import json
import os
import runpy

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
normalize = mod['_normalize_config']

cfg = {
    "model_id": "pipeline",
    "workflow": {
        "id": "recipe_conflict",
        "kind": "pipeline",
        "steps": [
            {"id": "workflow_step", "action": "separate", "model_id": "workflow-model"}
        ]
    },
    "pipeline_config": [
        {"step_name": "explicit_step", "action": "separate", "model_id": "explicit-model"}
    ]
}

normalized = normalize(cfg)
assert normalized["pipeline_config"][0]["step_name"] == "explicit_step"
assert normalized["pipeline_config"][0]["model_id"] == "explicit-model"
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_normalize_config_rebuilds_pipeline_when_explicit_config_is_empty():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
normalize = mod['_normalize_config']

cfg = {
    "model_id": "pipeline",
    "workflow": {
        "id": "recipe_fallback",
        "kind": "pipeline",
        "steps": [
            {"id": "workflow_step", "action": "phase_fix", "model_id": "workflow-model"}
        ]
    },
    "pipeline_config": []
}

normalized = normalize(cfg)
assert normalized["pipeline_config"][0]["step_name"] == "workflow_step"
assert normalized["pipeline_config"][0]["action"] == "phase_fix"
assert normalized["pipeline_config"][0]["model_id"] == "workflow-model"
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_main_uses_explicit_pipeline_config_for_execution():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import asyncio
import json
import os
import runpy
import sys
import types

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
main = mod['main']
captured = {}

class FakeJob:
    def __init__(self):
        self.status = "completed"
        self.output_files = {"vocals": "vocals.wav", "instrumental": "instrumental.wav"}
        self.job_id = "job-123"
        self.progress_callback = None
        self.on_complete = None

class FakeSeparationManager:
    def __init__(self, *args, **kwargs):
        self.job = FakeJob()

    def create_pipeline_job(self, **kwargs):
        captured["pipeline_config"] = kwargs.get("pipeline_config")
        return self.job.job_id

    def create_job(self, **kwargs):
        captured["create_job"] = kwargs
        return self.job.job_id

    def get_job(self, job_id):
        return self.job

    def start_job(self, job_id):
        return True

    def cancel_job(self, job_id):
        return True

class FakeModelManager:
    def __init__(self, *args, **kwargs):
        pass

def fake_send_ipc(event):
    captured.setdefault("events", []).append(event.get("type"))

main.__globals__['SeparationManager'] = FakeSeparationManager
main.__globals__['ModelManager'] = FakeModelManager
main.__globals__['send_ipc'] = fake_send_ipc
main.__globals__['setup_logging'] = lambda: None
main.__globals__['_validate_output_lengths'] = lambda file_path, outputs: (True, "ok")
main.__globals__['_audio_sanity_check'] = lambda file_path, outputs: ({"ok": True}, "ok")
main.__globals__['shutdown_event'] = types.SimpleNamespace(is_set=lambda: False)

cfg = {
    "file_path": "input.wav",
    "model_id": "pipeline",
    "output_dir": "out",
    "device": "cpu",
    "workflow": {
        "id": "recipe_conflict",
        "kind": "pipeline",
        "steps": [
            {"id": "workflow_step", "action": "separate", "model_id": "workflow-model"}
        ]
    },
    "pipeline_config": [
        {"step_name": "explicit_step", "action": "separate", "model_id": "explicit-model"}
    ]
}

sys.argv = ['inference.py', json.dumps(cfg)]
asyncio.run(main())

assert "create_job" not in captured, captured
assert captured["pipeline_config"][0]["step_name"] == "explicit_step", captured
assert captured["pipeline_config"][0]["model_id"] == "explicit-model", captured
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)
