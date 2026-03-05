from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "registry" / "sync_guide_knowledge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_guide_knowledge", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize_model_token():
    mod = _load_module()
    assert mod.normalize_model_token(" BS-Roformer-ViperX-1297 ") == "bs-roformer-viperx-1297"
    assert mod.normalize_model_token("`SCNet XL`") == "scnet-xl"


def test_extract_candidates_prefers_model_like_tokens():
    mod = _load_module()
    text = """
    Good: bs-roformer-viperx-1297 and mel-band-karaoke-becruily.
    Ignore numbers: 0.25 and 0.4-0.5.
    Ignore paths: C:\\models\\foo.ckpt and /tmp/bar.py.
    """
    got = mod.extract_guide_candidates(
        text=text,
        registry_ids={"bs-roformer-viperx-1297"},
        alias_keys={"scnet-xl"},
    )
    assert "bs-roformer-viperx-1297" in got
    assert "mel-band-karaoke-becruily" in got
    assert "0.25" not in got


def test_gate_report_thresholds():
    mod = _load_module()
    report = {
        "divergence": {
            "days_since_guide_revision": 20,
            "missing_in_registry_count": 30,
        }
    }
    status, reasons = mod.gate_report(report, max_age_days=14, max_unsynced_models=25)
    assert status == "fail"
    assert reasons
