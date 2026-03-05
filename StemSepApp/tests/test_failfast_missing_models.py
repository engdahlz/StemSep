import asyncio
import os
from pathlib import Path

import pytest

from stemsep.models.model_manager import ModelManager
from stemsep.core.separation_manager import SeparationManager


def test_create_ensemble_job_fails_when_phase_fix_enabled_and_any_model_missing(tmp_path: Path):
    mm = ModelManager(models_dir=tmp_path)

    # Ensure registry has at least one known model id for the test.
    # If the shipped registry changes, this test should be updated to pick an existing id.
    all_models = list(mm.models.keys())
    assert len(all_models) > 0

    existing_id = all_models[0]
    missing_id = "definitely_missing_model_id_for_test"

    mgr = SeparationManager(mm, gpu_enabled=False)

    with pytest.raises(RuntimeError) as e:
        mgr.create_ensemble_job(
            file_path=str(tmp_path / "in.wav"),
            ensemble_config=[
                {"model_id": existing_id, "weight": 1.0},
                {"model_id": missing_id, "weight": 1.0},
            ],
            output_dir=str(tmp_path / "out"),
            phase_fix_enabled=True,
        )

    msg = str(e.value).lower()
    assert "cannot start ensemble" in msg
    assert "missing models" in msg


def test_engine_load_model_fails_fast_without_auto_download(tmp_path: Path):
    mm = ModelManager(models_dir=tmp_path)
    mgr = SeparationManager(mm, gpu_enabled=False)

    # Pick any valid model id from registry, but ensure it's not installed locally.
    all_models = list(mm.models.keys())
    assert len(all_models) > 0

    model_id = all_models[0]

    # models_dir is empty => not installed.
    assert not mm.is_model_installed(model_id)

    with pytest.raises(FileNotFoundError) as e:
        asyncio.run(mgr.engine._load_model(model_id))

    assert "not installed" in str(e.value).lower()
