import os

from stemsep.core.separation_manager import SeparationManager


def test_map_pipeline_final_outputs_prefers_canonical_stems():
    last_outputs = {
        "target": "t.wav",
        "residual": "r.wav",
    }

    pipeline_config = [
        {"step_name": "step_0"},
        {"step_name": "step_1"},
        {"step_name": "step_2"},
    ]

    step_outputs = {
        "step_0": {"vocals": "v0.wav", "instrumental": "i0.wav"},
        "step_1": {"instrumental": "i1.wav"},
        "step_2": last_outputs,
    }

    mapped = SeparationManager._map_pipeline_final_outputs(
        last_outputs=last_outputs,
        step_outputs=step_outputs,
        pipeline_config=pipeline_config,
    )

    assert mapped == {"instrumental": "t.wav", "vocals": "v0.wav"}


def test_map_pipeline_final_outputs_residual_fallback_to_vocals():
    last_outputs = {
        "target": "t.wav",
        "residual": "r.wav",
    }

    pipeline_config = [
        {"step_name": "step_0"},
        {"step_name": "step_1"},
    ]

    step_outputs = {
        "step_0": {"instrumental": "i0.wav"},
        "step_1": last_outputs,
    }

    mapped = SeparationManager._map_pipeline_final_outputs(
        last_outputs=last_outputs,
        step_outputs=step_outputs,
        pipeline_config=pipeline_config,
    )

    assert mapped == {"instrumental": "t.wav", "vocals": "r.wav"}


def test_map_pipeline_final_outputs_noop_when_already_canonical():
    last_outputs = {"instrumental": "i.wav", "vocals": "v.wav"}

    mapped = SeparationManager._map_pipeline_final_outputs(
        last_outputs=last_outputs,
        step_outputs={},
        pipeline_config=[],
    )

    assert mapped == last_outputs
