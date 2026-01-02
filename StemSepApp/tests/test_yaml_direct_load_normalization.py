import pytest

pytest.importorskip("onnxruntime")


def _contains_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, list) or isinstance(obj, tuple):
        return any(_contains_key(v, key) for v in obj)
    return False


def test_roformer_build_strips_num_subbands_everywhere(tmp_path):
    from stemsep.audio.simple_separator import SimpleSeparator

    sep = SimpleSeparator(models_dir=str(tmp_path))

    model_data = {
        "audio": {"num_subbands": 4, "num_channels": 2},
        "model": {"num_subbands": 60, "dim": 384, "flash_attn": True},
        "training": {
            "instruments": ["Vocals", "Instrumental"],
            "target_instrument": "",
        },
    }

    built = sep._build_audio_separator_model_data(model_data, is_roformer=True)

    assert not _contains_key(built, "num_subbands")
    assert _contains_key(built, "num_bands")


def test_non_roformer_build_preserves_num_subbands_by_default(tmp_path):
    from stemsep.audio.simple_separator import SimpleSeparator

    sep = SimpleSeparator(models_dir=str(tmp_path))

    model_data = {
        "audio": {"num_subbands": 4, "num_channels": 2},
        "model": {"num_subbands": 4, "dim": 384},
    }

    built = sep._build_audio_separator_model_data(model_data, is_roformer=False)

    assert _contains_key(built, "num_subbands")
