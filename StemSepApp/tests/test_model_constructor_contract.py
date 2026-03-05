import asyncio
import builtins
from pathlib import Path

import stemsep.audio.separation_engine as separation_engine_module
from stemsep.audio.separation_engine import SeparationEngine
from stemsep.models.model_factory import ModelFactory
import pytest


class _DummyManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.installed_models = {}

    def is_model_installed(self, model_id: str) -> bool:
        return True

    def ensure_model_ready(self, model_id: str):
        return None


def test_load_model_does_not_inject_model_path_for_factory_models(
    tmp_path: Path, monkeypatch
):
    model_id = "demo-model"
    (tmp_path / f"{model_id}.ckpt").write_bytes(b"stub")
    (tmp_path / f"{model_id}.yaml").write_text("model: {}\n", encoding="utf-8")

    engine = SeparationEngine(model_manager=_DummyManager(tmp_path), gpu_enabled=False)
    monkeypatch.setattr(
        engine,
        "_get_model_info",
        lambda _id: {
            "architecture": "BS-Roformer",
            "links": {"checkpoint": ""},
            "runtime": {},
        },
    )

    class DummyCtorLoadedModel:
        pass

    # Ensure _load_model takes the "constructor-loaded model" branch and skips torch.load().
    monkeypatch.setattr(separation_engine_module, "MDXNetModel", DummyCtorLoadedModel)
    monkeypatch.setattr(
        separation_engine_module, "DemucsModel", type("DummyDemucs", (), {})
    )

    captured = {}

    def fake_create_model(architecture: str, config: dict):
        captured["architecture"] = architecture
        captured["config"] = dict(config)
        return DummyCtorLoadedModel()

    monkeypatch.setattr(ModelFactory, "create_model", staticmethod(fake_create_model))

    loaded = asyncio.run(engine._load_model(model_id))

    assert isinstance(loaded, DummyCtorLoadedModel)
    assert captured["architecture"] == "BS-Roformer"
    assert "model_path" not in captured["config"]


def test_load_model_fno_variant_fails_fast_when_fno1d_unavailable(
    tmp_path: Path, monkeypatch
):
    model_id = "fno-model"
    (tmp_path / f"{model_id}.ckpt").write_bytes(b"stub")
    (tmp_path / f"{model_id}.yaml").write_text("model: {}\n", encoding="utf-8")

    engine = SeparationEngine(model_manager=_DummyManager(tmp_path), gpu_enabled=False)
    monkeypatch.setattr(
        engine,
        "_get_model_info",
        lambda _id: {
            "architecture": "BS-Roformer",
            "links": {"checkpoint": ""},
            "runtime": {"variant": "fno"},
        },
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "neuralop.models":
            raise ImportError("forced missing neuralop.models")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        ModelFactory,
        "create_model",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())),
    )

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(engine._load_model(model_id))

    assert "FNO1d" in str(exc.value)
