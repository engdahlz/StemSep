import os
import sys
from pathlib import Path

import pytest

# Ensure StemSepApp/src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


torch = pytest.importorskip("torch")

from stemsep.models.model_factory import ModelFactory


def test_scnet_forward_shape():
    model = ModelFactory.create_model("SCNet", {})
    x = torch.randn(1, 2, 8192)
    y = model(x)
    assert y.ndim == 4
    assert y.shape[0] == 1
    assert y.shape[2] == 2


def test_apollo_forward_shape():
    model = ModelFactory.create_model(
        "Apollo", {"sr": 44100, "win": 32, "feature_dim": 32, "layer": 1}
    )
    x = torch.randn(1, 2, 8192)
    y = model(x)
    assert y.ndim == 3
    assert y.shape[0] == 1
    assert y.shape[1] == 2


def test_bandit_forward_shape_inference_wrapper():
    yaml = pytest.importorskip("yaml")

    # ZFTurbo configs sometimes include `!!python/tuple` (e.g. mixup_probs). For
    # inference we can safely treat it as a plain list.
    if not getattr(yaml.SafeLoader, "_stemsep_tuple_patch", False):
        yaml.SafeLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            lambda loader, node: loader.construct_sequence(node),
        )
        yaml.SafeLoader._stemsep_tuple_patch = True

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "docs"
        / "vendor"
        / "ZFTurbo Repo"
        / "Music-Source-Separation-Training-main"
        / "configs"
        / "config_dnr_bandit_bsrnn_multi_mus64.yaml"
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]

    model = ModelFactory.create_model("BandIt", model_cfg)

    # BandIt expects waveform shape (B, C, T) where C == config.model.in_channel
    x = torch.randn(1, int(model_cfg["in_channel"]), 8192)
    y = model(x)
    assert y.ndim == 4
    assert y.shape[0] == 1
    assert y.shape[1] == len(model_cfg["stems"])
    assert y.shape[2] == int(model_cfg["in_channel"])
