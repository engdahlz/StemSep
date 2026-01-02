from stemsep.core.separation_manager import SeparationManager


def test_canonicalize_stem_keys_maps_other_to_instrumental():
    inp = {"other": "other.wav", "vocals": "vocals.wav"}
    out = SeparationManager._canonicalize_stem_keys(inp)
    assert out["instrumental"] == "other.wav"
    assert "other" not in out
    assert out["vocals"] == "vocals.wav"


def test_canonicalize_stem_keys_maps_vocal_to_vocals():
    inp = {"instrumental": "inst.wav", "vocal": "v.wav"}
    out = SeparationManager._canonicalize_stem_keys(inp)
    assert out["instrumental"] == "inst.wav"
    assert out["vocals"] == "v.wav"
    assert "vocal" not in out
