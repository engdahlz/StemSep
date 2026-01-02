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


def test_validate_output_lengths_duration_based_tolerates_sr_change():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy
import types

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
validate = mod['_validate_output_lengths']

class Info:
    def __init__(self, frames, samplerate):
        self.frames = frames
        self.samplerate = samplerate

# 48k input, 44.1k output with same duration (23.256s)
def fake_info(path):
    if 'in.wav' in path:
        return Info(1116288, 48000)
    return Info(1025590, 44100)

sf = types.SimpleNamespace(info=fake_info)
validate.__globals__['sf'] = sf

ok, reason = validate('in.wav', {'target': 'out.wav'})
assert ok, reason
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_validate_output_lengths_fails_on_duration_mismatch():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy
import types

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
validate = mod['_validate_output_lengths']

class Info:
    def __init__(self, frames, samplerate):
        self.frames = frames
        self.samplerate = samplerate

def fake_info(path):
    if 'in.wav' in path:
        return Info(1116288, 48000)
    # Shorter duration to trigger mismatch
    return Info(1000000, 44100)

sf = types.SimpleNamespace(info=fake_info)
validate.__globals__['sf'] = sf

ok, reason = validate('in.wav', {'target': 'out.wav'})
assert (not ok), reason
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)
