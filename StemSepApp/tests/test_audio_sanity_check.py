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


def test_audio_sanity_check_skips_without_soundfile():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
check = mod['_audio_sanity_check']

# Force soundfile unavailable
check.__globals__['sf'] = None

report, reason = check('in.wav', {'vocals': 'vocals.wav'})
assert report.get('ok') is True
assert report.get('skipped') is True
assert 'soundfile' in reason
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_audio_sanity_check_flags_silent_stem():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy
import types

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
check = mod['_audio_sanity_check']

class Info:
    def __init__(self, frames, samplerate):
        self.frames = frames
        self.samplerate = samplerate

def fake_info(path):
    return Info(48000 * 10, 48000)

# Create deterministic arrays
import numpy as np

def fake_read(path, always_2d=True, frames=None):
    n = 48000 * 10
    if frames is not None:
        n = min(n, int(frames))

    if 'in.wav' in path:
        # non-silent input
        x = np.ones((n, 2), dtype=np.float32) * 0.1
        return x, 48000
    if 'vocals.wav' in path:
        # near silent
        x = np.zeros((n, 2), dtype=np.float32)
        return x, 48000
    if 'instrumental.wav' in path:
        x = np.ones((n, 2), dtype=np.float32) * 0.1
        return x, 48000
    x = np.ones((n, 2), dtype=np.float32) * 0.1
    return x, 48000

sf = types.SimpleNamespace(info=fake_info, read=fake_read)
check.__globals__['sf'] = sf

report, reason = check('in.wav', {'vocals': 'vocals.wav', 'instrumental': 'instrumental.wav'})
assert report.get('ok') is False
assert any('near-silent' in w for w in report.get('warnings', []))
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_audio_sanity_check_reconstruction_reports_rel_diff():
    repo_root = Path(__file__).resolve().parents[2]

    code = r'''
import os
import runpy
import types

mod = runpy.run_path(os.path.join('scripts', 'inference.py'))
check = mod['_audio_sanity_check']

class Info:
    def __init__(self, frames, samplerate):
        self.frames = frames
        self.samplerate = samplerate

def fake_info(path):
    return Info(44100 * 5, 44100)

import numpy as np

def fake_read(path, always_2d=True, frames=None):
    n = 44100 * 5
    if frames is not None:
        n = min(n, int(frames))

    if 'in.wav' in path:
        # mix is 0.2
        x = np.ones((n, 2), dtype=np.float32) * 0.2
        return x, 44100
    if 'vocals.wav' in path:
        x = np.ones((n, 2), dtype=np.float32) * 0.1
        return x, 44100
    if 'instrumental.wav' in path:
        x = np.ones((n, 2), dtype=np.float32) * 0.1
        return x, 44100
    x = np.ones((n, 2), dtype=np.float32) * 0.1
    return x, 44100

sf = types.SimpleNamespace(info=fake_info, read=fake_read)
check.__globals__['sf'] = sf

report, reason = check('in.wav', {'vocals': 'vocals.wav', 'instrumental': 'instrumental.wav'})
assert report.get('ok') is True
assert 'reconstruction' in report
assert report['reconstruction']['rel_diff'] is not None
assert report['reconstruction']['rel_diff'] < 1e-6
'''

    result = _run_python(code, cwd=repo_root)
    assert result.returncode == 0, (result.stdout, result.stderr)
