import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from stemsep.audio.separation_engine import SeparationJob
from stemsep.core.separation_manager import SeparationManager


def _mark_old(path: Path, *, age_seconds: int = 90000):
    old_ts = time.time() - age_seconds
    os.utime(path, (old_ts, old_ts))


def test_startup_cleanup_skips_active_and_locked_dirs(tmp_path: Path, monkeypatch, caplog):
    base = tmp_path / "tmp"
    base.mkdir()

    deletable = base / "stemsep_deletable"
    deletable.mkdir()
    _mark_old(deletable)

    active = base / "stemsep_active"
    active.mkdir()
    _mark_old(active)
    with open(active / SeparationJob.ACTIVE_LOCK_FILENAME, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "created_at": time.time()}, f)

    locked = base / "stemsep_locked"
    locked.mkdir()
    _mark_old(locked)

    old_checkpoint = base / "stemsep_checkpoint_abc.json"
    old_checkpoint.write_text("{}", encoding="utf-8")
    _mark_old(old_checkpoint)

    original_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        if str(path) == str(locked):
            raise PermissionError("locked for test")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(base))
    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    mgr = SeparationManager.__new__(SeparationManager)
    mgr.logger = logging.getLogger("test.startup.cleanup")

    with caplog.at_level(logging.INFO):
        mgr._cleanup_old_temp_dirs()

    assert not deletable.exists()
    assert active.exists()
    assert locked.exists()
    assert not old_checkpoint.exists()
    assert "Startup temp cleanup summary:" in caplog.text
