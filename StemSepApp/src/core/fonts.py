"""
Font utilities to improve CustomTkinter rendering reliability on Linux.

We do not ship fonts or download them here to avoid bloat and network needs.
Instead, we ensure the user's existing font files (if present) are readable,
which fixes common "Permission denied" issues for fonts under ~/.fonts.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
import logging


FONTS_DIR = Path.home() / ".fonts"
EXPECTED_FILES = [
    "Roboto-Regular.ttf",
    "Roboto-Medium.ttf",
    "CustomTkinter_shapes_font.otf",
]


def ensure_font_permissions(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger("StemSep")
    try:
        if not FONTS_DIR.exists():
            return
        for name in EXPECTED_FILES:
            p = FONTS_DIR / name
            if not p.exists():
                continue
            try:
                # Ensure user-readable, group/world-readable: 0644
                mode = p.stat().st_mode
                desired = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
                if (mode & 0o777) != desired:
                    os.chmod(p, desired)
            except Exception as e:
                log.debug(f"Font permission adjust failed for {p}: {e}")
    except Exception as e:
        log.debug(f"Font check failed: {e}")

