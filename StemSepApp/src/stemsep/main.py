"""
Canonical StemSep desktop entry point.

Electron is the primary product UI. This module keeps the legacy Tkinter UI
available for internal debugging and fallback workflows without exposing it as
the main product surface.
"""

from __future__ import annotations

from stemsep.core.config import Config
from stemsep.core.fonts import ensure_font_permissions
from stemsep.core.gpu_detector import GPUDetector
from stemsep.core.logger import setup_logging
from stemsep.models.model_manager import ModelManager
from stemsep.ui_legacy.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    _ = argv

    logger = setup_logging()
    config = Config()

    try:
        ensure_font_permissions()
    except Exception:
        logger.debug("Font permission check failed (non-fatal).", exc_info=True)

    try:
        GPUDetector()
    except Exception:
        logger.debug("GPU detector initialization encountered an error.", exc_info=True)

    try:
        ModelManager(config=config)
    except TypeError:
        ModelManager()  # type: ignore[call-arg]
    except Exception:
        logger.debug(
            "ModelManager initialization failed (may be deferred).", exc_info=True
        )

    app = MainWindow(config=config)
    app.mainloop()
    return 0
