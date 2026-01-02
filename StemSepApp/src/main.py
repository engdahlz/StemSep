"""
Legacy GUI entry module.

This file is kept for backwards compatibility with older workflows that run
`python src/main.py` or import `main`. The robust, long-term entry point is:

    python -m stemsep

The application code now lives under the `stemsep` import namespace to avoid
top-level package name collisions (e.g. `audio`, `core`, `models`, `ui`).

Important:
- No sys.path hacks here. Editable installs and normal installs should work.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """
    Start the StemSep desktop GUI application.

    Args:
        argv: Optional CLI args. Currently ignored (GUI app).

    Returns:
        Process exit code.
    """
    _ = argv

    # Import lazily so `import main` stays cheap and doesn't error in environments
    # where GUI deps aren't installed.
    from stemsep.core.config import Config
    from stemsep.core.fonts import ensure_font_permissions
    from stemsep.core.gpu_detector import GPUDetector
    from stemsep.core.logger import setup_logging
    from stemsep.models.model_manager import ModelManager
    from stemsep.ui.main_window import MainWindow

    # Setup logging early
    logger = setup_logging()

    # Load configuration
    config = Config()

    # Ensure font readability on Linux to avoid CustomTkinter fallback rendering
    try:
        ensure_font_permissions()
    except Exception:
        logger.debug("Font permission check failed (non-fatal).", exc_info=True)

    # Initialize GPU detector early, so device errors are surfaced deterministically
    try:
        GPUDetector()
    except Exception:
        # Device policy is enforced deeper in the pipeline; we log here for visibility.
        logger.debug("GPU detector initialization encountered an error.", exc_info=True)

    # Ensure models are discovered/initialized
    try:
        ModelManager(config=config)
    except TypeError:
        # Some versions/branches may have a different signature.
        ModelManager()  # type: ignore[call-arg]
    except Exception:
        logger.debug(
            "ModelManager initialization failed (may be deferred).", exc_info=True
        )

    # Start UI
    app = MainWindow(config=config)
    app.mainloop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
