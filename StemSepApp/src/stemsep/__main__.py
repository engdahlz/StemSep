"""
`python -m stemsep` entry point.

This module starts the desktop GUI application in the most robust way we can:
- Prefer importing the app entry (historically `main.py`, now modernized) by module.
- Avoid any path-hacking; editable installs and normal installs should both work.

Notes:
- We keep this file lightweight; the actual GUI imports (customtkinter, etc.)
  are done in the application module.
"""

from __future__ import annotations

import sys


def _run_gui() -> int:
    """
    Start the GUI application.

    We support both of these implementations (depending on how the repo is laid out):
    1) Preferred: `stemsep.main` module (after namespace migration)
    2) Fallback: legacy `main` module under src-layout (older layout)

    Returns a process exit code.
    """
    # Preferred: `stemsep.main` (if/when `main.py` is moved under the package)
    try:
        from stemsep.main import main as app_main  # type: ignore[attr-defined]

        return int(app_main() or 0)
    except ModuleNotFoundError:
        pass
    except AttributeError:
        # Module exists but doesn't expose `main()`
        pass

    # Fallback: legacy `main.py` at src root (older layout)
    try:
        from main import main as app_main  # type: ignore[attr-defined]

        return int(app_main() or 0)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Unable to locate the StemSep GUI entry point.\n"
            "Expected either `stemsep.main:main` (preferred) or `main:main` (legacy).\n"
            "If you're running from source, ensure you installed editable:\n"
            "  python -m pip install -e StemSepApp\n"
        ) from e
    except AttributeError as e:
        raise SystemExit(
            "Found the legacy `main` module but it does not define a `main()` function.\n"
            "Please expose `main()` in the GUI entry module (recommended) or update __main__.py."
        ) from e


def main(argv: list[str] | None = None) -> int:
    """
    Console-style entry. `argv` is accepted for future CLI flags,
    but currently ignored (GUI app).
    """
    _ = argv
    return _run_gui()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
