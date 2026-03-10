"""
`python -m stemsep` entry point.

This starts the internal legacy desktop UI. Product work should target the
Electron app; this entry point is kept for debugging and compatibility.
"""

from __future__ import annotations

import sys


def _run_gui() -> int:
    try:
        from stemsep.main import main as app_main  # type: ignore[attr-defined]

        return int(app_main() or 0)
    except ModuleNotFoundError:
        pass
    except AttributeError:
        # Module exists but doesn't expose `main()`
        pass

    try:
        from main import main as app_main  # type: ignore[attr-defined]

        return int(app_main() or 0)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Unable to locate the StemSep legacy UI entry point.\n"
            "Expected either `stemsep.main:main` (preferred) or `main:main` (compat).\n"
            "If you're running from source, ensure you installed editable:\n"
            "  python -m pip install -e StemSepApp\n"
        ) from e
    except AttributeError as e:
        raise SystemExit(
            "Found the compat `main` module but it does not define a `main()` function.\n"
            "Expose `main()` in `stemsep.main` or keep `src/main.py` as a thin wrapper."
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
