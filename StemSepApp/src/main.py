"""
Backward-compatible entry wrapper for older `python src/main.py` workflows.

The canonical package entry point lives at `stemsep.main`.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Delegate to the packaged StemSep entry point."""
    from stemsep.main import main as app_main

    return int(app_main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
