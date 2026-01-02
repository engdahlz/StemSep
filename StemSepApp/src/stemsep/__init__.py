"""
StemSep package root.

This package provides a stable, non-colliding import namespace for the StemSep
application code. Historically, StemSep used top-level packages like `audio`,
`core`, `models`, and `ui`. Those are now organized under `stemsep.*` to avoid
namespace collisions and to align with the distribution name (`stemsep`).

Public API policy:
- Keep this file lightweight and free of heavy imports (e.g. torch) at import time.
- Re-export only the most stable, high-level entry points here.
- Prefer importing submodules directly for implementation details.

Examples:
    from stemsep import Config
    from stemsep.core.separation_manager import SeparationManager
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# NOTE:
# Do NOT import large/heavy dependencies here. Keep import side-effects minimal.

__all__ = [
    "__version__",
    # High-level, stable entry points
    "Config",
    "SeparationManager",
    "ModelManager",
]

try:
    __version__ = _pkg_version("stemsep")
except PackageNotFoundError:
    # Package is being used from source without installed metadata
    __version__ = "0.0.0+local"

# Stable re-exports (keep these imports shallow; they should not trigger heavy deps).
from stemsep.core.config import Config  # noqa: E402
from stemsep.core.separation_manager import SeparationManager  # noqa: E402
from stemsep.models.model_manager import ModelManager  # noqa: E402
