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
from typing import TYPE_CHECKING, Any

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

_LAZY_EXPORTS = {
    "Config": ("stemsep.core.config", "Config"),
    "SeparationManager": ("stemsep.core.separation_manager", "SeparationManager"),
    "ModelManager": ("stemsep.models.model_manager", "ModelManager"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily resolve public exports to keep `import stemsep` lightweight.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'stemsep' has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


if TYPE_CHECKING:
    from stemsep.core.config import Config
    from stemsep.core.separation_manager import SeparationManager
    from stemsep.models.model_manager import ModelManager
