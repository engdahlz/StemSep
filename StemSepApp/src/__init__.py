"""
Make `src/` a *non-package*.

This repository uses a src-layout where importable code lives under `src/stemsep/`.
Having an `__init__.py` at the `src/` root makes Python treat `src` as its own
top-level package, which can introduce confusing import behavior and namespace
collisions.

Important:
- This file is intentionally empty of imports/side-effects.
- Do not add exports here. Import from `stemsep.*` instead.
"""

# Intentionally no imports.
