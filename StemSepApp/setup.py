"""
setup.py compatibility shim.

This project uses `pyproject.toml` with the setuptools PEP 517/518 backend as the
single source of truth for build/packaging metadata.

Some tooling still expects a `setup.py` file to exist, and setuptools' backend
may consult it when building an sdist. Keeping a non-functional stub (or raising
in setup()) can break builds with errors like "No distribution was found".

This shim delegates to setuptools using the configuration in `pyproject.toml`.
"""

from __future__ import annotations

from setuptools import setup

if __name__ == "__main__":
    setup()
