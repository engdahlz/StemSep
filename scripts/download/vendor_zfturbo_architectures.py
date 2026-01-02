"""Vendor selected model architectures from ZFTurbo/Music-Source-Separation-Training.

This repo supports several model families (SCNet, BandIt, Apollo) whose reference
implementations live in ZFTurbo's MSST repo. Rather than copying that code into
this repository manually, this script downloads the required Python source files
into `StemSepApp/src/models/architectures/zfturbo_vendor/`.

Why a script?
- Keeps our repo diffs small and reviewable.
- Makes it easy to update vendored code to upstream.

Usage (PowerShell):
  python scripts/vendor_zfturbo_architectures.py

Notes:
- Requires internet access.
- Only downloads `.py` files and creates missing `__init__.py` files.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Iterable

import requests

REPO = "ZFTurbo/Music-Source-Separation-Training"
REF = "main"

# MSST paths to vendor (directories are downloaded recursively).
PATHS: list[str] = [
    "models/scnet",
    "models/bandit",
    "models/look2hear/models",
]

DEST_ROOT = Path("StemSepApp/src/models/architectures/zfturbo_vendor")


def _api_url(path: str) -> str:
    return f"https://api.github.com/repos/{REPO}/contents/{path}?ref={REF}"


def _ensure_init_files(dest_dir: Path) -> None:
    """Ensure every directory up the tree is a Python package."""
    cur = dest_dir
    while True:
        init_path = cur / "__init__.py"
        if not init_path.exists():
            init_path.write_text("\n", encoding="utf-8")
        if cur == DEST_ROOT:
            break
        cur = cur.parent


def _download_file(download_url: str, dest_path: Path) -> None:
    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(resp.text, encoding="utf-8")


def _download_dir(path: str, dest_dir: Path) -> None:
    resp = requests.get(_api_url(path), timeout=60)
    resp.raise_for_status()
    items = resp.json()

    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected GitHub API response for {path}: {type(items)}")

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        item_name = item.get("name")
        item_path = item.get("path")

        if not item_type or not item_name or not item_path:
            continue

        if item_type == "dir":
            _download_dir(item_path, dest_dir / item_name)
            continue

        if item_type != "file":
            continue

        # Only vendor Python sources.
        if not str(item_name).endswith(".py"):
            continue

        download_url = item.get("download_url")
        if not download_url:
            # Fallback: some API responses omit download_url; use contents API.
            file_resp = requests.get(_api_url(item_path), timeout=60)
            file_resp.raise_for_status()
            file_obj = file_resp.json()
            content_b64 = file_obj.get("content")
            if not content_b64:
                raise RuntimeError(f"Unable to download {item_path}")
            data = base64.b64decode(content_b64)
            dest_path = dest_dir / item_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)
        else:
            dest_path = dest_dir / item_name
            _download_file(download_url, dest_path)

    _ensure_init_files(dest_dir)


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_init_files(DEST_ROOT)

    print(f"Vendoring MSST architectures into: {DEST_ROOT}")
    for p in PATHS:
        rel = Path(p)
        # Strip leading "models/" so we can import under `...zfturbo_vendor.<arch>`.
        if rel.parts and rel.parts[0] == "models":
            rel = Path(*rel.parts[1:])
        dest = DEST_ROOT / rel
        print(f"- Downloading {p} -> {dest}")
        _download_dir(p, dest)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
