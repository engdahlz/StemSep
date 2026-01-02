#!/usr/bin/env python3
"""
Patch model registry to point to mirrored GitHub Release assets.
"""

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "reports" / "mirror_upload_manifest.json"
REGISTRY_PATH = (
    REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
)


def main():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Create backup
    backup_path = REGISTRY_PATH.with_suffix(".bak.json")
    shutil.copy2(REGISTRY_PATH, backup_path)
    print(f"Backed up registry to {backup_path}")

    mirror_repo = manifest.get("mirror_repo", "engdahlz/stemsep-models")
    tag = manifest.get("release_tag", "models-v2-2026-01-01")

    updates = 0

    # Map manifest needs to model_id
    manifest_models = {m["model_id"]: m for m in manifest.get("models", [])}

    for model_entry in registry.get("models", []):
        model_id = model_entry.get("id")
        if not model_id or model_id not in manifest_models:
            continue

        man_entry = manifest_models[model_id]
        needs = man_entry.get("needs", [])

        links = model_entry.setdefault("links", {})

        for need in needs:
            kind = need["kind"]  # checkpoint or config
            filename = need["filename"]
            mirror_url = need.get("mirror_url")

            if not mirror_url:
                # Fallback construction if not in manifest
                mirror_url = f"https://github.com/{mirror_repo}/releases/download/{tag}/{filename}"

            # Update Link
            if kind in ["checkpoint", "config"]:
                old_url = links.get(kind)
                if old_url != mirror_url:
                    print(f"[{model_id}] Updating {kind} URL -> {mirror_url}")
                    links[kind] = mirror_url
                    updates += 1

    if updates > 0:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Trailing newline
        print(f"Registry updated with {updates} changes.")
    else:
        print("No changes needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
