# coding: utf-8
"""
Download original YAML configs from HuggingFace URLs for all models.
This is the correct approach - use the original configs that match the checkpoints.
"""

import json
import os
import ssl
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "StemSepApp" / "assets" / "models"
MODELS_DIR = Path.home() / ".stemsep" / "models"

# Create SSL context that doesn't verify (for corporate proxies)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_config(url: str, dest_path: Path) -> bool:
    """Download a YAML config from URL."""
    try:
        print(f"  Downloading from: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            content = response.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            print(f"  ✓ Saved to: {dest_path.name}")
            return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("DOWNLOADING ORIGINAL YAML CONFIGS FROM HUGGINGFACE")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    skipped = 0

    # Iterate through all model JSON files
    for json_file in sorted(ASSETS_DIR.glob("*.json")):
        model_id = json_file.stem

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                model_def = json.load(f)
        except Exception as e:
            print(f"\n⚠️ Could not read {json_file.name}: {e}")
            failed += 1
            continue

        # Check if there's a config URL
        links = model_def.get("links", {})
        config_url = links.get("config", "")

        if not config_url:
            print(f"\n⚠️ {model_id}: No config URL")
            skipped += 1
            continue

        # Check if we already have the checkpoint (otherwise no point downloading config)
        ckpt_exists = any(
            [
                (MODELS_DIR / f"{model_id}.ckpt").exists(),
                (MODELS_DIR / f"{model_id}.th").exists(),
                (MODELS_DIR / f"{model_id}.pth").exists(),
            ]
        )

        if not ckpt_exists:
            # Skip models where we don't have the checkpoint
            skipped += 1
            continue

        print(f"\n{model_id}")

        dest_yaml = MODELS_DIR / f"{model_id}.yaml"

        if download_config(config_url, dest_yaml):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Downloaded: {success}")
    print(f"⚠️ Skipped (no checkpoint): {skipped}")
    print(f"❌ Failed: {failed}")

    if success > 0:
        print("\nRun scripts/dev/diagnose_model.py again to verify config match!")


if __name__ == "__main__":
    main()
