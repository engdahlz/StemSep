#!/usr/bin/env python3
"""
Upload staged assets to GitHub Release.
"""

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_DIR = REPO_ROOT / "reports" / "release_assets_stage"
DEFAULT_MANIFEST = REPO_ROOT / "reports" / "mirror_upload_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--repo", default="engdahlz/stemsep-models")
    parser.add_argument("--tag", default="models-v2-2026-01-01")
    parser.add_argument("--token", default=os.environ.get("STEMSEP_GH_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        print("ERROR: STEMSEP_GH_TOKEN not set")
        return 1

    stage_dir = Path(args.stage_dir)
    if not stage_dir.exists():
        print(f"Stage dir not found: {stage_dir}")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    # 1. Get Release ID
    headers = {
        "Authorization": f"token {args.token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check if release exists
    print(f"Checking release {args.repo} @ {args.tag}...")
    api_base = f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}"
    try:
        r = requests.get(api_base, headers=headers, timeout=30)
        if r.status_code == 404:
            print(f"Release not found: {api_base}")
            return 1
        if r.status_code == 401:
            print("Authentication failed (401). Check STEMSEP_GH_TOKEN.")
            return 1
        r.raise_for_status()
    except Exception as e:
        print(f"Error checking release: {e}")
        return 1

    release_data = r.json()
    release_id = release_data["id"]
    upload_url_template = release_data[
        "upload_url"
    ]  # e.g. https://uploads.github.com/...{?name,label}
    existing_assets = {a["name"]: a for a in release_data.get("assets", [])}

    print(f"Release ID: {release_id}")
    print(f"Existing assets on GitHub: {len(existing_assets)}")

    # 2. Identify files to upload
    # We filter stage dir by what is in the manifest to ensure we only upload what we intend
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    allowed_filenames = set()
    for m in manifest.get("models", []):
        for n in m.get("needs", []):
            allowed_filenames.add(n["filename"])

    files_to_upload = []
    # Sort for deterministic order
    for f in sorted(stage_dir.iterdir()):
        if f.is_file() and f.name in allowed_filenames:
            if f.name in existing_assets:
                # Optionally check size? But for now trust name.
                print(f"Skipping {f.name} (already exists)")
            else:
                files_to_upload.append(f)

    print(f"Files to upload: {len(files_to_upload)}")

    if not files_to_upload:
        print("Nothing to upload.")
        return 0

    # 3. Upload
    upload_url_base = upload_url_template.split("{")[0]

    for i, fpath in enumerate(files_to_upload):
        print(f"[{i + 1}/{len(files_to_upload)}] Uploading {fpath.name}...")
        mime_type, _ = mimetypes.guess_type(fpath)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Read file into memory? Or stream? Requests supports streaming upload if data is a file-like object.
        with open(fpath, "rb") as f_data:
            params = {"name": fpath.name}
            headers_up = headers.copy()
            headers_up["Content-Type"] = mime_type

            try:
                # 10 min timeout for large files
                r_up = requests.post(
                    upload_url_base,
                    headers=headers_up,
                    params=params,
                    data=f_data,
                    timeout=600,
                )
                if r_up.status_code == 201:
                    print("  Success")
                else:
                    print(f"  Failed: {r_up.status_code} {r_up.text}")
                    # Don't abort, try next
            except Exception as e:
                print(f"  Exception: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
