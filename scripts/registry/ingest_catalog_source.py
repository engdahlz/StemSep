#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import requests

from catalog_v3_common import (
    DEFAULT_CATALOG_FRAGMENTS_ROOT,
    artifact_extension,
    infer_source_provider_resolver,
    is_direct_artifact_url,
    normalize_source_entry,
    slugify,
    source_locator_is_deterministic,
    url_basename,
    write_json,
)

CHUNK_SIZE = 1024 * 256


def classify_url(url: str) -> str:
    provider, resolver, locator = infer_source_provider_resolver(url)
    if provider in {"google_drive", "proton_drive"} and source_locator_is_deterministic(
        provider, resolver, locator, url
    ):
        return "folder_entry" if resolver.endswith("_entry") else "artifact"
    if is_direct_artifact_url(url):
        return "artifact"
    if resolver in {"github_release_asset", "github_raw", "huggingface_resolve"}:
        return "artifact"
    if any(token in url.lower() for token in ("github.com/", "huggingface.co/", "drive.google.com/drive/folders/", "drive.proton.me/urls/")):
        return "repo_page"
    return "reference"


def probe_url(url: str, compute_sha256: bool) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "StemSepCatalogIngest/4.1"})
    response = session.get(url, allow_redirects=True, timeout=30, stream=compute_sha256)
    response.raise_for_status()
    content_length = response.headers.get("content-length")
    sha256 = None
    if compute_sha256:
        digest = hashlib.sha256()
        total = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            digest.update(chunk)
        sha256 = digest.hexdigest()
        content_length = str(total)
    return {
        "final_url": response.url,
        "content_type": response.headers.get("content-type"),
        "size_bytes": int(content_length) if str(content_length or "").isdigit() else None,
        "sha256": sha256,
    }


def build_source_fragment(url: str, compute_sha256: bool) -> dict[str, Any]:
    provider, resolver, locator = infer_source_provider_resolver(url)
    classification = classify_url(url)
    basename = url_basename(url)
    probe = {}
    resolver_viable = classification in {"artifact", "folder_entry"} and source_locator_is_deterministic(
        provider, resolver, locator, url
    )
    verified = False
    try:
        probe = probe_url(url, compute_sha256=compute_sha256)
        verified = True
    except Exception as exc:
        probe = {"error": str(exc)}
        verified = False

    install_policy = "direct" if resolver_viable and verified else "manual"
    catalog_tier = "verified" if install_policy == "direct" and verified else "advanced_manual"

    fragment = normalize_source_entry(
        {
            "url": url,
            "provider": provider,
            "resolver": resolver,
            "locator": locator,
            "verified": verified,
            "resolver_viable": resolver_viable and verified,
            "size_bytes": probe.get("size_bytes"),
            "sha256": probe.get("sha256"),
            "last_checked": probe.get("checked_at"),
            "final_url": probe.get("final_url"),
            "content_type": probe.get("content_type"),
            "classification": classification,
            "install_policy": install_policy,
            "catalog_tier": catalog_tier,
        },
        fallback_filename=basename,
    )
    fragment["suggested_fragment_path"] = f"sources/{provider}/{fragment['source_id']}.json"
    fragment["suggested_model_id"] = slugify(Path(basename or fragment["source_id"]).stem)
    fragment["suggested_artifact"] = {
        "filename": basename,
        "canonical_path": None,
        "kind": artifact_extension(basename) or None,
        "source_ids": [fragment["source_id"]],
    }
    return fragment


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a single source URL into a catalog source fragment suggestion.")
    parser.add_argument("url")
    parser.add_argument("--fragments-root", default=str(DEFAULT_CATALOG_FRAGMENTS_ROOT))
    parser.add_argument("--out")
    parser.add_argument("--compute-sha256", action="store_true")
    args = parser.parse_args()

    fragment = build_source_fragment(args.url, compute_sha256=args.compute_sha256)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.fragments_root) / "sources" / str(fragment["provider"]) / f"{fragment['source_id']}.json"
    write_json(out_path, fragment)
    print(f"Wrote source fragment -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
