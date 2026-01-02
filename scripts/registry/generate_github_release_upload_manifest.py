#!/usr/bin/env python3
"""
Generate a GitHub Releases upload manifest for broken registry entries.

Why:
- The v2 registry may contain vendor URLs that are broken (404), gated (401), or placeholders ("MISSING").
- We want the app registry to point at stable, public direct-download URLs hosted on GitHub Releases.
- This script discovers broken entries and emits a manifest of *which files you need to upload* to a given
  GitHub repo release tag, with deterministic filenames.

Inputs:
- StemSepApp/assets/registry/models.v2.source.json

Outputs (by default into repo's reports/):
- reports/mirror_upload_manifest.json
- reports/mirror_upload_manifest.md

Deterministic filename policy (default):
- checkpoint: {model_id}{ext}   where ext is inferred from URL (.ckpt/.pth/.onnx) or defaults to .ckpt
- config:      {model_id}.yaml

Usage:
  python scripts/registry/generate_github_release_upload_manifest.py \
    --mirror-owner engdahlz --mirror-repo stemsep-models --tag models-v2-2026-01-01

Notes:
- This script does NOT download files and does NOT upload release assets.
- It only produces a manifest you can use to collect/rename files and upload them to GitHub Releases.

Exit codes:
- 0: success
- 2: invalid args / missing files
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    # Python 3.10+
    import urllib.error
    import urllib.request
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import urllib modules: {e}") from e


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_V2 = (
    REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
)
DEFAULT_OUT_JSON = REPO_ROOT / "reports" / "mirror_upload_manifest.json"
DEFAULT_OUT_MD = REPO_ROOT / "reports" / "mirror_upload_manifest.md"


ALLOWED_CHECKPOINT_EXTS = (".ckpt", ".pth", ".onnx")
DEFAULT_CHECKPOINT_EXT = ".ckpt"


@dataclass(frozen=True)
class LinkFailure:
    model_id: str
    kind: str  # "checkpoint" | "config"
    url: str
    status: Optional[int]
    host: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _url_host(url: str) -> str:
    try:
        return urlparse(url).netloc if "://" in url else "INVALID"
    except Exception:
        return "INVALID"


def _infer_checkpoint_ext_from_url(url: str) -> Optional[str]:
    """
    Infer checkpoint extension from URL/path.
    Returns None for placeholders or if no known extension is present.
    """
    if not isinstance(url, str):
        return None
    if url.upper() == "MISSING":
        return None
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return None
    for ext in ALLOWED_CHECKPOINT_EXTS:
        if path.endswith(ext):
            return ext
    return None


def _head_status(url: str, timeout_s: int) -> Tuple[Optional[int], Optional[str]]:
    """
    Perform best-effort HEAD with a user agent.
    Returns (status_code, location_header_if_any) or (None, error_string).
    """
    if not isinstance(url, str):
        return None, "non-string url"
    if url.upper() == "MISSING":
        return None, "placeholder MISSING"
    if "://" not in url:
        return None, "unknown url type"
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "StemSepRegistryAudit/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.getheader("Location")
    except urllib.error.HTTPError as e:
        # HTTPError is also a valid response; keep code and any redirect location.
        try:
            return int(e.code), e.headers.get("Location")
        except Exception:
            return int(getattr(e, "code", 0) or 0), None
    except Exception as e:
        return None, str(e)


def _collect_link_failures(
    registry: Dict[str, Any],
    *,
    timeout_s: int,
    include_redirects_as_failures: bool,
) -> List[LinkFailure]:
    """
    Audit all models[].links.{checkpoint,config} and return failures where:
    - status is None (invalid URL or network error)
    - status >= 400
    - optionally: redirects (301/302/307/308) when include_redirects_as_failures is True
    """
    failures: List[LinkFailure] = []

    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError("Registry JSON must contain top-level 'models' list")

    for m in models:
        if not isinstance(m, dict):
            continue
        model_id = m.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        links = m.get("links") or {}
        if not isinstance(links, dict):
            links = {}

        for kind in ("checkpoint", "config"):
            url = links.get(kind)
            if not url:
                continue

            status, location = _head_status(str(url), timeout_s=timeout_s)
            is_redirect = status in (301, 302, 307, 308)

            if status is None:
                failures.append(
                    LinkFailure(
                        model_id=model_id,
                        kind=kind,
                        url=str(url),
                        status=None,
                        host=_url_host(str(url)),
                    )
                )
                continue

            if status >= 400 or (include_redirects_as_failures and is_redirect):
                # Keep the original URL; location is useful but not required in output schema.
                failures.append(
                    LinkFailure(
                        model_id=model_id,
                        kind=kind,
                        url=str(url),
                        status=int(status),
                        host=_url_host(str(url)),
                    )
                )

    return failures


def _mirror_url(owner: str, repo: str, tag: str, filename: str) -> str:
    """
    GitHub Release download URL format.
    """
    return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}"


def _build_manifest(
    failures: List[LinkFailure],
    *,
    mirror_owner: str,
    mirror_repo: str,
    tag: str,
) -> Dict[str, Any]:
    """
    Build a JSON manifest containing:
    - per-model needed artifacts (checkpoint/config)
    - inferred deterministic filenames
    - original source URLs + statuses for traceability
    """
    by_model: Dict[str, Dict[str, Any]] = {}

    for f in failures:
        entry = by_model.setdefault(
            f.model_id,
            {
                "model_id": f.model_id,
                "needs": [],  # to be filled later
                "sources": [],  # all failing sources
            },
        )
        entry["sources"].append(
            {
                "kind": f.kind,
                "url": f.url,
                "status": f.status,
                "host": f.host,
            }
        )

    # Determine per-model "needs":
    # - if any checkpoint link failed, we assume checkpoint must be mirrored
    # - if any config link failed, we assume config must be mirrored
    # For checkpoint extension, infer from the failing checkpoint URL if possible.
    models_out: List[Dict[str, Any]] = []

    for model_id, entry in sorted(by_model.items(), key=lambda kv: kv[0]):
        sources: List[Dict[str, Any]] = entry["sources"]

        checkpoint_sources = [s for s in sources if s.get("kind") == "checkpoint"]
        config_sources = [s for s in sources if s.get("kind") == "config"]

        needs: List[Dict[str, Any]] = []
        if checkpoint_sources:
            # Infer ext from the first checkpoint source URL.
            ext = _infer_checkpoint_ext_from_url(
                str(checkpoint_sources[0].get("url", ""))
            )
            if ext is None:
                ext = DEFAULT_CHECKPOINT_EXT
            filename = f"{model_id}{ext}"
            needs.append(
                {
                    "kind": "checkpoint",
                    "filename": filename,
                    "mirror_url": _mirror_url(mirror_owner, mirror_repo, tag, filename),
                }
            )

        if config_sources:
            filename = f"{model_id}.yaml"
            needs.append(
                {
                    "kind": "config",
                    "filename": filename,
                    "mirror_url": _mirror_url(mirror_owner, mirror_repo, tag, filename),
                }
            )

        # Enrich with quick summary flags
        entry_out = {
            "model_id": model_id,
            "needs": needs,
            "sources": sources,
        }
        models_out.append(entry_out)

    manifest: Dict[str, Any] = {
        "version": 1,
        "kind": "github-release-upload-manifest",
        "mirror_repo": f"{mirror_owner}/{mirror_repo}",
        "release_tag": tag,
        "filename_policy": {
            "checkpoint_filename_template": "{model_id}{ext}",
            "config_filename_template": "{model_id}.yaml",
            "allowed_checkpoint_exts": list(ALLOWED_CHECKPOINT_EXTS),
            "default_checkpoint_ext": DEFAULT_CHECKPOINT_EXT,
        },
        "models": models_out,
    }
    return manifest


def _render_markdown(manifest: Dict[str, Any]) -> str:
    """
    Render a human-friendly markdown checklist.
    """
    mirror_repo = manifest.get("mirror_repo")
    tag = manifest.get("release_tag")
    lines: List[str] = []
    lines.append("# StemSep GitHub Releases upload manifest")
    lines.append("")
    lines.append(f"- Mirror repo: `{mirror_repo}`")
    lines.append(f"- Release tag: `{tag}`")
    lines.append("")
    lines.append("## Files to upload (deterministic filenames)")
    lines.append("")
    lines.append("Upload these as **Release Assets** under the tag above.")
    lines.append("")

    models: List[Dict[str, Any]] = manifest.get("models") or []
    total_files = 0

    for m in models:
        model_id = m.get("model_id")
        needs = m.get("needs") or []
        if not needs:
            continue
        lines.append(f"### `{model_id}`")
        lines.append("")
        for n in needs:
            total_files += 1
            kind = n.get("kind")
            filename = n.get("filename")
            mirror_url = n.get("mirror_url")
            lines.append(f"- [ ] **{kind}**: `{filename}`")
            lines.append(f"  - mirror URL: {mirror_url}")
        lines.append("")
        lines.append("<details><summary>Broken source URLs (for reference)</summary>")
        lines.append("")
        lines.append("| kind | status | url |")
        lines.append("|---|---:|---|")
        for s in m.get("sources") or []:
            lines.append(f"| {s.get('kind')} | {s.get('status')} | {s.get('url')} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Models needing mirror/fix: **{len(models)}**")
    lines.append(f"- Total files to upload: **{total_files}**")
    lines.append("")
    lines.append("## After upload")
    lines.append("")
    lines.append(
        "Once assets are uploaded, update the model registry to point `links.checkpoint` / `links.config` "
        "at the mirror URLs, and ensure `artifacts.*.filename` matches these filenames."
    )
    lines.append("")

    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a GitHub Releases upload manifest for broken registry entries."
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_V2),
        help="Path to v2 registry source JSON (default: StemSepApp/assets/registry/models.v2.source.json).",
    )
    p.add_argument(
        "--mirror-owner",
        required=True,
        help="GitHub owner/org for the mirror repo (e.g. engdahlz).",
    )
    p.add_argument(
        "--mirror-repo",
        required=True,
        help="GitHub repo name for the mirror repo (e.g. stemsep-models).",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="GitHub release tag to upload assets to (e.g. models-v2-2026-01-01).",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=20,
        help="HEAD request timeout in seconds (default: 20).",
    )
    p.add_argument(
        "--include-redirects-as-failures",
        action="store_true",
        help="Treat 3xx redirect responses as failures (default: false).",
    )
    p.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_JSON),
        help="Output path for JSON manifest (default: reports/mirror_upload_manifest.json).",
    )
    p.add_argument(
        "--out-md",
        default=str(DEFAULT_OUT_MD),
        help="Output path for Markdown checklist (default: reports/mirror_upload_manifest.md).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}", file=sys.stderr)
        return 2

    try:
        registry = _read_json(registry_path)
    except Exception as e:
        print(f"Failed to parse registry JSON: {registry_path}: {e}", file=sys.stderr)
        return 2

    try:
        failures = _collect_link_failures(
            registry,
            timeout_s=int(args.timeout_s),
            include_redirects_as_failures=bool(args.include_redirects_as_failures),
        )
    except Exception as e:
        print(f"Failed to audit registry links: {e}", file=sys.stderr)
        return 2

    manifest = _build_manifest(
        failures,
        mirror_owner=str(args.mirror_owner),
        mirror_repo=str(args.mirror_repo),
        tag=str(args.tag),
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    _write_json(out_json, manifest)
    _write_text(out_md, _render_markdown(manifest))

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"Broken link entries: {len(failures)}")
    print(f"Models needing mirror/fix: {len(manifest.get('models') or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
