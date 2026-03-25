#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import requests

from catalog_v3_common import (
    DEFAULT_CATALOG_FRAGMENTS_ROOT,
    DEFAULT_CATALOG_V3_SOURCE,
    REGISTRY_DIR,
    collect_model_source_urls,
    coerce_list,
    load_json,
    normalize_source_entry,
    now_rfc3339,
    write_json,
)


DEFAULT_REPORT = REGISTRY_DIR / "report_catalog_v3_sources.json"
CHUNK_SIZE = 1024 * 256


def _head_or_get(session: requests.Session, url: str, *, compute_sha256: bool) -> dict[str, Any]:
    try:
        response = session.head(url, allow_redirects=True, timeout=20)
        if response.status_code >= 400 or response.status_code == 405:
            response = session.get(url, allow_redirects=True, timeout=20, stream=compute_sha256)
        result = {
            "url": url,
            "ok": response.ok,
            "status": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "checked_at": now_rfc3339(),
            "error": None,
            "method": response.request.method,
        }
        if compute_sha256 and response.ok and response.request.method == "GET":
            digest = hashlib.sha256()
            total_bytes = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                digest.update(chunk)
            result["sha256"] = digest.hexdigest()
            result["content_length"] = str(total_bytes)
        return result
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "status": None,
            "final_url": None,
            "content_type": None,
            "content_length": None,
            "checked_at": now_rfc3339(),
            "error": str(exc),
            "method": None,
            "sha256": None,
        }


def _load_fragment_entries(root: Path, subdir: str) -> list[tuple[Path, dict[str, Any]]]:
    target = root / subdir
    if not target.exists():
        return []
    entries: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(target.rglob("*.json")):
        loaded = load_json(path)
        if isinstance(loaded, dict):
            entries.append((path, loaded))
    return entries


def _collect_inline_source_entries(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for model in coerce_list(catalog.get("models")):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id") or model.get("id") or "").strip()
        for artifact in coerce_list(model.get("artifacts")):
            if not isinstance(artifact, dict):
                continue
            filename = str(artifact.get("filename") or "").strip() or None
            for source in coerce_list(artifact.get("sources")):
                if not isinstance(source, dict):
                    continue
                normalized = normalize_source_entry(source, fallback_filename=filename)
                normalized.setdefault("model_id", model_id)
                normalized.setdefault("artifact_filename", filename)
                entries.append(normalized)
    return entries


def _collect_urls(catalog: dict[str, Any], source_entries: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for item in source_entries:
        value = str(item.get("url") or "").strip()
        if value:
            urls.append(value)
    for model in coerce_list(catalog.get("models")):
        if not isinstance(model, dict):
            continue
        urls.extend(collect_model_source_urls(model))
    for record in coerce_list(catalog.get("external_records")):
        if isinstance(record, dict):
            for url in coerce_list(record.get("urls")):
                if isinstance(url, str) and url.strip():
                    urls.append(url.strip())
    return list(dict.fromkeys(urls))


def _update_source_fragments(
    fragments_root: Path,
    fragment_entries: list[tuple[Path, dict[str, Any]]],
    report_map: dict[str, dict[str, Any]],
) -> None:
    for path, entry in fragment_entries:
        normalized = normalize_source_entry(entry)
        report = report_map.get(str(normalized.get("url") or "").strip())
        if not report:
            continue
        normalized["verified"] = bool(report.get("ok"))
        normalized["resolver_viable"] = bool(report.get("ok"))
        normalized["size_bytes"] = (
            int(report["content_length"])
            if str(report.get("content_length") or "").isdigit()
            else normalized.get("size_bytes")
        )
        normalized["sha256"] = report.get("sha256") or normalized.get("sha256")
        normalized["last_checked"] = report.get("checked_at")
        normalized["final_url"] = report.get("final_url")
        normalized["content_type"] = report.get("content_type")
        normalized["status"] = report.get("status")
        write_json(path, normalized)


def verify_catalog_sources(
    source_path: Path,
    fragments_root: Path,
    report_path: Path,
    update_source: bool,
    update_source_fragments: bool,
    compute_sha256: bool,
) -> dict[str, Any]:
    catalog = load_json(source_path)
    fragment_entries = _load_fragment_entries(fragments_root, "sources")
    source_entries = [normalize_source_entry(entry) for _, entry in fragment_entries]
    source_entries.extend(_collect_inline_source_entries(catalog))
    urls = _collect_urls(catalog, source_entries)

    session = requests.Session()
    session.headers.update({"User-Agent": "StemSepCatalogVerifier/4.1"})
    report_items = [_head_or_get(session, url, compute_sha256=compute_sha256) for url in urls]
    summary = {
        "checked_at": now_rfc3339(),
        "total": len(report_items),
        "ok": sum(1 for item in report_items if item["ok"]),
        "failed": sum(1 for item in report_items if not item["ok"]),
        "sources": len(source_entries),
        "source_fragments": len(fragment_entries),
    }
    payload = {
        "summary": summary,
        "items": report_items,
    }
    write_json(report_path, payload)

    report_map = {item["url"]: item for item in report_items}

    if update_source:
        for bucket in ("models", "recipes", "workflows", "external_records"):
            updated: list[Any] = []
            for item in coerce_list(catalog.get(bucket)):
                if not isinstance(item, dict):
                    updated.append(item)
                    continue
                verification = dict(item.get("verification") or {})
                if bucket == "models":
                    urls_for_item = collect_model_source_urls(item)
                else:
                    urls_for_item = [
                        url.strip()
                        for url in coerce_list(item.get("urls"))
                        if isinstance(url, str) and url.strip()
                    ]
                evidence = [report_map[url] for url in dict.fromkeys(urls_for_item) if url in report_map]
                verification["checked_at"] = summary["checked_at"]
                verification["reachable"] = all(entry.get("ok") for entry in evidence) if evidence else verification.get("reachable")
                verification["evidence"] = evidence
                if evidence:
                    verification["content_types"] = sorted(
                        {
                            str(entry.get("content_type") or "").strip()
                            for entry in evidence
                            if str(entry.get("content_type") or "").strip()
                        }
                    )
                item["verification"] = verification
                updated.append(item)
            catalog[bucket] = updated
        write_json(source_path, catalog)

    if update_source_fragments:
        _update_source_fragments(fragments_root, fragment_entries, report_map)

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify catalog sources, including authored source fragments.")
    parser.add_argument("--source", default=str(DEFAULT_CATALOG_V3_SOURCE))
    parser.add_argument("--fragments-root", default=str(DEFAULT_CATALOG_FRAGMENTS_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--update-source", action="store_true")
    parser.add_argument("--update-source-fragments", action="store_true")
    parser.add_argument("--compute-sha256", action="store_true")
    args = parser.parse_args()

    payload = verify_catalog_sources(
        Path(args.source),
        Path(args.fragments_root),
        Path(args.report),
        args.update_source,
        args.update_source_fragments,
        args.compute_sha256,
    )
    print(
        f"Verified {payload['summary']['total']} URLs "
        f"({payload['summary']['ok']} ok, {payload['summary']['failed']} failed) -> {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
