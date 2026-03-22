#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from catalog_v3_common import (
    DEFAULT_CATALOG_V3_SOURCE,
    REGISTRY_DIR,
    collect_model_source_urls,
    coerce_list,
    load_json,
    now_rfc3339,
    write_json,
)


DEFAULT_REPORT = REGISTRY_DIR / "report_catalog_v3_sources.json"


def _head_or_get(session: requests.Session, url: str) -> dict[str, Any]:
    try:
        response = session.head(url, allow_redirects=True, timeout=20)
        if response.status_code >= 400 or response.status_code == 405:
            response = session.get(url, allow_redirects=True, timeout=20, stream=True)
        return {
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
        }


def _collect_urls(catalog: dict[str, Any]) -> list[str]:
    urls: list[str] = []
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


def verify_catalog_sources(source_path: Path, report_path: Path, update_source: bool) -> dict[str, Any]:
    catalog = load_json(source_path)
    urls = _collect_urls(catalog)
    session = requests.Session()
    session.headers.update({"User-Agent": "StemSepCatalogVerifier/3"})
    report_items = [_head_or_get(session, url) for url in urls]
    summary = {
        "checked_at": now_rfc3339(),
        "total": len(report_items),
        "ok": sum(1 for item in report_items if item["ok"]),
        "failed": sum(1 for item in report_items if not item["ok"]),
    }
    payload = {
        "summary": summary,
        "items": report_items,
    }
    write_json(report_path, payload)

    if update_source:
        report_map = {item["url"]: item for item in report_items}
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
                    urls_for_item = []
                    for url in coerce_list(item.get("urls")):
                        if isinstance(url, str) and url.strip():
                            urls_for_item.append(url.strip())
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

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every URL referenced by catalog.v3.source.json.")
    parser.add_argument("--source", default=str(DEFAULT_CATALOG_V3_SOURCE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--update-source", action="store_true")
    args = parser.parse_args()

    payload = verify_catalog_sources(
        Path(args.source),
        Path(args.report),
        args.update_source,
    )
    print(
        f"Verified {payload['summary']['total']} URLs "
        f"({payload['summary']['ok']} ok, {payload['summary']['failed']} failed) -> {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
