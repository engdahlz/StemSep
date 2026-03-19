#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
ALIASES_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "report_model_aliases.json"
OUTPUT_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "report_model_coverage.json"
DOCX_PATH = REPO_ROOT / "Modellista för ljudseparation.docx"
REPORT_MD_PATH = REPO_ROOT / "deep-research-report.md"

W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_key(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("&amp;", "and")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\((.*?)\)", r" \1 ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def infer_coverage_status(model: dict[str, Any]) -> str:
    availability = model.get("availability") if isinstance(model.get("availability"), dict) else {}
    availability_class = str(availability.get("class") or "").strip()
    if availability_class:
        return availability_class

    download = model.get("download") if isinstance(model.get("download"), dict) else {}
    if str(download.get("strategy") or "").strip():
        return str(download["strategy"])

    install = model.get("install") if isinstance(model.get("install"), dict) else {}
    runtime = model.get("runtime") if isinstance(model.get("runtime"), dict) else {}
    status = model.get("status") if isinstance(model.get("status"), dict) else {}
    if install.get("mode") == "manual" or runtime.get("requires_manual_assets"):
        return "manual_import"
    if status.get("readiness") == "blocked" or model.get("catalog_status") == "blocked":
        return "blocked_non_public"

    artifacts = download.get("artifacts") if isinstance(download.get("artifacts"), list) else []
    if artifacts:
        if any(
            isinstance(artifact, dict)
            and any(
                isinstance(source, dict) and source.get("channel") == "mirror"
                for source in artifact.get("sources") or []
            )
            for artifact in artifacts
        ):
            return "mirror_fallback"
        if any(
            isinstance(artifact, dict)
            and artifact.get("required", True)
            and artifact.get("sources")
            for artifact in artifacts
        ):
            return "direct"

    links = model.get("links") if isinstance(model.get("links"), dict) else {}
    if links.get("checkpoint"):
        return "direct"
    if links.get("homepage"):
        return "manual_import"
    return "blocked_non_public"


def split_grouped_name(name: str) -> list[str]:
    compact = re.sub(r"\s+", " ", name).strip()
    if not compact:
        return []
    if " & " in compact and re.search(r"\bv\d+\b", compact, flags=re.IGNORECASE):
        base, variants = compact.split("(", 1)[0].strip(), compact
        creator = ""
        creator_match = re.search(r"\(([^)]+)\)\s*$", compact)
        if creator_match:
            creator = f" ({creator_match.group(1).strip()})"
        head = re.sub(r"\bv\d+\b.*$", "", base, flags=re.IGNORECASE).strip(" -")
        versions = re.findall(r"\bv\d+\b", compact, flags=re.IGNORECASE)
        return [f"{head} {version}{creator}".strip() for version in versions]
    if ", " in compact:
        creator_match = re.search(r"\(([^)]+)\)\s*$", compact)
        creator = f" ({creator_match.group(1).strip()})" if creator_match else ""
        leading = re.sub(r"\([^)]*\)\s*$", "", compact).strip()
        parts = [part.strip() for part in leading.split(",") if part.strip()]
        if len(parts) > 1:
            return [f"{part}{creator}".strip() for part in parts]
    if " & " in compact:
        creator_match = re.search(r"\(([^)]+)\)\s*$", compact)
        creator = f" ({creator_match.group(1).strip()})" if creator_match else ""
        leading = re.sub(r"\([^)]*\)\s*$", "", compact).strip()
        parts = [part.strip() for part in leading.split("&") if part.strip()]
        if len(parts) > 1:
            return [f"{part}{creator}".strip() for part in parts]
    return [compact]


@dataclass
class ReportEntry:
    report: str
    source_name: str
    display_name: str
    normalized_name: str
    status_text: str
    url: str | None


def parse_docx_rows(path: Path) -> list[ReportEntry]:
    with zipfile.ZipFile(path) as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))

    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    rows: list[list[dict[str, Any]]] = []
    for tr in doc.findall(".//w:tr", W_NS):
        row: list[dict[str, Any]] = []
        for tc in tr.findall("w:tc", W_NS):
            texts: list[str] = []
            links: list[str] = []
            for element in tc.iter():
                if element.tag.endswith("}t") and element.text:
                    texts.append(element.text)
                if element.tag.endswith("}hyperlink"):
                    rel_id = element.attrib.get(f"{{{W_NS['r']}}}id")
                    if rel_id and rel_id in rel_map:
                        links.append(rel_map[rel_id])
            text = re.sub(r"\s+", " ", " ".join(texts)).strip()
            row.append({"text": text, "links": links})
        if any(cell["text"] for cell in row):
            rows.append(row)

    entries: list[ReportEntry] = []
    for row in rows[1:]:
        if len(row) < 3:
            continue
        model_name = row[0]["text"].strip()
        status_text = row[2]["text"].strip()
        url = row[2]["links"][0] if row[2]["links"] else None
        for variant in split_grouped_name(model_name):
            entries.append(
                ReportEntry(
                    report="docx",
                    source_name=model_name,
                    display_name=variant,
                    normalized_name=normalize_key(variant),
                    status_text=status_text,
                    url=url,
                )
            )
    return entries


def parse_markdown_examples(path: Path) -> list[ReportEntry]:
    text = path.read_text(encoding="utf-8")
    entries: list[ReportEntry] = []
    table_pattern = re.compile(r"^\| `([^`]+)` \| .*? \| `([^`]+)` \| .*?$", re.MULTILINE)
    for name, url in table_pattern.findall(text):
        entries.append(
            ReportEntry(
                report="deep_research_report",
                source_name=name,
                display_name=name,
                normalized_name=normalize_key(name),
                status_text="report table example",
                url=url.strip(),
            )
        )
    repo_pattern = re.compile(r"\| `([^`/]+/[^`]+)` \(repo\) \| HF repo \| `([^`]+)` \|", re.MULTILINE)
    for name, url in repo_pattern.findall(text):
        entries.append(
            ReportEntry(
                report="deep_research_report",
                source_name=name,
                display_name=name,
                normalized_name=normalize_key(name),
                status_text="report repo example",
                url=url.strip(),
            )
        )
    return entries


def build_registry_indexes(models: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        by_id[model_id] = model
        keys = {
            normalize_key(model_id),
            normalize_key(str(model.get("name") or "")),
            normalize_key(str(model.get("description") or "")),
        }
        for key in keys:
            if key:
                by_key[key].append(model)
    return by_id, by_key


def resolve_model_ids(
    entry: ReportEntry,
    aliases: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_key: dict[str, list[dict[str, Any]]],
) -> list[str]:
    override_keys = [
        f"{entry.report}:{entry.source_name}",
        f"{entry.report}:{entry.display_name}",
        entry.source_name,
        entry.display_name,
        entry.normalized_name,
    ]

    row_overrides = aliases.get("row_overrides") if isinstance(aliases.get("row_overrides"), dict) else {}
    name_aliases = aliases.get("name_aliases") if isinstance(aliases.get("name_aliases"), dict) else {}

    for key in override_keys:
        override = row_overrides.get(key) or name_aliases.get(key) or name_aliases.get(normalize_key(key))
        if override:
            if isinstance(override, str):
                return [override]
            if isinstance(override, list):
                return [model_id for model_id in override if model_id in by_id]

    direct = by_key.get(entry.normalized_name)
    if direct:
        return [model["id"] for model in direct]

    tokens = set(entry.normalized_name.split())
    if not tokens:
        return []

    candidates: list[tuple[int, str]] = []
    for key, models in by_key.items():
        key_tokens = set(key.split())
        overlap = len(tokens & key_tokens)
        if overlap < 2:
            continue
        for model in models:
            candidates.append((overlap, model["id"]))
    if not candidates:
        return []

    best = max(score for score, _ in candidates)
    model_ids = []
    seen = set()
    for score, model_id in sorted(candidates, reverse=True):
        if score != best or model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)
    return model_ids


def classify_unmatched(entry: ReportEntry) -> str:
    compact = f"{entry.status_text} {entry.url or ''}".lower()
    if "går ej att ladda ner" in compact or "ingen specifik nerladdnings-url" in compact:
        return "blocked_non_public"
    if entry.url:
        return "missing_research"
    return "blocked_non_public"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report coverage for the local model reports.")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--aliases", default=str(ALIASES_PATH))
    parser.add_argument("--docx", default=str(DOCX_PATH))
    parser.add_argument("--report-md", default=str(REPORT_MD_PATH))
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    registry = load_json(Path(args.registry))
    models = registry.get("models") if isinstance(registry.get("models"), list) else []
    aliases = load_json(Path(args.aliases)) if Path(args.aliases).exists() else {}
    by_id, by_key = build_registry_indexes(models)

    entries = parse_docx_rows(Path(args.docx)) + parse_markdown_examples(Path(args.report_md))
    coverage_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)

    for entry in entries:
        model_ids = resolve_model_ids(entry, aliases, by_id, by_key)
        if not model_ids:
            status = classify_unmatched(entry)
            counts[status] += 1
            coverage_rows.append(
                {
                    "report": entry.report,
                    "source_name": entry.source_name,
                    "display_name": entry.display_name,
                    "model_ids": [],
                    "coverage_status": status,
                    "url": entry.url,
                    "status_text": entry.status_text,
                }
            )
            continue

        for model_id in model_ids:
            model = by_id[model_id]
            status = infer_coverage_status(model)
            counts[status] += 1
            coverage_rows.append(
                {
                    "report": entry.report,
                    "source_name": entry.source_name,
                    "display_name": entry.display_name,
                    "model_ids": [model_id],
                    "coverage_status": status,
                    "availability": model.get("availability"),
                    "url": entry.url,
                    "status_text": entry.status_text,
                }
            )

    output = {
        "generated_at": now_rfc3339(),
        "registry": str(Path(args.registry)),
        "reports": [str(Path(args.docx)), str(Path(args.report_md))],
        "counts": dict(sorted(counts.items())),
        "rows": coverage_rows,
    }
    write_json(Path(args.out), output)
    print(json.dumps({"rows": len(coverage_rows), "counts": output["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
