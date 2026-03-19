#!/usr/bin/env python3
"""
Generate StemSep's runtime registry (models.json.bak) from the v2 source.

The generated registry is the backend/runtime source of truth. It carries forward
legacy fields for compatibility, but also normalizes:
- availability.class / availability.reason
- download.strategy
- download.artifacts[].sources[]
- canonical relative paths for every artifact
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
DEFAULT_SCHEMA = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.schema.json"
DEFAULT_OUT = REPO_ROOT / "StemSepApp" / "assets" / "models.json.bak"


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _dump_json(path: Path, obj: Any, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def _maybe_validate_with_jsonschema(data: Any, schema_path: Path) -> Tuple[bool, str]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, "skipped (jsonschema not installed)"

    try:
        schema = _load_json(schema_path)
    except Exception as exc:
        return False, f"failed to load schema: {exc}"

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, "ok"
    except Exception as exc:
        return False, f"schema validation failed: {exc}"


def _get_env_commit() -> Optional[str]:
    return (
        os.environ.get("GITHUB_SHA")
        or os.environ.get("CI_COMMIT_SHA")
        or os.environ.get("COMMIT_SHA")
        or None
    )


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_metrics(
    v2_model: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    metrics = v2_model.get("metrics") if isinstance(v2_model.get("metrics"), dict) else {}

    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    return (
        _as_float(_coalesce(metrics.get("sdr"), v2_model.get("sdr"))),
        _as_float(_coalesce(metrics.get("fullness"), v2_model.get("fullness"))),
        _as_float(_coalesce(metrics.get("bleedless"), v2_model.get("bleedless"))),
    )


def _normalize_links(v2_model: Dict[str, Any]) -> Dict[str, Any]:
    links = v2_model.get("links") if isinstance(v2_model.get("links"), dict) else {}
    out: Dict[str, Any] = {}
    for key in ("checkpoint", "config", "homepage"):
        value = links.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


def _infer_model_family(v2_model: Dict[str, Any]) -> str:
    runtime = v2_model.get("runtime") if isinstance(v2_model.get("runtime"), dict) else {}
    engine = str(runtime.get("engine") or "").strip().lower()
    model_type = str(runtime.get("model_type") or "").strip().lower()
    architecture = str(v2_model.get("architecture") or "").strip().lower()
    if engine == "demucs_native" or "demucs" in architecture or "demucs" in model_type:
        return "demucs"
    if architecture == "vr" or model_type == "vr":
        return "vr"
    if "mdx" in architecture or "mdx" in model_type:
        return "mdx"
    if (
        engine in {"msst_builtin", "custom_builtin_variant"}
        or "roformer" in architecture
        or "roformer" in model_type
        or "scnet" in architecture
        or "scnet" in model_type
        or "apollo" in architecture
        or "apollo" in model_type
        or "bandit" in architecture
        or "bandit" in model_type
    ):
        return "msst"
    return "other"


def _relative_path(v2_model: Dict[str, Any], model_id: str, filename: str) -> str:
    return f"{_infer_model_family(v2_model)}/{model_id}/{filename}".replace("\\", "/")


def _dedupe_by(items: Iterable[Dict[str, Any]], key_fields: tuple[str, ...]) -> List[Dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _infer_channel(url: str) -> str:
    lower = url.lower()
    if "github.com/engdahlz/stemsep-models" in lower:
        return "mirror"
    return "upstream"


def _infer_auth(url: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return "none"


def _infer_host(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def _normalize_source_entry(
    *,
    url: str,
    priority: int,
    explicit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = explicit if isinstance(explicit, dict) else {}
    return {
        "url": url,
        "channel": str(payload.get("channel") or _infer_channel(url)),
        "priority": int(payload.get("priority") or priority),
        "auth": _infer_auth(url, payload.get("auth")),
        "verified": bool(payload.get("verified", True)),
        "host": str(payload.get("host") or _infer_host(url)),
    }


def _old_download_sources(v2_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    download = v2_model.get("download") if isinstance(v2_model.get("download"), dict) else {}
    out: List[Dict[str, Any]] = []
    for index, source in enumerate(download.get("sources") or []):
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        out.append(_normalize_source_entry(url=url, priority=index + 1, explicit=source))
    return _dedupe_by(out, ("url",))


def _build_artifact_sources(
    *,
    kind: str,
    item: Optional[Dict[str, Any]],
    links: Dict[str, Any],
    global_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    item = item if isinstance(item, dict) else {}
    sources: List[Dict[str, Any]] = []
    explicit_sources = item.get("sources") if isinstance(item.get("sources"), list) else []
    for index, source in enumerate(explicit_sources):
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        sources.append(_normalize_source_entry(url=url, priority=index + 1, explicit=source))

    legacy_source = str(item.get("source") or "").strip()
    if legacy_source:
        sources.append(
            _normalize_source_entry(
                url=legacy_source,
                priority=len(sources) + 1,
                explicit={"channel": item.get("channel"), "auth": item.get("auth"), "verified": item.get("verified", True)},
            )
        )

    if not sources:
        if kind == "checkpoint" and isinstance(links.get("checkpoint"), str):
            sources.append(_normalize_source_entry(url=links["checkpoint"], priority=1))
        elif kind == "config" and isinstance(links.get("config"), str):
            sources.append(_normalize_source_entry(url=links["config"], priority=1))
        elif len(global_sources) == 1:
            sources.append(dict(global_sources[0]))

    return _dedupe_by(sorted(sources, key=lambda entry: int(entry.get("priority", 0))), ("url",))


def _base_artifacts(v2_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts_block = v2_model.get("artifacts") if isinstance(v2_model.get("artifacts"), dict) else {}
    out: List[Dict[str, Any]] = []
    primary = artifacts_block.get("primary") if isinstance(artifacts_block.get("primary"), dict) else None
    if primary and primary.get("filename"):
        out.append(dict(primary, required=True))
    config = artifacts_block.get("config") if isinstance(artifacts_block.get("config"), dict) else None
    if config and config.get("filename"):
        out.append(dict(config, required=True))
    for item in artifacts_block.get("additional") or []:
        if isinstance(item, dict) and item.get("filename"):
            out.append(dict(item, required=item.get("required", True)))
    return out


def _normalize_artifacts(v2_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_id = str(v2_model.get("id") or "").strip()
    links = _normalize_links(v2_model)
    download = v2_model.get("download") if isinstance(v2_model.get("download"), dict) else {}
    global_sources = _old_download_sources(v2_model)
    artifact_items = download.get("artifacts") if isinstance(download.get("artifacts"), list) else None
    candidate_items = artifact_items if artifact_items else _base_artifacts(v2_model)
    runtime = v2_model.get("runtime") if isinstance(v2_model.get("runtime"), dict) else {}

    normalized: List[Dict[str, Any]] = []
    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "").strip()
        if not filename or ".MISSING" in filename:
            continue
        kind = str(item.get("kind") or "aux").strip()
        artifact_sources = _build_artifact_sources(kind=kind, item=item, links=links, global_sources=global_sources)
        relative_path = str(item.get("relative_path") or "").strip() or _relative_path(v2_model, model_id, filename)
        normalized.append(
            {
                "kind": kind,
                "filename": filename,
                "relative_path": relative_path.replace("\\", "/"),
                "required": bool(item.get("required", True)),
                "manual": bool(item.get("manual", False)),
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
                "sources": artifact_sources,
            }
        )

    required_files = runtime.get("required_files") if isinstance(runtime.get("required_files"), list) else []
    known_filenames = {artifact["filename"] for artifact in normalized}
    for filename in required_files:
        if not isinstance(filename, str) or not filename.strip():
            continue
        if filename in known_filenames:
            continue
        inferred_kind = "config" if filename.lower().endswith((".yaml", ".yml", ".json")) else "aux"
        normalized.append(
            {
                "kind": inferred_kind,
                "filename": filename.strip(),
                "relative_path": _relative_path(v2_model, model_id, filename.strip()),
                "required": True,
                "manual": True,
                "sha256": None,
                "size_bytes": None,
                "sources": [],
            }
        )

    return _dedupe_by(normalized, ("relative_path",))


def _manual_instructions(v2_model: Dict[str, Any]) -> List[str]:
    instructions: List[str] = []
    install = v2_model.get("install") if isinstance(v2_model.get("install"), dict) else {}
    requirements = v2_model.get("requirements") if isinstance(v2_model.get("requirements"), dict) else {}
    runtime = v2_model.get("runtime") if isinstance(v2_model.get("runtime"), dict) else {}
    for value in install.get("notes") or []:
        if isinstance(value, str) and value.strip():
            instructions.append(value.strip())
    for value in requirements.get("manual_steps") or []:
        if isinstance(value, str) and value.strip():
            instructions.append(value.strip())
    if runtime.get("requires_manual_assets") and not instructions:
        instructions.append("Manual model assets are required before this model can be used.")
    return list(dict.fromkeys(instructions))


def _derive_availability(v2_model: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    explicit = v2_model.get("availability") if isinstance(v2_model.get("availability"), dict) else {}
    if explicit.get("class"):
        return {
            "class": str(explicit.get("class")),
            "reason": explicit.get("reason"),
        }

    install = v2_model.get("install") if isinstance(v2_model.get("install"), dict) else {}
    runtime = v2_model.get("runtime") if isinstance(v2_model.get("runtime"), dict) else {}
    catalog_status = str(v2_model.get("catalog_status") or "").strip()
    status = v2_model.get("status") if isinstance(v2_model.get("status"), dict) else {}

    required_artifacts = [artifact for artifact in artifacts if artifact.get("required", True)]
    artifact_sources_ready = all(
        artifact.get("sources") and any(source.get("verified", True) for source in artifact.get("sources", []))
        for artifact in required_artifacts
    ) if required_artifacts else False

    manual_required = (
        install.get("mode") == "manual"
        or runtime.get("install_mode") == "manual"
        or runtime.get("requires_manual_assets")
    )
    blocked_reason = (
        explicit.get("reason")
        or status.get("blocking_reason")
        or runtime.get("blocking_reason")
        or None
    )

    if catalog_status == "blocked" or status.get("readiness") == "blocked":
        return {"class": "blocked_non_public", "reason": blocked_reason or "Registry marks the model as blocked."}
    if manual_required:
        return {"class": "manual_import", "reason": blocked_reason or "Model requires manual asset import."}
    if artifact_sources_ready:
        if any(
            any(source.get("channel") == "mirror" for source in artifact.get("sources") or [])
            for artifact in required_artifacts
        ):
            return {"class": "mirror_fallback", "reason": blocked_reason}
        return {"class": "direct", "reason": blocked_reason}
    return {"class": "blocked_non_public", "reason": blocked_reason or "Required artifacts do not expose verified direct sources."}


def _normalize_download(v2_model: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = _normalize_artifacts(v2_model)
    availability = _derive_availability(v2_model, artifacts)
    direct_count = sum(1 for artifact in artifacts if artifact.get("required", True) and artifact.get("sources"))
    flattened_sources = _dedupe_by(
        [
            {
                "role": artifact.get("kind"),
                **source,
            }
            for artifact in artifacts
            for source in artifact.get("sources") or []
        ],
        ("url",),
    )
    install = v2_model.get("install") if isinstance(v2_model.get("install"), dict) else {}
    runtime = v2_model.get("runtime") if isinstance(v2_model.get("runtime"), dict) else {}
    strategy = availability["class"]
    if strategy == "blocked_non_public" and runtime.get("engine") == "demucs_native" and direct_count > 0:
        strategy = "direct"
    mode = "unavailable"
    if strategy == "manual_import":
        mode = "manual"
    elif strategy in {"direct", "mirror_fallback"}:
        mode = "multi_artifact_direct" if direct_count > 1 else "direct"
    download = {
        "strategy": strategy,
        "mode": mode,
        "install_mode": str(install.get("mode") or runtime.get("install_mode") or "direct"),
        "family": _infer_model_family(v2_model),
        "artifact_count": len(artifacts),
        "downloadable_artifact_count": direct_count,
        "manual_instructions": _manual_instructions(v2_model),
        "sources": flattened_sources,
        "artifacts": artifacts,
    }
    return download


def _legacy_model_entry(v2_model: Dict[str, Any]) -> Dict[str, Any]:
    model_id = v2_model.get("id")
    name = v2_model.get("name")
    architecture = v2_model.get("architecture")
    if not model_id or not name or not architecture:
        raise ValueError(f"model missing required fields: id/name/architecture (id={model_id!r})")

    sdr, fullness, bleedless = _extract_metrics(v2_model)
    recommended_settings = (
        v2_model.get("recommended_settings")
        if isinstance(v2_model.get("recommended_settings"), dict)
        else None
    )
    recommended_overlap = None
    if isinstance(recommended_settings, dict) and isinstance(recommended_settings.get("overlap"), int):
        recommended_overlap = recommended_settings["overlap"]

    download = _normalize_download(v2_model)
    availability = _derive_availability(v2_model, download.get("artifacts") or [])
    links = _normalize_links(v2_model)

    out: Dict[str, Any] = {
        "id": model_id,
        "name": name,
        "type": _coalesce(v2_model.get("type"), ""),
        "architecture": architecture,
        "description": _coalesce(v2_model.get("description"), ""),
        "links": links,
        "vram_required": v2_model.get("vram_required"),
        "sdr": sdr,
        "fullness": fullness,
        "bleedless": bleedless,
        "availability": availability,
        "download": download,
    }

    if recommended_overlap is not None:
        out["recommended_overlap"] = recommended_overlap
    if isinstance(recommended_settings, dict):
        out["recommended_settings"] = recommended_settings
    if isinstance(v2_model.get("stems"), list):
        out["stems"] = v2_model.get("stems")
    if isinstance(v2_model.get("tags"), list):
        out["tags"] = v2_model.get("tags")

    for key in (
        "guide_rank",
        "guide_notes",
        "status",
        "runtime",
        "compatibility",
        "requirements",
        "phase_fix",
        "capabilities",
        "artifacts",
        "settings_semantics",
        "metrics",
        "card_metrics",
        "catalog_status",
        "metrics_status",
        "metrics_evidence",
        "notes",
        "quality_role",
        "best_for",
        "artifacts_risk",
        "vram_profile",
        "chunk_overlap_policy",
        "workflow_groups",
        "quality_axes",
        "workflow_roles",
        "operating_profiles",
        "content_fit",
        "quality_profile",
        "hardware_tiers",
        "stability_notes",
        "install",
        "guide_revision",
    ):
        if key in v2_model and v2_model.get(key) is not None:
            out[key] = v2_model.get(key)

    return out


def generate_legacy_registry(v2: Dict[str, Any]) -> Dict[str, Any]:
    version = v2.get("version")
    if version != "2.0":
        raise ValueError(f"Unsupported v2 registry version: {version!r} (expected '2.0')")

    models = v2.get("models")
    if not isinstance(models, list):
        raise ValueError("v2 registry must contain 'models' as a list")

    out_models: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("v2 registry models must be objects")
        entry = _legacy_model_entry(model)
        model_id = entry["id"]
        if model_id in seen:
            raise ValueError(f"Duplicate model id: {model_id}")
        seen.add(model_id)
        out_models.append(entry)

    return {
        "_meta": {
            "generated_from": "models.v2.source.json",
            "generated_at": _now_rfc3339(),
            "source_generated_at": v2.get("generated_at"),
            "source": v2.get("source"),
            "commit": _get_env_commit(),
            "strict_spec": True,
            "notes": "Generated by scripts/generate_model_registry.py. Edit the v2 source, not models.json.bak.",
        },
        "models": out_models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate StemSepApp/assets/models.json.bak from v2 registry source.")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Path to v2 source JSON")
    parser.add_argument("--schema", dest="schema_path", default=str(DEFAULT_SCHEMA), help="Path to v2 JSON schema")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="Path to output models.json.bak")
    parser.add_argument("--validate", action="store_true", help="Validate v2 source against schema (best-effort)")
    parser.add_argument("--pretty", action="store_true", help="Write pretty-printed JSON")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output; print summary only")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    schema_path = Path(args.schema_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"ERROR: v2 source not found: {in_path}", file=sys.stderr)
        return 2

    try:
        v2 = _load_json(in_path)
    except Exception as exc:
        print(f"ERROR: failed to read v2 source: {exc}", file=sys.stderr)
        return 2

    if args.validate:
        ok, msg = _maybe_validate_with_jsonschema(v2, schema_path)
        if not ok:
            print(f"ERROR: v2 schema validation failed: {msg}", file=sys.stderr)
            return 2
        print(f"v2 schema validation: {msg}")

    try:
        legacy = generate_legacy_registry(v2)
    except Exception as exc:
        print(f"ERROR: failed to generate legacy registry: {exc}", file=sys.stderr)
        return 2

    models = legacy.get("models", [])
    print(f"Generated runtime registry with {len(models)} model(s).")

    if args.dry_run:
        return 0

    try:
        _dump_json(out_path, legacy, pretty=args.pretty)
    except Exception as exc:
        print(f"ERROR: failed to write output: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
