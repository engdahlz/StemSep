#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from catalog_v3_common import (
    DEFAULT_CATALOG_V3_SOURCE,
    DEFAULT_EXTERNAL_MARKDOWN,
    DEFAULT_LINK_REPORT,
    DEFAULT_PRESETS,
    DEFAULT_RECIPES,
    DEFAULT_V2_SOURCE,
    ARTIFACT_EXTENSIONS,
    collect_model_source_urls,
    build_verification_from_report,
    classify_external_record,
    coerce_dict,
    coerce_list,
    extract_urls,
    infer_catalog_tier,
    infer_install_policy,
    infer_model_family,
    infer_quality_role,
    infer_runtime_adapter,
    infer_source_kind_from_urls,
    is_direct_artifact_url,
    link_report_map,
    load_json,
    now_rfc3339,
    parse_uvr_markdown,
    selection_envelope,
    slugify,
    url_basename,
    write_json,
)


def _artifact_sources_from_urls(urls: list[str], report_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for priority, url in enumerate(urls):
        sources.append(
            {
                "url": url,
                "host": report_map.get(url, {}).get("domain") or "",
                "channel": "mirror" if "engdahlz/stemsep-models" in url.lower() else "upstream",
                "priority": priority,
                "auth": "public",
                "verified": bool(report_map.get(url, {}).get("ok")),
            }
        )
    return sources


def _build_model_artifacts(
    model: dict[str, Any],
    report_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    links = coerce_dict(model.get("links"))
    legacy_artifacts = coerce_dict(model.get("artifacts"))
    family = infer_model_family(model)
    model_id = str(model.get("id") or "").strip()
    artifacts: list[dict[str, Any]] = []

    def push(kind: str, artifact_obj: dict[str, Any] | None, fallback_url: str | None, required: bool = True) -> None:
        obj = artifact_obj or {}
        filename = str(obj.get("filename") or url_basename(fallback_url or "") or "").strip()
        if not filename:
            return
        relative_path = str(obj.get("relative_path") or f"{family}/{model_id}/{filename}").replace("\\", "/")
        urls: list[str] = []
        if fallback_url:
            urls.append(fallback_url)
        for item in coerce_list(obj.get("sources")):
            if isinstance(item, dict) and str(item.get("url") or "").strip():
                urls.append(str(item["url"]).strip())
        urls = list(dict.fromkeys(urls))
        artifacts.append(
            {
                "kind": kind,
                "filename": filename,
                "required": required,
                "manual": infer_install_policy(model) != "direct" and not any(is_direct_artifact_url(url) for url in urls),
                "relative_path": relative_path,
                "canonical_path": relative_path,
                "sha256": obj.get("sha256"),
                "size_bytes": obj.get("size_bytes"),
                "verified": all(bool(report_map.get(url, {}).get("ok")) for url in urls) if urls else False,
                "sources": _artifact_sources_from_urls(urls, report_map),
            }
        )

    primary_obj = legacy_artifacts.get("primary") if isinstance(legacy_artifacts.get("primary"), dict) else None
    config_obj = legacy_artifacts.get("config") if isinstance(legacy_artifacts.get("config"), dict) else None
    additional = [item for item in coerce_list(legacy_artifacts.get("additional")) if isinstance(item, dict)]

    push("checkpoint", primary_obj, links.get("checkpoint"))
    push("config", config_obj, links.get("config"), required=False if config_obj is None and not links.get("config") else True)
    for idx, item in enumerate(additional):
        item_kind = str(item.get("kind") or f"aux_{idx + 1}")
        fallback = None
        if item_kind == "config":
            fallback = links.get("config")
        elif item_kind in {"checkpoint", "weights", "model"}:
            fallback = links.get("checkpoint")
        push(item_kind, item, fallback, required=bool(item.get("required", False)))

    if not artifacts and isinstance(links.get("checkpoint"), str):
        push("checkpoint", None, str(links["checkpoint"]))

    download_manifest = {
        "mode": infer_install_policy(model),
        "strategy": "direct" if infer_install_policy(model) == "direct" else "manual",
        "artifacts": artifacts,
        "sources": [
            {
                "role": key,
                "url": str(links.get(key)).strip(),
                "manual": infer_install_policy(model) != "direct",
            }
            for key in ("checkpoint", "config", "homepage")
            if isinstance(links.get(key), str) and str(links.get(key)).strip()
        ],
        "manual_instructions": coerce_list(coerce_dict(model.get("install")).get("notes")),
    }
    return artifacts, download_manifest


def _build_model_entry(model: dict[str, Any], report_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model_id = str(model.get("id") or "").strip()
    if not model_id:
        raise ValueError("model is missing id")
    links = coerce_dict(model.get("links"))
    checkpoint_urls = [
        str(links.get(key)).strip()
        for key in ("checkpoint", "config", "homepage")
        if isinstance(links.get(key), str) and str(links.get(key)).strip()
    ]
    source_urls = collect_model_source_urls(model)
    artifacts, download_manifest = _build_model_artifacts(model, report_map)
    catalog_tier = infer_catalog_tier(model)
    install_policy = infer_install_policy(model)
    source_kind = infer_source_kind_from_urls(source_urls)
    verification = build_verification_from_report(
        source_urls,
        report_map,
        source="models.v2.source.json",
        notes=[
            f"Install policy inferred as {install_policy}.",
            f"Runtime adapter inferred as {infer_runtime_adapter(model) or 'unknown'}.",
            f"Source URLs checked: {len(source_urls)}.",
        ],
    )
    verification["artifact_count"] = len(artifacts)
    verification["direct_artifact_count"] = sum(
        1 for artifact in artifacts if any(src.get("url") and is_direct_artifact_url(str(src["url"])) for src in artifact.get("sources", []))
    )

    entry = dict(model)
    entry["model_id"] = model_id
    entry["selection_type"] = "model"
    entry["selection_id"] = model_id
    entry["catalog_tier"] = catalog_tier
    entry["source_kind"] = source_kind
    entry["quality_role"] = infer_quality_role(model)
    entry["runtime_adapter"] = infer_runtime_adapter(model)
    entry["install_policy"] = install_policy
    entry["provenance"] = {
        "source_file": "models.v2.source.json",
        "guide_revision": model.get("guide_revision"),
        "legacy_model_id": model_id,
        "primary_links": checkpoint_urls,
    }
    entry["verification"] = verification
    entry["catalog"] = {
        "tier": catalog_tier,
        "sourceKind": source_kind,
        "installPolicy": install_policy,
        "verification": verification,
    }
    entry["selection_envelope"] = selection_envelope(
        selection_type="model",
        selection_id=model_id,
        catalog_tier=catalog_tier,
        source_kind=source_kind,
        install_policy=install_policy,
        verification=verification,
    )
    entry["legacy_artifacts"] = entry.get("artifacts")
    entry["artifacts"] = artifacts
    entry["download"] = download_manifest
    return entry


def _required_model_ids_from_workflow(item: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for step in coerce_list(item.get("steps")):
        if not isinstance(step, dict):
            continue
        for key in ("model_id", "source_model"):
            value = step.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in ids:
                ids.append(value.strip())
    return ids


def _build_workflow_entry(recipe: dict[str, Any], model_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    recipe_id = str(recipe.get("id") or "").strip()
    required_model_ids = _required_model_ids_from_workflow(recipe)
    all_verified = all(
        model_index.get(model_id, {}).get("catalog_tier") == "verified"
        and model_index.get(model_id, {}).get("install_policy") == "direct"
        for model_id in required_model_ids
    )
    catalog_tier = "verified" if all_verified and recipe.get("qa_status") == "verified" else "advanced_manual"
    install_policy = (
        "direct"
        if required_model_ids
        and all(model_index.get(model_id, {}).get("install_policy") == "direct" for model_id in required_model_ids)
        else "manual"
    )
    verification = {
        "source": "recipes.json",
        "checked_at": now_rfc3339(),
        "reachable": True,
        "notes": [
            f"Workflow depends on {len(required_model_ids)} model(s).",
            f"QA status: {recipe.get('qa_status') or 'unknown'}.",
        ],
    }
    entry = dict(recipe)
    entry["selection_type"] = "workflow"
    entry["selection_id"] = recipe_id
    entry["catalog_tier"] = catalog_tier
    entry["source_kind"] = "guide"
    entry["install_policy"] = install_policy
    entry["runtime_adapter"] = "workflow"
    entry["required_model_ids"] = required_model_ids
    entry["verification"] = verification
    entry["selection_envelope"] = selection_envelope(
        selection_type="workflow",
        selection_id=recipe_id,
        catalog_tier=catalog_tier,
        source_kind="guide",
        install_policy=install_policy,
        verification=verification,
    )
    return entry


def _build_recipe_entry(preset: dict[str, Any], model_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    preset_id = str(preset.get("id") or "").strip()
    required_model_ids: list[str] = []
    model_id = preset.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        required_model_ids.append(model_id.strip())
    for item in coerce_list(preset.get("models")):
        if isinstance(item, dict):
            value = item.get("model_id")
            if isinstance(value, str) and value.strip() and value.strip() not in required_model_ids:
                required_model_ids.append(value.strip())

    direct_ready = required_model_ids and all(
        model_index.get(model_id, {}).get("install_policy") == "direct" for model_id in required_model_ids
    )
    verification = {
        "source": "presets.json",
        "checked_at": now_rfc3339(),
        "reachable": True,
        "notes": [f"Preset depends on {len(required_model_ids)} model(s)."],
    }
    entry = dict(preset)
    entry["selection_type"] = "recipe"
    entry["selection_id"] = preset_id
    entry["catalog_tier"] = "verified" if direct_ready else "advanced_manual"
    entry["source_kind"] = "guide"
    entry["install_policy"] = "direct" if direct_ready else "manual"
    entry["runtime_adapter"] = "preset"
    entry["required_model_ids"] = required_model_ids
    entry["verification"] = verification
    entry["selection_envelope"] = selection_envelope(
        selection_type="recipe",
        selection_id=preset_id,
        catalog_tier=entry["catalog_tier"],
        source_kind="guide",
        install_policy=entry["install_policy"],
        verification=verification,
    )
    return entry


def _build_external_record(
    record: dict[str, Any],
    report_map: dict[str, dict[str, Any]],
    model_url_index: dict[str, list[str]],
) -> dict[str, Any]:
    source_kind = infer_source_kind_from_urls(record["urls"])
    record_type, install_policy = classify_external_record(record)
    verification = build_verification_from_report(
        record["urls"],
        report_map,
        source="uvr_modelllista_ralankar.md",
        notes=[record.get("note") or "No extra note."],
    )
    linked_model_ids = []
    for url in record["urls"]:
        linked_model_ids.extend(model_url_index.get(url, []))
    linked_model_ids = list(dict.fromkeys(linked_model_ids))
    return {
        "record_id": f"external-{slugify(record['section'])}-{slugify(record['name'])}",
        "section": record["section"],
        "section_order": record["section_order"],
        "line_order": record["line_order"],
        "name": record["name"],
        "raw": record["raw"],
        "note": record.get("note") or None,
        "record_type": record_type,
        "catalog_tier": "advanced_manual",
        "source_kind": source_kind,
        "install_policy": install_policy,
        "urls": record["urls"],
        "verification": verification,
        "linked_model_ids": linked_model_ids,
    }


def bootstrap_catalog_v3(
    *,
    v2_source_path: Path,
    recipes_path: Path,
    presets_path: Path,
    external_markdown_path: Path,
    link_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    v2_source = load_json(v2_source_path)
    recipes_payload = load_json(recipes_path)
    presets_payload = load_json(presets_path)
    markdown_text = external_markdown_path.read_text(encoding="utf-8")
    link_report = load_json(link_report_path) if link_report_path.exists() else []
    report_map = link_report_map(link_report)

    models = [_build_model_entry(model, report_map) for model in coerce_list(v2_source.get("models")) if isinstance(model, dict)]
    model_index = {str(model["model_id"]): model for model in models}
    model_url_index: dict[str, list[str]] = {}
    for model in models:
        for url in collect_model_source_urls(model):
            model_url_index.setdefault(url, []).append(str(model["model_id"]))

    workflows = [
        _build_workflow_entry(recipe, model_index)
        for recipe in coerce_list(recipes_payload.get("recipes"))
        if isinstance(recipe, dict)
    ]

    recipe_entries = []
    for group_key in ("presets", "ensembles"):
        for preset in coerce_list(presets_payload.get(group_key)):
            if isinstance(preset, dict):
                recipe_entries.append(_build_recipe_entry(preset, model_index))

    external_records = [
        _build_external_record(record, report_map, model_url_index)
        for record in parse_uvr_markdown(markdown_text)
    ]

    catalog = {
        "version": "3",
        "generated_at": now_rfc3339(),
        "source": {
            "bootstrap": True,
            "origin_files": {
                "models_v2_source": str(v2_source_path),
                "recipes": str(recipes_path),
                "presets": str(presets_path),
                "external_markdown": str(external_markdown_path),
                "link_report": str(link_report_path) if link_report_path.exists() else None,
            },
        },
        "metadata": {
            "catalog_tiers": ["verified", "advanced_manual"],
            "record_types": ["artifact", "repo", "access_page", "unverified_note"],
            "artifact_extensions": sorted(ARTIFACT_EXTENSIONS),
        },
        "models": models,
        "recipes": recipe_entries,
        "workflows": workflows,
        "external_records": external_records,
        "summary": {
            "models": len(models),
            "recipes": len(recipe_entries),
            "workflows": len(workflows),
            "external_records": len(external_records),
            "verified_models": sum(1 for model in models if model.get("catalog_tier") == "verified"),
            "advanced_manual_models": sum(1 for model in models if model.get("catalog_tier") != "verified"),
        },
    }

    write_json(output_path, catalog)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the v3 model catalog source from current registries and external research.")
    parser.add_argument("--v2-source", default=str(DEFAULT_V2_SOURCE))
    parser.add_argument("--recipes", default=str(DEFAULT_RECIPES))
    parser.add_argument("--presets", default=str(DEFAULT_PRESETS))
    parser.add_argument("--external-markdown", default=str(DEFAULT_EXTERNAL_MARKDOWN))
    parser.add_argument("--link-report", default=str(DEFAULT_LINK_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_CATALOG_V3_SOURCE))
    args = parser.parse_args()

    catalog = bootstrap_catalog_v3(
        v2_source_path=Path(args.v2_source),
        recipes_path=Path(args.recipes),
        presets_path=Path(args.presets),
        external_markdown_path=Path(args.external_markdown),
        link_report_path=Path(args.link_report),
        output_path=Path(args.output),
    )
    print(
        f"Bootstrapped catalog v3 source with {catalog['summary']['models']} models, "
        f"{catalog['summary']['workflows']} workflows, {catalog['summary']['recipes']} recipes, "
        f"and {catalog['summary']['external_records']} external records -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
