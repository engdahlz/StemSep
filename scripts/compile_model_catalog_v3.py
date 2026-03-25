#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from registry.catalog_v3_common import (
    DEFAULT_BOOTSTRAP_RUNTIME,
    DEFAULT_CATALOG_FRAGMENTS_ROOT,
    DEFAULT_CATALOG_RUNTIME,
    DEFAULT_CATALOG_V3_SOURCE,
    DEFAULT_LEGACY_RUNTIME,
    coerce_dict,
    coerce_list,
    deep_copy_json,
    load_json,
    normalize_source_entry,
    now_rfc3339,
    selection_envelope,
    write_json,
)


def _load_fragment_entries(root: Path, subdir: str) -> list[dict[str, Any]]:
    target = root / subdir
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*.json")):
        loaded = load_json(path)
        if isinstance(loaded, dict):
            entries.append(loaded)
    return entries


def _merge_entries(
    base_entries: list[dict[str, Any]],
    fragment_entries: list[dict[str, Any]],
    id_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged = [deep_copy_json(entry) for entry in base_entries if isinstance(entry, dict)]
    key_to_index: dict[str, int] = {}
    for index, entry in enumerate(merged):
        for key in id_keys:
            value = str(entry.get(key) or "").strip()
            if value:
                key_to_index[value] = index
                break

    for entry in fragment_entries:
        if not isinstance(entry, dict):
            continue
        fragment = deep_copy_json(entry)
        resolved_key = ""
        for key in id_keys:
            value = str(fragment.get(key) or "").strip()
            if value:
                resolved_key = value
                break
        if resolved_key and resolved_key in key_to_index:
            merged[key_to_index[resolved_key]] = fragment
        else:
            if resolved_key:
                key_to_index[resolved_key] = len(merged)
            merged.append(fragment)
    return merged


def _merged_source_payload(source: dict[str, Any], fragments_root: Path | None) -> dict[str, Any]:
    merged = deep_copy_json(source)
    if not fragments_root or not fragments_root.exists():
        return merged

    merged["models"] = _merge_entries(
        coerce_list(merged.get("models")),
        _load_fragment_entries(fragments_root, "models"),
        ("model_id", "id"),
    )
    merged["recipes"] = _merge_entries(
        coerce_list(merged.get("recipes")),
        _load_fragment_entries(fragments_root, "recipes"),
        ("selection_id", "id"),
    )
    merged["workflows"] = _merge_entries(
        coerce_list(merged.get("workflows")),
        _load_fragment_entries(fragments_root, "workflows"),
        ("selection_id", "id"),
    )
    merged["sources"] = _merge_entries(
        coerce_list(merged.get("sources")),
        _load_fragment_entries(fragments_root, "sources"),
        ("source_id", "id"),
    )
    merged["external_records"] = _merge_entries(
        coerce_list(merged.get("external_records")),
        _load_fragment_entries(fragments_root, "external-records"),
        ("record_id", "id"),
    )
    return merged


def _source_generated_at(source: dict[str, Any]) -> str:
    value = str(source.get("generated_at") or "").strip()
    return value or now_rfc3339()


def _availability_class(model: dict[str, Any]) -> str:
    policy = str(model.get("install_policy") or "").strip().lower()
    if policy == "manual":
        return "manual_import"
    if policy == "custom_runtime":
        return "custom_runtime"
    if policy == "unavailable":
        return "blocked_non_public"
    return "direct"


def _catalog_status(model: dict[str, Any]) -> str:
    tier = str(model.get("catalog_tier") or "").strip().lower()
    policy = str(model.get("install_policy") or "").strip().lower()
    if tier == "verified" and policy == "direct":
        return "verified"
    if policy == "unavailable":
        return "blocked"
    if policy in {"manual", "custom_runtime"}:
        return "manual_only"
    return "candidate"


def _legacy_artifacts_from_array(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    primary = None
    config = None
    additional = []
    for artifact in artifacts:
        kind = str(artifact.get("kind") or "").strip().lower()
        legacy = {
            "kind": artifact.get("kind"),
            "filename": artifact.get("filename"),
            "sha256": artifact.get("sha256"),
        }
        if kind in {"checkpoint", "weights", "model"} and primary is None:
            primary = legacy
        elif kind == "config" and config is None:
            config = legacy
        else:
            additional.append(legacy)
    return {
        "primary": primary,
        "config": config,
        "additional": additional,
    }


def _index_authored_sources(source_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in source_entries:
        if not isinstance(entry, dict):
            continue
        normalized = normalize_source_entry(entry)
        indexed[normalized["source_id"]] = normalized
    return indexed


def _materialize_artifact_sources(
    artifact: dict[str, Any],
    source_registry: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    resolved_source_ids: list[str] = []
    resolved_sources: list[dict[str, Any]] = []

    for source_id in coerce_list(artifact.get("source_ids")):
        source_key = str(source_id or "").strip()
        if not source_key:
            continue
        source_entry = source_registry.get(source_key)
        if source_entry:
            resolved_source_ids.append(source_key)
            resolved_sources.append(deep_copy_json(source_entry))

    for inline_source in coerce_list(artifact.get("sources")):
        if not isinstance(inline_source, dict):
            continue
        normalized = normalize_source_entry(
            inline_source,
            fallback_filename=str(artifact.get("filename") or "").strip() or None,
        )
        source_registry.setdefault(normalized["source_id"], normalized)
        resolved_source_ids.append(normalized["source_id"])
        resolved_sources.append(deep_copy_json(source_registry[normalized["source_id"]]))

    deduped_ids: list[str] = []
    deduped_sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_id, source in zip(resolved_source_ids, resolved_sources):
        if source_id in seen:
            continue
        seen.add(source_id)
        deduped_ids.append(source_id)
        deduped_sources.append(source)
    return deduped_ids, deduped_sources


def _compile_model(model: dict[str, Any], source_registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compiled = deep_copy_json(model)
    model_id = str(compiled.get("model_id") or compiled.get("id") or "").strip()
    compiled["id"] = model_id
    compiled["model_id"] = model_id
    compiled["selection_type"] = "model"
    compiled["selection_id"] = model_id
    compiled["catalog_status"] = _catalog_status(compiled)
    compiled["availability"] = {
        "class": _availability_class(compiled),
        "reason": coerce_dict(compiled.get("verification")).get("notes", [None])[0],
    }
    compiled["selection_envelope"] = selection_envelope(
        selection_type="model",
        selection_id=model_id,
        catalog_tier=compiled.get("catalog_tier"),
        source_kind=compiled.get("source_kind"),
        install_policy=compiled.get("install_policy"),
        verification=compiled.get("verification"),
    )
    compiled["catalog"] = {
        "tier": compiled.get("catalog_tier"),
        "sourceKind": compiled.get("source_kind"),
        "installPolicy": compiled.get("install_policy"),
        "verification": compiled.get("verification"),
    }

    artifacts = []
    for artifact in coerce_list(compiled.get("artifacts")):
        if not isinstance(artifact, dict):
            continue
        entry = deep_copy_json(artifact)
        relative_path = str(entry.get("canonical_path") or entry.get("relative_path") or "")
        entry["relative_path"] = relative_path
        entry["canonical_path"] = relative_path
        source_ids, normalized_sources = _materialize_artifact_sources(entry, source_registry)
        entry["source_ids"] = source_ids
        entry["sources"] = normalized_sources
        entry["primary_source_id"] = source_ids[0] if source_ids else None
        entry["source"] = next(
            (
                source.get("url")
                for source in normalized_sources
                if isinstance(source, dict) and source.get("url")
            ),
            None,
        )
        entry["source_host"] = next(
            (
                source.get("host")
                for source in normalized_sources
                if isinstance(source, dict) and source.get("host")
            ),
            None,
        )
        artifacts.append(entry)

    download = coerce_dict(compiled.get("download"))
    download["mode"] = compiled.get("install_policy") or download.get("mode") or "direct"
    download["strategy"] = download.get("strategy") or ("direct" if download["mode"] == "direct" else "manual")
    download["artifacts"] = artifacts
    if "manual_instructions" not in download:
        download["manual_instructions"] = coerce_list(coerce_dict(compiled.get("install")).get("notes"))
    compiled["download"] = download

    legacy_artifacts = compiled.get("legacy_artifacts")
    if not isinstance(legacy_artifacts, dict):
        legacy_artifacts = _legacy_artifacts_from_array(artifacts)
    compiled["artifacts"] = legacy_artifacts
    return compiled


def _compile_selection_entry(entry: dict[str, Any], selection_type: str) -> dict[str, Any]:
    compiled = deep_copy_json(entry)
    selection_id = str(compiled.get("selection_id") or compiled.get("id") or "").strip()
    compiled["selection_type"] = selection_type
    compiled["selection_id"] = selection_id
    compiled["selection_envelope"] = selection_envelope(
        selection_type=selection_type,
        selection_id=selection_id,
        catalog_tier=compiled.get("catalog_tier"),
        source_kind=compiled.get("source_kind"),
        install_policy=compiled.get("install_policy"),
        verification=compiled.get("verification"),
    )
    return compiled


def compile_catalog_v3(
    *,
    source_path: Path,
    fragments_root: Path | None,
    runtime_path: Path,
    bootstrap_runtime_path: Path,
    legacy_runtime_path: Path,
) -> dict[str, Any]:
    source = _merged_source_payload(load_json(source_path), fragments_root)
    compiled_at = _source_generated_at(source)
    source_registry = _index_authored_sources(
        [entry for entry in coerce_list(source.get("sources")) if isinstance(entry, dict)]
    )
    compiled_models = [
        _compile_model(model, source_registry)
        for model in coerce_list(source.get("models"))
        if isinstance(model, dict)
    ]
    compiled_recipes = [
        _compile_selection_entry(recipe, "recipe")
        for recipe in coerce_list(source.get("recipes"))
        if isinstance(recipe, dict)
    ]
    compiled_workflows = [
        _compile_selection_entry(workflow, "workflow")
        for workflow in coerce_list(source.get("workflows"))
        if isinstance(workflow, dict)
    ]

    compiled_sources = sorted(
        (deep_copy_json(entry) for entry in source_registry.values()),
        key=lambda entry: str(entry.get("source_id") or ""),
    )

    selection_index = []
    for model in compiled_models:
        selection_index.append(
            {
                "selection_type": "model",
                "selection_id": model["selection_id"],
                "name": model.get("name"),
                "catalog_tier": model.get("catalog_tier"),
                "install_policy": model.get("install_policy"),
                "runtime_adapter": model.get("runtime_adapter"),
                "required_model_ids": [model["model_id"]],
            }
        )
    for recipe in compiled_recipes:
        selection_index.append(
            {
                "selection_type": "recipe",
                "selection_id": recipe["selection_id"],
                "name": recipe.get("name"),
                "catalog_tier": recipe.get("catalog_tier"),
                "install_policy": recipe.get("install_policy"),
                "runtime_adapter": recipe.get("runtime_adapter"),
                "required_model_ids": coerce_list(recipe.get("required_model_ids")),
            }
        )
    for workflow in compiled_workflows:
        selection_index.append(
            {
                "selection_type": "workflow",
                "selection_id": workflow["selection_id"],
                "name": workflow.get("name"),
                "catalog_tier": workflow.get("catalog_tier"),
                "install_policy": workflow.get("install_policy"),
                "runtime_adapter": workflow.get("runtime_adapter"),
                "required_model_ids": coerce_list(workflow.get("required_model_ids")),
            }
        )

    runtime = {
        "version": "4.1",
        "schema_version": "catalog-runtime-v4",
        "generated_at": compiled_at,
        "source": {
            "compiled_from": str(source_path),
            "fragments_root": str(fragments_root) if fragments_root else None,
            "source_generated_at": source.get("generated_at"),
        },
        "summary": {
            "models": len(compiled_models),
            "recipes": len(compiled_recipes),
            "workflows": len(compiled_workflows),
            "sources": len(compiled_sources),
            "external_records": len(coerce_list(source.get("external_records"))),
            "verified_models": sum(1 for model in compiled_models if model.get("catalog_tier") == "verified"),
            "advanced_manual_models": sum(1 for model in compiled_models if model.get("catalog_tier") != "verified"),
        },
        "models": compiled_models,
        "recipes": compiled_recipes,
        "workflows": compiled_workflows,
        "sources": compiled_sources,
        "external_records": coerce_list(source.get("external_records")),
        "selection_index": selection_index,
    }

    legacy = {
        "version": "4.1-legacy-compat",
        "schema_version": "catalog-runtime-legacy-v4",
        "generated_at": compiled_at,
        "source": {
            "compiled_from": str(source_path),
            "compatibility_mode": "models_only",
        },
        "models": compiled_models,
    }

    write_json(runtime_path, runtime)
    write_json(bootstrap_runtime_path, runtime)
    write_json(legacy_runtime_path, legacy)
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile catalog.v3.source.json into runtime manifests.")
    parser.add_argument("--source", default=str(DEFAULT_CATALOG_V3_SOURCE))
    parser.add_argument("--fragments-root", default=str(DEFAULT_CATALOG_FRAGMENTS_ROOT))
    parser.add_argument("--runtime-out", default=str(DEFAULT_CATALOG_RUNTIME))
    parser.add_argument("--bootstrap-out", default=str(DEFAULT_BOOTSTRAP_RUNTIME))
    parser.add_argument("--legacy-out", default=str(DEFAULT_LEGACY_RUNTIME))
    args = parser.parse_args()

    runtime = compile_catalog_v3(
        source_path=Path(args.source),
        fragments_root=Path(args.fragments_root) if args.fragments_root else None,
        runtime_path=Path(args.runtime_out),
        bootstrap_runtime_path=Path(args.bootstrap_out),
        legacy_runtime_path=Path(args.legacy_out),
    )
    print(
        f"Compiled catalog runtime with {runtime['summary']['models']} models, "
        f"{runtime['summary']['workflows']} workflows, {runtime['summary']['recipes']} recipes -> {args.runtime_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
