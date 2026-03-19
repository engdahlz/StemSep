#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
OVERRIDES_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "guide_model_overrides.json"
EXTRA_MODELS_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "extra_models.json"
LINK_FIXES_PATH = REPO_ROOT / "scripts" / "registry" / "registry_link_fixes.json"


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def merge_overlays_into_source(
    source_path: Path,
    overrides_path: Path,
    extra_models_path: Path,
    link_fixes_path: Path,
) -> dict[str, Any]:
    source = load_json(source_path)
    models = ensure_list(source.get("models"))
    by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        by_id[model_id] = deepcopy(model)
        ordered_ids.append(model_id)

    overrides = ensure_dict(load_json(overrides_path).get("models")) if overrides_path.exists() else {}
    for model_id, overlay in overrides.items():
        if not isinstance(overlay, dict):
            continue
        if model_id not in by_id:
            continue
        by_id[model_id] = deep_merge(by_id[model_id], overlay)

    link_fixes = ensure_dict(load_json(link_fixes_path)) if link_fixes_path.exists() else {}
    for model_id, overlay in link_fixes.items():
        if not isinstance(overlay, dict):
            continue
        if model_id not in by_id:
            continue
        by_id[model_id] = deep_merge(by_id[model_id], overlay)

    extra_models_payload = load_json(extra_models_path) if extra_models_path.exists() else {}
    extra_models = ensure_list(extra_models_payload.get("models"))
    for model in extra_models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        if model_id in by_id:
            by_id[model_id] = deep_merge(by_id[model_id], model)
        else:
            by_id[model_id] = deepcopy(model)
            ordered_ids.append(model_id)

    source["models"] = [by_id[model_id] for model_id in ordered_ids if model_id in by_id]
    source["generated_at"] = now_rfc3339()

    source_meta = ensure_dict(source.get("source"))
    notes = str(source_meta.get("notes") or "").strip()
    overlay_note = "Runtime overlays flattened into v2 source"
    if overlay_note not in notes:
        source_meta["notes"] = f"{notes} | {overlay_note}".strip(" |")
    source["source"] = source_meta
    return source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flatten runtime registry overlays and extra models into models.v2.source.json."
    )
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--overrides", default=str(OVERRIDES_PATH))
    parser.add_argument("--extra-models", default=str(EXTRA_MODELS_PATH))
    parser.add_argument("--link-fixes", default=str(LINK_FIXES_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    merged = merge_overlays_into_source(
        source_path=source_path,
        overrides_path=Path(args.overrides),
        extra_models_path=Path(args.extra_models),
        link_fixes_path=Path(args.link_fixes),
    )

    if args.dry_run:
        before = load_json(source_path)
        before_ids = {
            str(model.get("id") or "").strip()
            for model in ensure_list(before.get("models"))
            if isinstance(model, dict)
        }
        after_ids = {
            str(model.get("id") or "").strip()
            for model in ensure_list(merged.get("models"))
            if isinstance(model, dict)
        }
        print(
            json.dumps(
                {
                    "source_models_before": len(before_ids),
                    "source_models_after": len(after_ids),
                    "new_model_ids": sorted(after_ids - before_ids),
                },
                ensure_ascii=False,
            )
        )
        return 0

    write_json(source_path, merged)
    print(
        json.dumps(
            {
                "source": str(source_path),
                "models": len(ensure_list(merged.get("models"))),
                "generated_at": merged.get("generated_at"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
