#!/usr/bin/env python3
"""
Generate StemSep's legacy model registry (models.json.bak) from the v2 registry source.

Why:
- We want a robust, long-term schema (v2) that can represent:
  - deterministic runtime routing (audio-separator vs msst vs legacy)
  - compatibility constraints
  - special requirements (pip deps, manual steps)
  - phase-fix reference eligibility and recommended params
  - unambiguous settings semantics
- But existing app code expects the legacy registry shape in:
  StemSepApp/assets/models.json.bak

This script:
- Reads:  StemSepApp/assets/registry/models.v2.source.json
- Optionally validates against: StemSepApp/assets/registry/models.v2.schema.json (if jsonschema installed)
- Generates: StemSepApp/assets/models.json.bak
  - preserving legacy fields so the current app continues to work
  - carrying through strict-spec fields as additional keys (additive/backwards compatible)

Usage:
  python scripts/generate_model_registry.py
  python scripts/generate_model_registry.py --in StemSepApp/assets/registry/models.v2.source.json --out StemSepApp/assets/models.json.bak
  python scripts/generate_model_registry.py --validate
  python scripts/generate_model_registry.py --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
DEFAULT_SCHEMA = (
    REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.schema.json"
)
DEFAULT_OUT = REPO_ROOT / "StemSepApp" / "assets" / "models.json.bak"


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(obj, ensure_ascii=False, indent=4, sort_keys=False)
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def _maybe_validate_with_jsonschema(data: Any, schema_path: Path) -> Tuple[bool, str]:
    """
    Best-effort validation. If jsonschema isn't installed, we return (True, 'skipped').
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, "skipped (jsonschema not installed)"

    try:
        schema = _load_json(schema_path)
    except Exception as e:
        return False, f"failed to load schema: {e}"

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, "ok"
    except Exception as e:
        return False, f"schema validation failed: {e}"


def _get_env_commit() -> Optional[str]:
    # Best-effort: allow CI to inject a commit SHA.
    return (
        os.environ.get("GITHUB_SHA")
        or os.environ.get("CI_COMMIT_SHA")
        or os.environ.get("COMMIT_SHA")
        or None
    )


def _coalesce(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _extract_metrics(
    v2_model: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Legacy registry used top-level sdr/fullness/bleedless. v2 groups under metrics.
    """
    metrics = v2_model.get("metrics")
    if isinstance(metrics, dict):
        sdr = metrics.get("sdr")
        fullness = metrics.get("fullness")
        bleedless = metrics.get("bleedless")
    else:
        sdr = v2_model.get("sdr")
        fullness = v2_model.get("fullness")
        bleedless = v2_model.get("bleedless")

    def f(x: Any) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    return f(sdr), f(fullness), f(bleedless)


def _normalize_links(v2_model: Dict[str, Any]) -> Dict[str, Any]:
    links = v2_model.get("links") or {}
    if not isinstance(links, dict):
        links = {}
    out: Dict[str, Any] = {}
    if "checkpoint" in links and links["checkpoint"]:
        out["checkpoint"] = links["checkpoint"]
    if "config" in links and links["config"]:
        out["config"] = links["config"]
    # Keep homepage if present (non-legacy but harmless)
    if "homepage" in links and links["homepage"]:
        out["homepage"] = links["homepage"]
    return out


def _legacy_model_entry(v2_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single v2 model entry into the legacy registry shape.

    Legacy fields expected by current code:
      - id, name, type, architecture, description
      - links: {checkpoint, config}
      - vram_required
      - sdr/fullness/bleedless
      - recommended_overlap
      - recommended_settings

    We also forward strict-spec fields additively:
      - runtime, compatibility, requirements, phase_fix, capabilities, artifacts, settings_semantics, tags, stems, metrics
    """
    mid = v2_model.get("id")
    name = v2_model.get("name")
    arch = v2_model.get("architecture")
    if not mid or not name or not arch:
        raise ValueError(
            f"model missing required fields: id/name/architecture (id={mid!r})"
        )

    sdr, fullness, bleedless = _extract_metrics(v2_model)
    rec = v2_model.get("recommended_settings")
    if rec is not None and not isinstance(rec, dict):
        rec = None

    # Legacy: some code expects recommended_overlap separately (but registry already has recommended_settings.overlap).
    recommended_overlap = None
    if isinstance(rec, dict) and "overlap" in rec:
        # keep overlap in recommended_settings AND mirror to recommended_overlap when it's an int
        try:
            ov = rec.get("overlap")
            if isinstance(ov, int):
                recommended_overlap = ov
        except Exception:
            pass

    out: Dict[str, Any] = {
        "id": mid,
        "name": name,
        "type": _coalesce(v2_model.get("type"), ""),
        "architecture": arch,
        "description": _coalesce(v2_model.get("description"), ""),
        "links": _normalize_links(v2_model),
        "vram_required": v2_model.get("vram_required"),
        "sdr": sdr,
        "fullness": fullness,
        "bleedless": bleedless,
    }

    if recommended_overlap is not None:
        out["recommended_overlap"] = recommended_overlap
    if isinstance(rec, dict):
        out["recommended_settings"] = rec

    # Carry forward optional “classic” fields if present in v2 to preserve behavior
    if isinstance(v2_model.get("stems"), list):
        out["stems"] = v2_model.get("stems")
    if isinstance(v2_model.get("tags"), list):
        out["tags"] = v2_model.get("tags")

    # Forward strict-spec extensions as additive keys (harmless to old code)
    for k in (
        "runtime",
        "compatibility",
        "requirements",
        "phase_fix",
        "capabilities",
        "artifacts",
        "settings_semantics",
        "metrics",
        "notes",
    ):
        if k in v2_model and v2_model.get(k) is not None:
            out[k] = v2_model.get(k)

    return out


def generate_legacy_registry(v2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce the legacy registry object that will be written to models.json.bak.

    Legacy registry top-level is: {"models": [ ... ]}
    We add a `_meta` block to preserve provenance and enable debugging.
    """
    version = v2.get("version")
    if version != "2.0":
        raise ValueError(
            f"Unsupported v2 registry version: {version!r} (expected '2.0')"
        )

    models = v2.get("models")
    if not isinstance(models, list):
        raise ValueError("v2 registry must contain 'models' as a list")

    out_models: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for m in models:
        if not isinstance(m, dict):
            raise ValueError("v2 registry models must be objects")
        entry = _legacy_model_entry(m)
        mid = entry["id"]
        if mid in seen:
            raise ValueError(f"Duplicate model id: {mid}")
        seen.add(mid)
        out_models.append(entry)

    legacy = {
        "_meta": {
            "generated_from": "models.v2.source.json",
            "generated_at": _now_rfc3339(),
            "source_generated_at": v2.get("generated_at"),
            "source": v2.get("source"),
            "commit": _get_env_commit(),
            "strict_spec": True,
            "notes": "Generated by scripts/generate_model_registry.py. Do not hand-edit models.json.bak; edit v2 source instead.",
        },
        "models": out_models,
    }
    return legacy


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate StemSepApp/assets/models.json.bak from v2 registry source."
    )
    ap.add_argument(
        "--in", dest="in_path", default=str(DEFAULT_IN), help="Path to v2 source JSON"
    )
    ap.add_argument(
        "--schema",
        dest="schema_path",
        default=str(DEFAULT_SCHEMA),
        help="Path to v2 JSON schema",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default=str(DEFAULT_OUT),
        help="Path to output legacy models.json.bak",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate v2 source against schema (best-effort)",
    )
    ap.add_argument("--pretty", action="store_true", help="Write pretty-printed JSON")
    ap.add_argument(
        "--dry-run", action="store_true", help="Do not write output; print summary only"
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    schema_path = Path(args.schema_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"ERROR: v2 source not found: {in_path}", file=sys.stderr)
        return 2

    try:
        v2 = _load_json(in_path)
    except Exception as e:
        print(f"ERROR: failed to read v2 source: {e}", file=sys.stderr)
        return 2

    if args.validate:
        ok, msg = _maybe_validate_with_jsonschema(v2, schema_path)
        if not ok:
            print(f"ERROR: v2 schema validation failed: {msg}", file=sys.stderr)
            return 2
        print(f"v2 schema validation: {msg}")

    try:
        legacy = generate_legacy_registry(v2)
    except Exception as e:
        print(f"ERROR: failed to generate legacy registry: {e}", file=sys.stderr)
        return 2

    models = legacy.get("models", [])
    print(f"Generated legacy registry with {len(models)} model(s).")

    if args.dry_run:
        return 0

    try:
        _dump_json(out_path, legacy, pretty=args.pretty)
    except Exception as e:
        print(f"ERROR: failed to write output: {e}", file=sys.stderr)
        return 2

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
