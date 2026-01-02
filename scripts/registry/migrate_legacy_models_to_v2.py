#!/usr/bin/env python3
"""
Migrate StemSep legacy model registry -> v2 source-of-truth registry.

Purpose
- Reads the legacy registry JSON (typically: StemSepApp/assets/models.json.bak in older format)
- Converts each legacy model entry into the v2 schema shape
- Writes StemSepApp/assets/registry/models.v2.source.json

Design goals
- Be deterministic and idempotent
- Never "guess" semantics that aren't present in legacy
- Apply strict-spec safe defaults:
  - BS-Roformer + Mel-Roformer => runtime.allowed=["msst"] (Roformer are MSST-only)
  - Everything else => conservative placeholder runtime.allowed=["stemsep-legacy"]
    (you will refine via deton24 guide MUST rules later)

Usage (from repo root):
  python scripts/registry/migrate_legacy_models_to_v2.py \
    --legacy StemSepApp/assets/models.json.bak \
    --out StemSepApp/assets/registry/models.v2.source.json \
    --pretty

Notes
- This script intentionally does NOT call external network resources.
- homepage/config presence is preserved only if present in legacy; homepage defaults to null.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_ARCHITECTURES = {
    "BS-Roformer",
    "Mel-Roformer",
    "MDX23C",
    "MDX-Net",
    "VR",
    "Demucs",
    "HTDemucs",
    "SCNet",
    "Apollo",
    "BandIt",
    "Other",
}

RUNTIME_IDS = {"audio-separator", "stemsep-legacy", "msst"}


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any, pretty: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if pretty:
        txt = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    else:
        txt = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def _strip_qs_frag(url: str) -> str:
    # Remove querystring and fragment for filename derivation
    return re.split(r"[?#]", url, maxsplit=1)[0]


def _basename_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = _strip_qs_frag(url).rstrip("/")
    base = u.split("/")[-1]
    return base or None


def _infer_runtime(architecture: Optional[str]) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (runtime_obj, is_roformer).
    """
    if architecture in ("BS-Roformer", "Mel-Roformer"):
        return (
            {
                "allowed": ["msst"],
                "preferred": "msst",
                "blocking_reason": "Roformer models are MSST-only in strict-spec mode.",
            },
            True,
        )
    # Conservative placeholder; refine later via guide
    return (
        {
            "allowed": ["stemsep-legacy"],
            "preferred": "stemsep-legacy",
            "blocking_reason": None,
        },
        False,
    )


def _to_number_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    # Some registries store numbers as strings
    try:
        return float(v)
    except Exception:
        return None


def _canonical_arch(arch: Optional[str]) -> str:
    if not arch:
        return "Other"
    if arch in ALLOWED_ARCHITECTURES:
        return arch
    return "Other"


def _legacy_models_from_root(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("Legacy registry JSON root must be an object")
    models = obj.get("models")
    if not isinstance(models, list):
        raise ValueError("Legacy registry JSON must contain a 'models' array")
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            raise ValueError(f"Legacy model at index {i} is not an object")
        out.append(m)
    return out


def _validate_legacy_minimum(m: Dict[str, Any]) -> None:
    mid = m.get("id")
    if not isinstance(mid, str) or not mid.strip():
        raise ValueError("Legacy model missing non-empty string 'id'")
    links = m.get("links")
    if links is not None and not isinstance(links, dict):
        raise ValueError(f"Legacy model '{mid}' has non-object 'links'")


def _migrate_one_model(m: Dict[str, Any]) -> Dict[str, Any]:
    _validate_legacy_minimum(m)

    mid: str = m["id"]
    name: str = (
        m.get("name")
        if isinstance(m.get("name"), str) and m.get("name").strip()
        else mid
    )
    mtype: Optional[str] = m.get("type") if isinstance(m.get("type"), str) else None
    arch = _canonical_arch(
        m.get("architecture") if isinstance(m.get("architecture"), str) else None
    )
    desc: str = m.get("description") if isinstance(m.get("description"), str) else ""

    links_in = m.get("links") if isinstance(m.get("links"), dict) else {}
    chk_raw = links_in.get("checkpoint")
    chk: Optional[str] = (
        chk_raw if isinstance(chk_raw, str) and chk_raw.strip() else None
    )
    cfg_raw = links_in.get("config")
    cfg: Optional[str] = (
        cfg_raw if isinstance(cfg_raw, str) and cfg_raw.strip() else None
    )
    homepage_raw = links_in.get("homepage")
    homepage: Optional[str] = (
        homepage_raw if isinstance(homepage_raw, str) and homepage_raw.strip() else None
    )

    runtime, is_roformer = _infer_runtime(arch)

    chk_fn = _basename_from_url(chk) if chk else None
    cfg_fn = _basename_from_url(cfg) if cfg else None

    v2m: Dict[str, Any] = {
        "id": mid,
        "name": name,
        "type": mtype,
        "architecture": arch,
        "description": desc,
        "tags": [],
        "stems": [],
        "links": {
            "checkpoint": (chk or "MISSING"),
            "config": cfg,
            "homepage": homepage,
        },
        "artifacts": {
            "primary": {
                "kind": "checkpoint",
                "filename": (chk_fn or f"{mid}.MISSING"),
                "sha256": None,
            },
            "config": (
                {"kind": "config", "filename": cfg_fn, "sha256": None}
                if cfg_fn
                else None
            ),
            "additional": [],
        },
        "metrics": {
            "sdr": _to_number_or_none(m.get("sdr")),
            "fullness": _to_number_or_none(m.get("fullness")),
            "bleedless": _to_number_or_none(m.get("bleedless")),
            "aura_stft": None,
            "aura_mrstft": None,
            "log_wmse": None,
        },
        "vram_required": _to_number_or_none(m.get("vram_required")),
        "recommended_settings": {
            # Legacy registry doesn't reliably define these; keep nulls and fill via guide or research.
            "segment_size": None,
            "batch_size": None,
            "tta": None,
            "overlap": None,
            "shifts": None,
            "bit_depth": None,
        },
        "settings_semantics": {
            "overlap": {
                "meaning_by_runtime": {
                    "audio-separator": "auto",
                    "stemsep-legacy": "auto",
                    "msst": ("divisor" if is_roformer else "auto"),
                },
                "notes": (
                    "Strict default: treat overlap for Roformer under MSST as an overlap divisor. Tune per-guide/model as needed."
                    if is_roformer
                    else "Overlap semantics not yet specified for this architecture; refine via guide."
                ),
            }
        },
        "runtime": runtime,
        "compatibility": {
            "uvr_compatible": None,
            "audio_separator_compatible": (False if is_roformer else None),
            "msst_compatible": (True if is_roformer else None),
            "known_issues": [],
        },
        "requirements": {
            "python": {"min": "3.10", "max": None},
            "torch": {"min": None, "max": None, "cuda": [], "rocm": []},
            "pip": {"packages": []},
            "files": {"expects": [], "provides": []},
            "manual_steps": (
                [
                    "Legacy registry entry is missing a valid checkpoint URL. This model is kept as a placeholder; fill in links.checkpoint (and optionally links.config/homepage) before enabling execution."
                ]
                if not chk
                else []
            ),
        },
        "phase_fix": {
            "is_valid_reference": None,
            "recommended_params": {
                "lowHz": None,
                "highHz": None,
                "highFreqWeight": None,
            },
            "recommended_usage": [],
        },
        "capabilities": {
            "stems": [],
            "roles": ["separation"],
            "supports_mono": None,
            "supports_stereo": None,
        },
        "notes": {
            "migration": {
                "from": "legacy",
                "legacy_fields_present": sorted(list(m.keys())),
            },
            "strict_spec": {
                "routing_default": (
                    "msst_only_for_roformer"
                    if is_roformer
                    else "legacy_runtime_placeholder"
                )
            },
        },
    }

    return v2m


def _dedupe_keep_first(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for m in models:
        mid = m.get("id")
        if mid in seen:
            continue
        seen.add(mid)
        out.append(m)
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Migrate legacy models registry JSON into v2 source registry JSON."
    )
    ap.add_argument(
        "--legacy",
        default="StemSepApp/assets/models.json.bak",
        help="Path to legacy registry JSON (default: StemSepApp/assets/models.json.bak)",
    )
    ap.add_argument(
        "--out",
        default="StemSepApp/assets/registry/models.v2.source.json",
        help="Output path for v2 source registry JSON (default: StemSepApp/assets/registry/models.v2.source.json)",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ap.add_argument(
        "--generated-at",
        default="1970-01-01T00:00:00Z",
        help="Value for v2.generated_at (default keeps placeholder)",
    )
    args = ap.parse_args(argv)

    legacy_obj = _read_json(args.legacy)
    legacy_models = _legacy_models_from_root(legacy_obj)

    v2_models: List[Dict[str, Any]] = []
    errors: List[str] = []

    for i, m in enumerate(legacy_models):
        try:
            v2_models.append(_migrate_one_model(m))
        except Exception as e:
            mid = m.get("id", f"<index:{i}>")
            errors.append(f"{mid}: {e}")

    if errors:
        sys.stderr.write("Migration had errors; refusing to write partial v2 output.\n")
        for err in errors[:50]:
            sys.stderr.write(f"  - {err}\n")
        if len(errors) > 50:
            sys.stderr.write(f"  ... and {len(errors) - 50} more\n")
        return 2

    v2_models = _dedupe_keep_first(v2_models)
    v2_models.sort(key=lambda x: x["id"])

    v2_root: Dict[str, Any] = {
        "version": "2.0",
        "generated_at": args.generated_at,
        "source": {
            "tool": "scripts/registry/migrate_legacy_models_to_v2.py",
            "repo": "StemSep",
            "commit": None,
            "notes": "Migrated from legacy registry. Roformer families are MSST-only by default; non-Roformer models use conservative routing placeholders to be refined via deton24 guide (MUST).",
        },
        "models": v2_models,
    }

    _write_json(args.out, v2_root, pretty=args.pretty)
    sys.stdout.write(f"Migrated {len(v2_models)} model(s) from legacy -> v2.\n")
    sys.stdout.write(f"Wrote: {args.out}\n")
    sys.stdout.write(
        f"Tip: regenerate legacy via: python scripts/generate_model_registry.py --pretty\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
