"""Repair missing/broken download links in the StemSep aggregate model registry.

Why this exists
- The registry is an aggregate JSON at StemSepApp/assets/models.json.bak containing {"models": [...]}.
- Some entries may have empty links.checkpoint/config or gated/404 links.
- We must NOT delete models; only fill/replace links when we have authoritative replacements.

Typical usage
  # Dry-run (prints what would change)
  python scripts/repair_model_registry_links.py --dry-run

  # Apply mapping (creates a timestamped backup first)
  python scripts/repair_model_registry_links.py --apply

  # Apply only when audit says current URL is failing (401/404)
  python scripts/repair_model_registry_links.py --apply --apply-broken

  # Validate after applying
  python scripts/repair_model_registry_links.py --apply --validate

Mapping format
- JSON mapping keyed by model_id.
- Each entry may include:
    {
      "links": {"checkpoint": "...", "config": "...", "repository": "..."}
    }

Safety
- Defaults to only filling missing/empty fields.
- Never removes fields/models.
- Writes a backup unless --no-backup.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

WEIGHT_EXTENSIONS = (".ckpt", ".pth", ".pt", ".safetensors", ".onnx", ".chpt")
CONFIG_EXTENSIONS = (".yaml", ".yml")


@dataclass
class Change:
    model_id: str
    field_path: str
    before: Any
    after: Any
    reason: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: Any) -> None:
    # Keep escaping similar to existing file (ensure_ascii=True) and stable ordering.
    text = json.dumps(data, indent=4, ensure_ascii=True)
    path.write_text(text + "\n", encoding="utf-8")


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_probably_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return bool(u.scheme) and bool(u.netloc)
    except Exception:
        return False


def _ext_ok(url: str, kind: str) -> bool:
    lower = url.lower()
    if kind == "checkpoint":
        return lower.endswith(WEIGHT_EXTENSIONS)
    if kind == "config":
        return lower.endswith(CONFIG_EXTENSIONS)
    return True


def _index_models(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models = registry.get("models")
    if not isinstance(models, list):
        raise ValueError("Registry JSON must contain a top-level 'models' list")
    out: Dict[str, Dict[str, Any]] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if isinstance(mid, str):
            out[mid] = m
    return out


def _load_audit_index(audit_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return index: (model_id, kind) -> audit entry."""
    if not audit_path.exists():
        return {}
    audit = _load_json(audit_path)
    results = audit.get("results") if isinstance(audit, dict) else None
    if not isinstance(results, list):
        return {}
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        model_id = r.get("model_id")
        kind = r.get("kind")
        if isinstance(model_id, str) and isinstance(kind, str):
            idx[(model_id, kind)] = r
    return idx


def _should_replace_broken(
    audit_idx: Dict[Tuple[str, str], Dict[str, Any]],
    model_id: str,
    kind: str,
    current_url: Optional[str],
) -> bool:
    if not current_url:
        return False
    entry = audit_idx.get((model_id, kind))
    if not entry:
        return False
    ok = entry.get("ok")
    status = entry.get("status")
    # Replace if audit says it failed (401/403/404/etc)
    return ok is False or (isinstance(status, int) and status >= 400)


def apply_fixes(
    registry: Dict[str, Any],
    fixes: Dict[str, Any],
    *,
    only_missing: bool,
    apply_broken: bool,
    audit_idx: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Change]:
    changes: List[Change] = []
    model_map = _index_models(registry)

    for model_id, fix in fixes.items():
        if model_id not in model_map:
            continue
        if not isinstance(fix, dict):
            continue

        model = model_map[model_id]
        links_fix = fix.get("links")
        if not isinstance(links_fix, dict):
            continue

        links = model.get("links")
        if not isinstance(links, dict):
            links = {}
            model["links"] = links

        for kind in ("checkpoint", "config", "repository"):
            if kind not in links_fix:
                continue
            new_url = links_fix.get(kind)
            if not isinstance(new_url, str) or _is_empty(new_url):
                continue

            current_url = links.get(kind)
            if isinstance(current_url, str) and current_url.strip() == new_url.strip():
                continue

            if kind in ("checkpoint", "config"):
                if not _is_probably_url(new_url) or not _ext_ok(new_url, kind):
                    # Skip unsafe mapping entries.
                    continue

            allow = False
            reason = ""
            if only_missing:
                if _is_empty(current_url):
                    allow = True
                    reason = "fill-missing"
            else:
                allow = True
                reason = "overwrite"

            if (not allow) and apply_broken:
                if isinstance(current_url, str) and _should_replace_broken(
                    audit_idx, model_id, kind, current_url
                ):
                    allow = True
                    reason = "replace-broken"

            if not allow:
                continue

            before = current_url
            links[kind] = new_url
            changes.append(
                Change(
                    model_id=model_id,
                    field_path=f"links.{kind}",
                    before=before,
                    after=new_url,
                    reason=reason,
                )
            )

    return changes


def make_backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".{ts}.bak")
    backup.write_bytes(path.read_bytes())
    return backup


def run_validator(python_exe: str, repo_root: Path) -> int:
    # Validator was moved under scripts/registry. Prefer that location, but fall back
    # to the historical repo-root path for older checkouts.
    candidates = [
        repo_root / "scripts" / "registry" / "validate_model_registry.py",
        repo_root / "validate_model_registry.py",
    ]
    validator = next((p for p in candidates if p.exists()), candidates[0])

    cmd = [python_exe, str(validator), "--strict"]
    proc = subprocess.run(cmd, cwd=str(repo_root))
    return int(proc.returncode)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # This script lives in scripts/registry/, so repo root is two levels up.
    repo_root = Path(__file__).resolve().parents[2]
    default_registry = repo_root / "StemSepApp" / "assets" / "models.json.bak"

    # These files were moved under scripts/registry/
    default_fixes = repo_root / "scripts" / "registry" / "registry_link_fixes.json"
    default_audit = (
        repo_root / "scripts" / "registry" / "link_audit_models_json_bak.json"
    )

    p = argparse.ArgumentParser(
        description="Repair missing/broken links in models.json.bak"
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=default_registry,
        help="Path to aggregate registry JSON",
    )
    p.add_argument(
        "--fixes", type=Path, default=default_fixes, help="Path to fixes mapping JSON"
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=default_audit,
        help="Path to audit JSON (optional)",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Print changes only (default)"
    )
    mode.add_argument("--apply", action="store_true", help="Write changes to disk")

    p.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only fill empty/missing link fields (default)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting non-empty links (use with caution)",
    )
    p.add_argument(
        "--apply-broken",
        action="store_true",
        help="Replace current links if audit marks them as failing",
    )

    p.add_argument(
        "--no-backup", action="store_true", help="Do not create a timestamped backup"
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_model_registry.py --strict after applying",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # This script lives in scripts/registry/, so repo root is two levels up.
    repo_root = Path(__file__).resolve().parents[2]

    registry_path: Path = args.registry
    fixes_path: Path = args.fixes

    if not args.apply and not args.dry_run:
        args.dry_run = True

    if args.overwrite:
        args.only_missing = False

    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 2
    if not fixes_path.exists():
        print(f"ERROR: fixes mapping not found: {fixes_path}", file=sys.stderr)
        return 2

    registry = _load_json(registry_path)
    fixes = _load_json(fixes_path)
    if not isinstance(fixes, dict):
        print(
            "ERROR: fixes mapping must be a JSON object keyed by model_id",
            file=sys.stderr,
        )
        return 2

    audit_idx = _load_audit_index(args.audit) if args.apply_broken else {}

    changes = apply_fixes(
        registry,
        fixes,
        only_missing=bool(args.only_missing),
        apply_broken=bool(args.apply_broken),
        audit_idx=audit_idx,
    )

    if not changes:
        print("No changes.")
        return 0

    for c in changes:
        before = c.before if c.before is not None else "<missing>"
        print(f"{c.model_id}: {c.field_path}: {before!r} -> {c.after!r} ({c.reason})")

    if args.dry_run and not args.apply:
        print(f"\nDry-run: {len(changes)} change(s) would be applied.")
        return 0

    if args.apply:
        if not args.no_backup:
            backup = make_backup(registry_path)
            print(f"Backup written: {backup}")
        _dump_json(registry_path, registry)
        print(f"Wrote: {registry_path}")

        if args.validate:
            python_exe = sys.executable
            rc = run_validator(python_exe, repo_root)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
