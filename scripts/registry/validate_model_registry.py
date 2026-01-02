"""
Validate StemSep model registry consistency.

This script validates:
- The aggregate registry file under StemSepApp/assets/models.json or models.json.bak (preferred)
- Or, legacy per-model JSON entries under StemSepApp/assets/models/*.json
- Architecture strings are recognized (canonical allow-list + aliases)
- Link fields (checkpoint/config) are present/valid where expected
- "Expected local filename" can be derived from links.checkpoint (URL basename)
- If a model is installed locally, the installed file matches the expected name/extension

Usage (from repo root):
  python validate_model_registry.py
  python validate_model_registry.py --models-dir "C:\\Users\\you\\.stemsep\\models"
  python validate_model_registry.py --strict
  python validate_model_registry.py --show-missing

Exit codes:
  0 = success (no errors)
  1 = errors found

Notes:
- This script intentionally does NOT download anything.
- It reads registry JSON directly from StemSepApp/assets/models.json(.bak) or
    StemSepApp/assets/models/*.json to avoid relying on runtime initialization side-effects.
- Architecture handling is intentionally tolerant: registries may use aliases such as
  "MDXNet", "MDX-Net", "mdx_net", "HTDemucs", etc. These are normalized to canonical
  strings to match runtime loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# ---- Configuration ----

# Canonical architecture names (should match what the runtime uses after normalization).
RECOGNIZED_ARCHITECTURES = {
    # Roformer family
    "Mel-Roformer",
    "BS-Roformer",
    "BS-Roformer-HyperACE",
    "Roformer",
    # MDX family
    "MDX23C",
    "MDX-Net",
    "MDX",
    # Others
    "VR",
    "SCNet",
    "Apollo",
    "BandIt",
    "HTDemucs",
    "Demucs",
    "ONNX",
    # Fallback
    "Unknown",  # allow but warn by default
}

# Architectures that are NOT distributed as checkpoint URLs in our registry.
# These are resolved by their library/runtime (e.g. Demucs via torch.hub/embedded weights),
# so links.checkpoint may legitimately be null/absent.
ARCHITECTURES_WITHOUT_CHECKPOINT_URL = {
    "HTDemucs",
    "Demucs",
}

# Common aliases encountered in community registries / UI labels.
# These will be normalized to canonical values above.
ARCHITECTURE_ALIASES = {
    # MDX-Net aliases
    "mdxnet": "MDX-Net",
    "mdx-net": "MDX-Net",
    "mdx_net": "MDX-Net",
    "mdx net": "MDX-Net",
    # MDX23C aliases
    "mdx23c": "MDX23C",
    "mdx23c-8kfft": "MDX23C",
    "mdx23c 8kfft": "MDX23C",
    # Demucs / HTDemucs aliases
    "htdemucs": "HTDemucs",
    "ht-demucs": "HTDemucs",
    "demucs": "Demucs",
    # VR aliases
    "vr": "VR",
    "vr arch": "VR",
    "vr-arch": "VR",
    # SCNet aliases
    "scnet": "SCNet",
    # Roformer aliases
    "bsroformer": "BS-Roformer",
    "bs-roformer": "BS-Roformer",
    "bs_roformer": "BS-Roformer",
    "bs-roformer-hyperace": "BS-Roformer-HyperACE",
    "bsroformerhyperace": "BS-Roformer-HyperACE",
    "bs roformer hyperace": "BS-Roformer-HyperACE",
    "melroformer": "Mel-Roformer",
    "mel-roformer": "Mel-Roformer",
    "mel_roformer": "Mel-Roformer",
    "melbandroformer": "Mel-Roformer",
    "mel-band-roformer": "Mel-Roformer",
    "mel_band_roformer": "Mel-Roformer",
    "roformer": "Roformer",
    # ONNX label
    "onnx": "ONNX",
    # Apollo / BandIt
    "apollo": "Apollo",
    "bandit": "BandIt",
    "bandit_plus": "BandIt",
    "bandit-plus": "BandIt",
    # Unknown
    "unknown": "Unknown",
}

WEIGHT_EXTENSIONS = (".ckpt", ".pth", ".pt", ".safetensors", ".onnx", ".chpt")
CONFIG_EXTENSIONS = (".yaml", ".yml")


# ---- Results model ----


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN" | "INFO"
    model_id: str
    message: str
    file: Optional[Path] = None

    def format(self) -> str:
        where = f" [{self.file.name}]" if self.file else ""
        return f"{self.level}: {self.model_id}{where}: {self.message}"


# ---- Helpers ----


def _iter_model_json_files(repo_root: Path) -> List[Path]:
    assets_dir = repo_root / "StemSepApp" / "assets"

    # Preferred: single aggregate registry file.
    aggregate_candidates = [
        assets_dir / "models.json",
        assets_dir / "models.json.bak",
    ]
    aggregate_existing = [p for p in aggregate_candidates if p.exists() and p.is_file()]
    if aggregate_existing:
        return aggregate_existing

    # Legacy: one model per JSON file.
    models_dir = assets_dir / "models"
    if models_dir.exists():
        return sorted([p for p in models_dir.glob("*.json") if p.is_file()])

    raise FileNotFoundError(
        "Registry not found. Expected either an aggregate file at "
        f"{assets_dir / 'models.json'} (or models.json.bak), or a folder at {models_dir}."
    )


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _get_nested(obj: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _is_http_url(s: Any) -> bool:
    if not isinstance(s, str) or not s:
        return False
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _url_basename(url: str) -> Optional[str]:
    if not _is_http_url(url):
        return None
    parsed = urlparse(url)
    base = Path(parsed.path).name
    return base or None


def _looks_like_weight_filename(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in WEIGHT_EXTENSIONS)


def _normalize_architecture(raw: Any) -> str:
    """
    Normalize raw architecture strings to canonical values.

    This mirrors the runtime goal: treat registry architecture as user-supplied metadata
    that may contain aliases or formatting variants.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"

    key = raw.strip().lower()
    key = key.replace("_", "-")
    key = key.replace("  ", " ")

    key_compact = key.replace("-", "").replace(" ", "")

    if key in ARCHITECTURE_ALIASES:
        return ARCHITECTURE_ALIASES[key]
    if key_compact in ARCHITECTURE_ALIASES:
        return ARCHITECTURE_ALIASES[key_compact]

    # If it's already canonical (or near-canonical), keep it
    for canon in RECOGNIZED_ARCHITECTURES:
        if canon.lower() == key:
            return canon
        if canon.lower().replace("-", "").replace(" ", "") == key_compact:
            return canon

    return raw.strip()


def _infer_expected_local_filename(entry: Dict[str, Any]) -> Optional[str]:
    """
    Expected local filename is derived from links.checkpoint basename.
    This mirrors the resolver we use in runtime to avoid a static mapping table.
    """
    ckpt = _get_nested(entry, "links", "checkpoint")
    if isinstance(ckpt, str) and ckpt:
        return _url_basename(ckpt)
    # Some registries may store direct_download_url instead of links.checkpoint
    direct = entry.get("direct_download_url") or entry.get("download_url")
    if isinstance(direct, str) and direct:
        return _url_basename(direct)
    return None


def _find_installed_files(models_dir: Path, model_id: str) -> List[Path]:
    """
    Detect any plausible installed artifact for a model_id.
    We intentionally include:
    - <model_id>.<ext>
    - Any file matching expected URL basenames is checked separately.
    """
    hits: List[Path] = []
    for ext in WEIGHT_EXTENSIONS + CONFIG_EXTENSIONS:
        p = models_dir / f"{model_id}{ext}"
        if p.exists():
            hits.append(p)
    return hits


def _detect_repo_root() -> Path:
    # This script lives in scripts/registry/, so repo root is two levels up.
    return Path(__file__).resolve().parents[2]


# ---- Validation rules ----


def validate_entry(
    entry: Dict[str, Any],
    entry_file: Path,
    models_dir: Path,
    strict: bool,
) -> List[Issue]:
    issues: List[Issue] = []

    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        issues.append(
            Issue("ERROR", "<missing-id>", "Missing/invalid 'id' field", entry_file)
        )
        return issues

    model_id = model_id.strip()

    arch_raw = entry.get("architecture", "Unknown")
    arch = _normalize_architecture(arch_raw)

    if arch == "Unknown":
        if not isinstance(arch_raw, str) or not str(arch_raw).strip():
            issues.append(
                Issue(
                    "WARN", model_id, "Missing/invalid 'architecture' field", entry_file
                )
            )
        else:
            # It was provided but normalized to Unknown (e.g. weird label)
            issues.append(
                Issue(
                    "WARN",
                    model_id,
                    f"Architecture '{arch_raw}' normalized to 'Unknown' (consider fixing registry)",
                    entry_file,
                )
            )

    if arch not in RECOGNIZED_ARCHITECTURES:
        level = "ERROR" if strict else "WARN"
        issues.append(
            Issue(
                level,
                model_id,
                f"Unrecognized architecture '{arch}' (raw: {arch_raw!r})",
                entry_file,
            )
        )

    links_ckpt = _get_nested(entry, "links", "checkpoint")
    links_cfg = _get_nested(entry, "links", "config")

    # Registry consistency: checkpoint link should generally exist,
    # except for architectures that are resolved without a checkpoint URL (e.g. HTDemucs/Demucs).
    if arch in ARCHITECTURES_WITHOUT_CHECKPOINT_URL:
        # Accept missing checkpoint/config for these; they are resolved by the runtime/library.
        pass
    else:
        if not isinstance(links_ckpt, str) or not links_ckpt.strip():
            # Some "virtual" presets or non-download models could exist; treat as warn unless strict.
            level = "ERROR" if strict else "WARN"
            issues.append(
                Issue(level, model_id, "Missing links.checkpoint", entry_file)
            )
        elif not _is_http_url(links_ckpt):
            level = "ERROR" if strict else "WARN"
            issues.append(
                Issue(
                    level,
                    model_id,
                    f"Invalid checkpoint URL: {links_ckpt!r}",
                    entry_file,
                )
            )

    # Config URL is optional, but for Roformer-family models it is typically required.
    # Expected filename derivation is only meaningful when we have a checkpoint URL,
    # and it should be skipped for architectures that don't use checkpoint URLs.
    if (
        arch not in ARCHITECTURES_WITHOUT_CHECKPOINT_URL
        and isinstance(links_ckpt, str)
        and links_ckpt.strip()
    ):
        expected_name = _infer_expected_local_filename(entry)
        if expected_name is None:
            level = "ERROR" if strict else "WARN"
            issues.append(
                Issue(
                    level,
                    model_id,
                    "Could not derive expected local filename from checkpoint URL",
                    entry_file,
                )
            )
        else:
            # Extension sanity
            if not _looks_like_weight_filename(expected_name):
                level = "ERROR" if strict else "WARN"
                issues.append(
                    Issue(
                        level,
                        model_id,
                        f"Expected filename '{expected_name}' does not look like a weight file",
                        entry_file,
                    )
                )

    # For Roformer-ish models, config is often needed (but not always). Validate URL if present.
    if links_cfg is not None:
        if not isinstance(links_cfg, str) or not links_cfg.strip():
            issues.append(
                Issue(
                    "WARN",
                    model_id,
                    "links.config present but empty/invalid",
                    entry_file,
                )
            )
        elif not _is_http_url(links_cfg):
            level = "ERROR" if strict else "WARN"
            issues.append(
                Issue(level, model_id, f"Invalid config URL: {links_cfg!r}", entry_file)
            )
        else:
            cfg_base = _url_basename(links_cfg)
            if cfg_base and not cfg_base.lower().endswith(CONFIG_EXTENSIONS):
                issues.append(
                    Issue(
                        "WARN",
                        model_id,
                        f"Config URL basename '{cfg_base}' does not end with .yaml/.yml",
                        entry_file,
                    )
                )
    else:
        # Missing config is only a warning (strict can escalate for specific arches if you want later).
        pass

    # Installed file checks:
    expected = _infer_expected_local_filename(entry)
    if expected:
        expected_path = models_dir / expected
        installed_by_id = _find_installed_files(models_dir, model_id)

        if expected_path.exists():
            # Installed file matches expected name; good.
            pass
        else:
            # If any installed-by-id exists, warn that it doesn't match expected.
            # This catches the common "downloaded but named differently" bug.
            weight_by_id = [
                p for p in installed_by_id if p.suffix.lower() in WEIGHT_EXTENSIONS
            ]
            if weight_by_id:
                issues.append(
                    Issue(
                        "WARN",
                        model_id,
                        f"Installed weight exists as {weight_by_id[0].name!r} but expected {expected!r} (URL basename). Consider renaming/copying.",
                        entry_file,
                    )
                )

    return issues


def validate_registry(
    repo_root: Path,
    models_dir: Path,
    strict: bool,
) -> Tuple[List[Issue], List[Path], int]:
    issues: List[Issue] = []
    files = _iter_model_json_files(repo_root)
    entry_count = 0

    seen_ids: Dict[str, Path] = {}
    for f in files:
        try:
            root = _read_json(f)
        except Exception as e:
            issues.append(Issue("ERROR", "<parse>", f"Failed to parse JSON: {e}", f))
            continue

        # Support both layouts:
        # 1) Aggregate file: { "models": [ {..}, {..} ] }
        # 2) Single-entry file: { "id": "...", ... }
        entries: List[Dict[str, Any]]
        if isinstance(root.get("models"), list):
            entries = [e for e in root.get("models", []) if isinstance(e, dict)]
            if not entries:
                issues.append(
                    Issue(
                        "ERROR",
                        "<parse>",
                        "No valid model entries found under root.models[]",
                        f,
                    )
                )
                continue
        else:
            entries = [root]

        for entry in entries:
            entry_count += 1
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id.strip():
                mid = model_id.strip()
                if mid in seen_ids:
                    issues.append(
                        Issue(
                            "ERROR",
                            mid,
                            f"Duplicate model id (also in {seen_ids[mid].name})",
                            f,
                        )
                    )
                else:
                    seen_ids[mid] = f

            issues.extend(
                validate_entry(entry, f, models_dir=models_dir, strict=strict)
            )

    return issues, files, entry_count


# ---- CLI ----


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate StemSep model registry consistency and expected local filenames."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path.home() / ".stemsep" / "models"),
        help="Path to local models directory (default: ~/.stemsep/models)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat more issues as errors (unrecognized architecture, invalid URLs, missing checkpoint, etc.)",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Also print INFO lines for models that are not installed locally (best-effort).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = _detect_repo_root()
    models_dir = Path(args.models_dir).expanduser().resolve()

    if not models_dir.exists():
        # Not fatal: you might be validating registry without local installs.
        print(f"WARN: models-dir does not exist: {models_dir}")
    else:
        if not models_dir.is_dir():
            print(f"ERROR: models-dir is not a directory: {models_dir}")
            return 1

    issues, files, entry_count = validate_registry(
        repo_root=repo_root, models_dir=models_dir, strict=args.strict
    )

    # Optional: print missing installs (INFO)
    if args.show_missing and models_dir.exists():
        for f in files:
            try:
                root = _read_json(f)
            except Exception:
                continue

            if isinstance(root.get("models"), list):
                entries = [e for e in root.get("models", []) if isinstance(e, dict)]
            else:
                entries = [root]

            for entry in entries:
                mid = entry.get("id")
                if not isinstance(mid, str) or not mid.strip():
                    continue
                mid = mid.strip()
                expected = _infer_expected_local_filename(entry)
                if expected and (models_dir / expected).exists():
                    continue
                # If "installed by id" exists, it's not missing; it's mismatched (already warned).
                if _find_installed_files(models_dir, mid):
                    continue
                issues.append(Issue("INFO", mid, "Not installed locally", f))

    # Output grouped by severity
    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    issues_sorted = sorted(
        issues,
        key=lambda x: (
            severity_order.get(x.level, 99),
            x.model_id,
            (x.file.name if x.file else ""),
        ),
    )

    errors = [i for i in issues_sorted if i.level == "ERROR"]
    warns = [i for i in issues_sorted if i.level == "WARN"]
    infos = [i for i in issues_sorted if i.level == "INFO"]

    for i in issues_sorted:
        print(i.format())

    print("")
    print(f"Validated {entry_count} model entries across {len(files)} registry file(s)")
    print(f"Errors: {len(errors)}  Warnings: {len(warns)}  Info: {len(infos)}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
