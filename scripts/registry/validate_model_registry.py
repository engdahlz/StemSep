"""
Legacy audit for StemSep's historical models.json(.bak) registry.

This script validates:
- The aggregate legacy registry file under StemSepApp/assets/models.json or models.json.bak (preferred)
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
- This validator is now a non-blocking legacy audit helper. The default quality gate
  validates the remote-first v4 runtime catalog via `validate_runtime_catalog.py`.
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
from urllib.request import Request, urlopen

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
REPORT_COVERAGE_PATH = Path("StemSepApp/assets/registry/report_model_coverage.json")
ALLOWED_COVERAGE_STATUSES = {
    "direct",
    "mirror_fallback",
    "manual_import",
    "blocked_non_public",
}


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


def _download_artifacts(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    download = entry.get("download")
    if not isinstance(download, dict):
        return []
    artifacts = download.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _availability_class(entry: Dict[str, Any]) -> str:
    availability = entry.get("availability")
    if isinstance(availability, dict):
        value = availability.get("class")
        if isinstance(value, str):
            return value.strip()
    download = entry.get("download")
    if isinstance(download, dict):
        value = download.get("strategy")
        if isinstance(value, str):
            return value.strip()
    return ""


def _detect_repo_root() -> Path:
    # This script lives in scripts/registry/, so repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def _check_url_reachable(url: str, timeout_sec: int = 12) -> Tuple[bool, str]:
    try:
        req = Request(
            url,
            headers={"User-Agent": "StemSep-registry-validator/1.0"},
            method="HEAD",
        )
        with urlopen(req, timeout=timeout_sec) as resp:
            return True, str(getattr(resp, "status", 200))
    except Exception as exc:
        try:
            req = Request(
                url,
                headers={"User-Agent": "StemSep-registry-validator/1.0"},
                method="GET",
            )
            with urlopen(req, timeout=timeout_sec) as resp:
                return True, str(getattr(resp, "status", 200))
        except Exception as retry_exc:
            err = retry_exc if retry_exc is not None else exc
            return False, str(err)


# ---- Validation rules ----


def validate_entry(
    entry: Dict[str, Any],
    entry_file: Path,
    models_dir: Path,
    strict: bool,
    check_verified_urls: bool,
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
    catalog_status = entry.get("catalog_status")
    card_metrics = entry.get("card_metrics")
    availability_class = _availability_class(entry)
    artifacts = _download_artifacts(entry)

    if availability_class and availability_class not in ALLOWED_COVERAGE_STATUSES:
        issues.append(
            Issue(
                "ERROR" if strict else "WARN",
                model_id,
                f"Unknown availability class {availability_class!r}",
                entry_file,
            )
        )

    if availability_class in {"direct", "mirror_fallback"}:
        if not artifacts:
            issues.append(
                Issue(
                    "ERROR",
                    model_id,
                    "Runnable model is missing download.artifacts entries",
                    entry_file,
                )
            )
        for artifact in artifacts:
            if not artifact.get("required", True):
                continue
            source_urls: List[str] = []
            if isinstance(artifact.get("sources"), list):
                source_urls.extend(
                    [
                        str(source.get("url")).strip()
                        for source in artifact["sources"]
                        if isinstance(source, dict) and str(source.get("url") or "").strip()
                    ]
                )
            direct_source = artifact.get("source")
            if isinstance(direct_source, str) and direct_source.strip():
                source_urls.append(direct_source.strip())
            if not source_urls:
                issues.append(
                    Issue(
                        "ERROR",
                        model_id,
                        f"Required artifact {artifact.get('relative_path') or artifact.get('filename')!r} is missing sources",
                        entry_file,
                    )
                )
            if strict and not artifact.get("sha256"):
                issues.append(
                    Issue(
                        "ERROR",
                        model_id,
                        f"Required artifact {artifact.get('relative_path') or artifact.get('filename')!r} is missing sha256",
                        entry_file,
                    )
                )
    elif availability_class == "manual_import":
        download = entry.get("download")
        manual_instructions = (
            download.get("manual_instructions")
            if isinstance(download, dict)
            else None
        )
        if not isinstance(manual_instructions, list) or not manual_instructions:
            issues.append(
                Issue(
                    "ERROR" if strict else "WARN",
                    model_id,
                    "Manual-import model is missing manual_instructions",
                    entry_file,
                )
            )

    if catalog_status == "verified":
        if not isinstance(card_metrics, dict):
            issues.append(
                Issue(
                    "ERROR",
                    model_id,
                    "Verified model is missing card_metrics",
                    entry_file,
                )
            )
        else:
            labels = card_metrics.get("labels")
            values = card_metrics.get("values")
            if not isinstance(labels, list) or len(labels) != 3:
                issues.append(
                    Issue(
                        "ERROR",
                        model_id,
                        "Verified model card_metrics.labels must contain exactly 3 entries",
                        entry_file,
                    )
                )
            if not isinstance(values, list) or len(values) != 3:
                issues.append(
                    Issue(
                        "ERROR",
                        model_id,
                        "Verified model card_metrics.values must contain exactly 3 entries",
                        entry_file,
                    )
                )
            elif any(v in (None, "", "—") for v in values):
                issues.append(
                    Issue(
                        "ERROR",
                        model_id,
                        "Verified model card_metrics.values contains an empty slot",
                        entry_file,
                    )
                )

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
            elif catalog_status == "verified" and check_verified_urls:
                ok, detail = _check_url_reachable(links_ckpt)
                if not ok:
                    issues.append(
                        Issue(
                            "ERROR",
                            model_id,
                            f"Verified model checkpoint URL is not reachable: {detail}",
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
    check_verified_urls: bool,
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
                validate_entry(
                    entry,
                    f,
                    models_dir=models_dir,
                    strict=strict,
                    check_verified_urls=check_verified_urls,
                )
            )

    coverage_path = repo_root / REPORT_COVERAGE_PATH
    if coverage_path.exists():
        try:
            coverage = _read_json(coverage_path)
            rows = coverage.get("rows") if isinstance(coverage.get("rows"), list) else []
            missing_rows = [
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("coverage_status") or "").strip() == "missing_research"
            ]
            for row in missing_rows:
                issues.append(
                    Issue(
                        "ERROR",
                        "<coverage>",
                        f"Report coverage missing for {row.get('report')} / {row.get('display_name')}",
                        coverage_path,
                    )
                )
        except Exception as e:
            issues.append(
                Issue(
                    "ERROR" if strict else "WARN",
                    "<coverage>",
                    f"Failed to read report coverage file: {e}",
                    coverage_path,
                )
            )
    else:
        issues.append(
            Issue(
                "ERROR" if strict else "WARN",
                "<coverage>",
                f"Report coverage file not found: {coverage_path}",
                coverage_path,
            )
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
    parser.add_argument(
        "--check-verified-urls",
        action="store_true",
        help="Resolve verified model checkpoint URLs and fail if they are unreachable.",
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
        repo_root=repo_root,
        models_dir=models_dir,
        strict=args.strict,
        check_verified_urls=args.check_verified_urls,
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
