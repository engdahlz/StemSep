from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "StemSepApp" / "assets"
REGISTRY_DIR = ASSETS_DIR / "registry"
DEFAULT_V2_SOURCE = REGISTRY_DIR / "models.v2.source.json"
DEFAULT_RECIPES = ASSETS_DIR / "recipes.json"
DEFAULT_PRESETS = ASSETS_DIR / "presets.json"
DEFAULT_CATALOG_V3_SOURCE = REGISTRY_DIR / "catalog.v3.source.json"
DEFAULT_CATALOG_RUNTIME = ASSETS_DIR / "catalog.runtime.json"
DEFAULT_LEGACY_RUNTIME = ASSETS_DIR / "models.json.bak"
DEFAULT_EXTERNAL_MARKDOWN = Path(
    r"C:\Users\engdahlz\Downloads\uvr_modelllista_ralankar.md"
)
DEFAULT_LINK_REPORT = Path(
    r"C:\Users\engdahlz\AppData\Local\Temp\stemsep_uvr_link_check.json"
)

ARTIFACT_EXTENSIONS = {
    ".onnx",
    ".ckpt",
    ".pth",
    ".pt",
    ".th",
    ".yaml",
    ".yml",
    ".json",
    ".safetensors",
    ".bin",
    ".zip",
}


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def slugify(value: str) -> str:
    text = re.sub(r"[^\w\s-]+", " ", value or "", flags=re.UNICODE)
    text = re.sub(r"[_\s]+", "-", text.strip().lower(), flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "item"


def coerce_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def deep_copy_json(value: Any) -> Any:
    return deepcopy(value)


def url_host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def url_path(url: str) -> str:
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def url_basename(url: str) -> str | None:
    path = url_path(url)
    if not path:
        return None
    name = Path(path).name
    return name or None


def artifact_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def is_direct_artifact_url(url: str) -> bool:
    basename = url_basename(url)
    if not basename:
        return False
    ext = artifact_extension(basename)
    return ext in ARTIFACT_EXTENSIONS


def infer_source_kind_from_urls(urls: Iterable[str]) -> str:
    hosts = {url_host(url) for url in urls if url}
    if any("engdahlz/stemsep-models" in url.lower() for url in urls):
        return "mirror"
    if any("huggingface.co" in host for host in hosts):
        return "vendor"
    if any("github.com" in host for host in hosts):
        return "community"
    if any("mvsep.com" in host for host in hosts):
        return "guide"
    return "guide"


def infer_catalog_tier(model: dict[str, Any]) -> str:
    status = str(model.get("catalog_status") or "").strip().lower()
    if status == "verified":
        return "verified"
    return "advanced_manual"


def infer_install_policy(model: dict[str, Any]) -> str:
    install = coerce_dict(model.get("install"))
    runtime = coerce_dict(model.get("runtime"))
    availability = coerce_dict(model.get("availability"))
    catalog_status = str(model.get("catalog_status") or "").strip().lower()

    mode = (
        str(install.get("mode") or "")
        or str(runtime.get("install_mode") or "")
        or str(availability.get("class") or "")
    ).strip().lower()

    if catalog_status == "blocked" or mode in {"blocked_non_public", "missing_research"}:
        return "unavailable"
    if mode in {"manual", "manual_import"} or runtime.get("requires_manual_assets") is True:
        return "manual"
    if mode == "custom_runtime":
        return "custom_runtime"
    if mode:
        return "direct"
    return "direct"


def infer_runtime_adapter(model: dict[str, Any]) -> str | None:
    runtime = coerce_dict(model.get("runtime"))
    preferred = str(runtime.get("preferred") or "").strip().lower()
    allowed = [str(item).strip().lower() for item in coerce_list(runtime.get("allowed")) if str(item).strip()]
    engine = str(runtime.get("engine") or "").strip().lower()
    variant = str(runtime.get("variant") or "").strip().lower()
    model_type = str(runtime.get("model_type") or "").strip().lower()
    architecture = str(model.get("architecture") or "").strip().lower()

    if engine in {"msst_builtin", "demucs_native", "custom_builtin_variant", "native_stemsep"}:
        return engine
    if variant == "demucs" or "demucs" in architecture or "demucs" in model_type:
        return "demucs_native"
    if variant == "fno":
        return "custom_builtin_variant"
    if preferred == "msst" or any(entry == "msst" for entry in allowed):
        return "msst_builtin"
    if preferred in {"demucs", "demucs_native"} or any(
        entry in {"demucs", "demucs_native"} for entry in allowed
    ):
        return "demucs_native"
    if preferred in {"stemsep", "native_stemsep"} or any(
        entry in {"stemsep", "native_stemsep"} for entry in allowed
    ):
        return "native_stemsep"
    if preferred:
        return preferred
    return None


def infer_model_family(model: dict[str, Any]) -> str:
    runtime = coerce_dict(model.get("runtime"))
    engine = str(runtime.get("engine") or "").strip().lower()
    model_type = str(runtime.get("model_type") or "").strip().lower()
    architecture = str(model.get("architecture") or "").strip().lower()
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


def infer_quality_role(model: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(model.get("name") or ""),
            str(model.get("type") or ""),
            str(model.get("description") or ""),
            " ".join(str(item) for item in coerce_list(model.get("tags"))),
        ]
    ).lower()
    if "dereverb" in text or "de-reverb" in text:
        return "dereverb"
    if "denoise" in text or "noise" in text:
        return "denoise"
    if "karaoke" in text:
        return "karaoke"
    if "crowd" in text or "bleed" in text:
        return "cleanup"
    if "drum" in text:
        return "drums"
    if "guitar" in text:
        return "guitar"
    if "bass" in text:
        return "bass"
    if "vocal" in text:
        return "vocals"
    if "inst" in text or "instrumental" in text:
        return "instrumental"
    if "4-stem" in text or "5-stem" in text or "6-stem" in text or "stem separation" in text:
        return "multistem"
    return "general_separation"


def collect_model_source_urls(model: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    links = coerce_dict(model.get("links"))
    for key in ("checkpoint", "config", "homepage"):
        value = links.get(key)
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())
    for artifact in coerce_list(model.get("artifacts")):
        if not isinstance(artifact, dict):
            continue
        for source in coerce_list(artifact.get("sources")):
            if isinstance(source, dict):
                value = str(source.get("url") or "").strip()
                if value:
                    urls.append(value)
    return list(dict.fromkeys(urls))


def link_report_map(report_payload: Any) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in coerce_list(report_payload):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        mapping[url] = item
    return mapping


def build_verification_from_report(
    urls: Iterable[str],
    report_map: dict[str, dict[str, Any]],
    *,
    source: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    url_entries: list[dict[str, Any]] = []
    reachable = True
    content_kinds: set[str] = set()
    for url in urls:
        report = report_map.get(url)
        ok = bool(report.get("ok")) if report else False
        reachable = reachable and ok
        content_type = str(report.get("content_type") or "").strip() if report else ""
        if content_type:
            content_kinds.add(content_type)
        url_entries.append(
            {
                "url": url,
                "reachable": ok,
                "status": report.get("status") if report else None,
                "content_type": content_type or None,
                "final_url": report.get("final_url") if report else None,
                "method": report.get("method") if report else None,
            }
        )
    return {
        "source": source,
        "checked_at": now_rfc3339(),
        "reachable": reachable if url_entries else False,
        "evidence": url_entries,
        "notes": notes or [],
        "content_types": sorted(content_kinds),
    }


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s|)]+", text or "")


def parse_uvr_markdown(markdown_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_section = "Uncategorized"
    current_order = 0
    section_order = 0

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            section_order += 1
            current_order = 0
            continue
        if not line.startswith("- "):
            continue
        current_order += 1
        body = line[2:].strip()
        urls = extract_urls(body)
        name = body
        note = ""
        if " — " in body:
            name, note = body.split(" — ", 1)
        else:
            name = re.sub(r"https?://\S+", "", body).strip(" |")
        name = name.strip()
        note = note.strip()
        records.append(
            {
                "section": current_section,
                "section_order": section_order,
                "line_order": current_order,
                "name": name,
                "raw": body,
                "note": note,
                "urls": urls,
            }
        )
    return records


def classify_external_record(record: dict[str, Any]) -> tuple[str, str]:
    urls = record.get("urls") or []
    note = str(record.get("note") or "").lower()
    raw = str(record.get("raw") or "").lower()
    if not urls:
        return "unverified_note", "unavailable"
    if any(is_direct_artifact_url(url) for url in urls):
        return "artifact", "direct"
    if "mvsep" in raw or "algoritm" in raw or "release" in raw or "app/tjänst" in raw:
        return "access_page", "unavailable"
    if "ingen publik checkpointlänk verifierad" in note:
        return "unverified_note", "manual"
    return "repo", "manual"


def selection_envelope(
    *,
    selection_type: str,
    selection_id: str,
    catalog_tier: str | None,
    source_kind: str | None,
    install_policy: str | None,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    envelope = {
        "selectionType": selection_type,
        "selectionId": selection_id,
        "catalogTier": catalog_tier,
        "sourceKind": source_kind,
        "installPolicy": install_policy,
        "verification": verification,
    }
    return {key: value for key, value in envelope.items() if value is not None}
