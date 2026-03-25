from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "StemSepApp" / "assets"
REGISTRY_DIR = ASSETS_DIR / "registry"
DEFAULT_V2_SOURCE = REGISTRY_DIR / "models.v2.source.json"
DEFAULT_RECIPES = ASSETS_DIR / "recipes.json"
DEFAULT_PRESETS = ASSETS_DIR / "presets.json"
DEFAULT_CATALOG_V3_SOURCE = REGISTRY_DIR / "catalog.v3.source.json"
DEFAULT_CATALOG_FRAGMENTS_ROOT = REGISTRY_DIR / "catalog-fragments"
DEFAULT_CATALOG_RUNTIME = ASSETS_DIR / "catalog.runtime.json"
DEFAULT_BOOTSTRAP_RUNTIME = ASSETS_DIR / "catalog.runtime.bootstrap.json"
DEFAULT_LEGACY_RUNTIME = ASSETS_DIR / "models.json.bak"
DEFAULT_REMOTE_RUNTIME = REGISTRY_DIR / "remote" / "catalog.runtime.remote.json"
DEFAULT_REMOTE_SIGNATURE = REGISTRY_DIR / "remote" / "catalog.runtime.remote.json.sig"
DEFAULT_REMOTE_PUBLIC_KEY = REGISTRY_DIR / "remote" / "catalog.public.ed25519.txt"
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


def _path_parts(url: str) -> list[str]:
    return [part for part in (urlparse(url).path or "").split("/") if part]


def github_release_locator(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    parts = _path_parts(url)
    if "github.com" not in (parsed.netloc or "").lower():
        return None
    if len(parts) < 6 or parts[2:4] != ["releases", "download"]:
        return None
    return {
        "owner": parts[0],
        "repo": parts[1],
        "tag": parts[4],
        "asset_name": "/".join(parts[5:]),
    }


def github_raw_locator(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    parts = _path_parts(url)
    host = (parsed.netloc or "").lower()
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        return {
            "owner": parts[0],
            "repo": parts[1],
            "revision": parts[2],
            "file_path": "/".join(parts[3:]),
        }
    if host == "github.com" and len(parts) >= 5 and parts[2] == "raw":
        return {
            "owner": parts[0],
            "repo": parts[1],
            "revision": parts[3],
            "file_path": "/".join(parts[4:]),
        }
    return None


def huggingface_resolve_locator(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    parts = _path_parts(url)
    if (parsed.netloc or "").lower() != "huggingface.co":
        return None
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    return {
        "repo_id": f"{parts[0]}/{parts[1]}",
        "revision": parts[3],
        "file_path": "/".join(parts[4:]),
    }


def google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host and "drive.usercontent.google.com" not in host:
        return None
    query_id = parse_qs(parsed.query or "").get("id", [None])[0]
    if query_id:
        return query_id
    parts = _path_parts(url)
    for index, part in enumerate(parts):
        if part in {"d", "folders"} and index + 1 < len(parts):
            return parts[index + 1]
    return None


def proton_share_locator(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "drive.proton.me" not in host:
        return None
    locator: dict[str, Any] = {"share_url": url}
    if parsed.fragment:
        locator["share_token"] = parsed.fragment
    return locator


def source_id_for(
    *,
    url: str | None,
    provider: str | None,
    resolver: str | None,
    locator: dict[str, Any] | None = None,
    filename: str | None = None,
    explicit_id: str | None = None,
) -> str:
    if explicit_id and explicit_id.strip():
        return slugify(explicit_id.strip())
    pieces: list[str] = []
    if provider:
        pieces.append(provider)
    if resolver and resolver not in pieces:
        pieces.append(resolver)
    if locator:
        for key in ("repo_id", "asset_name", "file_path", "file_id", "folder_id", "item_id", "expected_name", "download_url", "share_token"):
            value = locator.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
                break
    if filename:
        pieces.append(filename)
    elif url:
        basename = url_basename(url)
        if basename:
            pieces.append(basename)
    if not pieces and url:
        pieces.append(url)
    return slugify("-".join(pieces or ["source"]))


def normalize_source_entry(
    source: dict[str, Any],
    *,
    fallback_url: str | None = None,
    fallback_filename: str | None = None,
) -> dict[str, Any]:
    normalized = deep_copy_json(source)
    url = str(normalized.get("url") or fallback_url or "").strip()
    locator = normalized.get("locator")
    if not isinstance(locator, dict):
        locator = None
    provider = str(normalized.get("provider") or "").strip().lower()
    resolver = str(normalized.get("resolver") or "").strip().lower()
    inferred_provider, inferred_resolver, inferred_locator = infer_source_provider_resolver(url) if url else ("static", "static_url", None)
    normalized["url"] = url or None
    normalized["provider"] = provider or inferred_provider
    normalized["resolver"] = resolver or inferred_resolver
    normalized["locator"] = locator or inferred_locator
    normalized["host"] = str(normalized.get("host") or url_host(url) or "").strip().lower() or None
    normalized["channel"] = str(normalized.get("channel") or "upstream").strip().lower()
    normalized["auth"] = str(normalized.get("auth") or "public").strip().lower()
    normalized["priority"] = int(normalized.get("priority") or 0)
    normalized["verified"] = bool(normalized.get("verified"))
    normalized["manual"] = bool(normalized.get("manual"))
    normalized["size_bytes"] = normalized.get("size_bytes")
    normalized["sha256"] = normalized.get("sha256")
    normalized["last_checked"] = normalized.get("last_checked") or normalized.get("checked_at")
    normalized["final_url"] = normalized.get("final_url")
    normalized["content_type"] = normalized.get("content_type")
    normalized["resolver_viable"] = normalized.get("resolver_viable")
    normalized["source_id"] = source_id_for(
        url=url or None,
        provider=normalized.get("provider"),
        resolver=normalized.get("resolver"),
        locator=normalized.get("locator") if isinstance(normalized.get("locator"), dict) else None,
        filename=fallback_filename,
        explicit_id=str(normalized.get("source_id") or normalized.get("id") or "").strip() or None,
    )
    return normalized


def infer_source_provider_resolver(url: str) -> tuple[str, str, dict[str, Any] | None]:
    if locator := huggingface_resolve_locator(url):
        return ("huggingface", "huggingface_resolve", locator)
    if locator := github_release_locator(url):
        return ("github", "github_release_asset", locator)
    if locator := github_raw_locator(url):
        return ("github", "github_raw", locator)
    if file_id := google_drive_file_id(url):
        resolver = "google_drive_folder_entry" if "/folders/" in url else "google_drive_file"
        return ("google_drive", resolver, {"file_id": file_id})
    if locator := proton_share_locator(url):
        return ("proton_drive", "proton_share_entry", locator)
    return ("static", "static_url", None)


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
