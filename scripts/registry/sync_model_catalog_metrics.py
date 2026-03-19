#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
SCORES_PATH = REPO_ROOT / "StemSepApp" / "dev_vendor" / "audio_separator" / "models-scores.json"
OUTPUT_DIR = REPO_ROOT / "reports" / "catalog_sync" / "latest"


def canonical_export_url(doc_id: str) -> str:
    return f"https://docs.google.com/document/d/{doc_id}/export?format=txt"


GUIDE_DOC_ID = "17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c"
GUIDE_URL = canonical_export_url(GUIDE_DOC_ID)
GUIDE_REVISION = "2026-03-14"
GUIDE_SOURCE_NAME = f"Curated research {GUIDE_REVISION}"

GUIDE_METRIC_RULES: dict[str, dict[str, Any]] = {
    "bs-roformer-large-inst": {"needle": "bs_large_v2_inst", "mode": "standard"},
    "unwa-big-beta-7": {"needle": "Big Beta 7", "mode": "standard"},
    "unwa-inst-fno": {"needle": "Unwa BS-Roformer-Inst-FNO", "mode": "standard"},
    "unwa-hyperace": {
        "needle": "Inst. fullness 36.91 | bleedless 38.77 | SDR 17.27",
        "mode": "standard",
    },
    "unwa-hyperace-v2-inst": {
        "needle": "Inst. bleedless 37.87, fullness 38.03, SDR 17.40",
        "mode": "standard",
    },
    "unwa-revive-3e": {"needle": "bs_roformer_revive3e", "mode": "standard"},
    "unwa-revive-2": {"needle": "Revive 2 variant", "mode": "standard"},
    "gabox-voc-fv7": {
        "needle": "Gabox released voc_fv7 Mel-Roformer",
        "mode": "standard",
    },
    "gabox-voc-fv7beta3": {"needle": "vocfv7beta3", "mode": "standard"},
    "mdx23c-drumsep-5stem": {"needle": "drumsep 5 stem model", "mode": "drumsep_5"},
    "anvuew-dereverb-bs-225050": {
        "needle": "dereverb_bs_roformer_anvuew_sdr_22.5050",
        "mode": "dereverb",
    },
}


CURATED_CARD_OVERRIDES: dict[str, dict[str, Any]] = {
    "bs-roformer-large-inst": {
        "catalog_status": "manual_only",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "instrumental",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [17.61, 32.06, 43.95],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "BS-Roformer-Large-Inst / bs_large_v2_inst guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 17.61, "fullness": 32.06, "bleedless": 43.95},
        "metrics_evidence": [
            {
                "source": GUIDE_SOURCE_NAME,
                "url": GUIDE_URL,
                "note": "Guide lists Fullness 32.06, Bleedless 43.95, SDR 17.61.",
                "verified_at": GUIDE_REVISION,
            }
        ],
    },
    "unwa-big-beta-7": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "vocals",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [11.2, 16.2, 38.77],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Big Beta 7 vocal guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 11.2, "fullness": 16.2, "bleedless": 38.77},
        "metrics_evidence": [
            {
                "source": GUIDE_SOURCE_NAME,
                "url": GUIDE_URL,
                "note": "Guide lists Bleedless 38.77, Fullness 16.20, SDR 11.20.",
                "verified_at": GUIDE_REVISION,
            }
        ],
    },
    "unwa-inst-fno": {
        "catalog_status": "manual_only",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "instrumental",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [17.6, 32.03, 42.87],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Unwa Inst FNO guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 17.6, "fullness": 32.03, "bleedless": 42.87},
        "metrics_evidence": [
            {
                "source": GUIDE_SOURCE_NAME,
                "url": GUIDE_URL,
                "note": "Guide lists Inst FNO with Fullness 32.03, Bleedless 42.87, SDR 17.60.",
                "verified_at": GUIDE_REVISION,
            }
        ],
    },
    "unwa-hyperace": {
        "catalog_status": "manual_only",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "instrumental",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [17.27, 36.91, 38.77],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Unwa HyperACE instrumental guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 17.27, "fullness": 36.91, "bleedless": 38.77},
    },
    "unwa-hyperace-v2-inst": {
        "catalog_status": "manual_only",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "instrumental",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [17.4, 38.03, 37.87],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "HyperACE v2 instrumental guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 17.4, "fullness": 38.03, "bleedless": 37.87},
    },
    "unwa-hyperace-v2-voc": {
        "catalog_status": "candidate",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "special",
            "primary_target": "vocals",
            "labels": ["VOC SDR", "FOCUS", "STATUS"],
            "values": [11.39, "Lead Vox", "Guide"],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Guide lists HyperACE v2 vocal SDR 11.39 but not the full triad nearby.",
            "last_verified": GUIDE_REVISION,
        },
    },
    "unwa-revive-3e": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "vocals",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [10.98, 21.43, 30.51],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Revive 3e guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 10.98, "fullness": 21.43, "bleedless": 30.51},
    },
    "unwa-revive-2": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "vocals",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [10.97, 15.13, 40.07],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Revive 2 guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 10.97, "fullness": 15.13, "bleedless": 40.07},
    },
    "gabox-voc-fv7": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "vocals",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [11.16, 17.99, 33.85],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "voc_fv7 guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 11.16, "fullness": 17.99, "bleedless": 33.85},
    },
    "gabox-voc-fv7beta3": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "standard",
            "primary_target": "vocals",
            "labels": ["SDR", "FULLNESS", "BLEED"],
            "values": [10.8, 21.82, 30.83],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "vocfv7beta3 guide entry.",
            "last_verified": GUIDE_REVISION,
        },
        "metrics": {"sdr": 10.8, "fullness": 21.82, "bleedless": 30.83},
    },
    "mdx23c-drumsep-5stem": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "special",
            "primary_target": "drums",
            "labels": ["KICK SDR", "SNARE FULL", "HH BLEED"],
            "values": [16.66, 25.0361, 12.347],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": GUIDE_URL,
            "evidence_note": "Guide lists per-stem drumsep metrics for the public 5-stem release.",
            "last_verified": GUIDE_REVISION,
        },
    },
    "anvuew-dereverb-room": {
        "catalog_status": "verified",
        "metrics_status": "verified",
        "links": {
            "checkpoint": "https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew_sdr_13.7432.ckpt",
            "config": "https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew.yaml",
            "homepage": "https://huggingface.co/anvuew/dereverb_room",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "dereverb_room_anvuew_sdr_13.7432.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "dereverb_room_anvuew.yaml", "sha256": None},
            "additional": [],
        },
        "card_metrics": {
            "kind": "special",
            "primary_target": "restoration",
            "labels": ["MODEL SDR", "FOCUS", "CHANNELS"],
            "values": [13.7432, "Room", "Mono"],
            "source": "HF README",
            "evidence_url": "https://huggingface.co/anvuew/dereverb_room",
            "evidence_note": "HF repo exposes dereverb_room_anvuew_sdr_13.7432.ckpt for mono room-reverb removal.",
            "last_verified": GUIDE_REVISION,
        },
    },
    "anvuew-dereverb-bs-225050": {
        "catalog_status": "verified",
        "metrics_status": "guide_curated",
        "card_metrics": {
            "kind": "special",
            "primary_target": "restoration",
            "labels": ["MODEL SDR", "BASE SDR", "CHANNELS"],
            "values": [22.505, 20.4029, "Mono"],
            "source": GUIDE_SOURCE_NAME,
            "evidence_url": "https://huggingface.co/anvuew/dereverb_bs_roformer",
            "evidence_note": "New BS-Roformer dereverb variant highlighted in the live guide.",
            "last_verified": GUIDE_REVISION,
        },
    },
    "small-karaoke-gaboxauf": {
        "catalog_status": "candidate",
        "metrics_status": "missing_evidence",
        "card_metrics": {
            "kind": "status",
            "primary_target": "karaoke",
            "labels": ["METRICS", "DOWNLOAD", "RUNTIME"],
            "values": ["Pending", "Direct", "MSST"],
            "source": "Guide mention",
            "evidence_url": GUIDE_URL,
            "evidence_note": "Guide says metrics exist, but the snippet does not include numeric values near the release note.",
            "last_verified": GUIDE_REVISION,
        },
    },
}


NEW_MODEL_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "bs-roformer-large-inst",
        "name": "BS-Roformer Large Inst",
        "type": "Roformer - Instrumental",
        "architecture": "BS-Roformer",
        "description": "Instrumental (Large v2)",
        "tags": ["instrumental"],
        "links": {
            "checkpoint": "https://huggingface.co/pcunwa/BS-Roformer-Large-Inst/resolve/main/bs_large_v2_inst.ckpt",
            "config": "https://huggingface.co/pcunwa/BS-Roformer-Large-Inst/resolve/main/config.yaml",
            "homepage": "https://huggingface.co/pcunwa/BS-Roformer-Large-Inst",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "bs_large_v2_inst.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "config.yaml", "sha256": None},
            "additional": [],
        },
        "metrics": {"sdr": 0.0, "fullness": None, "bleedless": None, "aura_stft": None, "aura_mrstft": None, "log_wmse": None},
        "recommended_settings": {"segment_size": 352800, "batch_size": None, "tta": False, "overlap": 4, "shifts": None, "bit_depth": None},
        "settings_semantics": {"overlap": {"meaning_by_runtime": {"audio-separator": "auto", "stemsep-legacy": "auto", "msst": "divisor"}, "notes": "Custom bs_roformer runtime variant."}},
        "runtime": {"allowed": ["msst"], "preferred": "msst", "blocking_reason": "Current runtime needs the attached custom bs_roformer.py for this model family."},
        "compatibility": {"uvr_compatible": False, "audio_separator_compatible": False, "msst_compatible": True, "known_issues": ["Requires the repo's custom bs_roformer.py."]},
        "requirements": {"python": {"min": "3.10", "max": None}, "torch": {"min": None, "max": None, "cuda": [], "rocm": []}, "pip": {"packages": []}, "files": {"expects": [], "provides": []}, "manual_steps": ["Use the repo-provided bs_roformer.py with MSST for this model."]},
        "phase_fix": {"is_valid_reference": None, "recommended_params": {"lowHz": None, "highHz": None, "highFreqWeight": None}, "recommended_usage": []},
        "capabilities": {"stems": [], "roles": ["separation"], "supports_mono": None, "supports_stereo": None},
        "vram_required": 8.0,
    },
    {
        "id": "unwa-big-beta-7",
        "name": "Unwa Big Beta 7",
        "type": "Roformer - Vocals",
        "architecture": "Mel-Roformer",
        "description": "Vocals",
        "tags": ["vocals"],
        "links": {
            "checkpoint": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta7.ckpt",
            "config": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta7.yaml",
            "homepage": "https://huggingface.co/pcunwa/Mel-Band-Roformer-big",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "big_beta7.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "big_beta7.yaml", "sha256": None},
            "additional": [],
        },
        "metrics": {"sdr": 0.0, "fullness": None, "bleedless": None, "aura_stft": None, "aura_mrstft": None, "log_wmse": None},
        "runtime": {"allowed": ["msst"], "preferred": "msst", "blocking_reason": None},
        "vram_required": 6.0,
    },
    {
        "id": "small-karaoke-gaboxauf",
        "name": "Small Karaoke GaboxAuf",
        "type": "Karaoke / Backing Vocals / Other Separations",
        "architecture": "Mel-Roformer",
        "description": "Small karaoke split",
        "tags": ["karaoke"],
        "links": {
            "checkpoint": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/small_karaoke_gaboxaufr.ckpt",
            "config": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/config_karaoke_small.yaml",
            "homepage": "https://huggingface.co/GaboxR67/MelBandRoformers",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "small_karaoke_gaboxaufr.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "config_karaoke_small.yaml", "sha256": None},
            "additional": [],
        },
        "metrics": {"sdr": 0.0, "fullness": None, "bleedless": None, "aura_stft": None, "aura_mrstft": None, "log_wmse": None},
        "runtime": {"allowed": ["msst"], "preferred": "msst", "blocking_reason": None},
        "vram_required": 6.0,
    },
    {
        "id": "anvuew-dereverb-bs-225050",
        "name": "Anvuew DeReverb BS 22.5050",
        "type": "Restoration / Denoise / Dereverb",
        "architecture": "BS-Roformer",
        "description": "De-Reverb (Mono)",
        "tags": ["dereverb", "restoration"],
        "links": {
            "checkpoint": "https://huggingface.co/anvuew/dereverb_bs_roformer/resolve/main/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
            "config": "https://huggingface.co/anvuew/dereverb_bs_roformer/resolve/main/config.yaml",
            "homepage": "https://huggingface.co/anvuew/dereverb_bs_roformer",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "config.yaml", "sha256": None},
            "additional": [],
        },
        "metrics": {"sdr": 0.0, "fullness": None, "bleedless": None, "aura_stft": None, "aura_mrstft": None, "log_wmse": None},
        "runtime": {"allowed": ["msst"], "preferred": "msst", "blocking_reason": None},
        "vram_required": 8.0,
    },
    {
        "id": "gabox-voc-fv7",
        "name": "Gabox Voc FV7",
        "type": "Roformer - Vocals",
        "architecture": "Mel-Roformer",
        "description": "Vocals",
        "tags": ["vocals"],
        "links": {
            "checkpoint": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv7.ckpt",
            "config": "https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/v7.yaml",
            "homepage": "https://huggingface.co/GaboxR67/MelBandRoformers",
        },
        "artifacts": {
            "primary": {"kind": "checkpoint", "filename": "voc_fv7.ckpt", "sha256": None},
            "config": {"kind": "config", "filename": "v7.yaml", "sha256": None},
            "additional": [],
        },
        "metrics": {"sdr": 0.0, "fullness": None, "bleedless": None, "aura_stft": None, "aura_mrstft": None, "log_wmse": None},
        "runtime": {"allowed": ["msst"], "preferred": "msst", "blocking_reason": None},
        "vram_required": 6.0,
    },
]

MANUAL_CANDIDATES = [
    {
        "id": "last-bs-roformer",
        "name": "Gabox Last BS-Roformer",
        "reason": "Guide mentions the model, but config/evidence are not ready for safe runtime integration.",
        "source": GUIDE_URL,
    }
]


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_text(url: str, timeout_sec: int = 45) -> str:
    req = Request(url, headers={"User-Agent": "StemSep-catalog-sync/1.0"}, method="GET")
    with urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="replace")


def url_filename(url: str | None) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    path = urlparse(url).path
    name = Path(path).name
    return name or None


def as_metric_value(value: Any) -> float | str | None:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def ensure_model_defaults(model: dict[str, Any]) -> None:
    model.setdefault("metrics", {})
    model.setdefault("tags", [])
    model.setdefault("stems", [])
    model.setdefault("recommended_settings", {})
    model.setdefault("guide_revision", GUIDE_REVISION)
    model.setdefault("catalog_status", "candidate")
    model.setdefault("metrics_status", "missing_evidence")
    model.setdefault("metrics_evidence", [])


def standard_card(values: dict[str, float], source: str, url: str | None, note: str) -> dict[str, Any]:
    return {
        "kind": "standard",
        "primary_target": "separation",
        "labels": ["SDR", "FULLNESS", "BLEED"],
        "values": [round(values["sdr"], 2), round(values["fullness"], 2), round(values["bleedless"], 2)],
        "source": source,
        "evidence_url": url,
        "evidence_note": note,
        "last_verified": GUIDE_REVISION,
    }


def vendor_card(voc_sdr: float | None, inst_sdr: float | None, filename: str) -> dict[str, Any]:
    vals = [v for v in [voc_sdr, inst_sdr] if isinstance(v, (int, float))]
    avg = round(sum(vals) / len(vals), 2) if vals else None
    return {
        "kind": "special",
        "primary_target": "separation",
        "labels": ["VOC SDR", "INST SDR", "AVG SDR"],
        "values": [as_metric_value(voc_sdr), as_metric_value(inst_sdr), as_metric_value(avg)],
        "source": "Vendor scores",
        "evidence_url": None,
        "evidence_note": f"Matched by artifact filename: {filename}",
        "last_verified": GUIDE_REVISION,
    }


def status_card(model: dict[str, Any]) -> dict[str, Any]:
    checkpoint = model.get("links", {}).get("checkpoint")
    runtime = model.get("runtime", {})
    blocking = runtime.get("blocking_reason")
    download_state = "Direct" if url_filename(checkpoint) else "Missing"
    runtime_state = "Manual" if blocking else runtime.get("preferred") or "Unknown"
    runtime_label = str(runtime_state).upper() if isinstance(runtime_state, str) else "UNKNOWN"
    return {
        "kind": "status",
        "primary_target": "catalog",
        "labels": ["METRICS", "DOWNLOAD", "RUNTIME"],
        "values": ["Pending", download_state, runtime_label],
        "source": "Catalog status",
        "evidence_url": model.get("links", {}).get("homepage"),
        "evidence_note": "No verified metric trio is currently attached to this model.",
        "last_verified": GUIDE_REVISION,
    }


def parse_standard_metrics(snippet: str) -> dict[str, float] | None:
    compact = " ".join(snippet.split())
    patterns = [
        re.compile(r"(?i)bleedless[: ]+([0-9]+(?:\.[0-9]+)?).*?fullness[: ]+([0-9]+(?:\.[0-9]+)?).*?sdr[: ]+([0-9]+(?:\.[0-9]+)?)"),
        re.compile(r"(?i)fullness[: ]+([0-9]+(?:\.[0-9]+)?).*?bleedless[: ]+([0-9]+(?:\.[0-9]+)?).*?sdr[: ]+([0-9]+(?:\.[0-9]+)?)"),
    ]
    for pattern in patterns:
        match = pattern.search(compact)
        if not match:
            continue
        if pattern.pattern.lower().startswith("(?i)bleedless"):
            bleedless, fullness, sdr = match.groups()
        else:
            fullness, bleedless, sdr = match.groups()
        return {"sdr": float(sdr), "fullness": float(fullness), "bleedless": float(bleedless)}
    return None


def extract_guide_metrics(guide_text: str) -> dict[str, dict[str, Any]]:
    lines = guide_text.splitlines()
    out: dict[str, dict[str, Any]] = {}
    for model_id, rule in GUIDE_METRIC_RULES.items():
        matches = [idx for idx, line in enumerate(lines) if rule["needle"].lower() in line.lower()]
        if not matches:
            continue
        idx = matches[0]
        snippet = "\n".join(lines[max(0, idx - 3) : min(len(lines), idx + 10)])
        if rule["mode"] == "standard":
            metrics = parse_standard_metrics(snippet)
            if metrics:
                out[model_id] = {"metrics": metrics, "snippet": snippet}
    return out


def build_vendor_index(scores: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for filename, payload in scores.items():
        if not isinstance(payload, dict):
            continue
        median = payload.get("median_scores") or {}
        voc = median.get("vocals") if isinstance(median.get("vocals"), dict) else {}
        inst = median.get("instrumental") if isinstance(median.get("instrumental"), dict) else {}
        voc_sdr = voc.get("SDR") if isinstance(voc.get("SDR"), (int, float)) else None
        inst_sdr = inst.get("SDR") if isinstance(inst.get("SDR"), (int, float)) else None
        if voc_sdr is not None or inst_sdr is not None:
            out[filename] = {"vocals_sdr": voc_sdr, "instrumental_sdr": inst_sdr}
    return out


def model_checkpoint_filenames(model: dict[str, Any]) -> list[str]:
    filenames: list[str] = []
    primary = model.get("artifacts", {}).get("primary", {})
    if isinstance(primary, dict) and isinstance(primary.get("filename"), str):
        filenames.append(primary["filename"].strip())
    link_name = url_filename(model.get("links", {}).get("checkpoint"))
    if link_name:
        filenames.append(link_name)
    return list(dict.fromkeys(name for name in filenames if name))


def build_candidate_reports(models: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    guide_candidates: list[dict[str, Any]] = []
    missing_download: list[dict[str, Any]] = []
    counts = {"verified": 0, "candidate": 0, "blocked": 0, "manual_only": 0, "online_only": 0}
    with_card_metrics = 0
    for model in models:
        status = model.get("catalog_status", "candidate")
        counts[status] = counts.get(status, 0) + 1
        if model.get("card_metrics"):
            with_card_metrics += 1
        if status != "verified":
            guide_candidates.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "catalog_status": status,
                    "metrics_status": model.get("metrics_status"),
                    "checkpoint": model.get("links", {}).get("checkpoint"),
                }
            )
        if not url_filename(model.get("links", {}).get("checkpoint")):
            missing_download.append({"id": model.get("id"), "name": model.get("name"), "reason": "Missing direct checkpoint link"})
    missing_download.extend(MANUAL_CANDIDATES)
    coverage = {
        "generated_at": now_rfc3339(),
        "guide_revision": GUIDE_REVISION,
        "models_total": len(models),
        "models_with_card_metrics": with_card_metrics,
        "counts_by_catalog_status": counts,
        "verified_with_renderable_card_metrics": sum(1 for model in models if model.get("catalog_status") == "verified" and model.get("card_metrics")),
    }
    return guide_candidates, missing_download, coverage


def add_or_replace_models(models: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {model["id"]: deepcopy(model) for model in models}
    for addition in additions:
        by_id[addition["id"]] = addition
    return [by_id[key] for key in sorted(by_id.keys())]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build manually curated catalog metrics and candidate reports."
    )
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--scores", default=str(SCORES_PATH))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--guide-url", default=GUIDE_URL)
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    registry = load_json(registry_path)
    models = registry.get("models", [])
    if not isinstance(models, list):
        raise SystemExit("Registry models must be a list")

    models = add_or_replace_models(models, NEW_MODEL_TEMPLATES)
    scores = load_json(Path(args.scores)) if Path(args.scores).exists() else {}
    vendor_index = build_vendor_index(scores if isinstance(scores, dict) else {})

    try:
        guide_text = fetch_text(args.guide_url)
        guide_source = args.guide_url
    except (HTTPError, URLError, TimeoutError, OSError):
        guide_text = ""
        guide_source = "unavailable"

    guide_metrics = extract_guide_metrics(guide_text) if guide_text else {}
    updated_models: list[dict[str, Any]] = []
    backfill_report: list[dict[str, Any]] = []

    for raw_model in models:
        model = deepcopy(raw_model)
        ensure_model_defaults(model)
        model["guide_revision"] = GUIDE_REVISION

        if model["id"] in guide_metrics and "metrics" in guide_metrics[model["id"]]:
            metrics = guide_metrics[model["id"]]["metrics"]
            model["metrics"] = {**(model.get("metrics") or {}), **metrics}
            model["card_metrics"] = standard_card(metrics, GUIDE_SOURCE_NAME, GUIDE_URL, f"Guide-extracted metrics for {model['id']}.")
            model["catalog_status"] = "verified"
            model["metrics_status"] = "guide_curated"
            model["metrics_evidence"] = [
                {
                    "source": GUIDE_SOURCE_NAME,
                    "url": GUIDE_URL,
                    "note": f"Guide excerpt matched for {model['id']}.",
                    "verified_at": GUIDE_REVISION,
                }
            ]

        filenames = model_checkpoint_filenames(model)
        vendor_filename = next((name for name in filenames if name in vendor_index), None)
        if vendor_filename and not model.get("card_metrics"):
            vendor_match = vendor_index[vendor_filename]
            model["card_metrics"] = vendor_card(vendor_match.get("vocals_sdr"), vendor_match.get("instrumental_sdr"), vendor_filename)
            model["metrics_status"] = "vendor_matched"
            model["metrics_evidence"] = [
                {
                    "source": "Vendor scores",
                    "url": None,
                    "note": f"Matched by artifact filename: {vendor_filename}",
                    "verified_at": GUIDE_REVISION,
                }
            ]

        override = CURATED_CARD_OVERRIDES.get(model["id"])
        if override:
            model = deep_merge(model, override)

        card = model.get("card_metrics")
        if isinstance(card, dict) and card.get("kind") == "standard":
            values = card.get("values") or []
            if len(values) == 3 and all(isinstance(v, (int, float)) for v in values):
                model.setdefault("metrics", {})
                model["metrics"]["sdr"] = float(values[0])
                model["metrics"]["fullness"] = float(values[1])
                model["metrics"]["bleedless"] = float(values[2])

        metrics = model.get("metrics") or {}
        model["sdr"] = metrics.get("sdr")
        model["fullness"] = metrics.get("fullness")
        model["bleedless"] = metrics.get("bleedless")

        if not model.get("card_metrics"):
            model["card_metrics"] = status_card(model)

        if model.get("catalog_status") == "candidate":
            runtime = model.get("runtime", {})
            if runtime.get("blocking_reason"):
                model["catalog_status"] = "manual_only"

        backfill_report.append(
            {
                "id": model.get("id"),
                "catalog_status": model.get("catalog_status"),
                "metrics_status": model.get("metrics_status"),
                "card_metrics_kind": model.get("card_metrics", {}).get("kind"),
                "checkpoint": model.get("links", {}).get("checkpoint"),
                "guide_source": guide_source,
            }
        )
        updated_models.append(model)

    registry["models"] = updated_models
    registry["guide_revision"] = GUIDE_REVISION
    registry["generated_at"] = now_rfc3339()

    output_dir = Path(args.output_dir)
    guide_candidates, missing_download, coverage = build_candidate_reports(updated_models)
    write_json(output_dir / "guide_model_candidates.json", guide_candidates)
    write_json(output_dir / "metric_backfill_report.json", backfill_report)
    write_json(output_dir / "missing_download_candidates.json", missing_download)
    write_json(output_dir / "coverage_summary.json", coverage)

    if args.write_registry:
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"guide_revision": GUIDE_REVISION, "models_total": len(updated_models), "reports": str(output_dir), "write_registry": bool(args.write_registry)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
