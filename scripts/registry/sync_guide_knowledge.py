#!/usr/bin/env python3
"""
Sync StemSep's external guide knowledge and generate machine-readable drift reports.

Primary goals:
- Fetch the Google Doc guide export (txt).
- Diff against the vendored/local snapshot.
- Produce a model coverage report:
  - model candidates seen in guide
  - candidates resolved to registry IDs
  - candidates missing in registry
  - candidates likely manual-only / gated
- Optionally enforce CI gate thresholds.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_ID = "1DuZkyucMLpCWfsjVNYxVOEr2s_mgRAhqUHRg2ndOFNs"
DEFAULT_EXPORT_URL = (
    f"https://docs.google.com/document/d/{DEFAULT_DOC_ID}/export?format=txt"
)
DEFAULT_LOCAL_SNAPSHOT = REPO_ROOT / "docs" / "separation_guide" / "deton24_guide.txt"
DEFAULT_REGISTRY = REPO_ROOT / "StemSepApp" / "assets" / "registry" / "models.v2.source.json"
DEFAULT_ALIAS_DOC = REPO_ROOT / "docs" / "separation_guide" / "stemsep_model_id_mapping.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "guide_sync" / "latest"

# Broad but conservative model-like token matcher:
# - lowercase/alnum start
# - contains at least one separator
# - length guard to avoid huge garbage captures
MODEL_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9._-]{2,127}\b")
BACKTICK_TOKEN_RE = re.compile(r"`([^`]{2,200})`")

# Heuristic terms for "manual/gated" models in guide text.
MANUAL_GATED_HINTS = (
    "manual",
    "gated",
    "private",
    "token",
    "patreon",
    "mvsep",
    "not downloadable",
    "not currently downloadable",
    "manual-download",
)

MODEL_KEYWORDS = (
    "roformer",
    "demucs",
    "mdx",
    "scnet",
    "apollo",
    "bandit",
    "karaoke",
    "dereverb",
    "denoise",
    "debleed",
    "drumsep",
    "viperx",
    "hyperace",
    "gabox",
    "becruily",
    "resurrection",
    "revive",
    "unwa",
    "anvuew",
    "aufr33",
)

MONTHS = {
    # English
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    # Swedish
    "januari": 1,
    "februari": 2,
    "mars": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "augusti": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}


@dataclasses.dataclass
class Snapshot:
    text: str
    source: str
    fetched: bool
    fetch_error: str | None

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8", errors="replace")).hexdigest()


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_model_token(raw: str) -> str:
    t = raw.strip().lower()
    # Common punctuation cleanup from prose contexts.
    t = t.strip("`.,;:!?()[]{}<>\"'")
    t = t.replace(" ", "-")
    t = re.sub(r"-{2,}", "-", t)
    return t


def parse_guide_revision_date(text: str) -> tuple[str | None, datetime | None]:
    """
    Best-effort date extraction from the top of the guide text.
    Returns (raw_match, parsed_utc_date_midnight).
    """
    head = "\n".join(text.splitlines()[:1200]).lower()

    # Pattern 1: "4 mars 2026" / "4 march 2026"
    m1 = re.search(r"\b(\d{1,2})\s+([a-zåäö]+)\s+(\d{4})\b", head)
    if m1:
        day = int(m1.group(1))
        mon_name = m1.group(2)
        year = int(m1.group(3))
        mon = MONTHS.get(mon_name)
        if mon:
            try:
                dt = datetime(year, mon, day, tzinfo=timezone.utc)
                return m1.group(0), dt
            except ValueError:
                pass

    # Pattern 2: "March 4, 2026"
    m2 = re.search(r"\b([a-zåäö]+)\s+(\d{1,2}),\s*(\d{4})\b", head)
    if m2:
        mon_name = m2.group(1)
        day = int(m2.group(2))
        year = int(m2.group(3))
        mon = MONTHS.get(mon_name)
        if mon:
            try:
                dt = datetime(year, mon, day, tzinfo=timezone.utc)
                return m2.group(0), dt
            except ValueError:
                pass

    # Pattern 3: ISO-like date
    m3 = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", head)
    if m3:
        year = int(m3.group(1))
        mon = int(m3.group(2))
        day = int(m3.group(3))
        try:
            dt = datetime(year, mon, day, tzinfo=timezone.utc)
            return m3.group(0), dt
        except ValueError:
            pass

    return None, None


def fetch_google_doc_export(url: str, timeout_sec: int = 45) -> str:
    req = Request(
        url=url,
        headers={
            "User-Agent": "StemSep-guide-sync/1.0 (+https://github.com/engdahlz/StemSep)",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def load_remote_or_fallback(
    export_url: str,
    local_snapshot_path: Path,
    timeout_sec: int,
    allow_fetch_failure: bool,
) -> Snapshot:
    try:
        txt = fetch_google_doc_export(export_url, timeout_sec=timeout_sec)
        return Snapshot(text=txt, source=export_url, fetched=True, fetch_error=None)
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        if not local_snapshot_path.exists():
            raise RuntimeError(
                f"Guide fetch failed and fallback snapshot missing at {local_snapshot_path}: {e}"
            ) from e
        if not allow_fetch_failure:
            raise RuntimeError(
                f"Guide fetch failed (no --allow-fetch-failure): {e}"
            ) from e
        txt = local_snapshot_path.read_text(encoding="utf-8", errors="replace")
        return Snapshot(
            text=txt,
            source=str(local_snapshot_path),
            fetched=False,
            fetch_error=str(e),
        )


def read_local_snapshot(path: Path) -> Snapshot | None:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    return Snapshot(text=txt, source=str(path), fetched=False, fetch_error=None)


def compute_diff_summary(remote: str, local: str) -> dict[str, Any]:
    remote_lines = remote.splitlines()
    local_lines = local.splitlines()

    matcher = difflib.SequenceMatcher(a=local_lines, b=remote_lines, autojunk=False)
    ratio = matcher.ratio()

    added = 0
    removed = 0
    changed_blocks: list[dict[str, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("insert", "replace"):
            added += max(0, j2 - j1)
        if tag in ("delete", "replace"):
            removed += max(0, i2 - i1)
        changed_blocks.append(
            {
                "tag": tag,
                "local_start": i1 + 1,
                "local_end": i2,
                "remote_start": j1 + 1,
                "remote_end": j2,
            }
        )

    return {
        "changed": sha256_text(remote) != sha256_text(local),
        "similarity_ratio": round(ratio, 6),
        "local_line_count": len(local_lines),
        "remote_line_count": len(remote_lines),
        "line_delta": len(remote_lines) - len(local_lines),
        "estimated_added_lines": added,
        "estimated_removed_lines": removed,
        "changed_blocks_preview": changed_blocks[:120],
    }


def load_registry_model_ids(registry_path: Path) -> set[str]:
    obj = load_json(registry_path)
    models = obj.get("models")
    if not isinstance(models, list):
        return set()
    out: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        mid = model.get("id")
        if isinstance(mid, str) and mid.strip():
            out.add(mid.strip().lower())
    return out


def parse_alias_doc(alias_doc_path: Path) -> tuple[dict[str, str], set[str]]:
    """
    Parse stemsep_model_id_mapping.md to recover:
    - alias -> canonical model id map from markdown table rows
    - manual-only names from "not currently downloadable" bullets
    """
    alias_map: dict[str, str] = {}
    manual_only: set[str] = set()
    if not alias_doc_path.exists():
        return alias_map, manual_only

    text = alias_doc_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_manual_section = False
    for line in lines:
        striped = line.strip()
        lower = striped.lower()

        if lower.startswith("##"):
            in_manual_section = "not currently downloadable" in lower

        # Markdown table row:
        # | Guide Name | model-id | note |
        if striped.startswith("|") and striped.count("|") >= 3:
            cols = [c.strip() for c in striped.strip("|").split("|")]
            if len(cols) >= 2:
                guide_name = cols[0]
                model_id = cols[1].strip("` ")
                if guide_name and model_id and re.match(r"^[a-z0-9._-]+$", model_id):
                    alias_map[normalize_model_token(guide_name)] = model_id.lower()
                    alias_map[normalize_model_token(model_id)] = model_id.lower()

        if in_manual_section and striped.startswith("-"):
            names = re.findall(r"`([^`]+)`", striped)
            for name in names:
                tok = normalize_model_token(name)
                if tok:
                    manual_only.add(tok)

    return alias_map, manual_only


def extract_guide_candidates(
    text: str, registry_ids: set[str], alias_keys: set[str]
) -> set[str]:
    """
    Extract model-like tokens from the guide.
    Heuristics:
    - markdown/code spans (`token`)
    - lower-case model-id-like words with separators
    """
    out: set[str] = set()
    lowered = text.lower()

    for raw in BACKTICK_TOKEN_RE.findall(lowered):
        tok = normalize_model_token(raw)
        if _looks_like_model_token(tok):
            out.add(tok)

    for raw in MODEL_TOKEN_RE.findall(lowered):
        tok = normalize_model_token(raw)
        if _looks_like_model_token(tok):
            out.add(tok)

    # Ensure known IDs/aliases are represented even when prose formatting differs.
    for known in registry_ids:
        if known and known in lowered:
            out.add(known)
    for alias in alias_keys:
        if alias and len(alias) >= 3 and alias in lowered:
            out.add(alias)

    return out


def _looks_like_model_token(token: str) -> bool:
    if not token:
        return False
    if len(token) > 80:
        return False
    # Must contain separator to avoid random words like "model".
    if "-" not in token and "_" not in token and "." not in token:
        return False
    # Must include at least one letter.
    if not any(ch.isalpha() for ch in token):
        return False
    # Avoid pure numeric/version patterns.
    if re.fullmatch(r"[0-9][0-9.\-]*", token):
        return False
    # Most model ids don't start with numeric prefixes in this ecosystem.
    if token[0].isdigit():
        return False
    # Avoid file paths / code snippets.
    if "/" in token or "\\" in token or ":" in token:
        return False
    # Avoid obvious non-model artifacts.
    if token.startswith("http") or token.startswith("www."):
        return False
    if token.endswith(
        (
            ".com",
            ".org",
            ".net",
            ".txt",
            ".pdf",
            ".json",
            ".yaml",
            ".yml",
            ".dll",
            ".exe",
            ".zip",
            ".py",
            ".md",
        )
    ):
        return False
    # Require domain keyword to avoid generic technical tokens.
    return any(k in token for k in MODEL_KEYWORDS)


def classify_manual_or_gated_candidates(
    text: str, unresolved_tokens: set[str], known_manual_only: set[str]
) -> set[str]:
    lowered = text.lower()
    out: set[str] = set()

    # Directly tagged in mapping doc.
    out.update({t for t in unresolved_tokens if t in known_manual_only})

    # Context heuristic: token appears near manual/gated hints.
    for token in unresolved_tokens:
        if token in out:
            continue
        # Keep search bounded to avoid pathological behavior.
        idx = lowered.find(token)
        if idx < 0:
            continue
        start = max(0, idx - 220)
        end = min(len(lowered), idx + len(token) + 220)
        ctx = lowered[start:end]
        if any(h in ctx for h in MANUAL_GATED_HINTS):
            out.add(token)
    return out


def resolve_guide_tokens(
    guide_tokens: set[str],
    registry_ids: set[str],
    alias_map: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    resolved: dict[str, str] = {}
    unresolved: set[str] = set()

    for tok in sorted(guide_tokens):
        if tok in registry_ids:
            resolved[tok] = tok
            continue
        alias = alias_map.get(tok)
        if alias and alias in registry_ids:
            resolved[tok] = alias
            continue
        unresolved.add(tok)

    return resolved, unresolved


def gate_report(
    report: dict[str, Any], max_age_days: int, max_unsynced_models: int
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    divergence = report.get("divergence", {})
    age_days = divergence.get("days_since_guide_revision")
    missing = divergence.get("missing_in_registry_count", 0)

    if isinstance(age_days, (int, float)) and age_days > max_age_days:
        reasons.append(
            f"Guide revision age {age_days}d exceeds threshold {max_age_days}d."
        )
    if isinstance(missing, int) and missing > max_unsynced_models:
        reasons.append(
            f"Missing guide model candidates {missing} exceeds threshold {max_unsynced_models}."
        )
    status = "fail" if reasons else "pass"
    return status, reasons


def build_report(
    *,
    remote_snapshot: Snapshot,
    local_snapshot: Snapshot | None,
    registry_ids: set[str],
    alias_map: dict[str, str],
    known_manual_only: set[str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    guide_text = remote_snapshot.text
    local_text = local_snapshot.text if local_snapshot else ""

    diff = compute_diff_summary(guide_text, local_text) if local_snapshot else {
        "changed": True,
        "similarity_ratio": 0.0,
        "local_line_count": 0,
        "remote_line_count": remote_snapshot.line_count,
        "line_delta": remote_snapshot.line_count,
        "estimated_added_lines": remote_snapshot.line_count,
        "estimated_removed_lines": 0,
        "changed_blocks_preview": [],
    }

    guide_tokens = extract_guide_candidates(
        guide_text, registry_ids=registry_ids, alias_keys=set(alias_map.keys())
    )
    resolved_map, unresolved = resolve_guide_tokens(guide_tokens, registry_ids, alias_map)
    manual_or_gated = classify_manual_or_gated_candidates(
        guide_text, unresolved, known_manual_only
    )
    missing_in_registry = sorted(unresolved - manual_or_gated)

    revision_raw, revision_dt = parse_guide_revision_date(guide_text)
    age_days: int | None = None
    if revision_dt is not None:
        age_days = int((datetime.now(timezone.utc) - revision_dt).total_seconds() // 86400)

    # Build frequency table for unresolved tokens (helps prioritization).
    unresolved_counts: dict[str, int] = defaultdict(int)
    lowered = guide_text.lower()
    for token in unresolved:
        unresolved_counts[token] = lowered.count(token)

    coverage_report = {
        "registry_model_count": len(registry_ids),
        "guide_candidate_count": len(guide_tokens),
        "resolved_candidate_count": len(resolved_map),
        "resolution_rate": round(
            (len(resolved_map) / len(guide_tokens)) if guide_tokens else 1.0,
            6,
        ),
        "resolved_candidates": [
            {"guide_token": k, "registry_id": v}
            for k, v in sorted(resolved_map.items())
        ],
        "missing_in_registry": [
            {"token": t, "mentions": unresolved_counts.get(t, 0)}
            for t in missing_in_registry
        ],
        "manual_or_gated": [
            {"token": t, "mentions": unresolved_counts.get(t, 0)}
            for t in sorted(manual_or_gated)
        ],
    }

    report = {
        "generated_at": now_rfc3339(),
        "source": {
            "guide_export": remote_snapshot.source,
            "guide_fetched": remote_snapshot.fetched,
            "guide_fetch_error": remote_snapshot.fetch_error,
            "local_snapshot_path": str(local_snapshot.source) if local_snapshot else None,
            "registry_path": str(DEFAULT_REGISTRY),
            "alias_doc_path": str(DEFAULT_ALIAS_DOC),
        },
        "guide": {
            "line_count": remote_snapshot.line_count,
            "sha256": remote_snapshot.sha256,
            "revision_raw": revision_raw,
            "revision_iso_utc": revision_dt.isoformat().replace("+00:00", "Z")
            if revision_dt
            else None,
        },
        "local_snapshot": {
            "exists": local_snapshot is not None,
            "line_count": local_snapshot.line_count if local_snapshot else 0,
            "sha256": local_snapshot.sha256 if local_snapshot else None,
        },
        "diff": diff,
        "divergence": {
            "days_since_guide_revision": age_days,
            "missing_in_registry_count": len(missing_in_registry),
            "manual_or_gated_count": len(manual_or_gated),
            "guide_candidates_unresolved_count": len(unresolved),
        },
        "coverage_summary": {
            "guide_candidates_total": len(guide_tokens),
            "resolved_total": len(resolved_map),
            "missing_in_registry_total": len(missing_in_registry),
            "manual_or_gated_total": len(manual_or_gated),
        },
    }

    # Keep a small human-readable diff preview for quick CI logs.
    diff_preview_lines = []
    if local_snapshot:
        raw_diff = difflib.unified_diff(
            local_snapshot.text.splitlines(),
            remote_snapshot.text.splitlines(),
            fromfile="local_snapshot",
            tofile="remote_guide",
            n=2,
            lineterm="",
        )
        for i, line in enumerate(raw_diff):
            if i >= 240:
                diff_preview_lines.append("... diff preview truncated ...")
                break
            diff_preview_lines.append(line)
    diff_preview_text = "\n".join(diff_preview_lines) + ("\n" if diff_preview_lines else "")

    return report, coverage_report, diff_preview_text


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sync Google guide export and generate model coverage + drift reports."
    )
    ap.add_argument("--doc-id", default=DEFAULT_DOC_ID, help="Google doc id")
    ap.add_argument(
        "--export-url",
        default=None,
        help="Explicit export URL (overrides --doc-id).",
    )
    ap.add_argument(
        "--local-snapshot",
        default=str(DEFAULT_LOCAL_SNAPSHOT),
        help="Path to local snapshot txt used for diff fallback.",
    )
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to v2 registry source JSON.",
    )
    ap.add_argument(
        "--alias-doc",
        default=str(DEFAULT_ALIAS_DOC),
        help="Path to markdown model-id mapping doc.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated reports.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=45,
        help="Network timeout for guide fetch.",
    )
    ap.add_argument(
        "--allow-fetch-failure",
        action="store_true",
        help="If guide fetch fails, use local snapshot and continue.",
    )
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=14,
        help="CI gate threshold for guide revision age.",
    )
    ap.add_argument(
        "--max-unsynced-models",
        type=int,
        default=20,
        help="CI gate threshold for unresolved non-manual guide model candidates.",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Exit with code 1 when thresholds are exceeded.",
    )
    ap.add_argument(
        "--write-remote-snapshot",
        action="store_true",
        help="Write fetched remote guide text into output directory.",
    )
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    export_url = args.export_url or f"https://docs.google.com/document/d/{args.doc_id}/export?format=txt"
    local_snapshot_path = Path(args.local_snapshot)
    registry_path = Path(args.registry)
    alias_doc_path = Path(args.alias_doc)
    output_dir = Path(args.output_dir)

    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    remote_snapshot = load_remote_or_fallback(
        export_url=export_url,
        local_snapshot_path=local_snapshot_path,
        timeout_sec=args.timeout_sec,
        allow_fetch_failure=args.allow_fetch_failure,
    )
    local_snapshot = read_local_snapshot(local_snapshot_path)

    registry_ids = load_registry_model_ids(registry_path)
    alias_map, known_manual_only = parse_alias_doc(alias_doc_path)

    report, coverage_report, diff_preview = build_report(
        remote_snapshot=remote_snapshot,
        local_snapshot=local_snapshot,
        registry_ids=registry_ids,
        alias_map=alias_map,
        known_manual_only=known_manual_only,
    )

    gate_status, gate_reasons = gate_report(
        report,
        max_age_days=args.max_age_days,
        max_unsynced_models=args.max_unsynced_models,
    )
    report["gate"] = {
        "enabled": bool(args.gate),
        "status": gate_status,
        "reasons": gate_reasons,
        "thresholds": {
            "max_age_days": args.max_age_days,
            "max_unsynced_models": args.max_unsynced_models,
        },
    }

    write_json(output_dir / "guide_sync_report.json", report)
    write_json(output_dir / "model_coverage_report.json", coverage_report)
    (output_dir / "guide_diff_preview.patch").write_text(diff_preview, encoding="utf-8")

    if args.write_remote_snapshot:
        (output_dir / "guide_remote_snapshot.txt").write_text(
            remote_snapshot.text, encoding="utf-8"
        )

    # Console summary for CI logs.
    print(
        json.dumps(
            {
                "gate_status": gate_status,
                "guide_fetched": remote_snapshot.fetched,
                "guide_line_count": report["guide"]["line_count"],
                "missing_in_registry_count": report["divergence"]["missing_in_registry_count"],
                "manual_or_gated_count": report["divergence"]["manual_or_gated_count"],
                "days_since_guide_revision": report["divergence"]["days_since_guide_revision"],
                "report": str(output_dir / "guide_sync_report.json"),
                "coverage": str(output_dir / "model_coverage_report.json"),
            },
            ensure_ascii=False,
        )
    )

    if args.gate and gate_status == "fail":
        for reason in gate_reasons:
            print(f"GATE_FAIL: {reason}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
