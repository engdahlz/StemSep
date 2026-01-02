"""Suggest fixes for failing Hugging Face URLs in reports/link_audit.json.

Reads the link audit output (from scripts/registry/audit_model_links.py) and, for any failing
Hugging Face `resolve/...` URLs, queries the Hugging Face model API to list files
(siblings) and suggests the closest matching filename.

This is intended as a helper for updating StemSepApp/assets/models.json.bak.

Usage:
  python scripts/registry/suggest_hf_link_fixes.py --audit reports/link_audit.json --out reports/hf_link_suggestions.json


Auth:
  Optionally set one of STEMSEP_HF_TOKEN / HF_TOKEN / HUGGINGFACE_HUB_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

HF_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def _token_from_env() -> Optional[str]:
    for k in ("STEMSEP_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def _http_get_json(url: str, token: Optional[str], timeout_s: float = 15.0) -> Any:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "StemSepHFLinkSuggest/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class Suggestion:
    model_id: str
    kind: str
    status: Optional[int]
    url: str
    repo: str
    rev: str
    requested_path: str
    best_match_path: Optional[str]
    best_match_score: float
    reason: str


def suggest_for_url(
    model_id: str, kind: str, status: Optional[int], url: str, token: Optional[str]
) -> Optional[Suggestion]:
    m = HF_RESOLVE_RE.match(url)
    if not m:
        return None

    repo = m.group("repo")
    rev = m.group("rev")
    requested_path = m.group("path")

    api_url = f"https://huggingface.co/api/models/{repo}"
    try:
        meta = _http_get_json(api_url, token=token)
    except Exception as e:
        return Suggestion(
            model_id=model_id,
            kind=kind,
            status=status,
            url=url,
            repo=repo,
            rev=rev,
            requested_path=requested_path,
            best_match_path=None,
            best_match_score=0.0,
            reason=f"Failed to fetch model API: {type(e).__name__}: {e}",
        )

    siblings = meta.get("siblings") if isinstance(meta, dict) else None
    if not isinstance(siblings, list):
        return Suggestion(
            model_id=model_id,
            kind=kind,
            status=status,
            url=url,
            repo=repo,
            rev=rev,
            requested_path=requested_path,
            best_match_path=None,
            best_match_score=0.0,
            reason="No siblings list in API response",
        )

    files: List[str] = []
    for s in siblings:
        if isinstance(s, dict) and isinstance(s.get("rfilename"), str):
            files.append(s["rfilename"])

    if not files:
        return Suggestion(
            model_id=model_id,
            kind=kind,
            status=status,
            url=url,
            repo=repo,
            rev=rev,
            requested_path=requested_path,
            best_match_path=None,
            best_match_score=0.0,
            reason="No rfilename entries in siblings",
        )

    # Prefer exact basename match (ignoring directories) if possible.
    req_base = requested_path.split("/")[-1]
    base_matches = [f for f in files if f.split("/")[-1].lower() == req_base.lower()]
    if base_matches:
        # If multiple, pick the one that matches directory best.
        best = max(base_matches, key=lambda f: _similarity(requested_path, f))
        return Suggestion(
            model_id=model_id,
            kind=kind,
            status=status,
            url=url,
            repo=repo,
            rev=rev,
            requested_path=requested_path,
            best_match_path=best,
            best_match_score=1.0,
            reason="Exact basename match found",
        )

    # Otherwise, pick best similarity across all file paths.
    scored: List[Tuple[float, str]] = [
        (_similarity(requested_path, f), f) for f in files
    ]
    scored.sort(reverse=True)
    best_score, best_file = scored[0]

    return Suggestion(
        model_id=model_id,
        kind=kind,
        status=status,
        url=url,
        repo=repo,
        rev=rev,
        requested_path=requested_path,
        best_match_path=best_file,
        best_match_score=float(best_score),
        reason="Best fuzzy match",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit", default="reports/link_audit.json", help="Path to link audit json"
    )
    ap.add_argument(
        "--out",
        default="reports/hf_link_suggestions.json",
        help="Write suggestions json",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.65,
        help="Minimum similarity score to include",
    )
    args = ap.parse_args()

    audit_path = args.audit
    if not os.path.exists(audit_path):
        print(f"ERROR: audit not found: {audit_path}", file=sys.stderr)
        return 2

    audit = json.load(open(audit_path, "r", encoding="utf-8"))
    results = audit.get("results", []) if isinstance(audit, dict) else []

    token = _token_from_env()

    suggestions: List[Suggestion] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("ok") is True:
            continue
        url = r.get("url")
        if not (isinstance(url, str) and "huggingface.co" in url):
            continue

        s = suggest_for_url(
            model_id=str(r.get("model_id") or ""),
            kind=str(r.get("kind") or ""),
            status=r.get("status"),
            url=url,
            token=token,
        )
        if not s:
            continue
        if s.best_match_path is None:
            suggestions.append(s)
            continue
        if s.best_match_score >= args.min_score:
            suggestions.append(s)

    # Write output
    out_payload = {
        "audit": audit_path,
        "count": len(suggestions),
        "suggestions": [asdict(s) for s in suggestions],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Wrote {len(suggestions)} suggestions to {args.out}")

    # Print quick summary for interactive use
    by_model: Dict[str, List[Suggestion]] = {}
    for s in suggestions:
        by_model.setdefault(s.model_id, []).append(s)

    for mid in sorted(by_model.keys()):
        print(f"\n{mid}:")
        for s in sorted(
            by_model[mid], key=lambda x: (x.kind, -(x.best_match_score or 0.0))
        ):
            if s.best_match_path:
                fixed = f"https://huggingface.co/{s.repo}/resolve/{s.rev}/{s.best_match_path}"
                print(f"  - {s.kind} {s.status}: {s.url}")
                print(f"    -> {fixed} (score={s.best_match_score:.2f}, {s.reason})")
            else:
                print(f"  - {s.kind} {s.status}: {s.url}")
                print(f"    -> no suggestion ({s.reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
