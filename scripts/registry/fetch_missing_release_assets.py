#!/usr/bin/env python3
"""
Download missing model assets into the GitHub Release staging folder.

This script is designed to support the "snapshot + deterministic filenames" mirroring workflow.

High-level behavior:
- Reads `reports/mirror_upload_manifest.json` to learn which deterministic filenames are required.
- Reads `reports/link_audit.json` (and optionally `reports/hf_link_suggestions.json`) to find
  candidate upstream URLs for each required artifact kind.
- Compares required filenames to what's already in the target GitHub release (optional).
- Downloads missing files into `reports/release_assets_stage/` using deterministic filenames.
- Uses a HEAD preflight by default (recommended) to avoid waiting on large downloads that will fail.
- Supports redirects commonly used by Hugging Face and GitHub.

Notes / assumptions:
- This script only stages files locally. Uploading to GitHub Releases is handled separately.
- Some items may not be resolvable automatically if they are absent from `link_audit.json`
  and no explicit mapping is provided; the script will report those as unresolved.
- Hugging Face "resolve" URLs often redirect to signed blob URLs; allowing redirects is required.

Examples:
  python scripts/registry/fetch_missing_release_assets.py

  python scripts/registry/fetch_missing_release_assets.py --skip-release-check
  python scripts/registry/fetch_missing_release_assets.py --no-head
  python scripts/registry/fetch_missing_release_assets.py --max-bytes 500000000

Exit codes:
  0 if all required assets are present or successfully downloaded
  1 if any required assets remain missing/unresolved
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "reports" / "mirror_upload_manifest.json"
DEFAULT_LINK_AUDIT = REPO_ROOT / "reports" / "link_audit.json"
DEFAULT_HF_SUGGESTIONS = REPO_ROOT / "reports" / "hf_link_suggestions.json"
DEFAULT_STAGE_DIR = REPO_ROOT / "reports" / "release_assets_stage"

DEFAULT_RELEASE_REPO = "engdahlz/stemsep-models"
DEFAULT_RELEASE_TAG = "models-v2-2026-01-01"


@dataclass(frozen=True)
class Need:
    model_id: str
    kind: str  # "checkpoint" or "config"
    filename: str


@dataclass(frozen=True)
class CandidateURL:
    url: str
    source: str  # e.g. "link_audit", "link_audit_final", "hf_suggestion"
    notes: str = ""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _http_session(user_agent: str, timeout_s: int) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
        }
    )
    # requests timeout is per request, passed per call.
    return s


def _head_ok(
    sess: requests.Session,
    url: str,
    timeout_s: int,
    allow_redirects: bool,
) -> Tuple[bool, Optional[int], Optional[str], Optional[int], Optional[str]]:
    """
    Returns (ok, status_code, final_url, content_length, content_type).
    """
    try:
        r = sess.head(url, timeout=timeout_s, allow_redirects=allow_redirects)
        status = r.status_code
        final_url = r.url
        ct = r.headers.get("Content-Type")
        cl = r.headers.get("Content-Length")
        content_length = int(cl) if cl and cl.isdigit() else None
        ok = 200 <= status < 300
        return ok, status, final_url, content_length, ct
    except requests.RequestException as e:
        return False, None, None, None, f"HEAD error: {e!s}"


def _download_to_file(
    sess: requests.Session,
    url: str,
    out_path: Path,
    timeout_s: int,
    allow_redirects: bool,
    max_bytes: Optional[int],
    chunk_size: int = 1024 * 1024,
) -> Tuple[bool, Optional[int], Optional[str], Optional[int], Optional[str]]:
    """
    Stream download to a temp file then move into place.
    Returns (ok, status_code, final_url, bytes_written, error).
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    # Handle .part files: Clean up old partials to start fresh
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError as e:
            # If we can't delete the part file, we probably can't write to it either.
            return False, None, None, 0, f"Could not clear existing .part file: {e}"

    start_time = time.time()
    last_print = start_time
    last_bytes = 0

    try:
        print(f"    Starting download: {url}")
        with sess.get(
            url, timeout=timeout_s, allow_redirects=allow_redirects, stream=True
        ) as r:
            status = r.status_code
            final_url = r.url
            if not (200 <= status < 300):
                return False, status, final_url, None, f"GET status {status}"

            total = 0
            _safe_mkdir(out_path.parent)

            # Watchdog
            last_activity = time.time()
            watchdog_timeout = 300  # 5 minutes without data

            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    # Watchdog check
                    now = time.time()
                    if now - last_activity > watchdog_timeout:
                        return (
                            False,
                            status,
                            final_url,
                            total,
                            f"Watchdog timeout: no data for {watchdog_timeout}s",
                        )
                    last_activity = now

                    f.write(chunk)
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        return (
                            False,
                            status,
                            final_url,
                            total,
                            f"Exceeded --max-bytes ({max_bytes})",
                        )

                    # Progress log every ~5 seconds
                    if now - last_print > 5.0:
                        diff_time = now - last_print
                        diff_bytes = total - last_bytes
                        speed_mb = (
                            (diff_bytes / (1024 * 1024)) / diff_time
                            if diff_time > 0
                            else 0
                        )
                        print(
                            f"    ... {total / (1024 * 1024):.1f} MB downloaded ({speed_mb:.2f} MB/s)"
                        )
                        last_print = now
                        last_bytes = total

            print(
                f"    Download complete. Total: {total / (1024 * 1024):.2f} MB. Moving to final path..."
            )

            # On Windows, antivirus/indexers/etc can hold a transient lock on the .part file.
            # Retry a few times instead of crashing the entire run.
            max_rename_attempts = 20
            for attempt in range(1, max_rename_attempts + 1):
                try:
                    tmp_path.replace(out_path)
                    print(f"    Successfully saved to {out_path.name}")
                    break
                except PermissionError as e:
                    if attempt >= max_rename_attempts:
                        return (
                            False,
                            status,
                            final_url,
                            total,
                            f"Rename failed (file locked) after {max_rename_attempts} attempts: {e!s}",
                        )
                    print(
                        f"    [WinError 32] File locked, retrying {attempt}/{max_rename_attempts}..."
                    )
                    time.sleep(0.5 * attempt)  # Backoff

            return True, status, final_url, total, None
    except requests.RequestException as e:
        return False, None, None, None, f"GET error: {e!s}"
    finally:
        # best-effort cleanup if something failed mid-stream
        if tmp_path.exists() and not out_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _parse_manifest_needs(manifest: Dict[str, Any]) -> List[Need]:
    needs: List[Need] = []
    models = manifest.get("models") or []
    for m in models:
        model_id = m.get("model_id")
        for n in m.get("needs") or []:
            needs.append(
                Need(
                    model_id=model_id,
                    kind=n.get("kind"),
                    filename=n.get("filename"),
                )
            )
    # deterministic order
    return sorted(needs, key=lambda x: (x.model_id, x.kind, x.filename))


def _load_link_audit_candidates(
    link_audit: Dict[str, Any],
) -> Dict[Tuple[str, str], List[CandidateURL]]:
    """
    Builds candidates keyed by (model_id, kind) from reports/link_audit.json.

    We include:
    - original `url`
    - (if present) `final_url` from the audit. This may be a signed URL for HF/GitHub assets,
      which is time-limited. It's useful for immediate runs but shouldn't be written back to a registry.
    """
    out: Dict[Tuple[str, str], List[CandidateURL]] = {}
    for r in link_audit.get("results") or []:
        model_id = r.get("model_id")
        kind = r.get("kind")
        url = r.get("url")
        final_url = r.get("final_url")
        ok = r.get("ok")
        status = r.get("status")

        if not model_id or not kind:
            continue

        key = (model_id, kind)
        out.setdefault(key, [])

        if url:
            out[key].append(
                CandidateURL(
                    url=url, source="link_audit", notes=f"ok={ok} status={status}"
                )
            )
        if final_url:
            out[key].append(
                CandidateURL(
                    url=final_url,
                    source="link_audit_final",
                    notes="time-limited final_url",
                )
            )
    return out


def _load_hf_suggestions(
    hf_suggestions: Dict[str, Any],
) -> Dict[Tuple[str, str], List[CandidateURL]]:
    """
    Builds candidates keyed by (model_id, kind) from reports/hf_link_suggestions.json.

    Uses the suggested best_match_path to build a corrected HF resolve URL.
    """
    out: Dict[Tuple[str, str], List[CandidateURL]] = {}
    for s in hf_suggestions.get("suggestions") or []:
        model_id = s.get("model_id")
        kind = s.get("kind")
        repo = s.get("repo")
        rev = s.get("rev") or "main"
        best_path = s.get("best_match_path")
        status = s.get("status")
        if not model_id or not kind or not repo or not best_path:
            continue

        # Construct a HF resolve URL
        # Example: https://huggingface.co/<repo>/resolve/<rev>/<path>
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{best_path}"
        key = (model_id, kind)
        out.setdefault(key, []).append(
            CandidateURL(
                url=url, source="hf_suggestion", notes=f"suggested_from_status={status}"
            )
        )
    return out


def _github_release_asset_names(repo: str, tag: str) -> Optional[set]:
    """
    Query GitHub API for release assets to avoid downloading things already present on the mirror.

    Unauthenticated GitHub API is rate-limited; if it fails we return None and proceed without this check.

    Note: This checks the mirror release contents, not local staging contents.
    """
    api = f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag)}"
    try:
        r = requests.get(api, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        assets = data.get("assets") or []
        return set(a.get("name") for a in assets if a.get("name"))
    except requests.RequestException:
        return None
    except ValueError:
        return None


def _pick_candidates_for_need(
    need: Need,
    by_model_kind: Dict[Tuple[str, str], List[CandidateURL]],
    prefer_sources: List[str],
) -> List[CandidateURL]:
    """
    Return candidates in a preference order:
    - prefer_sources order
    - then stable ordering by url
    """
    cands = list(by_model_kind.get((need.model_id, need.kind), []))
    if not cands:
        return []

    pref_rank = {s: i for i, s in enumerate(prefer_sources)}

    def key(c: CandidateURL) -> Tuple[int, str]:
        return (pref_rank.get(c.source, 999), c.url)

    return sorted(cands, key=key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to reports/mirror_upload_manifest.json",
    )
    ap.add_argument(
        "--link-audit",
        default=str(DEFAULT_LINK_AUDIT),
        help="Path to reports/link_audit.json",
    )
    ap.add_argument(
        "--hf-suggestions",
        default=str(DEFAULT_HF_SUGGESTIONS),
        help="Path to reports/hf_link_suggestions.json (optional)",
    )
    ap.add_argument(
        "--stage-dir",
        default=str(DEFAULT_STAGE_DIR),
        help="Output directory for staged assets",
    )
    ap.add_argument(
        "--release-repo",
        default=DEFAULT_RELEASE_REPO,
        help="Mirror repo, e.g. engdahlz/stemsep-models",
    )
    ap.add_argument(
        "--release-tag",
        default=DEFAULT_RELEASE_TAG,
        help="Mirror release tag (snapshot)",
    )
    ap.add_argument(
        "--skip-release-check",
        action="store_true",
        help="Do not query GitHub Release assets before staging downloads",
    )
    ap.add_argument(
        "--no-head",
        action="store_true",
        help="Skip HEAD preflight (not recommended); attempt downloads directly",
    )
    ap.add_argument(
        "--allow-redirects",
        action="store_true",
        default=True,
        help="Allow redirects (required for HF and GitHub release assets). Default true.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60)",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Safety limit: abort a download if it exceeds this many bytes",
    )
    ap.add_argument(
        "--user-agent",
        default="StemSepRegistryFetcher/1.0",
        help="User-Agent header for HTTP requests",
    )
    ap.add_argument(
        "--prefer",
        default="link_audit,hf_suggestion,link_audit_final",
        help="Comma-separated source preference order (default: link_audit,hf_suggestion,link_audit_final)",
    )
    ap.add_argument(
        "--report-out",
        default=str(REPO_ROOT / "reports" / "fetch_missing_release_assets_report.json"),
        help="Where to write a JSON report with results",
    )

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    link_audit_path = Path(args.link_audit)
    hf_suggestions_path = Path(args.hf_suggestions)
    stage_dir = Path(args.stage_dir)
    report_out = Path(args.report_out)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not link_audit_path.exists():
        print(f"ERROR: link audit not found: {link_audit_path}", file=sys.stderr)
        return 1

    manifest = _read_json(manifest_path)
    needs = _parse_manifest_needs(manifest)

    link_audit = _read_json(link_audit_path)
    by_model_kind = _load_link_audit_candidates(link_audit)

    if hf_suggestions_path.exists():
        hf_suggestions = _read_json(hf_suggestions_path)
        hf_map = _load_hf_suggestions(hf_suggestions)
        for k, v in hf_map.items():
            by_model_kind.setdefault(k, []).extend(v)

    release_names: Optional[set] = None
    if not args.skip_release_check:
        release_names = _github_release_asset_names(args.release_repo, args.release_tag)

    sess = _http_session(user_agent=args.user_agent, timeout_s=args.timeout)
    prefer_sources = [p.strip() for p in str(args.prefer).split(",") if p.strip()]

    _safe_mkdir(stage_dir)

    print("=" * 60)
    print("StemSep Asset Downloader")
    print("=" * 60)
    print(f"CWD: {os.getcwd()}")
    print(f"Manifest: {manifest_path}")
    print(f"Link Audit: {link_audit_path}")
    print(f"Stage Dir: {stage_dir}")
    print(f"Target Release: {args.release_repo} @ {args.release_tag}")
    print(f"Check Release Content: {not args.skip_release_check}")
    print(f"Allow Redirects: {args.allow_redirects}")
    print("=" * 60)

    results: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "link_audit": str(link_audit_path),
        "hf_suggestions": str(hf_suggestions_path)
        if hf_suggestions_path.exists()
        else None,
        "stage_dir": str(stage_dir),
        "release_repo": args.release_repo,
        "release_tag": args.release_tag,
        "release_check_enabled": not args.skip_release_check,
        "release_asset_count": len(release_names)
        if release_names is not None
        else None,
        "needs_total": len(needs),
        "staged_already": [],
        "skipped_present_in_release": [],
        "downloaded": [],
        "failed": [],
        "unresolved": [],
        "started_at": time.time(),
        "finished_at": None,
    }

    any_missing = False

    for i, need in enumerate(needs):
        out_path = stage_dir / need.filename

        prefix = (
            f"[{i + 1}/{len(needs)}] {need.model_id}/{need.kind} -> {need.filename}"
        )

        # If already staged locally, skip.
        if out_path.exists() and out_path.is_file():
            print(f"{prefix}: ALREADY STAGED")
            results["staged_already"].append(
                {
                    "model_id": need.model_id,
                    "kind": need.kind,
                    "filename": need.filename,
                    "path": str(out_path),
                    "sha256": _sha256_file(out_path),
                    "bytes": out_path.stat().st_size,
                }
            )
            continue

        # Optional: if the mirror release already contains the deterministic filename, skip staging
        if release_names is not None and need.filename in release_names:
            print(f"{prefix}: PRESENT IN RELEASE (Skipping)")
            results["skipped_present_in_release"].append(
                {
                    "model_id": need.model_id,
                    "kind": need.kind,
                    "filename": need.filename,
                    "reason": "present_in_release",
                }
            )
            continue

        candidates = _pick_candidates_for_need(need, by_model_kind, prefer_sources)
        if not candidates:
            print(f"{prefix}: NO CANDIDATE URLS")
            any_missing = True
            results["unresolved"].append(
                {
                    "model_id": need.model_id,
                    "kind": need.kind,
                    "filename": need.filename,
                    "reason": "no_candidate_urls",
                }
            )
            continue

        chosen: Optional[CandidateURL] = None
        head_info: Optional[Dict[str, Any]] = None

        print(f"{prefix}: Searching candidates...")

        # HEAD preflight unless disabled
        if not args.no_head:
            for c in candidates:
                ok, status, final_url, content_length, content_type = _head_ok(
                    sess,
                    c.url,
                    timeout_s=args.timeout,
                    allow_redirects=args.allow_redirects,
                )
                head_info = {
                    "ok": ok,
                    "status": status,
                    "final_url": final_url,
                    "content_length": content_length,
                    "content_type": content_type,
                    "candidate_source": c.source,
                    "candidate_url": c.url,
                    "candidate_notes": c.notes,
                }
                if ok:
                    # optional size guard
                    if (
                        args.max_bytes is not None
                        and content_length is not None
                        and content_length > args.max_bytes
                    ):
                        print(
                            f"    Skipping candidate {c.url}: size {content_length} > max {args.max_bytes}"
                        )
                        continue
                    chosen = c
                    print(f"    Selected candidate: {c.url} (size: {content_length})")
                    break
                else:
                    print(f"    Candidate failed HEAD: {c.url} ({status})")

            if chosen is None:
                print(f"{prefix}: ALL CANDIDATES FAILED HEAD")
                any_missing = True
                results["failed"].append(
                    {
                        "model_id": need.model_id,
                        "kind": need.kind,
                        "filename": need.filename,
                        "reason": "no_head_ok_candidate",
                        "candidates": [c.__dict__ for c in candidates],
                        "last_head": head_info,
                    }
                )
                continue
        else:
            chosen = candidates[0]
            print(f"    Selected candidate (no HEAD): {chosen.url}")

        # Download
        ok, status, final_url, bytes_written, err = _download_to_file(
            sess,
            chosen.url,
            out_path,
            timeout_s=args.timeout,
            allow_redirects=args.allow_redirects,
            max_bytes=args.max_bytes,
        )

        if ok:
            print(f"{prefix}: SUCCESS")
            results["downloaded"].append(
                {
                    "model_id": need.model_id,
                    "kind": need.kind,
                    "filename": need.filename,
                    "url": chosen.url,
                    "final_url": final_url,
                    "status": status,
                    "bytes": bytes_written,
                    "sha256": _sha256_file(out_path),
                    "source": chosen.source,
                    "notes": chosen.notes,
                    "head": head_info,
                }
            )
        else:
            print(f"{prefix}: FAILED - {err}")
            any_missing = True
            # remove any partial file if created
            if out_path.exists() and out_path.stat().st_size == 0:
                try:
                    out_path.unlink()
                except OSError:
                    pass

            results["failed"].append(
                {
                    "model_id": need.model_id,
                    "kind": need.kind,
                    "filename": need.filename,
                    "url": chosen.url if chosen else None,
                    "final_url": final_url,
                    "status": status,
                    "bytes": bytes_written,
                    "error": err,
                    "source": chosen.source if chosen else None,
                    "notes": chosen.notes if chosen else None,
                    "head": head_info,
                }
            )

    results["finished_at"] = time.time()

    _safe_mkdir(report_out.parent)
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Wrote report: {report_out}")
    print(f"Stage dir: {stage_dir}")
    print(
        "Summary:",
        f"needs={results['needs_total']}",
        f"staged_already={len(results['staged_already'])}",
        f"skipped_present_in_release={len(results['skipped_present_in_release'])}",
        f"downloaded={len(results['downloaded'])}",
        f"failed={len(results['failed'])}",
        f"unresolved={len(results['unresolved'])}",
    )

    # Return non-zero if anything is still missing / unresolved / failed
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
