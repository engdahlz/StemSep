"""Audit model download links in StemSep registries.

This script reads a registry JSON (supports StemSepApp/assets/models.json.bak format)
extracts all URLs (links.checkpoint + links.config), and probes them with a cheap
HTTP request (HEAD when possible, otherwise GET with Range: bytes=0-0).

Goal: identify which models are publicly downloadable (no auth) vs gated/private
(401/403) vs broken links.

Usage:
  python scripts/registry/audit_model_links.py --registry StemSepApp/assets/models.json.bak
  python scripts/registry/audit_model_links.py --registry StemSepApp/assets/models.json.bak --out link_audit.json

Optional:
  Set HF_TOKEN / HUGGINGFACE_HUB_TOKEN / STEMSEP_HF_TOKEN to re-check 401/403
  Hugging Face URLs with auth and report whether token resolves it.

Notes:
- This does not download model weights; it only fetches headers / first byte.
- Some hosts may block HEAD; script will fall back to a ranged GET.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class LinkItem:
    model_id: str
    kind: str  # checkpoint | config | other
    url: str


@dataclass
class ProbeResult:
    model_id: str
    kind: str
    url: str
    ok: bool
    status: Optional[int]
    final_url: Optional[str]
    error: Optional[str]
    elapsed_ms: int
    requires_auth_without_token: bool = False
    ok_with_token: Optional[bool] = None
    status_with_token: Optional[int] = None


def _is_http_url(s: Any) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    try:
        u = urllib.parse.urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_links_from_models_json_bak(data: Any) -> List[LinkItem]:
    out: List[LinkItem] = []
    models = []
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        models = data["models"]

    for m in models:
        if not isinstance(m, dict):
            continue
        model_id = str(m.get("id") or "")
        links = m.get("links") if isinstance(m.get("links"), dict) else {}
        ckpt = links.get("checkpoint")
        cfg = links.get("config")

        if _is_http_url(ckpt):
            out.append(LinkItem(model_id=model_id, kind="checkpoint", url=str(ckpt)))
        if _is_http_url(cfg):
            out.append(LinkItem(model_id=model_id, kind="config", url=str(cfg)))

    return out


def _hf_token_from_env() -> Optional[str]:
    for key in ("STEMSEP_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    return None


def _looks_like_hf(url: str) -> bool:
    return "huggingface.co" in url or "cdn-lfs.huggingface.co" in url


def _probe_url(
    url: str, timeout_s: float, bearer_token: Optional[str]
) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """Return (ok, status, final_url, error)."""

    # 1) HEAD (cheap) with redirects.
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "StemSepLinkAudit/1.0")
    if bearer_token and _looks_like_hf(url):
        req.add_header("Authorization", f"Bearer {bearer_token}")

    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())

    try:
        with opener.open(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", None) or getattr(resp, "code", None)
            final_url = getattr(resp, "url", None)
            ok = status is not None and 200 <= int(status) < 300
            return ok, int(status) if status is not None else None, final_url, None
    except urllib.error.HTTPError as e:
        # Some servers reject HEAD with 403/405; fall back to ranged GET.
        status = int(getattr(e, "code", 0) or 0) or None
        if status in (403, 405, 400):
            pass
        else:
            return False, status, getattr(e, "url", None), f"HTTPError: {e}"
    except Exception as e:
        # Network errors: also try GET for better signal.
        pass

    # 2) GET with Range to avoid downloading.
    req2 = urllib.request.Request(url, method="GET")
    req2.add_header("User-Agent", "StemSepLinkAudit/1.0")
    req2.add_header("Range", "bytes=0-0")
    if bearer_token and _looks_like_hf(url):
        req2.add_header("Authorization", f"Bearer {bearer_token}")

    try:
        with opener.open(req2, timeout=timeout_s) as resp2:
            status2 = getattr(resp2, "status", None) or getattr(resp2, "code", None)
            final_url2 = getattr(resp2, "url", None)
            ok2 = status2 is not None and 200 <= int(status2) < 300
            # Consume at most 1 byte.
            try:
                resp2.read(1)
            except Exception:
                pass
            return ok2, int(status2) if status2 is not None else None, final_url2, None
    except urllib.error.HTTPError as e2:
        status2 = int(getattr(e2, "code", 0) or 0) or None
        return False, status2, getattr(e2, "url", None), f"HTTPError: {e2}"
    except Exception as e2:
        return False, None, None, f"Error: {type(e2).__name__}: {e2}"


def probe_item(item: LinkItem, timeout_s: float, token: Optional[str]) -> ProbeResult:
    t0 = time.perf_counter()
    ok, status, final_url, err = _probe_url(
        item.url, timeout_s=timeout_s, bearer_token=None
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    res = ProbeResult(
        model_id=item.model_id,
        kind=item.kind,
        url=item.url,
        ok=ok,
        status=status,
        final_url=final_url,
        error=err,
        elapsed_ms=elapsed_ms,
    )

    if status in (401, 403) and _looks_like_hf(item.url):
        res.requires_auth_without_token = True
        if token:
            t1 = time.perf_counter()
            ok2, status2, final_url2, err2 = _probe_url(
                item.url, timeout_s=timeout_s, bearer_token=token
            )
            elapsed2 = int((time.perf_counter() - t1) * 1000)
            # Prefer token-probe details, but keep original final_url/error.
            res.ok_with_token = ok2
            res.status_with_token = status2
            # If token probe succeeded, update final_url for clarity.
            if ok2 and final_url2:
                res.final_url = final_url2
            # If token probe failed with a better error, append it.
            if err2 and err2 != res.error:
                res.error = (
                    (res.error or "").strip()
                    + ("; " if res.error else "")
                    + f"with_token: {err2} ({elapsed2}ms)"
                )

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        required=True,
        help="Path to registry JSON (e.g. StemSepApp/assets/models.json.bak)",
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0, help="Per-request timeout seconds"
    )
    ap.add_argument("--workers", type=int, default=12, help="Concurrent probes")
    ap.add_argument("--out", default="", help="Write full JSON report to this path")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 2

    data = _read_json(registry_path)
    items = _extract_links_from_models_json_bak(data)
    if not items:
        print("No URLs found in registry (unexpected).")
        return 1

    token = _hf_token_from_env()

    results: List[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(probe_item, it, args.timeout, token) for it in items]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Summary
    total = len(results)
    ok_n = sum(1 for r in results if r.ok)
    by_status: Dict[str, int] = {}
    for r in results:
        key = str(r.status) if r.status is not None else "ERR"
        by_status[key] = by_status.get(key, 0) + 1

    gated = [r for r in results if r.requires_auth_without_token]

    print(f"Checked {total} URLs")
    print(f"OK: {ok_n}  Fail: {total - ok_n}")
    print("Status breakdown:")
    for k in sorted(
        by_status.keys(), key=lambda x: (x == "ERR", int(x) if x.isdigit() else 9999, x)
    ):
        print(f"  {k}: {by_status[k]}")

    if gated:
        print(f"\nHugging Face auth-required (401/403) without token: {len(gated)}")
        # Show unique model_ids for faster action.
        seen = set()
        for r in sorted(gated, key=lambda x: (x.model_id, x.kind)):
            if r.model_id in seen:
                continue
            seen.add(r.model_id)
            suffix = ""
            if r.ok_with_token is True:
                suffix = " (OK with token)"
            elif r.ok_with_token is False:
                suffix = f" (still failing with token: {r.status_with_token})"
            print(f"  - {r.model_id}{suffix}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registry": str(registry_path),
            "checked": total,
            "ok": ok_n,
            "fail": total - ok_n,
            "has_token": bool(token),
            "results": [
                asdict(r)
                for r in sorted(
                    results, key=lambda x: (x.ok, x.status or 0, x.model_id, x.kind)
                )
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
