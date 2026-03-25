#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from catalog_v3_common import ASSETS_DIR, now_rfc3339, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_MANIFEST = REPO_ROOT / "stemsep-backend" / "Cargo.toml"
DEFAULT_REPORT = REPO_ROOT / "StemSep-catalog" / "reports" / "provider_acceptance.json"
DEFAULT_CATALOG_REPO_ROOT = REPO_ROOT.parent / "StemSep-catalog"

ACCEPTANCE_CASES: list[dict[str, Any]] = [
    {
        "provider": "huggingface",
        "selection_type": "model",
        "selection_id": "anvuew-dereverb-room",
        "required_model_ids": ["anvuew-dereverb-room"],
        "expect_install": True,
    },
    {
        "provider": "github_release",
        "selection_type": "model",
        "selection_id": "bs-roformer-viperx-1297",
        "required_model_ids": ["bs-roformer-viperx-1297"],
        "expect_install": True,
    },
    {
        "provider": "github_raw",
        "selection_type": "model",
        "selection_id": "htdemucs",
        "required_model_ids": ["htdemucs"],
        "expect_install": True,
    },
    {
        "provider": "google_drive",
        "selection_type": "model",
        "selection_id": "drypd-side-extraction",
        "required_model_ids": ["drypd-side-extraction"],
        "expect_install": True,
    },
    {
        "provider": "proton_drive",
        "selection_type": "source_reference",
        "selection_id": "proton-rifforge-metal-roformer-reference",
        "required_model_ids": [],
        "expect_install": False,
        "expected_status": "blocked_external",
        "reason": "Known Proton share from the source list is no longer resolvable to a deterministic download URL.",
    },
]


def detect_backend_binary(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in (
        REPO_ROOT / "stemsep-backend" / "target" / "release" / "stemsep-backend.exe",
        REPO_ROOT / "stemsep-backend" / "target" / "debug" / "stemsep-backend.exe",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate stemsep-backend.exe under target/(release|debug).")


class BackendSession:
    def __init__(
        self,
        backend_path: Path,
        models_dir: Path,
        assets_dir: Path,
        *,
        catalog_url: str | None = None,
        catalog_sig_url: str | None = None,
    ) -> None:
        env = dict(os.environ)
        env.setdefault("STEMSEP_PROXY_PYTHON", "0")
        env.setdefault("STEMSEP_PREFER_RUST_SEPARATION", "1")
        if catalog_url:
            env["STEMSEP_CATALOG_URL"] = catalog_url
        if catalog_sig_url:
            env["STEMSEP_CATALOG_SIG_URL"] = catalog_sig_url
        self.proc = subprocess.Popen(
            [
                str(backend_path),
                "--assets-dir",
                str(assets_dir),
                "--models-dir",
                str(models_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._next_id = 1
        self.events: list[dict[str, Any]] = []
        self._wait_for_bridge_ready()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _wait_for_bridge_ready(self) -> None:
        deadline = time.time() + 20
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            payload = json.loads(line)
            if payload.get("type") == "bridge_ready":
                return
            self.events.append(payload)
        stderr = ""
        if self.proc.stderr is not None:
            try:
                stderr = self.proc.stderr.read()
            except Exception:
                stderr = ""
        raise RuntimeError(f"Backend did not emit bridge_ready. stderr={stderr[:1000]}")

    def request(self, command: str, **extra: Any) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        payload = {"id": req_id, "command": command, **extra}
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Backend exited while waiting for response to {command}")
            message = json.loads(line)
            if message.get("id") == req_id:
                if message.get("success") is False:
                    raise RuntimeError(
                        f"{command} failed: {message.get('error') or message.get('message') or message}"
                    )
                return message.get("data") or {}
            self.events.append(message)

    def wait_for_install(
        self,
        selection_type: str,
        selection_id: str,
        required_model_ids: list[str],
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            installation = self.request(
                "get_selection_installation",
                selection_type=selection_type,
                selection_id=selection_id,
            )
            if installation.get("installed") and installation.get("canonical_ready"):
                return installation
            time.sleep(1.0)
        verification = self.request(
            "verify_selection_artifacts",
            selection_type=selection_type,
            selection_id=selection_id,
        )
        raise TimeoutError(
            f"Timed out waiting for installation of {selection_type}:{selection_id}. "
            f"Verification={json.dumps(verification)[:2000]}"
        )


class LocalCatalogServer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None

    def start(self) -> None:
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
            *args,
            directory=str(self.root),
            **kwargs,
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)


def run_acceptance_case(session: BackendSession, case: dict[str, Any]) -> dict[str, Any]:
    provider = case["provider"]
    if not case.get("expect_install", True):
        return {
            "provider": provider,
            "selection_type": case["selection_type"],
            "selection_id": case["selection_id"],
            "status": case.get("expected_status", "blocked_external"),
            "reason": case.get("reason"),
            "performed_at": now_rfc3339(),
        }

    selection_type = case["selection_type"]
    selection_id = case["selection_id"]
    install_plan = session.request(
        "resolve_install_plan",
        selection_type=selection_type,
        selection_id=selection_id,
    )
    install_result = session.request(
        "install_selection",
        selection_type=selection_type,
        selection_id=selection_id,
    )
    installation = session.wait_for_install(
        selection_type,
        selection_id,
        case.get("required_model_ids") or [],
    )
    verification = session.request(
        "verify_selection_artifacts",
        selection_type=selection_type,
        selection_id=selection_id,
    )
    required_models = install_plan.get("required_models") or []
    return {
        "provider": provider,
        "selection_type": selection_type,
        "selection_id": selection_id,
        "status": "installed",
        "performed_at": now_rfc3339(),
        "required_model_ids": case.get("required_model_ids") or [],
        "required_models": required_models,
        "install_result": install_result,
        "installation": installation,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run live provider acceptance for selection-first catalog installs."
    )
    parser.add_argument("--backend")
    parser.add_argument("--assets-dir", default=str(ASSETS_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--catalog-repo-root", default=str(DEFAULT_CATALOG_REPO_ROOT))
    parser.add_argument("--use-local-catalog-server", action="store_true")
    args = parser.parse_args()

    backend_path = detect_backend_binary(args.backend)
    assets_dir = Path(args.assets_dir)
    catalog_repo_root = Path(args.catalog_repo_root)
    local_catalog_server = None
    catalog_url = None
    catalog_sig_url = None
    if args.use_local_catalog_server:
        local_catalog_server = LocalCatalogServer(catalog_repo_root)
        local_catalog_server.start()
        catalog_url = f"{local_catalog_server.base_url}/catalog.runtime.remote.json"
        catalog_sig_url = f"{local_catalog_server.base_url}/catalog.runtime.remote.json.sig"

    with tempfile.TemporaryDirectory(prefix="stemsep_provider_acceptance_") as tmp_dir:
        models_dir = Path(tmp_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        session = BackendSession(
            backend_path,
            models_dir,
            assets_dir,
            catalog_url=catalog_url,
            catalog_sig_url=catalog_sig_url,
        )
        try:
            catalog_status = session.request("get_catalog_status")
            results = [run_acceptance_case(session, case) for case in ACCEPTANCE_CASES]
        finally:
            session.close()
            if local_catalog_server:
                local_catalog_server.close()

        payload = {
            "performed_at": now_rfc3339(),
            "backend": str(backend_path),
            "assets_dir": str(assets_dir),
            "catalog_url": catalog_url,
            "catalog_sig_url": catalog_sig_url,
            "catalog_status": catalog_status,
            "results": results,
            "summary": {
                "installed": sum(1 for item in results if item.get("status") == "installed"),
                "blocked_external": sum(
                    1 for item in results if item.get("status") == "blocked_external"
                ),
            },
        }
        write_json(Path(args.report), payload)
        print(f"Wrote provider acceptance report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
