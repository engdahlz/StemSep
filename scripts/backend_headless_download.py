#!/usr/bin/env python3
"""Headless model downloader for stemsep-backend (JSONL protocol).

Purpose: download a single model without Electron UI, with clear progress + log capture.

Usage (PowerShell):
  python scripts\backend_headless_download.py \
    --model-id anvuew-dereverb-room \
    --models-dir "D:\\StemSep Models"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import queue
from typing import Any, Optional


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _default_backend_exe() -> pathlib.Path:
    root = _repo_root()
    return root / "stemsep-backend" / "target" / "release" / "stemsep-backend.exe"


def _safe_mkdir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Model ID to download")
    parser.add_argument("--models-dir", required=False, help="Models directory")
    parser.add_argument("--assets-dir", required=False, help="Assets directory (StemSepApp/assets)")
    parser.add_argument("--output-dir", required=False, help="Directory for logs")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Optional hard timeout")

    args = parser.parse_args()

    models_dir = args.models_dir or os.environ.get("STEMSEP_MODELS_DIR")
    if not models_dir:
        if pathlib.Path("D:/StemSep Models").exists():
            models_dir = "D:/StemSep Models"

    backend_exe = _default_backend_exe()
    if not backend_exe.exists():
        print(f"ERROR: backend exe not found: {backend_exe}", file=sys.stderr)
        print("Hint: build it with: cd stemsep-backend; cargo build --release", file=sys.stderr)
        return 2

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = _repo_root() / "_backend_out" / f"download-{args.model_id}-{stamp}"

    _safe_mkdir(output_dir)

    events_log_path = output_dir / "events.jsonl"
    stderr_log_path = output_dir / "backend_stderr.log"

    events_f = events_log_path.open("a", encoding="utf-8")
    stderr_f = stderr_log_path.open("a", encoding="utf-8")

    cmd = [str(backend_exe)]
    if models_dir:
        cmd += ["--models-dir", str(models_dir)]
    if args.assets_dir:
        cmd += ["--assets-dir", str(args.assets_dir)]

    child_env = os.environ.copy()
    # Force Rust-native code paths for download_model in stemsep-backend.
    # This avoids needing python-bridge.py (the Python proxy) for downloads.
    child_env.setdefault("STEMSEP_PREFER_RUST_SEPARATION", "1")
    child_env.setdefault("STEMSEP_PROXY_PYTHON", "0")

    # Keep Python wiring available for commands that still use it in other contexts.
    if "STEMSEP_PYTHON" not in child_env:
        venv_python = _repo_root() / "StemSepApp" / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            child_env["STEMSEP_PYTHON"] = str(venv_python)

    if "STEMSEP_PYTHON_BRIDGE" not in child_env:
        bridge = _repo_root() / "electron-poc" / "python-bridge.py"
        if bridge.exists():
            child_env["STEMSEP_PYTHON_BRIDGE"] = str(bridge)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_q: "queue.Queue[str]" = queue.Queue()
    stderr_lines: list[str] = []

    def _pump_stdout() -> None:
        try:
            for raw in proc.stdout:  # type: ignore[assignment]
                stdout_q.put(raw)
        except Exception:
            return

    def _pump_stderr() -> None:
        try:
            for raw in proc.stderr:  # type: ignore[assignment]
                stderr_lines.append(raw)
                try:
                    stderr_f.write(raw)
                    stderr_f.flush()
                except Exception:
                    pass
                if len(stderr_lines) > 2000:
                    del stderr_lines[:500]
        except Exception:
            return

    threading.Thread(target=_pump_stdout, daemon=True).start()
    threading.Thread(target=_pump_stderr, daemon=True).start()

    req = {"id": 1, "command": "download_model", "model_id": args.model_id}
    proc.stdin.write(_json_dumps(req) + "\n")
    proc.stdin.flush()

    deadline: Optional[float] = None
    if args.timeout_seconds and args.timeout_seconds > 0:
        deadline = time.time() + args.timeout_seconds

    try:
        started_at = time.time()
        last_print = 0.0
        got_response = False

        while True:
            if deadline and time.time() > deadline:
                raise TimeoutError(f"Timed out after {args.timeout_seconds} seconds")

            try:
                line = stdout_q.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                # heartbeat
                now = time.time()
                if now - last_print >= 10.0:
                    last_print = now
                    print(f"...downloading {args.model_id} (elapsed {now - started_at:.0f}s)")
                continue

            line = line.strip()
            if not line:
                continue

            try:
                events_f.write(line + "\n")
                events_f.flush()
            except Exception:
                pass

            try:
                msg = json.loads(line)
            except Exception:
                continue

            if isinstance(msg, dict) and msg.get("id") == 1 and "success" in msg:
                got_response = True
                print(f"Response: {msg}")
                if not msg.get("success"):
                    raise RuntimeError(msg.get("error") or "download failed")
                continue

            if not isinstance(msg, dict):
                continue

            mtype = msg.get("type")
            if mtype in ("download_progress", "progress") and msg.get("model_id") == args.model_id:
                progress = msg.get("progress")
                if progress is not None:
                    print(f"{args.model_id}: {progress}%")
                continue

            if mtype in ("download_complete", "complete") and msg.get("model_id") == args.model_id:
                print("Download complete")
                print(f"Logs: {output_dir}")
                return 0

            if mtype in ("download_error", "error") and msg.get("model_id") == args.model_id:
                raise RuntimeError(msg.get("error") or "download_error")

    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass

        if stderr_lines:
            print("\n--- backend stderr (tail) ---\n" + "".join(stderr_lines[-200:]), file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"Logs: {output_dir}", file=sys.stderr)
        return 1
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            events_f.close()
        except Exception:
            pass
        try:
            stderr_f.close()
        except Exception:
            pass

    if not got_response:
        print("ERROR: backend exited without a response", file=sys.stderr)
    else:
        print("ERROR: backend exited before download_complete", file=sys.stderr)

    if stderr_lines:
        print("\n--- backend stderr (tail) ---\n" + "".join(stderr_lines[-200:]), file=sys.stderr)

    try:
        proc.kill()
    except Exception:
        pass

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
