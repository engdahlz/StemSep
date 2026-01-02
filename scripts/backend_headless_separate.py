#!/usr/bin/env python3
"""Headless separation runner for stemsep-backend (JSONL protocol).

Purpose: run a single separation job without the Electron UI, for faster debugging.

Usage (PowerShell):
  python scripts\backend_headless_separate.py \
    --input "C:\\path\\to\\file.mp3" \
    --output-dir "C:\\path\\to\\out" \
    --models-dir "D:\\StemSep Models" \
    --model-id htdemucs \
    --stems vocals,instrumental
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import queue
from typing import Any, Dict, Optional


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
    parser.add_argument("--input", required=True, help="Input audio path")
    parser.add_argument("--output-dir", required=False, help="Output directory")
    parser.add_argument("--models-dir", required=False, help="Models directory")
    parser.add_argument("--assets-dir", required=False, help="Assets directory (StemSepApp/assets)")
    parser.add_argument("--model-id", default="htdemucs", help="Model ID (e.g. htdemucs, ensemble)")
    parser.add_argument(
        "--stems",
        default="vocals,instrumental",
        help="Comma-separated stems (e.g. vocals,instrumental)",
    )
    parser.add_argument("--device", default="cpu", help="Device string (cpu, cuda:0, etc.)")
    parser.add_argument("--segment-size", type=int, default=0, help="Segment size override (0 = default)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap override (0 = default)")
    parser.add_argument("--shifts", type=int, default=0, help="Shifts override (0 = default)")
    parser.add_argument("--tta", action="store_true", help="Enable TTA")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Optional hard timeout")

    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    models_dir = args.models_dir or os.environ.get("STEMSEP_MODELS_DIR")
    if not models_dir:
        # Common dev default used by the Electron main process in this repo.
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
        output_dir = _repo_root() / "_backend_out" / f"separation-{stamp}"

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

    # Ensure the backend uses the repo venv (avoids issues with system Python versions).
    child_env = os.environ.copy()
    if "STEMSEP_PYTHON" not in child_env:
        # Prefer the current interpreter (this script is typically run from the desired venv).
        if sys.executable:
            child_env["STEMSEP_PYTHON"] = sys.executable
        else:
            # Fallbacks for older setups.
            venv_python = _repo_root() / ".venv" / "Scripts" / "python.exe"
            if venv_python.exists():
                child_env["STEMSEP_PYTHON"] = str(venv_python)
            else:
                venv_python = _repo_root() / "StemSepApp" / ".venv" / "Scripts" / "python.exe"
                if venv_python.exists():
                    child_env["STEMSEP_PYTHON"] = str(venv_python)

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
                # Keep a small rolling buffer for error context
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

    t_out = threading.Thread(target=_pump_stdout, daemon=True)
    t_err = threading.Thread(target=_pump_stderr, daemon=True)
    t_out.start()
    t_err.start()

    # Build request payload using the Rust backend's contract.
    stems = [s.strip() for s in str(args.stems).split(",") if s.strip()]

    extra: Dict[str, Any] = {
        "file_path": str(input_path),
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "stems": stems,
        "device": args.device,
    }

    if args.segment_size and args.segment_size > 0:
        extra["segment_size"] = args.segment_size
    if args.overlap and args.overlap > 0:
        extra["overlap"] = args.overlap
    if args.shifts and args.shifts > 0:
        extra["shifts"] = args.shifts
    if args.tta:
        extra["tta"] = True

    req = {"id": 1, "command": "separate_audio", **extra}
    proc.stdin.write(_json_dumps(req) + "\n")
    proc.stdin.flush()

    job_id: Optional[str] = None
    deadline: Optional[float] = None
    if args.timeout_seconds and args.timeout_seconds > 0:
        deadline = time.time() + args.timeout_seconds

    # Read until we see completion/error for our job.
    try:
        started_at = time.time()
        last_heartbeat = 0.0
        while True:
            if deadline and time.time() > deadline:
                raise TimeoutError(f"Timed out after {args.timeout_seconds} seconds")

            # Heartbeat so long model-load phases don't look like a hang.
            now = time.time()
            if now - last_heartbeat >= 10.0:
                last_heartbeat = now
                elapsed = now - started_at
                if job_id:
                    print(f"[{job_id}] ...working (elapsed {elapsed:.0f}s)")
                else:
                    print(f"...waiting for job start (elapsed {elapsed:.0f}s)")

            try:
                line = stdout_q.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    break
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
                # Not a JSON line; ignore
                continue

            # Response envelope
            if isinstance(msg, dict) and "success" in msg and msg.get("id") == 1:
                if not msg.get("success"):
                    raise RuntimeError(msg.get("error") or "Unknown backend error")
                data = msg.get("data") or {}
                job_id = data.get("job_id")
                print(f"Started job_id={job_id}")
                continue

            # Events
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if not mtype:
                continue

            if mtype == "bridge_ready":
                continue

            if mtype == "error":
                # Generic IPC error from Python bootstrap (e.g. missing deps/import failure)
                raise RuntimeError(msg.get("error") or "inference error")

            if mtype == "separation_started":
                jid = msg.get("job_id")
                # If we already know our job id, ignore other jobs.
                if job_id and jid != job_id:
                    continue
                print(f"[{jid}] started")
                continue

            if mtype in ("separation_step_started", "separation_step_completed"):
                jid = msg.get("job_id")
                if job_id and jid != job_id:
                    continue
                meta = msg.get("meta") or {}
                phase = meta.get("phase") if isinstance(meta, dict) else None
                step = meta.get("step") if isinstance(meta, dict) else None
                status = msg.get("status")
                dur = msg.get("duration_seconds")
                label = f"{phase}/{step}" if phase or step else "step"
                if mtype.endswith("completed"):
                    extra = f" ({status})" if status else ""
                    if dur is not None:
                        extra = f"{extra} {dur}s".rstrip()
                    print(f"[{jid}] {label} completed{extra}")
                else:
                    print(f"[{jid}] {label} started")
                continue

            if mtype == "separation_progress":
                jid = msg.get("job_id")
                if job_id and jid != job_id:
                    continue
                progress = msg.get("progress")
                message = msg.get("message")
                if progress is not None:
                    try:
                        p = float(progress)
                        print(f"[{jid}] {p:.1f}% {message or ''}".rstrip())
                    except Exception:
                        print(f"[{jid}] {progress} {message or ''}".rstrip())
                else:
                    print(f"[{jid}] {message or 'progress'}")
                continue

            if mtype == "separation_complete":
                jid = msg.get("job_id")
                if job_id and jid != job_id:
                    continue
                output_files = msg.get("output_files") or {}
                copied: Dict[str, str] = {}
                if isinstance(output_files, dict):
                    for key, src in output_files.items():
                        if not src:
                            continue
                        src_path = pathlib.Path(str(src))
                        if not src_path.exists():
                            continue
                        # Preserve original stem filename when possible.
                        dst_path = output_dir / src_path.name
                        if dst_path.resolve() != src_path.resolve():
                            shutil.copy2(src_path, dst_path)
                        copied[str(key)] = str(dst_path)

                result = {
                    "job_id": jid,
                    "input": str(input_path),
                    "model_id": args.model_id,
                    "device": args.device,
                    "output_dir": str(output_dir),
                    "outputs": copied or output_files,
                }
                try:
                    (output_dir / "result.json").write_text(_json_dumps(result) + "\n", encoding="utf-8")
                except Exception:
                    pass

                print("Separation complete")
                print(f"Output dir: {output_dir}")
                print(_json_dumps(result.get("outputs")))
                return 0

            if mtype == "separation_error":
                jid = msg.get("job_id")
                if job_id and jid != job_id:
                    continue
                raise RuntimeError(msg.get("error") or "separation_error")

            if mtype == "separation_cancelled":
                jid = msg.get("job_id")
                if job_id and jid != job_id:
                    continue
                print("Separation cancelled")
                return 3

    except Exception as e:
        # Dump stderr for context
        try:
            proc.kill()
        except Exception:
            pass

        if stderr_lines:
            print("\n--- backend stderr (tail) ---\n" + "".join(stderr_lines[-200:]), file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
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

    # If we got here, backend exited without completion event.
    if stderr_lines:
        print("\n--- backend stderr (tail) ---\n" + "".join(stderr_lines[-200:]), file=sys.stderr)

    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass

    print("ERROR: backend exited before completion", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
