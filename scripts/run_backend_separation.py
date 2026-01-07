import argparse
import json
import os
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path


def _read_stderr(stream):
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if line:
            print(f"[backend:stderr] {line}", file=sys.stderr, flush=True)


def _send(proc: subprocess.Popen, obj: dict):
    line = json.dumps(obj, ensure_ascii=False)
    assert proc.stdin is not None
    proc.stdin.write((line + "\n").encode("utf-8"))
    proc.stdin.flush()


def _start_stdout_reader(proc: subprocess.Popen):
    assert proc.stdout is not None
    q: "queue.Queue[dict]" = queue.Queue()

    def _reader():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                print(f"[backend:raw] {line}", flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q


def _read_json_from_queue(proc: subprocess.Popen, q: "queue.Queue[dict]", timeout_s: float | None = None):
    try:
        return q.get(timeout=timeout_s)
    except queue.Empty:
        if proc.poll() is not None:
            raise RuntimeError("Backend exited unexpectedly while waiting for output")
        raise TimeoutError("Timed out waiting for backend output")


def _read_response(proc: subprocess.Popen, q: "queue.Queue[dict]", expected_id: int, timeout_s: float):
    """Read messages until we get a response envelope with the given id.

    The backend may emit asynchronous events (e.g. recipe_plan) interleaved with
    the request/response stream. This helper filters those out.
    """
    deadline = time.time() + timeout_s
    while True:
        remaining = max(0.1, deadline - time.time())
        msg = _read_json_from_queue(proc, q, timeout_s=remaining)
        # Responses have an explicit id field.
        if msg.get("id") == expected_id:
            return msg

        # Print/ignore asynchronous events while waiting for the response.
        t = msg.get("type")
        if t:
            print(f"event: {t} - {json.dumps(msg, ensure_ascii=False)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run a headless separation via stemsep-backend JSONL protocol")
    parser.add_argument("--backend", required=False, default=str(Path("stemsep-backend") / "target" / "release" / "stemsep-backend.exe"))
    parser.add_argument("--assets-dir", required=False, default=str(Path("StemSepApp") / "assets"))
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")

    # Mode selection
    parser.add_argument("--model-id", default="ensemble")
    parser.add_argument("--stems", default="vocals,instrumental")

    # Ensemble config (used when model-id=ensemble)
    parser.add_argument("--ensemble-models", default="unwa-hyperace,bs-roformer-viperx-1297")
    parser.add_argument("--ensemble-algorithm", default="max_spec")

    parser.add_argument("--timeout-min", type=int, default=60)

    args = parser.parse_args()

    backend_path = Path(args.backend)
    assets_dir = Path(args.assets_dir)
    models_dir = Path(args.models_dir)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not backend_path.exists():
        raise SystemExit(f"Backend not found: {backend_path}")
    if not assets_dir.exists():
        raise SystemExit(f"Assets dir not found: {assets_dir}")
    if not models_dir.exists():
        raise SystemExit(f"Models dir not found: {models_dir}")
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    ensemble_models = [m.strip() for m in args.ensemble_models.split(",") if m.strip()]
    ensemble_config = None
    ensemble_algorithm = None
    if args.model_id == "ensemble":
        ensemble_config = {
            "models": [{"model_id": mid, "weight": 1.0} for mid in ensemble_models],
            "algorithm": args.ensemble_algorithm,
        }
        ensemble_algorithm = args.ensemble_algorithm

    cmd = [
        str(backend_path),
        "--assets-dir",
        str(assets_dir),
        "--models-dir",
        str(models_dir),
    ]

    print("Starting backend:", " ".join(cmd), flush=True)

    repo_root = Path(__file__).resolve().parent.parent
    python_bridge = repo_root / "electron-poc" / "python-bridge.py"
    inference_script = repo_root / "scripts" / "inference.py"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"

    env = {**os.environ}
    # NOTE: We want to run without the legacy python-bridge proxy.
    # The Rust backend has a native `separation_preflight` handler, and `separate_audio`
    # runs via `scripts/inference.py` directly.
    env.setdefault("STEMSEP_PROXY_PYTHON", "0")
    if python_bridge.exists():
        env.setdefault("STEMSEP_PYTHON_BRIDGE", str(python_bridge))
    if inference_script.exists():
        env.setdefault("STEMSEP_INFERENCE_SCRIPT", str(inference_script))
    if venv_python.exists():
        env.setdefault("STEMSEP_PYTHON", str(venv_python))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        cwd=str(repo_root),
        env=env,
    )

    assert proc.stderr is not None
    stderr_thread = threading.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True)
    stderr_thread.start()

    out_q = _start_stdout_reader(proc)

    try:
        # Wait for bridge_ready
        while True:
            msg = _read_json_from_queue(proc, out_q, timeout_s=30)
            if msg.get("type") == "bridge_ready":
                print("bridge_ready:", json.dumps(msg, ensure_ascii=False), flush=True)
                break

        # 1) Preflight
        preflight_req = {
            "command": "separation_preflight",
            "id": 1,
            "file_path": str(input_path),
            "model_id": args.model_id,
            "output_dir": str(output_dir),
            "stems": stems,
            "device": args.device,
            "ensemble_config": ensemble_config,
            "ensemble_algorithm": ensemble_algorithm,
            "output_format": "wav",
        }
        _send(proc, preflight_req)

        preflight_resp = _read_json_from_queue(proc, out_q, timeout_s=60)
        if preflight_resp.get("id") != 1:
            print("Unexpected preflight response:", preflight_resp, flush=True)
        if not preflight_resp.get("success"):
            raise RuntimeError(preflight_resp.get("error") or "Preflight failed")

        preflight_data = preflight_resp.get("data") or {}
        print("preflight:", json.dumps(preflight_data, ensure_ascii=False, indent=2), flush=True)

        if not preflight_data.get("can_proceed", False):
            raise SystemExit("Preflight says can_proceed=false (see errors above)")

        # 2) Start job
        sep_req = {
            "command": "separate_audio",
            "id": 2,
            "file_path": str(input_path),
            "model_id": args.model_id,
            "output_dir": str(output_dir),
            "stems": stems,
            "device": args.device,
            "ensemble_config": ensemble_config,
            "ensemble_algorithm": ensemble_algorithm,
            "output_format": "wav",
        }
        _send(proc, sep_req)

        sep_resp = _read_response(proc, out_q, expected_id=2, timeout_s=60)
        if not sep_resp.get("success"):
            raise RuntimeError(sep_resp.get("error") or "separate_audio failed")

        data = sep_resp.get("data") or {}
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"No job_id in response: {sep_resp}")

        print(f"Job started: {job_id}", flush=True)

        # 3) Stream events until completion
        deadline = time.time() + (args.timeout_min * 60)
        last_progress_print = 0.0

        last_heartbeat = 0.0
        while time.time() < deadline:
            try:
                msg = _read_json_from_queue(proc, out_q, timeout_s=5)
            except TimeoutError:
                # No events for a bit; print a heartbeat occasionally.
                now = time.time()
                if now - last_heartbeat >= 30:
                    last_heartbeat = now
                    print("(waiting for backend events...)", flush=True)
                continue

            t = msg.get("type")
            mid = msg.get("job_id")
            if mid and mid != job_id:
                continue

            if t == "separation_progress":
                # throttle a bit
                now = time.time()
                if now - last_progress_print >= 0.5:
                    last_progress_print = now
                    p = msg.get("progress")
                    m = msg.get("message")
                    print(f"progress: {p} - {m}", flush=True)

            elif t == "separation_complete":
                files = msg.get("output_files") or {}
                print("complete:", json.dumps(files, ensure_ascii=False, indent=2), flush=True)
                print("Done.", flush=True)
                return 0

            elif t == "separation_error":
                raise RuntimeError(msg.get("error") or "separation_error")

        raise TimeoutError(f"Timed out waiting for completion after {args.timeout_min} minutes")

    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
