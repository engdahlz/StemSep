import subprocess
import json
import time
import os
import sys
import threading
from pathlib import Path

BACKEND_EXE = Path("stemsep-backend/target/debug/stemsep-backend.exe")
if not BACKEND_EXE.exists() and sys.platform != "win32":
    BACKEND_EXE = Path("stemsep-backend/target/debug/stemsep-backend")

MODELS_DIR = Path("test_models_dir")
ASSETS_DIR = Path("StemSepApp/assets")
SELECTION_ID = "bs-roformer-viperx-1297"
SELECTION_TYPE = "model"


def read_stdout(process, queue):
    for line in iter(process.stdout.readline, b""):
        try:
            line_str = line.decode("utf-8").strip()
            if line_str:
                queue.append(json.loads(line_str))
        except Exception as e:
            print(f"Error parsing line: {line}: {e}", file=sys.stderr)
    process.stdout.close()


def send_command(process, cmd_data):
    line = json.dumps(cmd_data) + "\n"
    process.stdin.write(line.encode("utf-8"))
    process.stdin.flush()
    print(f"-> Sent: {cmd_data['command']}")


def find_event(queue, condition):
    for event in queue:
        if condition(event):
            return event
    return None


def main():
    if not BACKEND_EXE.exists():
        print(f"Error: Backend executable not found at {BACKEND_EXE}")
        sys.exit(1)

    if MODELS_DIR.exists():
        import shutil

        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting backend: {BACKEND_EXE}")
    env = os.environ.copy()
    env["STEMSEP_PROXY_PYTHON"] = "0"

    process = subprocess.Popen(
        [str(BACKEND_EXE), "--assets-dir", str(ASSETS_DIR), "--models-dir", str(MODELS_DIR)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
    )

    response_queue = []
    stdout_thread = threading.Thread(target=read_stdout, args=(process, response_queue))
    stdout_thread.daemon = True
    stdout_thread.start()

    try:
        print("Waiting for bridge_ready...")
        time.sleep(2)
        ready = find_event(response_queue, lambda e: e.get("type") == "bridge_ready")
        if not ready:
            print("FAILED: Did not receive bridge_ready event")
            sys.exit(1)
        print("Backend ready.")

        print(f"\n[Step 1] Resolving install plan for {SELECTION_TYPE}/{SELECTION_ID}...")
        send_command(
            process,
            {"id": 1, "command": "resolve_install_plan", "selection_type": SELECTION_TYPE, "selection_id": SELECTION_ID},
        )
        time.sleep(1)
        plan_response = find_event(response_queue, lambda e: e.get("id") == 1 and e.get("success") is True)
        if not plan_response:
            print("FAILED: resolve_install_plan did not succeed")
            sys.exit(1)

        print(f"\n[Step 2] Installing selection {SELECTION_TYPE}/{SELECTION_ID}...")
        send_command(
            process,
            {"id": 2, "command": "install_selection", "selection_type": SELECTION_TYPE, "selection_id": SELECTION_ID},
        )
        time.sleep(2)

        install_response = find_event(response_queue, lambda e: e.get("id") == 2 and e.get("success") is True)
        if not install_response:
            print("FAILED: install_selection did not succeed")
            sys.exit(1)

        print(f"\n[Step 3] Reading installation status...")
        send_command(
            process,
            {"id": 3, "command": "get_selection_installation", "selection_type": SELECTION_TYPE, "selection_id": SELECTION_ID},
        )
        time.sleep(1)
        installation = find_event(response_queue, lambda e: e.get("id") == 3 and e.get("success") is True)
        if not installation:
            print("FAILED: get_selection_installation did not succeed")
            sys.exit(1)

        print("SUCCESS: selection-first install path is reachable.")
        print(json.dumps(installation.get("data"), indent=2))

    finally:
        print("Terminating backend...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
