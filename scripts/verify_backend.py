import subprocess
import json
import time
import os
import sys
import threading
from pathlib import Path

# Configuration
# Adjust extension based on OS if needed, but user context implies Windows (.exe)
BACKEND_EXE = Path("stemsep-backend/target/debug/stemsep-backend.exe")
if not BACKEND_EXE.exists() and sys.platform != "win32":
    BACKEND_EXE = Path("stemsep-backend/target/debug/stemsep-backend")

MODELS_DIR = Path("test_models_dir")
ASSETS_DIR = Path("StemSepApp/assets")
MODEL_ID = "bs-roformer-viperx-1297" # Large enough to test pause/resume

def read_stdout(process, queue):
    """Reads stdout line by line and puts it in a queue."""
    for line in iter(process.stdout.readline, b''):
        try:
            line_str = line.decode('utf-8').strip()
            if line_str:
                queue.append(json.loads(line_str))
        except Exception as e:
            print(f"Error parsing line: {line}: {e}", file=sys.stderr)
    process.stdout.close()

def send_command(process, cmd_data):
    """Sends a JSONL command to stdin."""
    line = json.dumps(cmd_data) + "\n"
    process.stdin.write(line.encode('utf-8'))
    process.stdin.flush()
    print(f"-> Sent: {cmd_data['command']}")

def find_event(queue, condition):
    """Finds an event in the queue matching a condition."""
    # Search from the end backwards might be more efficient for 'latest', 
    # but for state transitions we usually want the first occurrence after we started waiting.
    # Since we append to a list, we just iterate.
    # Ideally we would consume the queue, but for this simple test list is fine.
    for i, event in enumerate(queue):
        if condition(event):
            return event
    return None

def main():
    if not BACKEND_EXE.exists():
        print(f"Error: Backend executable not found at {BACKEND_EXE}")
        sys.exit(1)

    # Ensure clean state
    if MODELS_DIR.exists():
        import shutil
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting backend: {BACKEND_EXE}")
    # Force Rust backend logic (disable python proxy) to test the fix
    env = os.environ.copy()
    env["STEMSEP_PROXY_PYTHON"] = "0"
    
    # Run from workspace root so relative paths work if needed
    process = subprocess.Popen(
        [str(BACKEND_EXE), "--assets-dir", str(ASSETS_DIR), "--models-dir", str(MODELS_DIR)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # We might want to see stderr
        bufsize=0, # Unbuffered
        env=env
    )

    response_queue = []
    stdout_thread = threading.Thread(target=read_stdout, args=(process, response_queue))
    stdout_thread.daemon = True
    stdout_thread.start()

    try:
        # Wait for bridge_ready
        print("Waiting for bridge_ready...")
        time.sleep(2)
        ready = find_event(response_queue, lambda e: e.get("type") == "bridge_ready")
        if not ready:
            print("FAILED: Did not receive bridge_ready event")
            sys.exit(1)
        print("Backend ready.")

        # 1. Start Download
        print(f"\n[Step 1] Starting download for {MODEL_ID}...")
        send_command(process, {
            "id": 1,
            "command": "download_model",
            "model_id": MODEL_ID
        })

        # Wait for some progress
        print("Waiting for download to progress...")
        
        # Poll for progress
        download_started = False
        for _ in range(20):
            time.sleep(0.5)
            progress_events = [e for e in response_queue if e.get("type") == "progress" and e.get("model_id") == MODEL_ID]
            if progress_events:
                last_pct = progress_events[-1].get("progress", 0)
                if last_pct > 0:
                    print(f"Download progress: {last_pct}%")
                    download_started = True
                    break
        
        if not download_started:
             # Check for generic progress 0 (Starting)
             progress_events = [e for e in response_queue if e.get("type") == "progress" and e.get("model_id") == MODEL_ID]
             if not progress_events:
                 print("FAILED: No progress events received within timeout")
                 print("Received events dump:")
                 print(json.dumps(response_queue, indent=2))
                 sys.exit(1)
             print("Download started (0%). Waiting a bit more for data...")
             time.sleep(3)

        # Check file existence
        # The backend creates a .part file inside the model dir
        part_files = list(MODELS_DIR.glob("*.part"))
        if not part_files:
            print(f"FAILED: No .part file found in {MODELS_DIR}")
            sys.exit(1)
        
        part_file = part_files[0]
        size_before_pause = part_file.stat().st_size
        print(f"Found partial file: {part_file.name} ({size_before_pause} bytes)")

        if size_before_pause == 0:
             print("FAILED: Partial file size is 0")
             sys.exit(1)

        # 2. Pause Download
        print(f"\n[Step 2] Pausing download...")
        send_command(process, {
            "id": 2,
            "command": "pause_download",
            "model_id": MODEL_ID
        })
        
        # Wait a bit to ensure it actually pauses and file handle is potentially released/flushed
        time.sleep(2)

        # 3. Resume Download
        print(f"\n[Step 3] Resuming download...")
        send_command(process, {
            "id": 3,
            "command": "resume_download",
            "model_id": MODEL_ID
        })

        print("Waiting for download to continue...")
        time.sleep(5)

        size_after_resume = part_file.stat().st_size
        print(f"File size after resume: {size_after_resume} bytes")

        if size_after_resume <= size_before_pause:
            print(f"FAILED: File did not grow after resume. Before: {size_before_pause}, After: {size_after_resume}")
            sys.exit(1)
        else:
            print(f"SUCCESS: File grew by {size_after_resume - size_before_pause} bytes.")

        # 4. Remove Model (and cancel download implicitly or explicitly)
        print(f"\n[Step 4] Removing model (while potentially active)...")
        # Clear queue to catch new events easier if needed, though we use ID matching
        send_command(process, {
            "id": 4,
            "command": "remove_model",
            "model_id": MODEL_ID
        })

        # Wait for removal
        time.sleep(2) 

        # Check if file is gone
        if part_file.exists():
             print(f"FAILED: .part file still exists: {part_file}")
             # Check if we got a success response for remove_model
             remove_response = find_event(response_queue, lambda e: e.get("id") == 4)
             print(f"Remove response: {remove_response}")
             sys.exit(1)
        
        # Also check if the final file exists (unlikely, but good to check)
        final_file = MODELS_DIR / part_file.stem # remove .part
        if final_file.exists():
            print(f"FAILED: Final file exists (unexpected): {final_file}")
            sys.exit(1)

        print("SUCCESS: File successfully removed.")
        print("\nVerification PASSED: Resume worked and file deletion succeeded.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Terminating backend...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Cleanup test dir
        import shutil
        if MODELS_DIR.exists():
             try:
                shutil.rmtree(MODELS_DIR)
             except Exception as e:
                 print(f"Warning: Failed to clean up {MODELS_DIR}: {e}")

if __name__ == "__main__":
    main()