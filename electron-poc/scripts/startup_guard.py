import subprocess
import time
import sys
import os
import re
import threading
import signal

# Configuration
PORT = 5173
PROJECT_DIR = os.path.join(os.getcwd(), "electron-poc")
NPM_CMD = "npm.cmd" if sys.platform == "win32" else "npm"

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def kill_process_on_port(port):
    """Finds and kills process blocking the port (Windows specific for now)"""
    log(f"Checking port {port}...")
    try:
        # Find PID
        cmd = f"netstat -ano | findstr :{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if not result.stdout:
            log(f"Port {port} is free.")
            return

        lines = result.stdout.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.split()
            if len(parts) > 4:
                pid = parts[-1]
                if pid != "0":
                    pids.add(pid)
        
        for pid in pids:
            log(f"Killing blocking PID {pid}...", "WARN")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    except Exception as e:
        log(f"Failed to clear port: {e}", "ERROR")

def kill_zombies():
    """Kills lingering python and electron processes"""
    log("Sweeping for zombie processes...")
    subprocess.run("taskkill /F /IM python.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("taskkill /F /IM electron.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def monitor_output(process):
    """Reads output line by line and checks for errors"""
    success_markers = ["App is ready", "VITE v", "Local:   http://localhost"]
    failure_markers = ["Error:", "Exception", "exited with code 1", "EADDRINUSE"]
    
    vite_ready = False
    electron_ready = False
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='') # Echo to console
        
        # Analysis
        if "VITE v" in line and "ready in" in line:
            vite_ready = True
            log("✅ Vite Server Ready", "SUCCESS")
            
        if "App is ready" in line:
            electron_ready = True
            log("✅ Electron Main Process Ready", "SUCCESS")
            
        if any(fail in line for fail in failure_markers):
            log(f"❌ DETECTED FAILURE: {line.strip()}", "CRITICAL")
            # Don't kill immediately, let user see, but mark as failed
            
    return vite_ready and electron_ready

def main():
    log("--- STARTUP GUARD ACTIVATED ---")
    
    # 1. Clean environment
    kill_zombies()
    kill_process_on_port(PORT)
    
    # 2. Start Application
    log("Starting Application...")
    try:
        process = subprocess.Popen(
            [NPM_CMD, "run", "electron:dev"], 
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 3. Monitor
        monitor_output(process)
        
    except KeyboardInterrupt:
        log("Stopping...", "INFO")
        process.terminate()

if __name__ == "__main__":
    main()
