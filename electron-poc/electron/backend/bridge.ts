import { app, dialog, type BrowserWindow } from "electron";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import fs from "fs";
import path from "path";

type LogFn = (message: string, ...args: any[]) => void;

type CreateBackendBridgeArgs = {
  log: LogFn;
  getStoredHuggingFaceToken: () => string | null;
  getStoredModelsDir: () => string | null;
  resolveBundledPythonForRustBackend: () => string | null;
  shouldUseRustBackend: () => boolean;
  getIsAppQuitting: () => boolean;
  getMainWindow: () => BrowserWindow | null;
  isBackendBusy: () => boolean;
  onBackendMessage: (msg: any) => void;
};

type PendingBackendCommand = {
  command: string;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
  timeout: NodeJS.Timeout;
};

const MAX_BRIDGE_RESTARTS = 3;
const HEALTH_CHECK_INTERVAL_MS = 30_000;
const MAX_HEALTH_CHECK_FAILURES = 2;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function candidateRootsFromBackendModule() {
  const cwd = process.cwd();
  const packageRoot = cwd;
  const repoRoot = path.resolve(cwd, "..");
  const distRoot = path.resolve(__dirname, "..");
  const packageRootFromDist = path.resolve(__dirname, "../..");
  const repoRootFromDist = path.resolve(__dirname, "../../..");

  return Array.from(
    new Set([
      packageRoot,
      repoRoot,
      distRoot,
      packageRootFromDist,
      repoRootFromDist,
    ]),
  );
}

function findFirstExistingPath(candidates: string[]) {
  return (
    candidates.find((candidate) => {
      try {
        return fs.existsSync(candidate);
      } catch {
        return false;
      }
    }) || null
  );
}

export function createBackendBridge({
  log,
  getStoredHuggingFaceToken,
  getStoredModelsDir,
  resolveBundledPythonForRustBackend,
  shouldUseRustBackend,
  getIsAppQuitting,
  getMainWindow,
  isBackendBusy,
  onBackendMessage,
}: CreateBackendBridgeArgs) {
  let backendProcess: ChildProcessWithoutNullStreams | null = null;
  let pythonBridge: ChildProcessWithoutNullStreams | null = null;
  let bridgeRestartCount = 0;
  let lastBridgeRestart = 0;
  let manualBridgeRestartPending = false;
  let healthCheckInterval: NodeJS.Timeout | null = null;
  let consecutiveHealthCheckFailures = 0;
  let backendMessageRouter: ((data: Buffer) => void) | null = null;
  let commandIdCounter = 0;

  const pendingBackendCommands = new Map<string, PendingBackendCommand>();
  let bridgeReadyWaiters: Array<{
    resolve: () => void;
    reject: (reason?: any) => void;
    timeout: NodeJS.Timeout;
  }> = [];

  const getBackendStdioLogPath = () =>
    path.join(app.getPath("userData"), "backend-stdio.log");

  function rotateLogIfLarge(filePath: string, maxBytes: number) {
    try {
      if (!fs.existsSync(filePath)) return;
      const st = fs.statSync(filePath);
      if (!st.isFile()) return;
      if (st.size <= maxBytes) return;
      const dir = path.dirname(filePath);
      const base = path.basename(filePath, path.extname(filePath));
      const ext = path.extname(filePath) || ".log";
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      fs.renameSync(filePath, path.join(dir, `${base}.${stamp}${ext}`));
    } catch {
      // ignore
    }
  }

  function appendBackendStdio(prefix: string, chunk: any) {
    try {
      rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
      const ts = new Date().toISOString();
      const text =
        typeof chunk === "string"
          ? chunk
          : chunk?.toString?.("utf-8") ?? String(chunk);
      fs.appendFileSync(
        getBackendStdioLogPath(),
        `${ts} ${prefix} ${text}${text.endsWith("\n") ? "" : "\n"}`,
      );
    } catch {
      // ignore
    }
  }

  function attachBackendStdioLogging(
    label: string,
    process: ChildProcessWithoutNullStreams,
  ) {
    try {
      fs.mkdirSync(path.dirname(getBackendStdioLogPath()), { recursive: true });
      rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
    } catch {
      // ignore
    }

    process.stdout?.on("data", (data) => {
      appendBackendStdio(`[${label} stdout]`, data);
    });
    process.stderr?.on("data", (data) => {
      appendBackendStdio(`[${label} stderr]`, data);
    });
  }

  function rejectAllPendingBackendCommands(error: Error) {
    for (const [key, pending] of Array.from(pendingBackendCommands.entries())) {
      clearTimeout(pending.timeout);
      pendingBackendCommands.delete(key);
      pending.reject(error);
    }
  }

  function resolveBridgeReadyWaiters() {
    for (const waiter of bridgeReadyWaiters.splice(0)) {
      clearTimeout(waiter.timeout);
      waiter.resolve();
    }
  }

  function rejectBridgeReadyWaiters(error: Error) {
    for (const waiter of bridgeReadyWaiters.splice(0)) {
      clearTimeout(waiter.timeout);
      waiter.reject(error);
    }
  }

  function waitForBridgeReady(timeoutMs = 30_000): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        bridgeReadyWaiters = bridgeReadyWaiters.filter(
          (entry) => entry.resolve !== resolve,
        );
        reject(new Error(`Backend did not become ready within ${timeoutMs}ms`));
      }, timeoutMs);
      bridgeReadyWaiters.push({ resolve, reject, timeout });
    });
  }

  function createLineBuffer(onLine: (line: string) => void) {
    let buffer = "";
    return (data: Buffer) => {
      buffer += data.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.trim()) onLine(line);
      }
    };
  }

  function attachBackendMessageRouter(process: ChildProcessWithoutNullStreams) {
    if (backendMessageRouter) return;
    backendMessageRouter = createLineBuffer((line) => {
      try {
        const msg = JSON.parse(line);
        const hasResponseShape =
          msg &&
          Object.prototype.hasOwnProperty.call(msg, "id") &&
          typeof msg.success === "boolean";
        if (hasResponseShape) {
          const key = String(msg.id);
          const pending = pendingBackendCommands.get(key);
          if (pending) {
            clearTimeout(pending.timeout);
            pendingBackendCommands.delete(key);
            if (msg.success) pending.resolve(msg.data);
            else pending.reject(new Error(msg.error || "Unknown error from backend"));
          }
        }
        if (msg?.type === "bridge_ready") {
          resolveBridgeReadyWaiters();
        }
        onBackendMessage(msg);
      } catch {
        // Ignore non-JSON lines on stdout.
      }
    });
    process.stdout?.on("data", backendMessageRouter);
  }

  function detachBackendMessageRouter() {
    if (!backendMessageRouter) return;
    if (pythonBridge?.stdout) {
      pythonBridge.stdout.removeListener("data", backendMessageRouter);
    }
    backendMessageRouter = null;
  }

  function requestBridgeRestart(reason: string) {
    log("Manual backend restart requested:", reason);
    if (!backendProcess) {
      ensureBackend();
      return;
    }
    manualBridgeRestartPending = true;
    try {
      backendProcess.kill();
    } catch (e) {
      log("Failed to kill backend bridge:", e);
      backendProcess = null;
      ensureBackend();
    }
  }

  async function sendBackendCommand(
    command: string,
    payload: Record<string, any> = {},
    timeoutMs = 60_000,
  ): Promise<any> {
    const process = ensureBackend();
    if (!process) throw new Error("Backend not available");

    const cmdId = ++commandIdCounter;
    const cmdKey = String(cmdId);
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pendingBackendCommands.delete(cmdKey);
        reject(
          new Error(`Backend command '${command}' timed out after ${timeoutMs}ms`),
        );
      }, timeoutMs);

      pendingBackendCommands.set(cmdKey, {
        command,
        resolve,
        reject,
        timeout,
      });

      try {
        process.stdin?.write(
          JSON.stringify({ command, id: cmdId, ...payload }) + "\n",
        );
      } catch (e) {
        clearTimeout(timeout);
        pendingBackendCommands.delete(cmdKey);
        reject(new Error(`Failed to write command '${command}' to backend: ${e}`));
      }
    });
  }

  async function sendBackendCommandWithRetry(
    command: string,
    payload: Record<string, any> = {},
    timeoutMs = 60_000,
    maxRetries = 2,
  ): Promise<any> {
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          const delay = Math.min(1000 * attempt, 5000);
          log(
            `Retrying command '${command}' after ${delay}ms (attempt ${attempt + 1}/${maxRetries + 1})`,
          );
          await sleep(delay);
        }
        return await sendBackendCommand(command, payload, timeoutMs);
      } catch (error) {
        lastError = error as Error;
        const errorMsg = lastError.message.toLowerCase();
        if (
          errorMsg.includes("cancelled") ||
          errorMsg.includes("not found") ||
          errorMsg.includes("missing") ||
          errorMsg.includes("invalid")
        ) {
          throw lastError;
        }
        if (attempt < maxRetries) {
          log(
            `Command '${command}' failed (attempt ${attempt + 1}): ${lastError.message}`,
          );
        }
      }
    }
    throw lastError!;
  }

  async function restartBackendAndWait(reason: string, timeoutMs = 45_000): Promise<void> {
    const readyPromise = waitForBridgeReady(timeoutMs);
    requestBridgeRestart(reason);
    try {
      await readyPromise;
      return;
    } catch (eventError) {
      let lastError: unknown = eventError;
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        try {
          try {
            const status = await sendBackendCommand("get_backend_status", {}, 4000);
            if (status && status.backend_ready === false) {
              throw new Error("Backend reported not ready");
            }
          } catch {
            await sendBackendCommand("ping", {}, 4000);
          }
          return;
        } catch (error) {
          lastError = error;
          await sleep(400);
        }
      }
      throw lastError instanceof Error ? lastError : new Error(String(lastError));
    }
  }

  function shouldUseRustBinary(): boolean {
    const v = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
    if (v === "python") return false;
    if (v === "rust") return true;
    return shouldUseRustBackend();
  }

  function ensureBackend() {
    if (pythonBridge) return pythonBridge;

    const hfToken = getStoredHuggingFaceToken();
    const modelsDir = getStoredModelsDir();

    if (shouldUseRustBinary()) {
      const backendPref = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
      const rustExe = (() => {
        const candidates: string[] = [];
        const roots = candidateRootsFromBackendModule();
        if (process.platform === "win32") {
          candidates.push(
            path.join(process.resourcesPath, "stemsep-backend.exe"),
            ...roots.flatMap((root) => [
              path.join(root, "stemsep-backend", "target", "release", "stemsep-backend.exe"),
              path.join(root, "stemsep-backend", "target", "debug", "stemsep-backend.exe"),
            ]),
          );
        } else {
          candidates.push(
            path.join(process.resourcesPath, "stemsep-backend"),
            ...roots.flatMap((root) => [
              path.join(root, "stemsep-backend", "target", "release", "stemsep-backend"),
              path.join(root, "stemsep-backend", "target", "debug", "stemsep-backend"),
            ]),
          );
        }
        return findFirstExistingPath(candidates);
      })();

      if (rustExe) {
        const rustArgs: string[] = [];
        const assetsDir = (() => {
          const roots = candidateRootsFromBackendModule();
          const candidates: string[] = [
            path.join(process.resourcesPath, "StemSepApp", "assets"),
            ...roots.map((root) => path.join(root, "StemSepApp", "assets")),
          ];
          return findFirstExistingPath(candidates);
        })();

        if (assetsDir) rustArgs.push("--assets-dir", assetsDir);
        if (modelsDir) rustArgs.push("--models-dir", modelsDir);

        const explicitPython = resolveBundledPythonForRustBackend();
        if (explicitPython) {
          log("Pinning Rust backend Python via STEMSEP_PYTHON:", explicitPython);
        } else {
          log(
            "WARNING: Could not resolve explicit Python interpreter for Rust backend; it may fall back to 'py' launcher.",
          );
        }

        log("Spawning Rust backend:", rustExe, rustArgs);
        pythonBridge = spawn(rustExe, rustArgs, {
          cwd: path.dirname(rustExe),
          stdio: ["pipe", "pipe", "pipe"],
          env: {
            ...process.env,
            STEMSEP_PREFER_RUST_SEPARATION:
              (process.env.STEMSEP_PREFER_RUST_SEPARATION || "").trim() || "1",
            ...(explicitPython ? { STEMSEP_PYTHON: explicitPython } : {}),
            PYTHONIOENCODING: "utf-8",
            PYTHONUTF8: "1",
            ...(hfToken
              ? {
                  HF_TOKEN: hfToken,
                  HUGGINGFACE_HUB_TOKEN: hfToken,
                  STEMSEP_HF_TOKEN: hfToken,
                }
              : {}),
          },
        });

        attachBackendStdioLogging("rust", pythonBridge);
        pythonBridge.stderr?.on("data", (data) => {
          console.error("Rust backend stderr:", data.toString());
        });
        backendProcess = pythonBridge;
        pythonBridge.stdout?.setMaxListeners(50);
        pythonBridge.stderr?.setMaxListeners(50);
      } else if (backendPref === "rust") {
        log("CRITICAL: STEMSEP_BACKEND=rust but Rust binary not found.");
        dialog.showErrorBox(
          "Startup Error",
          "STEMSEP_BACKEND is set to 'rust' but stemsep-backend binary was not found.\n\nRebuild it (cargo build --release) or adjust STEMSEP_BACKEND.",
        );
        return null;
      } else {
        log("WARNING:", "Rust backend not found.", "Falling back to Python backend.");
      }
    }

    if (pythonBridge) {
      log("Backend bridge spawned (rust mode)");
    } else {
      const pythonPath = (() => {
        const roots = candidateRootsFromBackendModule();
        if (process.platform === "win32") {
          const candidates = [
            path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
            ...roots.flatMap((root) => [
              path.join(root, "StemSepApp", ".venv", "Scripts", "python.exe"),
              path.join(root, ".venv", "Scripts", "python.exe"),
            ]),
            "python",
            "python3",
            "py",
            "py.exe",
          ];
          return candidates.find((candidate) => {
            try {
              return candidate.includes("\\") || candidate.includes("/")
                ? fs.existsSync(candidate)
                : true;
            } catch {
              return false;
            }
          }) || "python";
        }

        const candidates = [
          path.join(process.resourcesPath, ".venv", "bin", "python"),
          ...roots.flatMap((root) => [
            path.join(root, "StemSepApp", ".venv", "bin", "python"),
            path.join(root, ".venv", "bin", "python"),
          ]),
          "python3",
          "python",
        ];
        return candidates.find((candidate) => {
          try {
            return candidate.includes("/")
              ? fs.existsSync(candidate)
              : true;
          } catch {
            return false;
          }
        }) || "python3";
      })();

      let scriptPath = app.isPackaged
        ? path.join(process.resourcesPath, "python-bridge.py")
        : path.join(process.cwd(), "python-bridge.py");

      if (!fs.existsSync(scriptPath)) {
        const roots = candidateRootsFromBackendModule();
        const candidates = [
          ...roots.map((root) => path.join(root, "python-bridge.py")),
          path.join(process.resourcesPath, "python-bridge.py"),
          path.join(process.resourcesPath, "app.asar.unpacked", "python-bridge.py"),
        ];
        const resolved = findFirstExistingPath(candidates);
        if (resolved) scriptPath = resolved;
      }

      if (!fs.existsSync(scriptPath)) {
        log("CRITICAL: python-bridge.py not found at:", scriptPath);
        dialog.showErrorBox(
          "Startup Error",
          "Python backend was selected/fell back, but python-bridge.py is missing.\n\nSet STEMSEP_BACKEND=rust or restore the python bridge file.",
        );
        return null;
      }

      const args = [scriptPath];
      if (modelsDir) args.push("--models-dir", modelsDir);

      log("Spawning Python bridge:", pythonPath, args.join(" "));
      pythonBridge = spawn(pythonPath, args, {
        cwd: path.dirname(scriptPath),
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          ...(typeof modelsDir === "string" && modelsDir.trim()
            ? { STEMSEP_MODELS_DIR: modelsDir }
            : {}),
          ...(() => {
            const roots = candidateRootsFromBackendModule();
            const candidates: string[] = [
              path.join(process.resourcesPath, "StemSepApp", "assets"),
              ...roots.map((root) => path.join(root, "StemSepApp", "assets")),
            ];
            const assetsDir = findFirstExistingPath(candidates);
            return assetsDir ? { STEMSEP_ASSETS_DIR: assetsDir } : {};
          })(),
          PYTHONIOENCODING: "utf-8",
          ...(hfToken
            ? {
                HF_TOKEN: hfToken,
                HUGGINGFACE_HUB_TOKEN: hfToken,
                STEMSEP_HF_TOKEN: hfToken,
              }
            : {}),
        },
      });

      attachBackendStdioLogging("python", pythonBridge);
      pythonBridge.stderr?.on("data", (data) => {
        console.error("Python stderr:", data.toString());
      });
      backendProcess = pythonBridge;
      pythonBridge.stdout?.setMaxListeners(50);
      pythonBridge.stderr?.setMaxListeners(50);
    }

    attachBackendMessageRouter(pythonBridge);
    pythonBridge.on("exit", (code) => {
      log("Python bridge exited with code:", code);
      rejectAllPendingBackendCommands(
        new Error(`Backend bridge exited with code ${code ?? "unknown"}`),
      );
      rejectBridgeReadyWaiters(
        new Error(`Backend bridge exited with code ${code ?? "unknown"}`),
      );
      detachBackendMessageRouter();
      pythonBridge = null;
      backendProcess = null;

      if (manualBridgeRestartPending) {
        manualBridgeRestartPending = false;
        bridgeRestartCount = 0;
        lastBridgeRestart = Date.now();
        if (!getIsAppQuitting()) {
          setTimeout(() => {
            ensureBackend();
            const mainWindow = getMainWindow();
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("bridge-reconnected");
            }
          }, 200);
        }
        return;
      }

      if (getIsAppQuitting()) return;

      const now = Date.now();
      if (now - lastBridgeRestart > 60_000) bridgeRestartCount = 0;

      if (bridgeRestartCount < MAX_BRIDGE_RESTARTS) {
        bridgeRestartCount++;
        lastBridgeRestart = now;
        log(
          `Attempting to restart Python bridge (attempt ${bridgeRestartCount}/${MAX_BRIDGE_RESTARTS})`,
        );
        setTimeout(() => {
          if (!getIsAppQuitting()) {
            ensureBackend();
            const mainWindow = getMainWindow();
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("bridge-reconnected");
            }
          }
        }, 1000 * bridgeRestartCount);
      } else {
        log("CRITICAL: Python bridge failed to restart after maximum attempts");
        const mainWindow = getMainWindow();
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("bridge-connection-failed");
          dialog.showErrorBox(
            "Backend Connection Failed",
            "The audio processing backend has stopped responding and could not be restarted. Please restart the application.",
          );
        }
      }
    });

    return pythonBridge;
  }

  function startHealthChecks() {
    if (healthCheckInterval) return;

    healthCheckInterval = setInterval(async () => {
      if (getIsAppQuitting() || !backendProcess) return;
      if (isBackendBusy()) {
        consecutiveHealthCheckFailures = 0;
        return;
      }

      try {
        try {
          const status = await sendBackendCommand("get_backend_status", {}, 5000);
          if (status && status.backend_ready === false) {
            throw new Error("Backend reported not ready");
          }
        } catch {
          await sendBackendCommand("ping", {}, 5000);
        }
        consecutiveHealthCheckFailures = 0;
      } catch (error) {
        consecutiveHealthCheckFailures++;
        log(
          `Health check failed (${consecutiveHealthCheckFailures}/${MAX_HEALTH_CHECK_FAILURES}): ${error}`,
        );

        if (consecutiveHealthCheckFailures >= MAX_HEALTH_CHECK_FAILURES) {
          log("Bridge unresponsive, forcing restart...");
          if (backendProcess) {
            backendProcess.kill("SIGKILL");
            backendProcess = null;
          }
          consecutiveHealthCheckFailures = 0;
        }
      }
    }, HEALTH_CHECK_INTERVAL_MS);
  }

  function stopHealthChecks() {
    if (healthCheckInterval) {
      clearInterval(healthCheckInterval);
      healthCheckInterval = null;
    }
  }

  function shutdown() {
    stopHealthChecks();
    rejectAllPendingBackendCommands(new Error("Backend shutdown"));
    rejectBridgeReadyWaiters(new Error("Backend shutdown"));
    if (backendProcess) {
      log("Killing backend process before quit");
      backendProcess.kill();
      backendProcess = null;
    }
  }

  return {
    ensureBackend,
    sendBackendCommand,
    sendBackendCommandWithRetry,
    requestBridgeRestart,
    restartBackendAndWait,
    startHealthChecks,
    stopHealthChecks,
    shutdown,
  };
}
