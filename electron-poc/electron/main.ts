import { app, BrowserWindow, ipcMain, dialog, shell, protocol } from "electron";
import path from "path";
import { spawn } from "child_process";
import fs from "fs";
import chokidar, { FSWatcher } from "chokidar";
// const __filename = fileURLToPath(import.meta.url)
// const __dirname = path.dirname(__filename)

// Simple file logger
const getLogPath = () => path.join(app.getPath("userData"), "app.log");

function log(message: string, ...args: any[]) {
  const timestamp = new Date().toISOString();
  const formattedMessage = `${timestamp} - ${message} ${args.length > 0 ? JSON.stringify(args) : ""}\n`;

  // Console log
  console.log(message, ...args);

  // File log
  try {
    fs.appendFileSync(getLogPath(), formattedMessage);
  } catch (e) {
    console.error("Failed to write to log file:", e);
  }
}

// Global error handlers
process.on("uncaughtException", (error) => {
  log("CRITICAL: Uncaught Exception:", error);
  // Optionally show a dialog to the user
  dialog.showErrorBox(
    "Application Error",
    `An unexpected error occurred: ${error.message}\n\nPlease check the logs for more details.`,
  );
});

process.on("unhandledRejection", (reason) => {
  log("CRITICAL: Unhandled Rejection:", reason);
});

let pythonBridge: ReturnType<typeof spawn> | null = null;

function getStoredModelsDir(): string | null {
  try {
    // Read zustand persist storage from localStorage file
    // On Windows: %APPDATA%/[appName]/Local Storage/leveldb
    // Simpler: read from a config file we create
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      return config.modelsDir || null;
    }
  } catch (e) {
    log("Could not read modelsDir from config:", e);
  }
  return null;
}

// Bridge auto-restart state
const MAX_BRIDGE_RESTARTS = 3;
let bridgeRestartCount = 0;
let lastBridgeRestart = 0;
let isAppQuitting = false;

function ensurePythonBridge() {
  if (pythonBridge) return pythonBridge;

  // Resolve a usable Python interpreter.
  // Prefer packaged/bundled venv if present, otherwise fall back to system Python on PATH.
  //
  // Why: hardcoding repo-root ".venv" breaks dev setups where the venv is located elsewhere
  // (e.g. StemSepApp/.venv) or not present at all.
  const pythonPath = (() => {
    if (process.platform === "win32") {
      const candidates = [
        // Packaged app: extraResources ships ".venv" next to the app
        path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
        // Dev repo: prefer backend venv if present
        path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
        // Dev repo: root venv (legacy)
        path.join(__dirname, "../../.venv/Scripts/python.exe"),
        // Fallback: system installs (common locations)
        "python",
        "python3",
        "py",
        "py.exe",
      ];

      for (const c of candidates) {
        try {
          if (c.includes("\\") || c.includes("/")) {
            if (fs.existsSync(c)) return c;
          } else {
            // command on PATH
            return c;
          }
        } catch {
          // ignore and continue
        }
      }
      return "python";
    }

    // macOS/Linux
    const candidates = [
      path.join(process.resourcesPath, ".venv", "bin", "python"),
      path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
      path.join(__dirname, "../../.venv/bin/python"),
      "python3",
      "python",
    ];
    for (const c of candidates) {
      try {
        if (c.includes("/")) {
          if (fs.existsSync(c)) return c;
        } else {
          return c;
        }
      } catch {
        // ignore and continue
      }
    }
    return "python3";
  })();

  // Resolve python-bridge.py location.
  // In dev: it lives next to the project root (electron-poc/python-bridge.py).
  // In packaged builds: electron-builder `extraResources` places it under `process.resourcesPath`.
  let scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, "python-bridge.py")
    : path.join(__dirname, "../python-bridge.py");

  if (!fs.existsSync(scriptPath)) {
    const candidates = [
      // Dev fallback
      path.join(__dirname, "../python-bridge.py"),
      // Packaged (normal)
      path.join(process.resourcesPath, "python-bridge.py"),
      // Packaged (some builders place extraResources under app.asar.unpacked)
      path.join(process.resourcesPath, "app.asar.unpacked", "python-bridge.py"),
    ];
    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) {
          scriptPath = c;
          break;
        }
      } catch {
        // ignore and continue
      }
    }
  }

  // Build arguments - include modelsDir if set
  const args = [scriptPath];
  const modelsDir = getStoredModelsDir();
  if (modelsDir) {
    args.push("--models-dir", modelsDir);
  }

  log("Spawning Python bridge:", pythonPath, args.join(" "));

  pythonBridge = spawn(pythonPath, args, {
    cwd: path.dirname(scriptPath),
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
    },
  });

  pythonBridge.stderr?.on("data", (data) => {
    console.error("Python stderr:", data.toString());
  });

  // Global stdout listener that always forwards download events to renderer
  // This ensures progress updates reach the UI even when other handlers are active
  const globalDownloadHandler = createLineBuffer((line) => {
    try {
      const msg = JSON.parse(line);

      // Handle bridge_ready event - signal frontend that Python bridge is fully initialized
      if (msg.type === "bridge_ready" && mainWindow && !mainWindow.isDestroyed()) {
        log(`Bridge ready! Capabilities: ${msg.capabilities?.join(', ')}, Models: ${msg.models_count}, Recipes: ${msg.recipes_count}`);
        mainWindow.webContents.send("bridge-ready", {
          capabilities: msg.capabilities,
          modelsCount: msg.models_count,
          recipesCount: msg.recipes_count
        });
      }
      // Forward download-related events to renderer
      else if (
        msg.type === "progress" &&
        msg.model_id &&
        mainWindow &&
        !mainWindow.isDestroyed()
      ) {
        console.log(
          `[Global] Download progress for ${msg.model_id}: ${msg.progress}%`,
        );
        mainWindow.webContents.send("download-progress", {
          modelId: msg.model_id,
          progress: msg.progress,
        });
      } else if (
        msg.type === "complete" &&
        msg.model_id &&
        mainWindow &&
        !mainWindow.isDestroyed()
      ) {
        console.log(`[Global] Download complete for ${msg.model_id}`);
        mainWindow.webContents.send("download-complete", {
          modelId: msg.model_id,
        });
      } else if (
        msg.type === "error" &&
        msg.model_id &&
        mainWindow &&
        !mainWindow.isDestroyed()
      ) {
        console.log(
          `[Global] Download error for ${msg.model_id}: ${msg.error}`,
        );
        mainWindow.webContents.send("download-error", {
          modelId: msg.model_id,
          error: msg.error,
        });
      }
    } catch (e) {
      // Not JSON or not a download message - ignore
    }
  });
  pythonBridge.stdout?.on("data", globalDownloadHandler);

  pythonBridge.on("exit", (code) => {
    log("Python bridge exited with code:", code);
    pythonBridge = null;

    // Skip restart logic if app is quitting
    if (isAppQuitting) {
      return;
    }

    // Reset restart count if it's been more than 60 seconds since last restart
    const now = Date.now();
    if (now - lastBridgeRestart > 60000) {
      bridgeRestartCount = 0;
    }

    // Attempt restart if we haven't exceeded the limit
    if (bridgeRestartCount < MAX_BRIDGE_RESTARTS) {
      bridgeRestartCount++;
      lastBridgeRestart = now;
      log(
        `Attempting to restart Python bridge (attempt ${bridgeRestartCount}/${MAX_BRIDGE_RESTARTS})`,
      );

      // Delay restart slightly to avoid rapid restart loops
      setTimeout(() => {
        if (!isAppQuitting) {
          ensurePythonBridge();
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("bridge-reconnected");
          }
        }
      }, 1000 * bridgeRestartCount); // Exponential backoff: 1s, 2s, 3s
    } else {
      log("CRITICAL: Python bridge failed to restart after maximum attempts");
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

// Health check state
let healthCheckInterval: NodeJS.Timeout | null = null;
let consecutiveHealthCheckFailures = 0;
const HEALTH_CHECK_INTERVAL_MS = 30000; // 30 seconds
const MAX_HEALTH_CHECK_FAILURES = 2;

function startHealthChecks() {
  if (healthCheckInterval) return;

  healthCheckInterval = setInterval(async () => {
    if (isAppQuitting || !pythonBridge) return;

    try {
      // Send ping with short timeout
      await sendPythonCommand("ping", {}, 5000);
      consecutiveHealthCheckFailures = 0;
    } catch (error) {
      consecutiveHealthCheckFailures++;
      log(
        `Health check failed (${consecutiveHealthCheckFailures}/${MAX_HEALTH_CHECK_FAILURES}): ${error}`,
      );

      if (consecutiveHealthCheckFailures >= MAX_HEALTH_CHECK_FAILURES) {
        log("Bridge unresponsive, forcing restart...");
        if (pythonBridge) {
          pythonBridge.kill("SIGKILL");
          pythonBridge = null;
        }
        // Auto-restart will be triggered by the exit handler
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

let mainWindow: InstanceType<typeof BrowserWindow> | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const isDev = !app.isPackaged;

  if (isDev) {
    const loadURLWithRetry = (url: string, retries = 10) => {
      mainWindow?.loadURL(url).catch((e) => {
        if (retries > 0) {
          log(`Failed to load URL, retrying... (${retries} attempts left)`);
          setTimeout(() => loadURLWithRetry(url, retries - 1), 1000);
        } else {
          log("Failed to load URL after multiple attempts:", e);
        }
      });
    };
    loadURLWithRetry("http://localhost:5173");
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow
      // Renderer output is built by Vite into dist-renderer/ (kept separate from electron-builder output dir).
      .loadFile(path.join(__dirname, "../dist-renderer/index.html"))
      .catch((e) => {
        log("CRITICAL: Failed to load index.html:", e);
        dialog.showErrorBox(
          "Startup Error",
          "Failed to load application resources. Please reinstall the application.",
        );
      });
  }

  mainWindow.on("unresponsive", () => {
    log("WARNING: Main window became unresponsive");
    dialog
      .showMessageBox(mainWindow!, {
        type: "warning",
        title: "App Unresponsive",
        message:
          "The application is not responding. You can wait or restart it.",
        buttons: ["Wait", "Restart"],
        defaultId: 0,
        cancelId: 0,
      })
      .then(({ response }) => {
        if (response === 1) {
          mainWindow?.reload();
        }
      });
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// Suppress Autofill warnings
app.commandLine.appendSwitch("disable-features", "AutofillServer");

app.whenReady().then(() => {
  // Register 'media' protocol to serve local files
  protocol.registerFileProtocol("media", (request, callback) => {
    // URL format: media:///C:/Users/... or media:///path/to/file
    let filePath = request.url.replace("media://", "");
    // Remove leading slash for Windows absolute paths (e.g., /C:/Users -> C:/Users)
    if (
      filePath.startsWith("/") &&
      filePath.length > 2 &&
      filePath[2] === ":"
    ) {
      filePath = filePath.slice(1);
    }
    try {
      const decodedPath = decodeURIComponent(filePath);
      console.log("[media protocol] Serving file:", decodedPath);
      return callback(decodedPath);
    } catch (error) {
      console.error("Failed to serve media file:", error);
    }
  });

  log("App is ready, creating window.");
  createWindow();
  startHealthChecks();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    log("All windows closed, quitting app.");
    app.quit();
  }
});

app.on("before-quit", () => {
  isAppQuitting = true;
  stopHealthChecks();
  if (pythonBridge) {
    log("Killing Python bridge before quit");
    pythonBridge.kill();
    pythonBridge = null;
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Helper to create a line buffer for processing stream data
const createLineBuffer = (onLine: (line: string) => void) => {
  let buffer = "";
  return (data: Buffer) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    // Keep the last incomplete line in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.trim()) {
        onLine(line);
      }
    }
  };
};

// Helper to send a command to Python and await a single JSON response
let commandIdCounter = 0;

async function sendPythonCommand(
  command: string,
  payload: Record<string, any> = {},
  timeoutMs: number = 60000,
): Promise<any> {
  const bridge = ensurePythonBridge();
  if (!bridge) throw new Error("Python bridge not available");

  const cmdId = ++commandIdCounter;

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(
        new Error(`Python command '${command}' timed out after ${timeoutMs}ms`),
      );
    }, timeoutMs);

    const dataHandler = createLineBuffer((line) => {
      try {
        const response = JSON.parse(line);

        // Check ID match (if response has ID)
        // Note: Legacy responses might not have ID, but we updated backend to send it.
        if (response.id === cmdId) {
          if (response.success !== undefined) {
            cleanup();
            if (response.success) {
              resolve(response.data);
            } else {
              reject(
                new Error(
                  response.error || "Unknown error from Python backend",
                ),
              );
            }
          }
        }
      } catch (e) {
        console.error(`Error parsing JSON response for '${command}':`, e);
      }
    });

    const cleanup = () => {
      clearTimeout(timeout);
      bridge.stdout?.removeListener("data", dataHandler);
    };

    bridge.stdout?.on("data", dataHandler);

    try {
      bridge.stdin?.write(
        JSON.stringify({ command, id: cmdId, ...payload }) + "\n",
      );
    } catch (e) {
      cleanup();
      reject(
        new Error(
          `Failed to write command '${command}' to Python bridge: ${e}`,
        ),
      );
    }
  });
}

/**
 * Wrapper for sendPythonCommand with automatic retry and exponential backoff.
 * Only retries transient failures (timeouts, bridge errors).
 * User cancellations and "not found" errors are not retried.
 */
async function sendPythonCommandWithRetry(
  command: string,
  payload: Record<string, any> = {},
  timeoutMs: number = 60000,
  maxRetries: number = 2,
): Promise<any> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Exponential backoff: 0ms, 1000ms, 2000ms
      if (attempt > 0) {
        const delay = Math.min(1000 * attempt, 5000);
        log(
          `Retrying command '${command}' after ${delay}ms (attempt ${attempt + 1}/${maxRetries + 1})`,
        );
        await new Promise((r) => setTimeout(r, delay));
      }

      return await sendPythonCommand(command, payload, timeoutMs);
    } catch (error) {
      lastError = error as Error;

      // Don't retry user cancellations or definitive failures
      const errorMsg = lastError.message.toLowerCase();
      if (
        errorMsg.includes("cancelled") ||
        errorMsg.includes("not found") ||
        errorMsg.includes("missing") ||
        errorMsg.includes("invalid")
      ) {
        throw lastError;
      }

      // Log the retry attempt
      if (attempt < maxRetries) {
        log(
          `Command '${command}' failed (attempt ${attempt + 1}): ${lastError.message}`,
        );
      }
    }
  }

  throw lastError!;
}

// IPC handler for Python separation
ipcMain.handle(
  "separate-audio",
  async (
    event,
    {
      inputFile,
      modelId,
      outputDir,
      stems,
      device,
      overlap,
      segmentSize,
      tta,
      outputFormat,
      exportMixes,
      shifts,
      bitrate,
      ensembleConfig,
      ensembleAlgorithm,
      invert,
      splitFreq,
      phaseParams,
    }: {
      inputFile: string;
      modelId: string;
      outputDir: string;
      stems?: string[];
      device?: string;
      overlap?: number;
      segmentSize?: number;
      tta?: boolean;
      outputFormat?: string;
      exportMixes?: string[];
      shifts?: number;
      bitrate?: string;
      ensembleConfig?: any;
      ensembleAlgorithm?: string;
      invert?: boolean;
      splitFreq?: number;
      phaseParams?: {
        enabled: boolean;
        lowHz: number;
        highHz: number;
        highFreqWeight: number;
      };
    },
  ) => {
    console.log("Received separate-audio request:", {
      inputFile,
      modelId,
      ensembleConfig,
      splitFreq,
      phaseParams,
    });
    const bridge = ensurePythonBridge();
    if (!bridge)
      return Promise.reject(new Error("Python bridge not available."));

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => {
          reject(new Error("Separation timeout (30 minutes)"));
        },
        30 * 60 * 1000,
      );

      let resolved = false;

      const messageHandler = createLineBuffer((line) => {
        try {
          const msg = JSON.parse(line);

          if (msg.type === "separation_progress") {
            console.log(
              `Separation Progress: ${msg.progress}% - ${msg.message}`,
            );
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("separation-progress", {
                progress: msg.progress,
                message: msg.message,
                device: msg.device,
              });
            }
          } else if (msg.type === "separation_complete") {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              bridge.stdout.removeListener("data", messageHandler);
              resolve({
                success: true,
                outputFiles: msg.output_files,
                jobId: msg.job_id,
              });
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send("separation-complete", {
                  outputFiles: msg.output_files,
                  jobId: msg.job_id,
                });
              }
            }
          } else if (msg.success === false) {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              bridge.stdout.removeListener("data", messageHandler);
              reject(new Error(msg.error || "Separation failed"));
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send("separation-error", {
                  error: msg.error || "Separation failed",
                });
              }
            }
          }
        } catch (e) {
          console.error("Error parsing JSON from Python:", e);
        }
      });

      bridge.stdout.on("data", messageHandler);

      // Send start command
      bridge.stdin.write(
        JSON.stringify({
          command: "separate_audio",
          file_path: inputFile, // Note: Python expects snake_case
          model_id: modelId,
          output_dir: outputDir,
          stems,
          device,
          shifts: shifts || 0,
          overlap,
          segment_size: segmentSize,
          tta,
          output_format: outputFormat,
          bitrate,
          ensemble_config: ensembleConfig,
          ensemble_algorithm: ensembleAlgorithm,
          invert,
          split_freq: splitFreq,
          phase_params: phaseParams, // Pass user-specified phase swap params to Python
        }) + "\n",
      );
    });
  },
);

// Resolve YouTube URL to a local temp audio file (WAV)
ipcMain.handle("resolve-youtube-url", async (event, { url }: { url: string }) => {
  const bridge = ensurePythonBridge();
  if (!bridge) {
    return Promise.reject(new Error("Python bridge not available."));
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("YouTube resolve timeout (10 minutes)"));
    }, 10 * 60 * 1000);

    const messageHandler = createLineBuffer((line) => {
      try {
        const msg = JSON.parse(line);

        if (msg.type === "youtube_progress") {
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("youtube-progress", msg);
          }
          return;
        }

        // Final response for this command
        if (msg.success !== undefined) {
          cleanup();
          if (msg.success) {
            resolve(msg.data);
          } else {
            reject(new Error(msg.error || "Failed to resolve YouTube URL"));
          }
        }
      } catch (e) {
        // ignore parse errors from partial lines
      }
    });

    const cleanup = () => {
      clearTimeout(timeout);
      bridge.stdout?.removeListener("data", messageHandler);
    };

    bridge.stdout?.on("data", messageHandler);

    bridge.stdin?.write(
      JSON.stringify({
        command: "resolve_youtube",
        url,
      }) + "\n",
    );
  });
});

// Cancel separation
ipcMain.handle("cancel-separation", async (event, jobId: string) => {
  return sendPythonCommand("cancel_job", { job_id: jobId });
});

// Save/Discard output
ipcMain.handle("save-job-output", async (event, jobId: string) => {
  return sendPythonCommand("save_output", { job_id: jobId });
});

ipcMain.handle(
  "export-output",
  async (
    event,
    {
      jobId,
      exportPath,
      format,
      bitrate,
    }: { jobId: string; exportPath: string; format: string; bitrate: string },
  ) => {
    return sendPythonCommand("export_output", {
      job_id: jobId,
      export_path: exportPath,
      format,
      bitrate,
    });
  },
);

ipcMain.handle("discard-job-output", async (event, jobId: string) => {
  return sendPythonCommand("discard_output", { job_id: jobId });
});

// Export files directly from paths (bypasses job registry - for historical exports)
ipcMain.handle(
  "export-files",
  async (
    event,
    {
      sourceFiles,
      exportPath,
      format,
      bitrate,
    }: {
      sourceFiles: Record<string, string>;
      exportPath: string;
      format: string;
      bitrate: string;
    },
  ) => {
    return sendPythonCommand("export_files", {
      source_files: sourceFiles,
      export_path: exportPath,
      format,
      bitrate,
    });
  },
);

// Queue Management
ipcMain.handle("pause-queue", async () => sendPythonCommand("pause_queue"));
ipcMain.handle("resume-queue", async () => sendPythonCommand("resume_queue"));
ipcMain.handle("reorder-queue", async (event, jobIds: string[]) =>
  sendPythonCommand("reorder_queue", { job_ids: jobIds }),
);

ipcMain.handle("open-audio-file-dialog", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile", "multiSelections"],
      filters: [
        {
          name: "Audio Files",
          extensions: [
            "mp3",
            "wav",
            "flac",
            "m4a",
            "ogg",
            "aac",
            "wma",
            "aiff",
          ],
        },
      ],
    });
    return result.filePaths || [];
  } catch (error) {
    log("Error opening file dialog:", error);
    throw new Error("Failed to open file dialog");
  }
});

ipcMain.handle("open-model-file-dialog", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: [
        {
          name: "Model Files",
          extensions: ["ckpt", "pth", "pt", "onnx", "safetensors"],
        },
      ],
    });
    return result.filePaths || [];
  } catch (error) {
    log("Error opening model file dialog:", error);
    throw new Error("Failed to open model file dialog");
  }
});

ipcMain.handle("select-output-directory", async () => {
  if (!mainWindow) return null;
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openDirectory"],
    });
    return result.filePaths[0] || null;
  } catch (error) {
    log("Error selecting output directory:", error);
    throw new Error("Failed to open directory dialog");
  }
});

// Scan directory for audio files
ipcMain.handle("scan-directory", async (event, folderPath: string) => {
  const audioExtensions = new Set([".mp3", ".wav", ".flac", ".ogg", ".m4a"]);
  const audioFiles: string[] = [];

  async function scan(dir: string) {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await scan(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (audioExtensions.has(ext)) {
          audioFiles.push(fullPath);
        }
      }
    }
  }

  try {
    await scan(folderPath);
    return audioFiles;
  } catch (error) {
    console.error("Error scanning directory:", error);
    log("Error scanning directory:", error);
    throw new Error(
      `Failed to scan directory: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
});

// Open folder
ipcMain.handle("open-folder", async (event, folderPath: string) => {
  await shell.openPath(folderPath);
});

// Read audio file as base64 for use with Blob URLs
// This works around browser security restrictions that block file:// URLs
ipcMain.handle("read-audio-file", async (_event, filePath: string) => {
  try {
    // Security: block arbitrary file reads if renderer is compromised.
    // Only allow a strict list of audio extensions.
    const ext = filePath.split(".").pop()?.toLowerCase();
    const allowedExtensions = new Set([
      "mp3",
      "wav",
      "flac",
      "ogg",
      "m4a",
      "aac",
      "wma",
      "aiff",
    ]);
    if (!ext || !allowedExtensions.has(ext)) {
      throw new Error(
        `Security violation: Attempted to read non-audio file extension '.${ext || ""}'`,
      );
    }

    const data = fs.readFileSync(filePath);
    const base64 = data.toString("base64");
    // Determine MIME type from extension
    const mimeTypes: Record<string, string> = {
      mp3: "audio/mpeg",
      wav: "audio/wav",
      flac: "audio/flac",
      ogg: "audio/ogg",
      m4a: "audio/mp4",
      aac: "audio/aac",
      wma: "audio/x-ms-wma",
      aiff: "audio/aiff",
    };
    const mimeType = mimeTypes[ext] || "audio/mpeg";
    return { success: true, data: base64, mimeType };
  } catch (error: any) {
    log(`Failed to read audio file: ${filePath}`, error);
    return { success: false, error: error.message };
  }
});

// Check preset models
ipcMain.handle(
  "check-preset-models",
  async (event, presetMappings: Record<string, string>) => {
    return sendPythonCommandWithRetry("check-preset-models", {
      preset_mappings: presetMappings,
    });
  },
);

// Get GPU devices
ipcMain.handle("get-gpu-devices", async () => {
  return sendPythonCommandWithRetry("get-gpu-devices", {}, 15000);
});

// Get workflow types (Live vs Studio)
ipcMain.handle("get-workflows", async () => {
  return sendPythonCommandWithRetry("get_workflows", {}, 10000);
});

// Get all available models
ipcMain.handle("get-models", async () => {
  return sendPythonCommandWithRetry("get_models", {}, 120000);
});

ipcMain.handle("get-model-tech", async (_event, modelId: string) => {
  return sendPythonCommandWithRetry("get_model_tech", { model_id: modelId }, 20000);
});

ipcMain.handle(
  "separation-preflight",
  async (
    _event,
    {
      inputFile,
      modelId,
      outputDir,
      stems,
      device,
      overlap,
      segmentSize,
      tta,
      outputFormat,
      exportMixes,
      shifts,
      bitrate,
      ensembleConfig,
      ensembleAlgorithm,
      invert,
      splitFreq,
      phaseParams,
    }: {
      inputFile: string;
      modelId: string;
      outputDir: string;
      stems?: string[];
      device?: string;
      overlap?: number;
      segmentSize?: number;
      tta?: boolean;
      outputFormat?: string;
      exportMixes?: string[];
      shifts?: number;
      bitrate?: string;
      ensembleConfig?: any;
      ensembleAlgorithm?: string;
      invert?: boolean;
      splitFreq?: number;
      phaseParams?: {
        enabled: boolean;
        lowHz: number;
        highHz: number;
        highFreqWeight: number;
      };
    },
  ) => {
    return sendPythonCommandWithRetry(
      "separation_preflight",
      {
        file_path: inputFile,
        model_id: modelId,
        output_dir: outputDir,
        stems,
        device,
        shifts: shifts || 0,
        overlap,
        segment_size: segmentSize,
        tta,
        output_format: outputFormat,
        bitrate,
        ensemble_config: ensembleConfig,
        ensemble_algorithm: ensembleAlgorithm,
        invert,
        split_freq: splitFreq,
        phase_params: phaseParams,
        export_mixes: exportMixes,
      },
      30000,
    );
  },
);

// Get recipes
ipcMain.handle("get-recipes", async () => {
  return sendPythonCommandWithRetry("get_recipes", {}, 10000);
});

// Download model
ipcMain.handle("download-model", async (event, modelId: string) => {
  const bridge = ensurePythonBridge();

  return new Promise((resolve, reject) => {
    let dataHandler: ((data: Buffer) => void) | null = null;

    const timeout = setTimeout(() => {
      // Clean up listener on timeout to prevent memory leak
      if (dataHandler) {
        bridge.stdout?.removeListener("data", dataHandler);
      }
      reject(new Error("Download timeout"));
    }, 600000); // 10 minute timeout for large downloads

    dataHandler = createLineBuffer((line) => {
      try {
        const response = JSON.parse(line);

        // Handle progress updates
        if (response.type === "progress") {
          event.sender.send("download-progress", {
            modelId: response.model_id,
            progress: response.progress,
          });
        }
        // Handle completion
        else if (response.type === "complete") {
          event.sender.send("download-complete", {
            modelId: response.model_id,
          });
        }
        // Handle errors
        else if (response.type === "error") {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);
          // Send error event to renderer
          event.sender.send("download-error", {
            modelId: response.model_id,
            error: response.error,
          });
          reject(new Error(response.error));
        }
        // Handle final response
        else if (response.success !== undefined) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);

          if (response.success) {
            resolve(response.data);
          } else {
            reject(new Error(response.error || "Download failed"));
          }
        }
      } catch (e) {
        console.error("Error parsing download response:", e);
      }
    });

    bridge.stdout.on("data", dataHandler);

    // Send download command
    bridge.stdin.write(
      JSON.stringify({
        command: "download_model",
        model_id: modelId,
      }) + "\n",
    );
  });
});

// Remove model
ipcMain.handle("remove-model", async (event, modelId: string) => {
  return sendPythonCommand("remove_model", { model_id: modelId }, 10000);
});

// Pause download
ipcMain.handle("pause-download", async (event, modelId: string) => {
  const bridge = ensurePythonBridge();

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Pause timeout"));
    }, 5000);

    const dataHandler = createLineBuffer((line) => {
      try {
        const response = JSON.parse(line);

        if (response.success !== undefined) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);

          if (response.success) {
            resolve(response.data);
          } else {
            reject(new Error(response.error || "Pause failed"));
          }
        }

        // Also handle paused event
        if (response.type === "paused" && response.model_id === modelId) {
          event.sender.send("download-paused", { modelId: response.model_id });
        }
      } catch (e) {
        // Not JSON or incomplete, continue
      }
    });

    bridge.stdout.on("data", dataHandler);

    // Send pause command
    bridge.stdin.write(
      JSON.stringify({
        command: "pause_download",
        model_id: modelId,
      }) + "\n",
    );
  });
});

// Resume download
ipcMain.handle("resume-download", async (event, modelId: string) => {
  const bridge = ensurePythonBridge();

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Resume timeout"));
      event.sender.send("download-error", { modelId, error: "Resume timeout" });
    }, 120000); // 2 min timeout like regular download

    const dataHandler = createLineBuffer((line) => {
      try {
        const response = JSON.parse(line);

        // Handle response messages (success/error)
        if (response.success !== undefined) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);

          if (response.success) {
            resolve(response.data);
          } else {
            event.sender.send("download-error", {
              modelId,
              error: response.error,
            });
            reject(new Error(response.error || "Resume failed"));
          }
        }

        // Handle progress events
        if (response.type === "progress" && response.model_id === modelId) {
          event.sender.send("download-progress", {
            modelId: response.model_id,
            progress: response.progress,
            downloaded: response.downloaded,
            total: response.total,
            speed: response.speed,
            eta: response.eta,
          });
        } else if (
          response.type === "complete" &&
          response.model_id === modelId
        ) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);
          event.sender.send("download-complete", {
            modelId: response.model_id,
          });
          resolve({ modelId: response.model_id });
        } else if (response.type === "error" && response.model_id === modelId) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);
          event.sender.send("download-error", {
            modelId: response.model_id,
            error: response.error,
          });
          reject(new Error(response.error));
        } else if (
          response.type === "paused" &&
          response.model_id === modelId
        ) {
          clearTimeout(timeout);
          bridge.stdout.removeListener("data", dataHandler);
          event.sender.send("download-paused", { modelId: response.model_id });
          resolve({ modelId: response.model_id, paused: true });
        }
      } catch (e) {
        // Not JSON or incomplete, continue
      }
    });

    bridge.stdout.on("data", dataHandler);

    // Send resume command
    bridge.stdin.write(
      JSON.stringify({
        command: "resume_download",
        model_id: modelId,
      }) + "\n",
    );
  });
});

// Import custom model
ipcMain.handle(
  "import-custom-model",
  async (
    event,
    {
      filePath,
      modelName,
      architecture,
    }: { filePath: string; modelName: string; architecture?: string },
  ) => {
    return sendPythonCommand("import_custom_model", {
      file_path: filePath,
      model_name: modelName,
      architecture: architecture || "Custom",
    });
  },
);

// Queue Persistence
const getQueuePath = () =>
  path.join(app.getPath("userData"), "queue_state.json");

ipcMain.handle("save-queue", async (event, queueData: any) => {
  try {
    await fs.promises.writeFile(
      getQueuePath(),
      JSON.stringify(queueData),
      "utf-8",
    );
    return { success: true };
  } catch (error) {
    console.error("Failed to save queue:", error);
    return { success: false, error: String(error) };
  }
});

ipcMain.handle("load-queue", async () => {
  try {
    const queuePath = getQueuePath();
    if (!fs.existsSync(queuePath)) {
      return null;
    }
    const data = await fs.promises.readFile(queuePath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    console.error("Failed to load queue:", error);
    return null;
  }
});

// Watch Folder Logic
let watcher: FSWatcher | null = null;

ipcMain.handle("start-watch-mode", async (event, folderPath: string) => {
  if (watcher) {
    await watcher.close();
  }

  log("Starting watch mode on:", folderPath);

  watcher = chokidar.watch(folderPath, {
    ignored: /(^|[\/\\])\../, // ignore dotfiles
    persistent: true,
    ignoreInitial: true, // Don't process existing files immediately
    awaitWriteFinish: {
      stabilityThreshold: 2000, // Wait 2s for file write to finish
      pollInterval: 100,
    },
  });

  watcher.on("add", (filePath) => {
    const ext = path.extname(filePath).toLowerCase();
    if ([".wav", ".mp3", ".flac", ".m4a", ".ogg"].includes(ext)) {
      log("New file detected:", filePath);
      mainWindow?.webContents.send("watch-file-detected", filePath);
    }
  });

  return true;
});

ipcMain.handle("stop-watch-mode", async () => {
  if (watcher) {
    await watcher.close();
    watcher = null;
    log("Watch mode stopped");
  }
  return true;
});

// App config persistence for main process settings (like modelsDir)
ipcMain.handle(
  "save-app-config",
  async (_event, config: Record<string, any>) => {
    try {
      const configPath = path.join(app.getPath("userData"), "app-config.json");
      let existingConfig: Record<string, any> = {};

      if (fs.existsSync(configPath)) {
        existingConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      }

      const newConfig = { ...existingConfig, ...config };
      fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
      log("Saved app config:", Object.keys(config).join(", "));
      return true;
    } catch (error) {
      log("Failed to save app config:", error);
      return false;
    }
  },
);

ipcMain.handle("get-app-config", async () => {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    }
  } catch (error) {
    log("Failed to read app config:", error);
  }
  return {};
});
