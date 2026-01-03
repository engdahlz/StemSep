import { app, BrowserWindow, ipcMain, dialog, shell, protocol, Menu, screen } from "electron";
import path from "path";
import { spawn } from "child_process";
import fs from "fs";
import { randomUUID, createHash } from "crypto";
import chokidar, { FSWatcher } from "chokidar";
// const __filename = fileURLToPath(import.meta.url)
// const __dirname = path.dirname(__filename)

// Simple file logger
const getLogPath = () => path.join(app.getPath("userData"), "app.log");
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
    const rotated = path.join(dir, `${base}.${stamp}${ext}`);
    fs.renameSync(filePath, rotated);
  } catch {
    // ignore
  }
}

function appendBackendStdio(prefix: string, chunk: any) {
  try {
    rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
    const ts = new Date().toISOString();
    const text = typeof chunk === "string" ? chunk : chunk?.toString?.("utf-8") ?? String(chunk);
    fs.appendFileSync(getBackendStdioLogPath(), `${ts} ${prefix} ${text}${text.endsWith("\n") ? "" : "\n"}`);
  } catch {
    // ignore
  }
}

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

function attachBackendStdioLogging(label: string) {
  if (!pythonBridge) return;
  // Make sure the file exists/rotated early.
  try {
    safeMkdir(path.dirname(getBackendStdioLogPath()));
    rotateLogIfLarge(getBackendStdioLogPath(), 10 * 1024 * 1024);
  } catch {
    // ignore
  }

  pythonBridge.stdout?.on("data", (data) => {
    appendBackendStdio(`[${label} stdout]`, data);
  });

  pythonBridge.stderr?.on("data", (data) => {
    appendBackendStdio(`[${label} stderr]`, data);
  });
}

function safeMkdir(dirPath: string) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
  } catch {
    // ignore
  }
}

function sanitizeForPathSegment(name: string) {
  return (name || "")
    .replace(/[<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80);
}

function getFfmpegExe(): string {
  // Prefer a bundled ffmpeg binary when available.
  // 1) build-electron.mjs copies ffmpeg next to dist-electron/main.js
  // 2) ffmpeg-static (node module)
  // 3) ffmpeg on PATH
  try {
    const local = path.join(
      __dirname,
      process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg",
    );
    if (fs.existsSync(local)) return local;
  } catch {
    // ignore
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ffmpegStaticPath = require("ffmpeg-static");
    if (typeof ffmpegStaticPath === "string" && ffmpegStaticPath) {
      return ffmpegStaticPath;
    }
  } catch {
    // ignore
  }

  return "ffmpeg";
}

async function runFfmpeg(args: string[]): Promise<void> {
  const exe = getFfmpegExe();

  await new Promise<void>((resolve, reject) => {
    const child = spawn(exe, args, { windowsHide: true });
    let stderr = "";
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      reject(
        new Error(
          `Failed to start ffmpeg (${exe}): ${err?.message || String(err)}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (exit=${code}). ${stderr}`.trim()));
    });
  });
}

function getPreviewCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "previews");
}

function createPreviewDirForInput(inputFile: string) {
  const base = getPreviewCacheBaseDir();
  safeMkdir(base);

  const stamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace("T", "_")
    .replace("Z", "");

  const baseName = sanitizeForPathSegment(
    path.basename(inputFile, path.extname(inputFile)),
  );

  const dirName = `${stamp}__${baseName || "audio"}__${randomUUID().slice(0, 8)}`;
  const full = path.join(base, dirName);
  safeMkdir(full);
  return full;
}

function resolveEffectiveModelId(modelId: any, ensembleConfig: any): string {
  // If ensembleConfig is present, the backend expects model_id="ensemble".
  if (ensembleConfig && Array.isArray(ensembleConfig.models) && ensembleConfig.models.length > 0) {
    return "ensemble";
  }
  if (typeof modelId === "string" && modelId.trim()) return modelId.trim();
  throw new Error(
    "Missing modelId. This is a preset/config bug (modelId must be a non-empty string).",
  );
}

function shortHash(input: string): string {
  return createHash("sha1").update(input).digest("hex").slice(0, 12);
}

async function ensureWavInput(inputFile: string, previewDir: string): Promise<string> {
  const ext = path.extname(inputFile || "").toLowerCase();
  if (ext === ".wav") return inputFile;

  // Stage a decoded WAV into the preview dir so the backend only ever sees WAV.
  const staged = path.join(previewDir, `input_${shortHash(inputFile)}.wav`);
  if (fs.existsSync(staged)) return staged;

  await runFfmpeg([
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputFile,
    "-vn",
    "-c:a",
    "pcm_s16le",
    staged,
  ]);

  if (!fs.existsSync(staged)) {
    throw new Error("Decoded WAV was not created.");
  }

  return staged;
}

async function exportFilesLocal({
  sourceFiles,
  exportPath,
  format,
  bitrate,
}: {
  sourceFiles: Record<string, string>;
  exportPath: string;
  format: string;
  bitrate: string;
}): Promise<{ exported: Record<string, string> }>{
  if (!exportPath || typeof exportPath !== "string") {
    throw new Error("Missing exportPath");
  }
  safeMkdir(exportPath);

  const fmt = String(format || "wav").toLowerCase();
  if (!new Set(["wav", "flac", "mp3"]).has(fmt)) {
    throw new Error(`Unsupported export format: ${format}`);
  }

  const exported: Record<string, string> = {};

  const entries = Object.entries(sourceFiles || {});
  for (const [stemRaw, inputFile] of entries) {
    const stem = sanitizeForPathSegment(stemRaw || "stem") || "stem";
    if (!inputFile || typeof inputFile !== "string") continue;
    if (!fs.existsSync(inputFile)) {
      throw new Error(`Source file missing for '${stemRaw}': ${inputFile}`);
    }

    const outBase = stem;
    let outFile = path.join(exportPath, `${outBase}.${fmt}`);
    for (let i = 2; fs.existsSync(outFile); i++) {
      outFile = path.join(exportPath, `${outBase}_${i}.${fmt}`);
    }

    const inExt = path.extname(inputFile || "").toLowerCase();
    if (fmt === "wav" && inExt === ".wav") {
      fs.copyFileSync(inputFile, outFile);
      exported[stemRaw] = outFile;
      continue;
    }

    if (fmt === "flac") {
      await runFfmpeg([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inputFile,
        "-vn",
        "-c:a",
        "flac",
        outFile,
      ]);
      exported[stemRaw] = outFile;
      continue;
    }

    if (fmt === "mp3") {
      const br = String(bitrate || "320k");
      await runFfmpeg([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inputFile,
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        br,
        outFile,
      ]);
      exported[stemRaw] = outFile;
      continue;
    }
  }

  if (Object.keys(exported).length === 0) {
    throw new Error("No files were exported (no valid source files)");
  }

  return { exported };
}

function cleanupPreviewCache() {
  try {
    const base = getPreviewCacheBaseDir();
    if (!fs.existsSync(base)) return;

    const entries = fs
      .readdirSync(base, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => {
        const full = path.join(base, e.name);
        let mtime = 0;
        try {
          mtime = fs.statSync(full).mtimeMs;
        } catch {
          // ignore
        }
        return { full, mtime };
      })
      .sort((a, b) => b.mtime - a.mtime);

    const keepLast = 20;
    const maxAgeDays = 7;
    const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (i < keepLast) continue;
      if (entry.mtime && entry.mtime > cutoff) continue;
      try {
        fs.rmSync(entry.full, { recursive: true, force: true });
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
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

let backendProcess: ReturnType<typeof spawn> | null = null;
// Historically this code called the backend process `pythonBridge`.
// Keep the name for compatibility with existing helper functions.
let pythonBridge: ReturnType<typeof spawn> | null = null;

let hfAuthWindow: BrowserWindow | null = null;

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

function readAppConfig(): Record<string, any> {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    }
  } catch (e) {
    log("Could not read app-config.json:", e);
  }
  return {};
}

function writeAppConfig(partial: Record<string, any>): boolean {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    const existingConfig = readAppConfig();
    const newConfig = { ...existingConfig, ...partial };
    fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
    return true;
  } catch (e) {
    log("Could not write app-config.json:", e);
    return false;
  }
}

function getStoredHuggingFaceToken(): string | null {
  try {
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      const token = config.hfToken;
      if (typeof token === "string" && token.trim()) return token.trim();
    }
  } catch (e) {
    log("Could not read hfToken from config:", e);
  }
  return null;
}

function setStoredHuggingFaceToken(token: string | null): { success: boolean; error?: string } {
  try {
    const trimmed = typeof token === "string" ? token.trim() : "";
    const configPath = path.join(app.getPath("userData"), "app-config.json");
    const existingConfig = readAppConfig();

    if (!trimmed) {
      if (Object.prototype.hasOwnProperty.call(existingConfig, "hfToken")) {
        delete existingConfig.hfToken;
        fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
      }
      return { success: true };
    }

    // Basic sanity check (HF tokens are typically fairly long). Avoid rejecting valid tokens.
    if (trimmed.length < 20) {
      return { success: false, error: "Token looks too short." };
    }

    existingConfig.hfToken = trimmed;
    fs.writeFileSync(configPath, JSON.stringify(existingConfig, null, 2));
    return { success: true };
  } catch (e: any) {
    log("Failed to set hfToken:", e);
    return { success: false, error: e?.message || "Failed to save token." };
  }
}

// Bridge auto-restart state
const MAX_BRIDGE_RESTARTS = 3;
let bridgeRestartCount = 0;
let lastBridgeRestart = 0;
let isAppQuitting = false;
let manualBridgeRestartPending = false;

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

function shouldUseRustBackend(): boolean {
  const v = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
  if (v === "python") return false;
  if (v === "rust") return true;
  // Default: prefer Rust backend.
  return true;
}

function resolveBundledPythonForRustBackend(): string | null {
  // The Rust backend spawns `scripts/inference.py`. On Windows, it defaults to using the `py` launcher,
  // which may select a Python without our dependencies. Provide an explicit interpreter when possible.
  try {
    if (process.platform === "win32") {
      const candidates = [
        // Packaged app: extraResources ships ".venv" next to the app
        path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
        // Dev repo: prefer StemSepApp venv if present
        path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
        // Dev repo: root venv (legacy)
        path.join(__dirname, "../../.venv/Scripts/python.exe"),
      ];

      for (const c of candidates) {
        try {
          if (fs.existsSync(c)) return c;
        } catch {
          // ignore
        }
      }
      return null;
    }

    const candidates = [
      path.join(process.resourcesPath, ".venv", "bin", "python"),
      path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
      path.join(__dirname, "../../.venv/bin/python"),
    ];

    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) return c;
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
  return null;
}

function ensureBackend() {
  return ensurePythonBridge();
}

function openHuggingFaceAuthWindow() {
  if (hfAuthWindow && !hfAuthWindow.isDestroyed()) {
    hfAuthWindow.focus();
    return;
  }

  hfAuthWindow = new BrowserWindow({
    width: 520,
    height: 360,
    resizable: false,
    minimizable: false,
    maximizable: false,
    modal: !!mainWindow,
    parent: mainWindow || undefined,
    title: "Authorize Hugging Face",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  hfAuthWindow.on("closed", () => {
    hfAuthWindow = null;
  });

  const html = `<!doctype html>
  <html>
    <head>
      <meta charset="utf-8" />
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
      <title>Authorize Hugging Face</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color: #111; }
        h1 { font-size: 18px; margin: 0 0 8px; }
        p { margin: 8px 0; line-height: 1.35; color: #333; }
        .box { border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa; }
        label { display: block; font-weight: 600; margin: 12px 0 6px; }
        input { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 13px; }
        .row { display: flex; gap: 10px; margin-top: 12px; }
        button { padding: 10px 12px; border-radius: 8px; border: 1px solid #bbb; background: white; cursor: pointer; font-weight: 600; }
        button.primary { background: #1677ff; border-color: #1677ff; color: white; }
        button.danger { background: #fff; border-color: #d33; color: #d33; }
        .status { margin-top: 10px; font-size: 12px; color: #444; }
        .hint { font-size: 12px; color: #555; }
        a { color: #1677ff; text-decoration: none; }
      </style>
    </head>
    <body>
      <h1>Authorize Hugging Face (optional)</h1>
      <p class="hint">Only needed for gated/private model downloads. Public models work without this.</p>
      <div class="box">
        <div id="current" class="status">Checking status…</div>
        <label for="token">Token</label>
        <input id="token" type="password" placeholder="Paste your Hugging Face access token" autocomplete="off" />
        <div class="row">
          <button class="primary" id="save">Save token</button>
          <button class="danger" id="clear">Clear token</button>
          <button id="close">Close</button>
        </div>
        <p class="hint">Create a token in <a href="#" id="openTokens">huggingface.co/settings/tokens</a>.</p>
        <div id="msg" class="status"></div>
      </div>
      <script>
        const current = document.getElementById('current');
        const msg = document.getElementById('msg');
        const tokenEl = document.getElementById('token');

        async function refresh() {
          try {
            const st = await window.electronAPI.getHuggingFaceAuthStatus();
            current.textContent = st && st.configured ? 'Status: Token configured' : 'Status: Not configured';
          } catch (e) {
            current.textContent = 'Status: Unknown';
          }
        }

        document.getElementById('openTokens').addEventListener('click', async (e) => {
          e.preventDefault();
          await window.electronAPI.openExternalUrl('https://huggingface.co/settings/tokens');
        });

        document.getElementById('save').addEventListener('click', async () => {
          msg.textContent = '';
          const t = tokenEl.value || '';
          const res = await window.electronAPI.setHuggingFaceToken(t);
          if (res && res.success) {
            tokenEl.value = '';
            msg.textContent = 'Saved. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to save token.';
          }
        });

        document.getElementById('clear').addEventListener('click', async () => {
          msg.textContent = '';
          const res = await window.electronAPI.clearHuggingFaceToken();
          if (res && res.success) {
            msg.textContent = 'Cleared. Backend will restart to apply.';
            await refresh();
          } else {
            msg.textContent = (res && res.error) ? res.error : 'Failed to clear token.';
          }
        });

        document.getElementById('close').addEventListener('click', () => window.close());
        refresh();
      </script>
    </body>
  </html>`;

  hfAuthWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(html)}`,
  );
}

function ensurePythonBridge() {
  if (pythonBridge) return pythonBridge;

  const hfToken = getStoredHuggingFaceToken();
  const modelsDir = getStoredModelsDir();

  // Optional: spawn Rust backend instead of Python.
  // This keeps renderer/UI unchanged while we migrate backend functionality.
  if (shouldUseRustBackend()) {
    const backendPref = (process.env.STEMSEP_BACKEND || "").toLowerCase().trim();
    const rustExe = (() => {
      const candidates: string[] = [];
      if (process.platform === "win32") {
        candidates.push(
          path.join(process.resourcesPath, "stemsep-backend.exe"),
          path.join(__dirname, "../../stemsep-backend/target/release/stemsep-backend.exe"),
          path.join(__dirname, "../../stemsep-backend/target/debug/stemsep-backend.exe"),
        );
      } else {
        candidates.push(
          path.join(process.resourcesPath, "stemsep-backend"),
          path.join(__dirname, "../../stemsep-backend/target/release/stemsep-backend"),
          path.join(__dirname, "../../stemsep-backend/target/debug/stemsep-backend"),
        );
      }

      for (const c of candidates) {
        try {
          if (fs.existsSync(c)) return c;
        } catch {
          // ignore
        }
      }
      return null;
    })();

    if (rustExe) {
      const rustArgs: string[] = [];

      const assetsDir = (() => {
        const candidates: string[] = [
          path.join(process.resourcesPath, "StemSepApp", "assets"),
          path.join(__dirname, "../../StemSepApp/assets"),
        ];

        for (const c of candidates) {
          try {
            if (fs.existsSync(c)) return c;
          } catch {
            // ignore
          }
        }
        return null;
      })();

      if (assetsDir) {
        rustArgs.push("--assets-dir", assetsDir);
      }
      if (modelsDir) {
        rustArgs.push("--models-dir", modelsDir);
      }

      const explicitPython = resolveBundledPythonForRustBackend();

      log("Spawning Rust backend:", rustExe, rustArgs);
      pythonBridge = spawn(rustExe, rustArgs, {
        cwd: path.dirname(rustExe),
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          // IMPORTANT: default to Rust-native separation/preflight.
          // This avoids requiring the python-bridge proxy for `separation_preflight`, which can
          // break dev runs when the bridge script isn't present.
          STEMSEP_PREFER_RUST_SEPARATION:
            (process.env.STEMSEP_PREFER_RUST_SEPARATION || "").trim() || "1",
          // Ensure inference uses the same Python environment as the app.
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

      attachBackendStdioLogging("rust");

      pythonBridge.stderr?.on("data", (data) => {
        console.error("Rust backend stderr:", data.toString());
      });

      backendProcess = pythonBridge;

      // Continue with shared stdout handling + restart logic below.
    } else {
      const msg = "Rust backend not found.";
      if (backendPref === "rust") {
        log("CRITICAL: STEMSEP_BACKEND=rust but Rust binary not found.");
        dialog.showErrorBox(
          "Startup Error",
          "STEMSEP_BACKEND is set to 'rust' but stemsep-backend binary was not found.\n\nRebuild it (cargo build --release) or adjust STEMSEP_BACKEND.",
        );
        return null;
      }
      log("WARNING:", msg, "Falling back to Python backend.");
    }
  }

  // If Rust backend was spawned above, skip Python-specific interpreter/script resolution.
  if (pythonBridge) {
    log("Backend bridge spawned (rust mode)");
  } else {

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

  if (!fs.existsSync(scriptPath)) {
    log("CRITICAL: python-bridge.py not found at:", scriptPath);
    dialog.showErrorBox(
      "Startup Error",
      "Python backend was selected/fell back, but python-bridge.py is missing.\n\nSet STEMSEP_BACKEND=rust or restore the python bridge file.",
    );
    return null;
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
      ...(typeof modelsDir === "string" && modelsDir.trim()
        ? { STEMSEP_MODELS_DIR: modelsDir }
        : {}),
      ...(() => {
        const candidates: string[] = [
          path.join(process.resourcesPath, "StemSepApp", "assets"),
          path.join(__dirname, "../../StemSepApp/assets"),
        ];

        for (const c of candidates) {
          try {
            if (fs.existsSync(c)) return { STEMSEP_ASSETS_DIR: c };
          } catch {
            // ignore
          }
        }

        return {};
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

  attachBackendStdioLogging("python");

  pythonBridge.stderr?.on("data", (data) => {
    console.error("Python stderr:", data.toString());
  });

  backendProcess = pythonBridge;
  }

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
      } else if (
        msg.type === "paused" &&
        msg.model_id &&
        mainWindow &&
        !mainWindow.isDestroyed()
      ) {
        console.log(`[Global] Download paused for ${msg.model_id}`);
        mainWindow.webContents.send("download-paused", {
          modelId: msg.model_id,
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
    backendProcess = null;

    // Manual restart: don't count towards crash restart budget.
    if (manualBridgeRestartPending) {
      manualBridgeRestartPending = false;
      bridgeRestartCount = 0;
      lastBridgeRestart = Date.now();

      if (!isAppQuitting) {
        setTimeout(() => {
          ensurePythonBridge();
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("bridge-reconnected");
          }
        }, 200);
      }
      return;
    }

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
    if (isAppQuitting || !backendProcess) return;

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
        if (backendProcess) {
          backendProcess.kill("SIGKILL");
          backendProcess = null;
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
let isCreatingMainWindow = false;

// Avoid multiple Electron instances (common in dev when scripts rerun), which otherwise
// results in multiple app windows.
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
  // IMPORTANT: Return immediately to prevent the rest of the script (e.g., app.whenReady)
  // from running and opening a window in this secondary process.
  process.exit(0);
} else {
  app.on("second-instance", () => {
    // In dev it’s common to accidentally launch Electron twice (e.g. script reruns).
    // If this happens before `whenReady` has created the window, calling createWindow()
    // here can race and produce two windows. Always wait for readiness and rely on
    // createWindow()’s internal guard.
    app
      .whenReady()
      .then(() => {
        try {
          if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.show();
            mainWindow.focus();
          } else {
            createWindow();
          }
        } catch {
          // ignore
        }
      })
      .catch(() => {
        // ignore
      });
  });
}

function ensureWindowOnScreen(win: BrowserWindow) {
  try {
    const bounds = win.getBounds();
    const displays = screen.getAllDisplays();

    const intersectsSomeDisplay = displays.some((d) => {
      const wa = d.workArea;
      const xOverlap =
        Math.min(bounds.x + bounds.width, wa.x + wa.width) -
        Math.max(bounds.x, wa.x);
      const yOverlap =
        Math.min(bounds.y + bounds.height, wa.y + wa.height) -
        Math.max(bounds.y, wa.y);
      return xOverlap > 20 && yOverlap > 20;
    });

    if (intersectsSomeDisplay) return;

    const wa = screen.getPrimaryDisplay().workArea;
    const width = Math.min(bounds.width, wa.width);
    const height = Math.min(bounds.height, wa.height);
    const x = wa.x + Math.round((wa.width - width) / 2);
    const y = wa.y + Math.round((wa.height - height) / 2);
    win.setBounds({ x, y, width, height });
  } catch {
    // ignore
  }
}

function createWindow() {
  // Hard guard to avoid duplicate windows (e.g. second-instance race in dev).
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
      return;
    }
  } catch {
    // ignore
  }

  if (isCreatingMainWindow) return;
  isCreatingMainWindow = true;

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Make sure the window is visible (helps if it was spawned off-screen/minimized).
  try {
    ensureWindowOnScreen(mainWindow);
    mainWindow.center();
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  } catch {
    // ignore
  }

  mainWindow.once("ready-to-show", () => {
    try {
      if (mainWindow) ensureWindowOnScreen(mainWindow);
      if (mainWindow?.isMinimized()) mainWindow.restore();
      mainWindow?.show();
      mainWindow?.focus();
    } catch {
      // ignore
    }
  });

  const isDev = !app.isPackaged;

  if (isDev) {
    // Briefly bring the window above others so it's obvious it launched.
    try {
      mainWindow.setAlwaysOnTop(true);
      setTimeout(() => {
        try {
          mainWindow?.setAlwaysOnTop(false);
        } catch {
          // ignore
        }
      }, 1500);
    } catch {
      // ignore
    }

    const loadURLWithRetry = (url: string, retries = 10) => {
      mainWindow?.loadURL(url)
        .then(() => {
          try {
            if (mainWindow) ensureWindowOnScreen(mainWindow);
            if (mainWindow?.isMinimized()) mainWindow.restore();
            mainWindow?.show();
            mainWindow?.focus();
          } catch {
            // ignore
          }
        })
        .catch((e) => {
        if (retries > 0) {
          log(`Failed to load URL, retrying... (${retries} attempts left)`);
          setTimeout(() => loadURLWithRetry(url, retries - 1), 1000);
        } else {
          log("Failed to load URL after multiple attempts:", e);
        }
      });
    };

    // On Windows, `localhost` often resolves to IPv6 (::1) first, while Vite typically
    // listens on IPv4 only (0.0.0.0/127.0.0.1). Prefer 127.0.0.1 to avoid
    // intermittent `ERR_CONNECTION_REFUSED` during dev startup.
    const envDevUrl = (process.env.VITE_DEV_SERVER_URL || "").trim();
    const rawDevUrl = envDevUrl || "http://127.0.0.1:5173/";
    const normalizedDevUrl = rawDevUrl
      .replace(/^http:\/\/localhost(?=[:/]|$)/i, "http://127.0.0.1")
      .replace(/^https:\/\/localhost(?=[:/]|$)/i, "https://127.0.0.1");

    log("Dev renderer URL", { envDevUrl: envDevUrl || null, normalizedDevUrl });

    loadURLWithRetry(normalizedDevUrl);
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
    isCreatingMainWindow = false;
  });

  mainWindow.webContents.once("did-fail-load", () => {
    // Avoid getting stuck “creating” if load fails and the user tries again.
    isCreatingMainWindow = false;
  });

  mainWindow.once("ready-to-show", () => {
    isCreatingMainWindow = false;
  });
}

// Suppress Autofill warnings
app.commandLine.appendSwitch("disable-features", "AutofillServer");

app.whenReady().then(() => {
  cleanupPreviewCache();

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

  // Minimal app menu with optional Hugging Face authorization (no React UI changes).
  try {
    const template: Electron.MenuItemConstructorOptions[] = [
      ...(process.platform === "darwin"
        ? ([
            {
              label: app.name,
              submenu: [
                { role: "about" },
                { type: "separator" },
                { role: "quit" },
              ],
            },
          ] as Electron.MenuItemConstructorOptions[])
        : []),
      {
        label: "File",
        submenu: [
          process.platform === "darwin" ? { role: "close" } : { role: "quit" },
        ],
      },
      {
        label: "Hugging Face",
        submenu: [
          {
            label: "Authorize Hugging Face…",
            click: () => openHuggingFaceAuthWindow(),
          },
          {
            label: "Clear Hugging Face Token",
            click: () => {
              const res = setStoredHuggingFaceToken(null);
              if (res.success) requestBridgeRestart("cleared huggingface token");
            },
          },
          { type: "separator" },
          {
            label: "Open Token Settings…",
            click: async () => {
              await shell.openExternal("https://huggingface.co/settings/tokens");
            },
          },
        ],
      },
    ];
    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  } catch (e) {
    log("Failed to set application menu:", e);
  }

  startHealthChecks();

  // Optional: run an automated smoke separation to validate end-to-end behavior
  // without relying on manual UI clicks.
  void maybeRunSmokeSeparation();
});

async function maybeRunSmokeSeparation(): Promise<void> {
  const enabled = (process.env.STEMSEP_SMOKE_SEPARATION || "").trim();
  if (!enabled || enabled === "0" || enabled.toLowerCase() === "false") {
    return;
  }

  const inputFile =
    (process.env.STEMSEP_SMOKE_INPUT_FILE || "").trim() ||
    path.join(__dirname, "../strip mall.wav");
  const modelId =
    (process.env.STEMSEP_SMOKE_MODEL_ID || "").trim() || "bs-roformer-viperx-1297";
  const device = (process.env.STEMSEP_SMOKE_DEVICE || "").trim() || undefined;

  const outRoot = path.join(app.getPath("userData"), "smoke_runs");
  const runDir = path.join(outRoot, `run-${Date.now()}`);
  try {
    fs.mkdirSync(runDir, { recursive: true });
  } catch {
    // ignore
  }

  log("[smoke] Starting smoke separation", { inputFile, modelId, runDir, device });

  try {
    // 1) Preflight (same command as UI)
    const pre = await sendPythonCommandWithRetry(
      "separation_preflight",
      {
        file_path: inputFile,
        model_id: modelId,
        output_dir: runDir,
        device,
      },
      60000,
    );
    log("[smoke] preflight ok", pre);

    // 2) Start separation
    const start = await sendPythonCommand(
      "separate_audio",
      {
        file_path: inputFile,
        model_id: modelId,
        output_dir: runDir,
        device,
      },
      60000,
    );

    const jobId = start?.job_id;
    if (!jobId) throw new Error("Smoke separation: backend did not return job_id");
    log("[smoke] job started", { jobId, start });

    // 3) Wait for completion event
    const process = ensureBackend();
    if (!process) throw new Error("Smoke separation: backend unavailable");

    let completeMsg: any = null;

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(new Error("Smoke separation timeout (60 minutes)"));
      }, 60 * 60 * 1000);

      const handler = createLineBuffer((line) => {
        try {
          const msg = JSON.parse(line);
          if (msg?.type === "separation_progress" && msg.job_id === jobId) {
            log(`[smoke] progress ${msg.progress}%`, msg.message || "");
          }
          if (msg?.job_id !== jobId) return;

          if (msg.type === "separation_complete") {
            completeMsg = msg;
            log("[smoke] complete", msg);
            cleanup();
            resolve();
          } else if (msg.type === "separation_error") {
            cleanup();
            reject(new Error(msg.error || "Smoke separation failed"));
          } else if (msg.type === "separation_cancelled") {
            cleanup();
            reject(new Error("Smoke separation cancelled"));
          }
        } catch {
          // ignore
        }
      });

      const cleanup = () => {
        clearTimeout(timeout);
        process.stdout?.removeListener("data", handler);
      };

      process.stdout?.on("data", handler);
    });

    // 4) Validate output exists
    const outputFiles: string[] = (() => {
      const byEvent = completeMsg?.output_files;
      if (byEvent && typeof byEvent === "object") {
        return Object.values(byEvent)
          .filter((v) => typeof v === "string")
          .map((v) => v as string);
      }
      return [];
    })();

    const existingFromEvent = outputFiles.filter((p) => {
      try {
        return fs.existsSync(p);
      } catch {
        return false;
      }
    });

    const listWavsRecursively = (baseDir: string, maxFiles = 50): string[] => {
      const out: string[] = [];
      const walk = (dir: string) => {
        if (out.length >= maxFiles) return;
        let entries: fs.Dirent[] = [];
        try {
          entries = fs.readdirSync(dir, { withFileTypes: true });
        } catch {
          return;
        }
        for (const e of entries) {
          if (out.length >= maxFiles) return;
          const full = path.join(dir, e.name);
          if (e.isDirectory()) {
            walk(full);
          } else if (e.isFile() && e.name.toLowerCase().endsWith(".wav")) {
            out.push(full);
          }
        }
      };
      walk(baseDir);
      return out;
    };

    const fallbackWavs = listWavsRecursively(runDir);

    log("[smoke] output_files (event)", completeMsg?.output_files || null);
    log("[smoke] output files (existing)", existingFromEvent);
    if (existingFromEvent.length === 0 && fallbackWavs.length === 0) {
      throw new Error(`Smoke separation completed but no wav outputs found under ${runDir}`);
    }

    log("[smoke] SUCCESS", { runDir });
  } catch (e: any) {
    log("[smoke] FAILED", e?.message || String(e));
  } finally {
    if ((process.env.STEMSEP_SMOKE_QUIT || "").trim() === "1") {
      log("[smoke] Quitting app (STEMSEP_SMOKE_QUIT=1)");
      app.quit();
    }
  }
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    log("All windows closed, quitting app.");
    app.quit();
  }
});

app.on("before-quit", () => {
  isAppQuitting = true;
  stopHealthChecks();
  if (backendProcess) {
    log("Killing backend process before quit");
    backendProcess.kill();
    backendProcess = null;
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
  const process = ensureBackend();
  if (!process) throw new Error("Backend not available");

  const cmdId = ++commandIdCounter;

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(
        new Error(`Backend command '${command}' timed out after ${timeoutMs}ms`),
      );
    }, timeoutMs);

    const dataHandler = createLineBuffer((line) => {
      try {
        const response = JSON.parse(line);

        // Check ID match (if response has ID)
        if (response.id === cmdId) {
          if (response.success !== undefined) {
            cleanup();
            if (response.success) {
              resolve(response.data);
            } else {
              reject(
                new Error(
                  response.error || "Unknown error from backend",
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
      process.stdout?.removeListener("data", dataHandler);
    };

    process.stdout?.on("data", dataHandler);

    try {
      process.stdin?.write(
        JSON.stringify({ command, id: cmdId, ...payload }) + "\n",
      );
    } catch (e) {
      cleanup();
      reject(
        new Error(
          `Failed to write command '${command}' to backend: ${e}`,
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

// IPC handler for audio separation
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
      postProcessingSteps,
      volumeCompensation,
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
      postProcessingSteps?: any[];
      volumeCompensation?: { enabled: boolean; stage?: "export" | "blend" | "both"; dbPerExtraModel?: number };
    },
  ) => {
    console.log("Received separate-audio request:", {
      inputFile,
      modelId,
      ensembleConfig,
      splitFreq,
      phaseParams,
    });
    const process = ensureBackend();
    if (!process)
      return Promise.reject(new Error("Backend not available."));

    // Always stage outputs into a stable preview cache for playback/preview.
    // Export is handled separately (Results -> Export).
    const previewDir = createPreviewDirForInput(inputFile);

    // Ensure model id is always valid (never null/undefined), and normalize ensembles.
    const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);

    // Ensure backend receives WAV input even if user dropped MP3/FLAC/M4A.
    const effectiveInputFile = await ensureWavInput(inputFile, previewDir);

    return new Promise((resolve, reject) => {
      // 60 minute timeout for separation jobs (some models are slow)
      const timeout = setTimeout(
        () => {
          reject(new Error("Separation timeout (60 minutes)"));
        },
        60 * 60 * 1000,
      );

      let resolved = false;

      const messageHandler = createLineBuffer((line) => {
        try {
          const msg = JSON.parse(line);

          // IMPORTANT: Rust backend emits 'separation_progress' events with job_id
          // We must ensure we're listening to the correct job if multiple are running,
          // though the current architecture mostly serializes or we rely on the bridge filtering.
          // For now, we forward all progress events to the UI, which filters by active job if needed.

          if (msg.type === "separation_progress") {
            console.log(
              `Separation Progress: ${msg.progress}% - ${msg.message}`,
            );
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("separation-progress", {
                jobId: msg.job_id,
                progress: msg.progress,
                message: msg.message,
                device: msg.device,
              });
            }
          } else if (msg.type === "separation_complete") {
            // Check if this completion matches our request?
            // The rust backend should send a response to the command with the job_id,
            // AND emit an event.
            // Wait, this specific block is for the *streaming event* approach.
            // The Rust backend responds to the initial "separate_audio" command with { "job_id": "...", "status": "started" }
            // Then it emits "separation_complete" events later.
            //
            // THE PROBLEM: The original code used a "request-response" pattern for `separate-audio` IPC that waited indefinitely
            // for completion.
            // Rust backend returns immediately with "started".
            // So we need to adapt this handler to resolve ONLY when the completion event arrives.

            // Note: We might need to filter by job_id if we had it.
            // But we don't have the job_id until we send the command.
            // Actually, we can just resolve when we get *a* completion event if we assume single-tasking,
            // but better is to read the immediate response to get job_id, then wait for event.
            //
            // HOWEVER, the existing frontend expects `ipcRenderer.invoke('separate-audio')` to resolve with the result.
            // So we must keep this Promise pending until the job finishes.

            if (msg.job_id) {
               // Good, we have a job_id. We should match it.
               // But we haven't sent the command yet, so we don't know our job_id.
               // We need to capture the job_id from the initial command response first.
            }

            if (!resolved) {
               // We'll optimistically resolve on the first completion event for now,
               // or we'd need to refactor to wait for the command response to get job_id.
               // Let's assume the backend won't spam completion events for unknown jobs.
               // Actually, let's fix this properly below.
            }
          }
        } catch (e) {
          // ignore
        }
      });

      // We attach the global listener temporarily.
      process.stdout?.on("data", messageHandler);

      // Send start command
      // We wrap the write in a separate async call effectively, but here we just write.
      // We need to get the immediate response to know the job_id.
      // But `bridge.stdin.write` is fire-and-forget.
      // We should use `sendPythonCommand` which waits for the response!
      //
      // Refactoring to use sendPythonCommand for the initial kick-off:
      const payload = {
          file_path: effectiveInputFile,
          model_id: effectiveModelId,
          output_dir: previewDir,
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
          post_processing_steps: postProcessingSteps,
          export_mixes: exportMixes,
          volume_compensation: volumeCompensation,
      };

      // We can't easily use sendPythonCommand inside this Promise because of the complex event listener setup needed *before* sending.
      // Instead, let's manually replicate the send-and-wait-for-completion logic more carefully.

      // 1. Send command and get Job ID.
      // 2. Wait for event with that Job ID.

      // Clean up the temporary global listener from above, it's not safe.
      process.stdout?.removeListener("data", messageHandler);

      // New approach:
      let myJobId: string | null = null;

      const smartHandler = createLineBuffer((line) => {
        try {
          const msg = JSON.parse(line);

          // 1. Capture Job ID from immediate response (if we used manual write, we'd need to track cmdId)
          // But here we are listening to ALL events.

          // Always forward progress events to the UI (UI can correlate by jobId).
          if (msg.type === "separation_progress") {
            console.log(
              `Separation Progress (event): job_id=${msg.job_id} progress=${msg.progress}% message=${msg.message}`,
            );
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("separation-progress", {
                jobId: msg.job_id,
                progress: msg.progress,
                message: msg.message,
                device: msg.device,
              });
            }
          }

          // Also forward errors globally for visibility (but keep Promise rejection job-scoped below).
          if (msg.type === "separation_error") {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("separation-error", {
                jobId: msg.job_id,
                error: msg.error || "Separation failed",
              });
            }
          }

          if (msg.type === "separation_complete") {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send("separation-complete", {
                outputFiles: msg.output_files,
                jobId: msg.job_id,
                outputDir: previewDir,
              });
            }
          }

          // If we see a progress/completion event matching our job ID:
          if (myJobId && msg.job_id === myJobId) {
             if (msg.type === "separation_complete") {
                cleanup();
                resolve({
                  success: true,
                  outputFiles: msg.output_files,
                  jobId: msg.job_id,
                outputDir: previewDir,
                });
             } else if (msg.type === "separation_error") {
                cleanup();
                reject(new Error(msg.error || "Separation failed"));
             } else if (msg.type === "separation_cancelled") {
                cleanup();
                reject(new Error("Separation cancelled"));
             }
          }
        } catch (e) { }
      });

      process.stdout?.on("data", smartHandler);

      const cleanup = () => {
        clearTimeout(timeout);
        process.stdout?.removeListener("data", smartHandler);
      };

      // Use the helper to send the command and get the immediate { job_id, status } response
      sendPythonCommand("separate_audio", payload)
        .then((response) => {
           if (response.job_id) {
             myJobId = response.job_id;
             console.log(`Started separation job: ${myJobId}`);

             // Immediately notify the renderer so it can bind this queue item to the backend job id
             // and show non-zero progress even if the first real progress event is delayed.
             if (mainWindow && !mainWindow.isDestroyed()) {
               mainWindow.webContents.send("separation-started", {
                 jobId: myJobId,
               });
               mainWindow.webContents.send("separation-progress", {
                 jobId: myJobId,
                 progress: 1,
                 message: "Starting separation...",
               });
             }
           } else {
             cleanup();
             reject(new Error("Backend did not return a job ID"));
           }
        })
        .catch((err) => {
          cleanup();
          reject(err);
        });
    });
  },
);

// Resolve YouTube URL to a local temp audio file (WAV)
ipcMain.handle("resolve-youtube-url", async (event, { url }: { url: string }) => {
  const process = ensureBackend();
  if (!process) {
    return Promise.reject(new Error("Backend not available."));
  }

  // Use the new smart handler approach or the simple command wrapper?
  // The YouTube resolver emits progress events that we need to capture.
  // Standard sendPythonCommand waits for the final response but doesn't expose a way to hook stream events during the wait.
  // So we need a custom implementation similar to separate-audio.

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
        
        // We can't rely on `msg.success` here because it might be a response to a DIFFERENT command if we are sharing the stream.
        // However, resolve_youtube is blocking in the legacy python bridge.
        // In the new Rust backend, it might be proxied.
        // If it's proxied, the ID matching in `sendPythonCommand` handles the final response.
        // But `sendPythonCommand` doesn't let us see the progress events.
        
        // Strategy:
        // 1. Attach a global listener for "youtube_progress" events (which don't have IDs usually).
        // 2. Use `sendPythonCommand` to await the final result.
      } catch (e) {
        // ignore
      }
    });

    const cleanup = () => {
      clearTimeout(timeout);
      process.stdout?.removeListener("data", messageHandler);
    };

    process.stdout?.on("data", messageHandler);

    // Send command via the standard helper to handle ID matching/response
    sendPythonCommand("resolve_youtube", { url }, 10 * 60 * 1000)
      .then((data) => {
        cleanup();
        resolve(data);
      })
      .catch((err) => {
        cleanup();
        reject(err);
      });
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
    // Export is a pure file operation (copy/transcode) and should not require a backend.
    // This also avoids failures when the legacy python-bridge proxy is not present.
    log("[export-files] local export requested", {
      stems: Object.keys(sourceFiles || {}),
      exportPath,
      format,
      bitrate,
    });
    try {
      const res = await exportFilesLocal({ sourceFiles, exportPath, format, bitrate });
      log("[export-files] local export complete", {
        exported: Object.keys(res?.exported || {}),
      });
      return res;
    } catch (e: any) {
      log("[export-files] local export FAILED", e?.message || String(e));
      throw e;
    }
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

    const resolveMissingPreviewAudioPath = (missingPath: string): string | null => {
      try {
        const parent = path.dirname(missingPath);
        if (!fs.existsSync(parent)) return null;

        const requestedName = path.basename(missingPath).toLowerCase();
        const requestedStem = requestedName.replace(path.extname(requestedName), "");

        const wavs: string[] = [];
        const maxFiles = 200;
        const walk = (dir: string) => {
          if (wavs.length >= maxFiles) return;
          let entries: fs.Dirent[] = [];
          try {
            entries = fs.readdirSync(dir, { withFileTypes: true });
          } catch {
            return;
          }
          for (const e of entries) {
            if (wavs.length >= maxFiles) return;
            const full = path.join(dir, e.name);
            if (e.isDirectory()) {
              walk(full);
            } else if (e.isFile() && e.name.toLowerCase().endsWith(".wav")) {
              wavs.push(full);
            }
          }
        };

        walk(parent);
        if (wavs.length === 0) return null;

        const score = (p: string): number => {
          const b = path.basename(p).toLowerCase();
          let s = 0;
          // Prefer canonical/clean filenames.
          if (b === `${requestedStem}.wav`) s += 100;
          // Prefer explicit stem markers used by audio-separator.
          if (requestedStem === "instrumental") {
            if (b.includes("(instrumental)") || b.includes("_instrumental_") || b.includes(" instrumental ")) s += 50;
            if (b.includes("(vocals)")) s -= 50;
          } else if (requestedStem === "vocals") {
            if (b.includes("(vocals)") || b.includes("_vocals_") || b.includes(" vocals ")) s += 50;
            if (b.includes("(instrumental)")) s -= 50;
          } else {
            if (b.includes(requestedStem)) s += 20;
          }
          // Prefer files closer to the parent directory (shallower depth).
          const rel = path.relative(parent, p);
          const depth = rel.split(path.sep).length;
          s += Math.max(0, 10 - depth);
          return s;
        };

        let best: string | null = null;
        let bestScore = -Infinity;
        for (const p of wavs) {
          const sc = score(p);
          if (sc > bestScore) {
            bestScore = sc;
            best = p;
          }
        }
        return best;
      } catch {
        return null;
      }
    };

    let resolvedPath = filePath;
    if (!fs.existsSync(resolvedPath)) {
      const fallback = resolveMissingPreviewAudioPath(resolvedPath);
      if (fallback && fs.existsSync(fallback)) {
        log("Resolved missing preview audio path", { from: filePath, to: fallback });
        resolvedPath = fallback;
      }
    }

    const data = fs.readFileSync(resolvedPath);
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
    return { success: true, data: base64, mimeType, resolvedPath };
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
      postProcessingSteps,
      volumeCompensation,
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
      postProcessingSteps?: any[];
      volumeCompensation?: { enabled: boolean; stage?: "export" | "blend" | "both"; dbPerExtraModel?: number };
    },
  ) => {
    // Preflight should mirror the real run: normalize model id and stage non-WAV inputs to WAV.
    const previewDir = createPreviewDirForInput(inputFile);
    const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);
    const effectiveInputFile = await ensureWavInput(inputFile, previewDir);

    return sendPythonCommandWithRetry(
      "separation_preflight",
      {
        file_path: effectiveInputFile,
        model_id: effectiveModelId,
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
        post_processing_steps: postProcessingSteps,
        export_mixes: exportMixes,
        volume_compensation: volumeCompensation,
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
  const process = ensureBackend();
  if (!process) throw new Error("Backend not available");

  // Similar to YouTube: Rust backend handles download in background and emits events.
  // We trigger it with a command that returns immediately (scheduled: true).
  
  // Note: Rust backend emits global events for progress. We don't need a specific listener attached here
  // because `globalDownloadHandler` in `ensureBackend` already forwards them to the window!
  
  // We just need to send the command.
  return sendPythonCommand("download_model", { model_id: modelId });
});

// Remove model
ipcMain.handle("remove-model", async (event, modelId: string) => {
  try {
    const res = await sendPythonCommandWithRetry("remove_model", { model_id: modelId }, 60000);
    log("remove-model completed", { modelId, res });
    return res;
  } catch (e: any) {
    log("remove-model failed", { modelId, error: e?.message || String(e) });
    throw e;
  }
});

// Pause download
ipcMain.handle("pause-download", async (event, modelId: string) => {
  // Simple command, events handled globally
  return sendPythonCommand("pause_download", { model_id: modelId });
});

// Resume download
ipcMain.handle("resume-download", async (event, modelId: string) => {
  // Simple command, events handled globally
  return sendPythonCommand("resume_download", { model_id: modelId });
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

      // Never allow saving secrets through the generic config endpoint.
      const { hfToken: _ignoredHfToken, ...safeConfig } = config || {};
      const newConfig = { ...existingConfig, ...safeConfig };
      fs.writeFileSync(configPath, JSON.stringify(newConfig, null, 2));
      log("Saved app config:", Object.keys(safeConfig).join(", "));
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
      const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      if (config && typeof config === "object") {
        delete config.hfToken;
      }
      return config;
    }
  } catch (error) {
    log("Failed to read app config:", error);
  }
  return {};
});

// Hugging Face auth (optional): store token in app-config.json and restart backend
ipcMain.handle("get-huggingface-auth-status", async () => {
  return { configured: !!getStoredHuggingFaceToken() };
});

ipcMain.handle("set-huggingface-token", async (_event, token: string) => {
  const res = setStoredHuggingFaceToken(token);
  if (res.success) {
    requestBridgeRestart("updated huggingface token");
  }
  return res;
});

ipcMain.handle("clear-huggingface-token", async () => {
  const res = setStoredHuggingFaceToken(null);
  if (res.success) {
    requestBridgeRestart("cleared huggingface token");
  }
  return res;
});

ipcMain.handle("open-external-url", async (_event, url: string) => {
  try {
    if (typeof url !== "string" || !url.trim()) return false;
    const u = new URL(url);
    if (u.protocol !== "https:" && u.protocol !== "http:") return false;
    await shell.openExternal(url);
    return true;
  } catch (e) {
    log("Failed to open external url:", e);
    return false;
  }
});
