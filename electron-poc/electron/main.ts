import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
  protocol,
  Menu,
  session,
  desktopCapturer,
} from "electron";
import path from "path";
import fs from "fs";
import { randomUUID } from "crypto";
import { createBackendBridge } from "./backend/bridge";
import {
  PREVIEW_CACHE_KEEP_LAST,
  PREVIEW_CACHE_MAX_AGE_DAYS,
  classifyMissingAudioPath,
  cleanupPreviewCache,
  createPreviewDirForInput,
  ensureWavInput,
  exportFilesLocal,
  getPreviewCacheBaseDir,
  probeAudioFile,
  resolveEffectiveModelId,
  resolvePlaybackFilePath,
  resolvePlaybackStems,
  type AudioSourceProfile,
  type ExportProgressPayload,
  type SourceStagingDecision,
} from "./media/local-audio";
import {
  computeCaptureQualityMode,
  resolveCaptureLaunchTarget,
} from "./playback/capture-metadata";
import { registerSelectionIpcHandlers } from "./backend/ipc-selection";
import { registerSeparationIpcHandlers } from "./backend/ipc-separation";
import { registerQueueIpcHandlers } from "./backend/ipc-queue";
import { registerRuntimeIpcHandlers } from "./backend/ipc-runtime";
import { registerModelIpcHandlers } from "./backend/ipc-models";
import { registerQualityIpcHandlers } from "./backend/ipc-quality";
import { registerDownloadIpcHandlers } from "./backend/ipc-downloads";
import { registerDialogIpcHandlers } from "./system/dialogs";
import { registerConfigIpcHandlers } from "./system/config";
import { registerWatchIpcHandlers } from "./system/watch";
import { registerLibraryCaptureIpcHandlers } from "./remote/ipc-library";
import { registerRemoteImportIpcHandlers } from "./remote/ipc-remote";
import { createRemoteDownloadController } from "./remote/download-audio";
import { createRemoteBrowserLibraryController } from "./remote/library-browser";
import { createRemoteLibraryController } from "./remote/library-core";
import { createQobuzCaptureController } from "./remote/qobuz-capture";
import { createRemoteWindowController } from "./remote/windows";
import { createMainWindow } from "./windows/main-window";
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

function safeMkdir(dirPath: string) {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
  } catch {
    // ignore
  }
}

const SYSTEM_RUNTIME_TTL_MS = 30_000;

function sanitizeForPathSegment(name: string) {
  return (name || "")
    .replace(/[<>:"/\\|?*]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80);
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

type RemoteLibraryProvider = "spotify" | "qobuz" | "bandcamp";
type ActiveLibraryProvider = "spotify" | "qobuz";
type RemoteSourceProvider = "youtube" | RemoteLibraryProvider;
type IngestMode = "local_file" | "remote_download" | "desktop_capture";
type PlaybackSurface = "desktop_app" | "browser" | "none";
type CaptureQualityMode = "best_available" | "verified_lossless";

type RemoteResolveProgressPayload = {
  provider: RemoteSourceProvider;
  status: string;
  detail?: string;
  percent?: string;
  progress?: number;
  speed?: string;
  eta?: string;
  error?: string;
};

type RemoteCatalogItem = {
  provider: RemoteLibraryProvider;
  trackId: string;
  albumId?: string;
  title: string;
  artist?: string;
  album?: string;
  trackNumber?: number;
  discNumber?: number;
  artworkUrl?: string;
  durationSec?: number;
  canonicalUrl?: string;
  qualityLabel?: string;
  isLossless?: boolean;
  downloadOrigin?: string;
  pageUrl?: string;
  variantId?: string;
  downloadUrlAvailable?: boolean;
  sourceUrl?: string;
  playbackUrl?: string;
  playbackUri?: string;
  playbackSurface?: PlaybackSurface;
  ingestMode?: IngestMode;
  qualityMode?: CaptureQualityMode;
  verifiedLossless?: boolean;
};

type PlaybackDevice = {
  id: string;
  label: string;
  kind: "render_endpoint" | "system_loopback" | "display_loopback";
  isDefault?: boolean;
};

type CaptureEnvironmentStatus = {
  windowsSupported: boolean;
  provider: "qobuz";
  authenticated: boolean;
  selectedDeviceId: string | null;
  selectedDeviceLabel: string | null;
  selectedDeviceReady: boolean;
  speakerSelectionAvailable: boolean;
  message?: string;
};

type PlaybackCaptureProgressPayload = {
  provider: ActiveLibraryProvider;
  captureId?: string;
  status: string;
  detail?: string;
  progress?: number;
  percent?: string;
  elapsedSec?: number;
  remainingSec?: number;
  error?: string;
};

type RemoteProviderConfig = {
  name: string;
  partition: string;
  loginUrl: string;
  libraryUrl: string;
  searchUrl?: (query: string) => string;
};

const REMOTE_PROVIDER_CONFIG: Record<RemoteLibraryProvider, RemoteProviderConfig> = {
  spotify: {
    name: "Spotify",
    partition: "persist:stemsep-spotify",
    loginUrl: "https://accounts.spotify.com/en/login",
    libraryUrl: "https://open.spotify.com/collection/tracks",
    searchUrl: (query) =>
      `https://open.spotify.com/search/${encodeURIComponent(query)}/tracks`,
  },
  qobuz: {
    name: "Qobuz",
    partition: "persist:stemsep-qobuz",
    loginUrl: "https://play.qobuz.com/login",
    libraryUrl: "https://play.qobuz.com/user/library/favorites/tracks",
    searchUrl: (query) =>
      `https://play.qobuz.com/search/tracks?searchEntry=${encodeURIComponent(query)}`,
  },
  bandcamp: {
    name: "Bandcamp",
    partition: "persist:stemsep-bandcamp",
    loginUrl: "https://bandcamp.com/login",
    libraryUrl: "https://bandcamp.com/collection",
  },
};

const REMOTE_AUDIO_EXTENSIONS = new Set([
  ".wav",
  ".flac",
  ".aif",
  ".aiff",
  ".mp3",
  ".m4a",
  ".aac",
  ".ogg",
  ".opus",
  ".alac",
]);

function emitExportProgress(payload: ExportProgressPayload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("export-progress", payload);
}

const remoteCatalogCache = new Map<RemoteLibraryProvider, RemoteCatalogItem[]>();
let markQobuzPlaySessionObserved = () => {};

function emitRemoteResolveProgress(payload: RemoteResolveProgressPayload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send("remote-resolve-progress", payload);
}


function getRemoteProviderConfig(provider: RemoteLibraryProvider): RemoteProviderConfig {
  return REMOTE_PROVIDER_CONFIG[provider];
}

function getRemoteSession(provider: RemoteLibraryProvider) {
  return session.fromPartition(getRemoteProviderConfig(provider).partition);
}

const remoteWindowController = createRemoteWindowController({
  getRemoteProviderConfig,
  getRemoteSession,
  getMainWindow: () => mainWindow,
  log,
  onObservedQobuzUrl: () => {
    markQobuzPlaySessionObserved();
  },
  onQobuzAutomationClosed: () => {
    qobuzSinkIdByPlaybackDeviceId.clear();
  },
});

const {
  getExistingAuthWindow,
  getExistingQobuzAutomationWindow,
  getQobuzAutomationWindow,
  openRemoteSourceAuthWindow,
  getOpenProviderProbeWindows,
  closeAllRemoteWindows,
} = remoteWindowController;

const remoteBrowserLibraryController = createRemoteBrowserLibraryController({
  getRemoteProviderConfig,
  getExistingAuthWindow,
  getMainWindow: () => mainWindow,
  log,
  remoteCatalogCache,
});

const { scrapeRemoteCollection, resolveRemoteDownloadInfo } =
  remoteBrowserLibraryController;

const remoteLibraryController = createRemoteLibraryController({
  getRemoteProviderConfig,
  getRemoteSession,
  getQobuzAutomationWindow,
  openRemoteSourceAuthWindow,
  getOpenProviderProbeWindows,
  remoteCatalogCache,
  log,
  isQobuzPlaybackCaptureActive: () => isQobuzPlaybackCaptureActive(),
});

const {
  markQobuzPlaySessionObserved: markObservedPlaySession,
  waitForDidFinishLoad,
  loadURLAndWait,
  checkLibraryProviderAuthenticated,
  scrapeProviderItems,
  searchQobuzCatalogViaApi,
  getCachedLibraryItem,
} = remoteLibraryController;

markQobuzPlaySessionObserved = markObservedPlaySession;

const remoteDownloadController = createRemoteDownloadController({
  log,
  getRemoteSession,
  getRemoteProviderCacheDir: (provider) => getRemoteProviderCacheDir(provider),
  getUserAgent: () =>
    mainWindow?.webContents.getUserAgent() || "Mozilla/5.0 StemSep Desktop",
  emitRemoteResolveProgress,
});

const {
  downloadRemoteFile,
  extractArchiveForRemoteTrack,
} = remoteDownloadController;

const qobuzCaptureController = createQobuzCaptureController({
  log,
  getMainWindow: () => mainWindow,
  getStoredCaptureOutputDeviceId,
  getQobuzLibraryUrl: () => getRemoteProviderConfig("qobuz").libraryUrl,
  getExistingQobuzAutomationWindow,
  getQobuzAutomationWindow,
  loadURLAndWait,
  waitForDidFinishLoad,
  checkLibraryProviderAuthenticated,
  searchQobuzCatalogViaApi,
  probeAudioFile,
  sendBackendCommand: (...args) => sendPythonCommand(...args),
  sendBackendCommandWithRetry: (...args) =>
    sendPythonCommandWithRetry(...args),
});

const {
  qobuzSinkIdByPlaybackDeviceId,
  lastPlaybackCaptureProgressById,
  playbackCaptureSessions,
  emitPlaybackCaptureProgress,
  isPlaybackCaptureActive,
  isQobuzPlaybackCaptureActive,
  getNativePlaybackDevices,
  refreshQobuzPlaybackDeviceMappings,
  getCaptureEnvironmentStatusForQobuz,
  stopHiddenQobuzPlayback,
  startQobuzPlaybackCaptureForItem,
  abortPlaybackCaptureSession,
  maybeRunQobuzCaptureSmokeTest,
} = qobuzCaptureController;

function getRemoteCacheBaseDir() {
  return path.join(app.getPath("userData"), "cache", "remote_sources");
}

function getRemoteProviderCacheDir(provider: RemoteLibraryProvider) {
  return path.join(getRemoteCacheBaseDir(), provider);
}

function getStoredCaptureOutputDeviceId(): string | null {
  const value = readAppConfig().captureOutputDeviceId;
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function toPercent(progress: number) {
  if (!Number.isFinite(progress)) return undefined;
  return `${Math.max(0, Math.min(100, Math.round(progress)))}%`;
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

let isAppQuitting = false;

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
    const venv = (process.env.VIRTUAL_ENV || "").trim();
    if (venv) {
      if (process.platform === "win32") {
        const p = path.join(venv, "Scripts", "python.exe");
        try {
          if (fs.existsSync(p)) return p;
        } catch {
          // ignore
        }
      } else {
        const p = path.join(venv, "bin", "python");
        try {
          if (fs.existsSync(p)) return p;
        } catch {
          // ignore
        }
      }
    }

    if (process.platform === "win32") {
      const candidates = [
        // Packaged app: extraResources ships ".venv" next to the app
        path.join(process.resourcesPath, ".venv", "Scripts", "python.exe"),
        // Dev repo: root venv (preferred)
        path.join(process.cwd(), ".venv", "Scripts", "python.exe"),
        // Dev repo: prefer StemSepApp venv if present
        path.join(__dirname, "../../StemSepApp/.venv/Scripts/python.exe"),
        // Dev repo: prefer StemSepApp venv from current working directory (more robust than __dirname)
        path.join(process.cwd(), "StemSepApp", ".venv", "Scripts", "python.exe"),
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
      path.join(process.cwd(), ".venv", "bin", "python"),
      path.join(__dirname, "../../StemSepApp/.venv/bin/python"),
      path.join(process.cwd(), "StemSepApp", ".venv", "bin", "python"),
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

function resolveAssetsDirForLocalOps(): string | null {
  const candidates: string[] = [
    path.join(process.resourcesPath, "StemSepApp", "assets"),
    path.join(__dirname, "../../StemSepApp/assets"),
    path.join(process.cwd(), "StemSepApp", "assets"),
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

type CatalogSelectionType = "model" | "recipe" | "workflow";

function resolveCatalogRuntimePath(): string | null {
  const assetsDir = resolveAssetsDirForLocalOps();
  if (!assetsDir) return null;

  const runtimePath = path.join(assetsDir, "catalog.runtime.json");
  try {
    return fs.existsSync(runtimePath) ? runtimePath : null;
  } catch {
    return null;
  }
}

function normalizeSelectionType(value: unknown): CatalogSelectionType | null {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "model" || normalized === "recipe" || normalized === "workflow") {
    return normalized;
  }
  return null;
}

function normalizeSelectionEnvelope(
  value: any,
): {
  selectionType: CatalogSelectionType;
  selectionId: string;
  selectionEnvelope: Record<string, any>;
} | null {
  const selectionType = normalizeSelectionType(
    value?.selectionType ?? value?.selection_type,
  );
  const selectionId = String(
    value?.selectionId ?? value?.selection_id ?? "",
  ).trim();
  if (!selectionType || !selectionId) {
    return null;
  }
  return {
    selectionType,
    selectionId,
    selectionEnvelope: {
      ...(typeof value === "object" && value !== null ? value : {}),
      selectionType,
      selectionId,
      selection_type: selectionType,
      selection_id: selectionId,
    },
  };
}

async function resolveSelectionExecutionPlan(
  selectionEnvelope: any,
  config?: Record<string, any>,
) {
  const normalized =
    normalizeSelectionEnvelope(selectionEnvelope) ||
    normalizeSelectionEnvelope({
      selectionType: config?.selectionType ?? config?.selection_type,
      selectionId: config?.selectionId ?? config?.selection_id,
    });
  if (!normalized) {
    return null;
  }
  const executionPlan = await sendPythonCommandWithRetry(
    "resolve_execution_plan",
    {
      selection_type: normalized.selectionType,
      selection_id: normalized.selectionId,
      config: config || {},
    },
    30000,
  );
  return executionPlan
    ? {
        ...executionPlan,
        config: config || {},
        selectionType:
          executionPlan.selectionType ??
          executionPlan.selection_type ??
          normalized.selectionType,
        selectionId:
          executionPlan.selectionId ??
          executionPlan.selection_id ??
          normalized.selectionId,
        selectionEnvelope:
          executionPlan.selectionEnvelope ??
          executionPlan.selection_envelope ??
          normalized.selectionEnvelope ??
          null,
        resolvedBundle:
          executionPlan.resolvedBundle ??
          executionPlan.resolved_bundle ??
          null,
      }
    : null;
}

function urlBasename(url: unknown): string | null {
  if (typeof url !== "string") return null;
  const u = url.split("?")[0].replace(/\/+$/, "");
  const parts = u.split("/");
  const last = parts[parts.length - 1];
  return last ? last : null;
}

function removeModelLocal(modelId: string): { success: true; removedFiles: string[] } {
  const modelsDir = getStoredModelsDir();
  if (!modelsDir) {
    throw new Error("Models directory is not configured.");
  }

  const assetsDir = resolveAssetsDirForLocalOps();
  if (!assetsDir) {
    throw new Error("Assets directory not found (cannot resolve model registry).");
  }

  const registryPath = resolveCatalogRuntimePath();
  let model: any = null;
  try {
    if (!registryPath) {
      throw new Error("No catalog runtime manifest found.");
    }
    const raw = fs.readFileSync(registryPath, "utf-8");
    const json = JSON.parse(raw);
    const models = Array.isArray(json?.models) ? json.models : [];
    model = models.find((m: any) => m && m.id === modelId) || null;
  } catch (e: any) {
    throw new Error(
      `Failed to read model registry at ${registryPath || "<missing>"}: ${e?.message || String(e)}`,
    );
  }

  const removed: string[] = [];
  const links = model?.links && typeof model.links === "object" ? model.links : null;
  const ckptBase = urlBasename(links?.checkpoint);
  const cfgBase = urlBasename(links?.config);

  // NOTE: Avoid deleting generic config.yaml/config.yml because it may be shared between models.
  const genericConfig = new Set(["config.yaml", "config.yml"]);

  const candidateNames = new Set<string>();

  if (ckptBase) candidateNames.add(ckptBase);
  if (cfgBase && !genericConfig.has(cfgBase.toLowerCase())) candidateNames.add(cfgBase);

  const addArtifactNames = (artifacts: any) => {
    for (const artifact of Array.isArray(artifacts) ? artifacts : []) {
      const filenames = [
        artifact?.filename,
        artifact?.relative_path,
        artifact?.relativePath,
        artifact?.canonical_path,
        artifact?.canonicalPath,
      ];
      for (const filename of filenames) {
        const basename = urlBasename(filename);
        if (!basename) continue;
        if (genericConfig.has(basename.toLowerCase())) continue;
        candidateNames.add(basename);
      }
    }
  };

  addArtifactNames(model?.download?.artifacts);
  addArtifactNames(model?.installation?.artifacts);
  addArtifactNames(model?.legacy_artifacts);

  // Per-model aliases we create / accept
  for (const ext of [
    ".ckpt",
    ".chpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".onnx",
    ".yaml",
    ".yml",
  ]) {
    candidateNames.add(`${modelId}${ext}`);
  }

  // Known special-case aliases
  if (modelId === "mel-band-roformer-kim") {
    candidateNames.add("MelBandRoformer.ckpt");
    candidateNames.add("vocals_mel_band_roformer.ckpt");
    candidateNames.add("vocals_mel_band_roformer.yaml");
  }

  for (const name of Array.from(candidateNames)) {
    const p = path.join(modelsDir, name);
    try {
      if (fs.existsSync(p) && fs.statSync(p).isFile()) {
        fs.unlinkSync(p);
        removed.push(p);
      }
    } catch {
      // ignore individual failures to keep batch delete resilient
    }
  }

  return { success: true, removedFiles: removed };
}

let mainWindow: InstanceType<typeof BrowserWindow> | null = null;
let isCreatingMainWindow = false;

const lastProgressByJobId = new Map<string, number>();
const activeSelectionJobIds = new Set<string>();
const modelInfoByJobId = new Map<
  string,
  {
    requestedModelId: string;
    effectiveModelId: string;
    inputFile: string;
    finalOutputDir: string;
    previewDir: string;
    sourceAudioProfile?: AudioSourceProfile;
    stagingDecision?: SourceStagingDecision;
    outputFiles?: Record<string, string>;
  }
>();

const backendBridge = createBackendBridge({
  log,
  getStoredHuggingFaceToken,
  getStoredModelsDir,
  resolveBundledPythonForRustBackend,
  shouldUseRustBackend,
  getIsAppQuitting: () => isAppQuitting,
  getMainWindow: () => mainWindow,
  isBackendBusy: () =>
    playbackCaptureSessions.size > 0 || activeSelectionJobIds.size > 0,
  onBackendMessage: (msg) => routeBackendMessage(msg),
});

const {
  ensureBackend,
  sendBackendCommand: sendPythonCommand,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
  requestBridgeRestart,
  restartBackendAndWait,
  startHealthChecks,
  stopHealthChecks,
  shutdown: shutdownBackendBridge,
} = backendBridge;

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

function createWindow() {
  return createMainWindow({
    log,
    getMainWindow: () => mainWindow,
    setMainWindow: (win) => {
      mainWindow = win;
    },
    getIsCreatingMainWindow: () => isCreatingMainWindow,
    setIsCreatingMainWindow: (value) => {
      isCreatingMainWindow = value;
    },
  });
}

// Suppress Autofill warnings
app.commandLine.appendSwitch("disable-features", "AutofillServer");
app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");

app.whenReady().then(() => {
  cleanupPreviewCache();
  session.defaultSession.setDisplayMediaRequestHandler(
    async (_request, callback) => {
      try {
        const sources = await desktopCapturer.getSources({
          types: ["screen"],
          thumbnailSize: { width: 1, height: 1 },
          fetchWindowIcons: false,
        });
        callback({
          video: sources[0],
          audio: "loopback",
        });
      } catch (error: any) {
        log("[capture] failed to set display media handler", {
          error: error?.message || String(error),
        });
        callback({ video: undefined, audio: undefined });
      }
    },
    { useSystemPicker: false },
  );

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
  if (process.env.STEMSEP_QOBUZ_CAPTURE_SMOKETEST_QUERY) {
    setTimeout(() => {
      void maybeRunQobuzCaptureSmokeTest();
    }, 2500);
  }

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
  closeAllRemoteWindows();
  shutdownBackendBridge();
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

function extractChunkCounters(message: string | undefined): {
  chunksDone?: number;
  chunksTotal?: number;
} {
  const text = String(message || "");
  const match = text.match(
    /\b(?:chunk|chunks|segment|segments)\D+(\d+)\D+(\d+)\b/i,
  );
  if (!match) return {};

  const chunksDone = Number(match[1]);
  const chunksTotal = Number(match[2]);
  if (!Number.isFinite(chunksDone) || !Number.isFinite(chunksTotal)) {
    return {};
  }
  return { chunksDone, chunksTotal };
}

function getStepLabelFromMeta(
  meta: Record<string, any> | undefined,
): string | undefined {
  if (!meta || typeof meta !== "object") return undefined;
  const step =
    meta.step && typeof meta.step === "object"
      ? (meta.step as Record<string, any>)
      : undefined;
  const phase = typeof meta.phase === "string" ? meta.phase : undefined;
  const stepName =
    step && typeof step.name === "string" && step.name.trim()
      ? step.name.trim()
      : undefined;
  const modelId =
    step && typeof step.model_id === "string" && step.model_id.trim()
      ? step.model_id.trim()
      : undefined;

  if (stepName && modelId) return `${stepName} (${modelId})`;
  if (stepName) return stepName;
  if (modelId) return modelId;
  if (!phase) return undefined;
  return phase
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const createLineBuffer = (onLine: (line: string) => void) => {
  let buffer = "";
  return (data: Buffer) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onLine(line);
    }
  };
};

function emitNormalizedSeparationEvent(msg: any) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const type = String(msg?.type || "");
  const jobId = typeof msg?.job_id === "string" ? msg.job_id : undefined;
  if (!jobId) return;

  const meta =
    msg?.meta && typeof msg.meta === "object"
      ? (msg.meta as Record<string, any>)
      : undefined;
  const step =
    meta?.step && typeof meta.step === "object"
      ? (meta.step as Record<string, any>)
      : undefined;
  const progress = Number(msg?.progress);
  const durationSeconds = Number(msg?.duration_seconds);
  const { chunksDone, chunksTotal } = extractChunkCounters(
    typeof msg?.message === "string" ? msg.message : undefined,
  );

  mainWindow.webContents.send("separation-progress-event", {
    jobId,
    kind:
      type === "separation_step_started"
        ? "step_started"
        : type === "separation_step_completed"
          ? "step_completed"
          : type === "separation_complete"
            ? "completed"
            : type === "separation_cancelled"
              ? "cancelled"
            : type === "separation_error"
              ? "error"
              : type === "separation_started"
                ? "job_started"
                : "progress",
    progress: Number.isFinite(progress) ? progress : undefined,
    message: typeof msg?.message === "string" ? msg.message : undefined,
    phase: typeof meta?.phase === "string" ? meta.phase : undefined,
    stepId:
      step && (typeof step.index === "number" || typeof step.index === "string")
        ? `${meta?.phase || "step"}:${String(step.index)}:${String(step.model_id || step.name || "")}`
        : undefined,
    stepLabel: getStepLabelFromMeta(meta),
    stepIndex:
      step && Number.isFinite(Number(step.index)) ? Number(step.index) : undefined,
    stepCount:
      step && Number.isFinite(Number(step.total)) ? Number(step.total) : undefined,
    modelId:
      step && typeof step.model_id === "string" ? step.model_id : undefined,
    chunksDone,
    chunksTotal,
    elapsedMs:
      Number.isFinite(durationSeconds) && durationSeconds >= 0
        ? Math.round(durationSeconds * 1000)
        : undefined,
    ts:
      Number.isFinite(Number(msg?.ts)) && Number(msg.ts) > 0
        ? Number(msg.ts)
        : undefined,
    meta,
    error: typeof msg?.error === "string" ? msg.error : undefined,
  });
}

function routeBackendMessage(msg: any) {
  if (msg?.type === "bridge_ready") {
    log(
      `Bridge ready! Capabilities: ${msg.capabilities?.join(", ")}, Models: ${msg.models_count}, Recipes: ${msg.recipes_count}`,
    );
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("bridge-ready", {
        capabilities: msg.capabilities,
        modelsCount: msg.models_count,
        recipesCount: msg.recipes_count,
      });
    }
  } else if (
    msg?.type === "progress" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-progress", {
      modelId: msg.model_id,
      progress: msg.progress,
      stage: msg.stage,
      artifactIndex: msg.artifactIndex,
      artifactCount: msg.artifactCount,
      currentFile: msg.currentFile,
      currentRelativePath: msg.currentRelativePath,
      currentSource: msg.currentSource,
      verified: msg.verified,
      message: msg.message,
    });
  } else if (
    msg?.type === "complete" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-complete", {
      modelId: msg.model_id,
      artifactCount: msg.artifactCount,
      stage: msg.stage,
      verified: msg.verified,
    });
  } else if (
    msg?.type === "error" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-error", {
      modelId: msg.model_id,
      error: msg.error,
    });
  } else if (
    msg?.type === "paused" &&
    msg?.model_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("download-paused", {
      modelId: msg.model_id,
      artifactIndex: msg.artifactIndex,
      artifactCount: msg.artifactCount,
      currentFile: msg.currentFile,
      currentRelativePath: msg.currentRelativePath,
      currentSource: msg.currentSource,
      stage: msg.stage,
      verified: msg.verified,
      progress: msg.progress,
    });
  } else if (
    msg?.type === "separation_progress" &&
    msg?.job_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    const jobId = msg.job_id;
    activeSelectionJobIds.add(jobId);
    const next = Number(msg.progress);
    const prev = lastProgressByJobId.get(jobId) ?? 0;
    const clamped = Number.isFinite(next)
      ? Math.max(prev, Math.max(0, Math.min(100, next)))
      : prev;
    lastProgressByJobId.set(jobId, clamped);
    mainWindow.webContents.send("separation-progress", {
      jobId,
      progress: clamped,
      message: msg.message,
      device: msg.device,
      meta: msg.meta,
    });
    emitNormalizedSeparationEvent({
      ...msg,
      progress: clamped,
    });
  } else if (
    (msg?.type === "separation_step_started" ||
      msg?.type === "separation_step_completed" ||
      msg?.type === "separation_started") &&
    msg?.job_id &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    activeSelectionJobIds.add(msg.job_id);
    emitNormalizedSeparationEvent(msg);
  } else if (
    msg?.type === "separation_error" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    emitNormalizedSeparationEvent(msg);
    if (msg?.job_id) {
      activeSelectionJobIds.delete(msg.job_id);
      lastProgressByJobId.delete(msg.job_id);
      modelInfoByJobId.delete(msg.job_id);
    }
    mainWindow.webContents.send("separation-error", {
      jobId: msg.job_id,
      error: msg.error || "Separation failed",
    });
  } else if (
    msg?.type === "separation_cancelled" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    emitNormalizedSeparationEvent({
      ...msg,
      type: "separation_cancelled",
    });
    if (msg?.job_id) {
      activeSelectionJobIds.delete(msg.job_id);
      lastProgressByJobId.delete(msg.job_id);
      modelInfoByJobId.delete(msg.job_id);
    }
  } else if (
    msg?.type === "separation_complete" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    let normalizedOutputFiles = msg.output_files || {};
    if (msg?.job_id) {
      activeSelectionJobIds.delete(msg.job_id);
      const state = modelInfoByJobId.get(msg.job_id);
      if (state) {
        const resolved = resolvePlaybackStems(msg.output_files || {}, {
          sourceKind: "preview_cache",
          previewDir: state.previewDir,
          savedDir: state.finalOutputDir,
        });
        normalizedOutputFiles =
          Object.keys(resolved.stems).length > 0 ? resolved.stems : normalizedOutputFiles;
        state.outputFiles = normalizedOutputFiles;
      }
    }
    emitNormalizedSeparationEvent(msg);
    if (msg?.job_id) {
      lastProgressByJobId.delete(msg.job_id);
    }
    mainWindow.webContents.send("separation-complete", {
      outputFiles: normalizedOutputFiles,
      jobId: msg.job_id,
    });
  } else if (
    msg?.type === "playback_capture_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    const sessionInfo =
      typeof msg?.capture_id === "string"
        ? playbackCaptureSessions.get(msg.capture_id)
        : null;
    emitPlaybackCaptureProgress({
      provider: sessionInfo?.provider || "qobuz",
      captureId: msg.capture_id,
      status: msg.status,
      detail: msg.detail,
      progress:
        typeof msg.progress === "number" ? msg.progress : undefined,
      percent:
        typeof msg.progress === "number"
          ? `${Math.round(msg.progress * 100)}%`
          : undefined,
      elapsedSec:
        typeof msg.elapsed_sec === "number" ? msg.elapsed_sec : undefined,
      error: typeof msg.error === "string" ? msg.error : undefined,
    });
  } else if (
    msg?.type === "youtube_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("youtube-progress", msg);
    mainWindow.webContents.send("remote-resolve-progress", {
      provider: "youtube",
      ...msg,
    });
  } else if (
    msg?.type === "quality_progress" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("quality-progress", msg);
  } else if (
    msg?.type === "quality_complete" &&
    mainWindow &&
    !mainWindow.isDestroyed()
  ) {
    mainWindow.webContents.send("quality-complete", msg);
  }
}

let getGpuDevicesInflight: Promise<any> | null = null;
let getRuntimeFingerprintInflight: Promise<any> | null = null;
let getSystemRuntimeInfoInflight: Promise<any> | null = null;

let gpuDevicesCache: { value: any; expiresAt: number } | null = null;
let runtimeFingerprintCache: { value: any; expiresAt: number } | null = null;

function getGpuDevicesDeduped(): Promise<any> {
  if (getGpuDevicesInflight) return getGpuDevicesInflight;
  getGpuDevicesInflight = sendPythonCommandWithRetry(
    "get-gpu-devices",
    {},
    30000,
  ).finally(() => {
    getGpuDevicesInflight = null;
  });
  return getGpuDevicesInflight;
}

function getRuntimeFingerprintDeduped(): Promise<any> {
  if (getRuntimeFingerprintInflight) return getRuntimeFingerprintInflight;
  getRuntimeFingerprintInflight = sendPythonCommandWithRetry(
    "get_runtime_fingerprint",
    {},
    20000,
    1,
  ).finally(() => {
    getRuntimeFingerprintInflight = null;
  });
  return getRuntimeFingerprintInflight;
}

async function getGpuDevicesCached(): Promise<{ data: any; fromCache: boolean }> {
  if (gpuDevicesCache && gpuDevicesCache.expiresAt > Date.now()) {
    return { data: gpuDevicesCache.value, fromCache: true };
  }
  if (isPlaybackCaptureActive() || activeSelectionJobIds.size > 0) {
    return { data: gpuDevicesCache?.value || [], fromCache: true };
  }
  const data = await getGpuDevicesDeduped();
  gpuDevicesCache = {
    value: data,
    expiresAt: Date.now() + SYSTEM_RUNTIME_TTL_MS,
  };
  return { data, fromCache: false };
}

async function getRuntimeFingerprintCached(): Promise<{
  data: any | null;
  fromCache: boolean;
  error: string | null;
}> {
  if (runtimeFingerprintCache && runtimeFingerprintCache.expiresAt > Date.now()) {
    return { data: runtimeFingerprintCache.value, fromCache: true, error: null };
  }
  if (isPlaybackCaptureActive() || activeSelectionJobIds.size > 0) {
    return {
      data: runtimeFingerprintCache?.value || null,
      fromCache: true,
      error: runtimeFingerprintCache
        ? null
        : activeSelectionJobIds.size > 0
          ? "Runtime fingerprint paused during active separation."
          : "Runtime fingerprint paused during playback capture.",
    };
  }

  try {
    const data = await getRuntimeFingerprintDeduped();
    runtimeFingerprintCache = {
      value: data,
      expiresAt: Date.now() + SYSTEM_RUNTIME_TTL_MS,
    };
    return { data, fromCache: false, error: null };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error || "unknown error");
    return { data: null, fromCache: false, error: message };
  }
}

async function getSystemRuntimeInfoCached(): Promise<any> {
  if (getSystemRuntimeInfoInflight) return getSystemRuntimeInfoInflight;

  getSystemRuntimeInfoInflight = (async () => {
    const [gpu, runtimeFingerprint] = await Promise.all([
      getGpuDevicesCached(),
      getRuntimeFingerprintCached(),
    ]);

    return {
      fetchedAt: new Date().toISOString(),
      cache: {
        ttlMs: SYSTEM_RUNTIME_TTL_MS,
        gpuSource: gpu.fromCache ? "cache" : "fresh",
        runtimeFingerprintSource: runtimeFingerprint.fromCache
          ? "cache"
          : runtimeFingerprint.error
            ? "error"
            : "fresh",
      },
      gpu: gpu.data,
      runtimeFingerprint: runtimeFingerprint.data,
      runtimeFingerprintError: runtimeFingerprint.error,
      previewCachePolicy: {
        baseDir: getPreviewCacheBaseDir(),
        keepLast: PREVIEW_CACHE_KEEP_LAST,
        maxAgeDays: PREVIEW_CACHE_MAX_AGE_DAYS,
        ephemeral: true,
      },
    };
  })().finally(() => {
    getSystemRuntimeInfoInflight = null;
  });

  return getSystemRuntimeInfoInflight;
}

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
      requestId,
    }: {
      sourceFiles: Record<string, string>;
      exportPath: string;
      format: string;
      bitrate: string;
      requestId?: string;
    },
  ) => {
    const resolvedRequestId = requestId || randomUUID().slice(0, 8);
    // Export is a pure file operation (copy/transcode) and should not require a backend.
    // This also avoids failures when the legacy python-bridge proxy is not present.
    log("[export-files] local export requested", {
      requestId: resolvedRequestId,
      stems: Object.keys(sourceFiles || {}),
      exportPath,
      format,
      bitrate,
    });
    try {
      const res = await exportFilesLocal({
        sourceFiles,
        exportPath,
        format,
        bitrate,
        requestId: resolvedRequestId,
        onProgress: emitExportProgress,
        log,
      });
      log("[export-files] local export complete", {
        requestId: resolvedRequestId,
        exported: Object.keys(res?.exported || {}),
      });
      return {
        success: true,
        exported: res.exported,
        requestId: resolvedRequestId,
      };
    } catch (e: any) {
      emitExportProgress({
        requestId: resolvedRequestId,
        status: "failed",
        error: e?.message || String(e),
      });
      log("[export-files] local export FAILED", {
        requestId: resolvedRequestId,
        error: e?.message || String(e),
      });
      return {
        success: false,
        error: e?.message || String(e),
        code: e?.code || "MISSING_SOURCE_FILE",
        hint:
          e?.hint ||
          "Export source is unavailable. Run a new separation to refresh cache files.",
        requestId: resolvedRequestId,
      };
    }
  },
);

registerQueueIpcHandlers({
  ipcMain,
  sendBackendCommand: sendPythonCommand,
});

registerRuntimeIpcHandlers({
  ipcMain,
  getGpuDevicesCached,
  getSystemRuntimeInfoCached,
});

registerSeparationIpcHandlers({
  ipcMain,
  log,
  createPreviewDirForInput,
  resolveEffectiveModelId,
  resolveSelectionExecutionPlan,
  ensureWavInput,
  normalizeSelectionType,
  sendBackendCommand: sendPythonCommand,
});

registerSelectionIpcHandlers({
  ipcMain,
  sendBackendCommand: sendPythonCommandWithRetry,
});

registerModelIpcHandlers({
  ipcMain,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
});

registerDownloadIpcHandlers({
  ipcMain,
  sendBackendCommand: sendPythonCommand,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
  removeModelLocal,
  log,
});

registerQualityIpcHandlers({
  ipcMain,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
});

registerLibraryCaptureIpcHandlers({
  ipcMain,
  log,
  refreshQobuzPlaybackDeviceMappings,
  getCaptureEnvironmentStatusForQobuz,
  writeAppConfig,
  qobuzSinkIdByPlaybackDeviceId,
  getRemoteProviderConfig,
  checkLibraryProviderAuthenticated,
  isQobuzPlaybackCaptureActive,
  openRemoteSourceAuthWindow,
  scrapeProviderItems,
  searchQobuzCatalogViaApi,
  remoteCatalogCache,
  isPlaybackCaptureActive,
  getCachedLibraryItem,
  getNativePlaybackDevices,
  resolveCaptureLaunchTarget,
  computeCaptureQualityMode,
  startQobuzPlaybackCaptureForItem,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
  playbackCaptureSessions,
  lastPlaybackCaptureProgressById,
  abortPlaybackCaptureSession,
});

registerRemoteImportIpcHandlers({
  ipcMain,
  log,
  getRemoteProviderConfig,
  openRemoteSourceAuthWindow,
  scrapeRemoteCollection,
  remoteCatalogCache,
  emitRemoteResolveProgress,
  resolveRemoteDownloadInfo,
  downloadRemoteFile,
  extractArchiveForRemoteTrack,
  probeAudioFile,
  sendPythonCommand,
});

registerDialogIpcHandlers({
  ipcMain,
  getMainWindow: () => mainWindow,
  log,
  resolvePlaybackFilePath,
  classifyMissingAudioPath,
  resolvePlaybackStems,
});

registerConfigIpcHandlers({
  ipcMain,
  log,
  writeAppConfig,
  getStoredHuggingFaceToken,
  setStoredHuggingFaceToken,
  requestBridgeRestart,
  restartBackendAndWait,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
});

registerWatchIpcHandlers({
  ipcMain,
  getMainWindow: () => mainWindow,
  log,
});

