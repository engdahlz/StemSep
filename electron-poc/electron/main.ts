import {
  app,
  BrowserWindow,
  ipcMain,
} from "electron";
import { createBackendBridge } from "./backend/bridge";
import { createBackendRuntimeController } from "./backend/runtime-status";
import {
  createSelectionPlanResolver,
  normalizeSelectionType,
} from "./backend/selection-plan";
import { createModelRemovalController } from "./backend/model-removal";
import { registerAppIpcHandlers } from "./backend/register-ipc";
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
import { createRemoteDownloadController } from "./remote/download-audio";
import { createRemoteBrowserLibraryController } from "./remote/library-browser";
import { createRemoteLibraryController } from "./remote/library-core";
import { createQobuzCaptureController } from "./remote/qobuz-capture";
import { createRemoteWindowController } from "./remote/windows";
import {
  getRemoteProviderConfig,
  getRemoteProviderCacheDir,
  getRemoteSession,
  type ActiveLibraryProvider,
  type RemoteLibraryProvider,
} from "./remote/provider-config";
import { createAppConfigStore } from "./system/app-config-store";
import {
  createHuggingFaceAuthWindowController,
  installApplicationMenu,
} from "./system/huggingface-auth";
import { createMainWindow } from "./windows/main-window";
import { createLogger } from "./system/logger";
import {
  resolveBundledPythonForRustBackend,
  shouldUseRustBackend,
} from "./backend/runtime-launch";
import { createSmokeSeparationRunner } from "./backend/smoke-separation";
import {
  enforceSingleInstance,
  initializeAppWhenReady,
  registerAppLifecycleHandlers,
  registerGlobalErrorHandlers,
} from "./windows/app-lifecycle";
// const __filename = fileURLToPath(import.meta.url)
// const __dirname = path.dirname(__filename)

const log = createLogger();

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

const appConfigStore = createAppConfigStore({ log });
const {
  writeAppConfig,
  getStoredModelsDir,
  getStoredCaptureOutputDeviceId,
  getStoredHuggingFaceToken,
  setStoredHuggingFaceToken,
} = appConfigStore;

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
  sendBackendCommand: (command, payload, timeoutMs) =>
    sendPythonCommand(command, payload, timeoutMs),
  sendBackendCommandWithRetry: (command, payload, timeoutMs, maxRetries) =>
    sendPythonCommandWithRetry(command, payload, timeoutMs, maxRetries),
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
registerGlobalErrorHandlers({ log });

let isAppQuitting = false;

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

const { openHuggingFaceAuthWindow } = createHuggingFaceAuthWindowController({
  getMainWindow: () => mainWindow,
});

const { removeModelLocal } = createModelRemovalController({
  getStoredModelsDir,
});

let sendPythonCommandWithRetryRef: (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any> = async () => {
  throw new Error("Backend bridge not ready");
};

const {
  routeBackendMessage,
  getGpuDevicesCached,
  getSystemRuntimeInfoCached,
} = createBackendRuntimeController({
  getMainWindow: () => mainWindow,
  activeSelectionJobIds,
  lastProgressByJobId,
  modelInfoByJobId,
  playbackCaptureSessions,
  emitPlaybackCaptureProgress,
  lastPlaybackCaptureProgressById,
  resolvePlaybackStems,
  sendBackendCommandWithRetry: (command, payload, timeoutMs, maxRetries) =>
    sendPythonCommandWithRetryRef(command, payload, timeoutMs, maxRetries),
  isPlaybackCaptureActive,
  getPreviewCacheBaseDir,
  previewCacheKeepLast: PREVIEW_CACHE_KEEP_LAST,
  previewCacheMaxAgeDays: PREVIEW_CACHE_MAX_AGE_DAYS,
});

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
  sendBackendCommand: sendPythonCommand,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
  requestBridgeRestart,
  restartBackendAndWait,
  startHealthChecks,
  shutdown: shutdownBackendBridge,
} = backendBridge;

sendPythonCommandWithRetryRef = sendPythonCommandWithRetry;

const { resolveSelectionExecutionPlan } = createSelectionPlanResolver({
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
});

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

const { maybeRunSmokeSeparation } = createSmokeSeparationRunner({
  log,
  sendBackendCommand: sendPythonCommand,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
});

enforceSingleInstance({
  getMainWindow: () => mainWindow,
  createWindow,
});

app.whenReady().then(() =>
  initializeAppWhenReady({
    cleanupPreviewCache,
    log,
    createWindow,
    maybeRunQobuzCaptureSmokeTest,
    installApplicationMenu: () =>
      installApplicationMenu({
        log,
        openHuggingFaceAuthWindow,
        setStoredHuggingFaceToken,
        requestBridgeRestart,
      }),
    startHealthChecks,
    maybeRunSmokeSeparation,
  }),
);

registerAppLifecycleHandlers({
  log,
  getMainWindow: () => mainWindow,
  createWindow,
  onBeforeQuit: () => {
    isAppQuitting = true;
    closeAllRemoteWindows();
    shutdownBackendBridge();
  },
});

registerAppIpcHandlers({
  ipcMain,
  log,
  sendBackendCommand: sendPythonCommand,
  sendBackendCommandWithRetry: sendPythonCommandWithRetry,
  createPreviewDirForInput,
  resolveEffectiveModelId,
  resolveSelectionExecutionPlan,
  ensureWavInput,
  normalizeSelectionType,
  getGpuDevicesCached,
  getSystemRuntimeInfoCached,
  removeModelLocal,
  exportFilesLocal,
  emitExportProgress,
  getMainWindow: () => mainWindow,
  resolvePlaybackFilePath,
  classifyMissingAudioPath,
  resolvePlaybackStems,
  writeAppConfig,
  getStoredHuggingFaceToken,
  setStoredHuggingFaceToken,
  requestBridgeRestart,
  restartBackendAndWait,
  getRemoteProviderConfig,
  openRemoteSourceAuthWindow,
  scrapeRemoteCollection,
  remoteCatalogCache,
  emitRemoteResolveProgress,
  resolveRemoteDownloadInfo,
  downloadRemoteFile,
  extractArchiveForRemoteTrack,
  probeAudioFile,
  refreshQobuzPlaybackDeviceMappings,
  getCaptureEnvironmentStatusForQobuz,
  qobuzSinkIdByPlaybackDeviceId,
  checkLibraryProviderAuthenticated,
  isQobuzPlaybackCaptureActive,
  scrapeProviderItems,
  searchQobuzCatalogViaApi,
  isPlaybackCaptureActive,
  getCachedLibraryItem,
  getNativePlaybackDevices,
  resolveCaptureLaunchTarget,
  computeCaptureQualityMode,
  startQobuzPlaybackCaptureForItem,
  playbackCaptureSessions,
  lastPlaybackCaptureProgressById,
  abortPlaybackCaptureSession,
});

