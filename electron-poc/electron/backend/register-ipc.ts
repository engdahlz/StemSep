import type { IpcMain, BrowserWindow } from "electron";
import { registerExportIpcHandlers } from "./ipc-export";
import { registerSelectionIpcHandlers } from "./ipc-selection";
import { registerSeparationIpcHandlers } from "./ipc-separation";
import { registerQueueIpcHandlers } from "./ipc-queue";
import { registerRuntimeIpcHandlers } from "./ipc-runtime";
import { registerModelIpcHandlers } from "./ipc-models";
import { registerQualityIpcHandlers } from "./ipc-quality";
import { registerDownloadIpcHandlers } from "./ipc-downloads";
import { registerDialogIpcHandlers } from "../system/dialogs";
import { registerConfigIpcHandlers } from "../system/config";
import { registerWatchIpcHandlers } from "../system/watch";
import { registerLibraryCaptureIpcHandlers } from "../remote/ipc-library";
import { registerRemoteImportIpcHandlers } from "../remote/ipc-remote";

type LogFn = (message: string, ...args: any[]) => void;

export function registerAppIpcHandlers({
  ipcMain,
  log,
  sendBackendCommand,
  sendBackendCommandWithRetry,
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
  getMainWindow,
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
}: {
  ipcMain: IpcMain;
  log: LogFn;
  sendBackendCommand: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
  ) => Promise<any>;
  sendBackendCommandWithRetry: (
    command: string,
    payload?: Record<string, any>,
    timeoutMs?: number,
    maxRetries?: number,
  ) => Promise<any>;
  createPreviewDirForInput: (inputFile: string) => string;
  resolveEffectiveModelId: (modelId: string, ensembleConfig?: any) => string;
  resolveSelectionExecutionPlan: (
    selectionEnvelope: Record<string, any> | undefined,
    context: Record<string, any>,
  ) => Promise<any>;
  ensureWavInput: (inputFile: string, previewDir: string) => Promise<any>;
  normalizeSelectionType: (
    selectionType?: string | null,
  ) => "model" | "recipe" | "workflow" | null;
  getGpuDevicesCached: () => Promise<any>;
  getSystemRuntimeInfoCached: () => Promise<any>;
  removeModelLocal: (modelId: string) => any;
  exportFilesLocal: (...args: any[]) => Promise<any>;
  emitExportProgress: (payload: any) => void;
  getMainWindow: () => BrowserWindow | null;
  resolvePlaybackFilePath: (...args: any[]) => any;
  classifyMissingAudioPath: (...args: any[]) => any;
  resolvePlaybackStems: (...args: any[]) => any;
  writeAppConfig: (patch: Record<string, any>) => any;
  getStoredHuggingFaceToken: () => string | null;
  setStoredHuggingFaceToken: (token: string | null) => any;
  requestBridgeRestart: (...args: any[]) => void;
  restartBackendAndWait: (...args: any[]) => Promise<any>;
  getRemoteProviderConfig: (provider: string) => any;
  openRemoteSourceAuthWindow: (provider: string) => any;
  scrapeRemoteCollection: (...args: any[]) => Promise<any>;
  remoteCatalogCache: Map<any, any>;
  emitRemoteResolveProgress: (payload: any) => void;
  resolveRemoteDownloadInfo: (...args: any[]) => Promise<any>;
  downloadRemoteFile: (...args: any[]) => Promise<any>;
  extractArchiveForRemoteTrack: (...args: any[]) => Promise<any>;
  probeAudioFile: (...args: any[]) => Promise<any>;
  refreshQobuzPlaybackDeviceMappings: () => Promise<any>;
  getCaptureEnvironmentStatusForQobuz: () => Promise<any>;
  qobuzSinkIdByPlaybackDeviceId: Map<string, string>;
  checkLibraryProviderAuthenticated: (...args: any[]) => Promise<any>;
  isQobuzPlaybackCaptureActive: () => boolean;
  scrapeProviderItems: (...args: any[]) => Promise<any>;
  searchQobuzCatalogViaApi: (...args: any[]) => Promise<any>;
  isPlaybackCaptureActive: () => boolean;
  getCachedLibraryItem: (...args: any[]) => any;
  getNativePlaybackDevices: () => Promise<any>;
  resolveCaptureLaunchTarget: (...args: any[]) => any;
  computeCaptureQualityMode: (...args: any[]) => any;
  startQobuzPlaybackCaptureForItem: (...args: any[]) => Promise<any>;
  playbackCaptureSessions: Map<string, any>;
  lastPlaybackCaptureProgressById: Map<string, any>;
  abortPlaybackCaptureSession: (...args: any[]) => Promise<any>;
}) {
  registerQueueIpcHandlers({
    ipcMain,
    sendBackendCommand,
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
    sendBackendCommand,
  });

  registerSelectionIpcHandlers({
    ipcMain,
    sendBackendCommand: sendBackendCommandWithRetry,
  });

  registerModelIpcHandlers({
    ipcMain,
    sendBackendCommandWithRetry,
  });

  registerDownloadIpcHandlers({
    ipcMain,
    sendBackendCommand,
    sendBackendCommandWithRetry,
    removeModelLocal,
    log,
  });

  registerExportIpcHandlers({
    ipcMain,
    exportFilesLocal,
    emitExportProgress,
    log,
  });

  registerQualityIpcHandlers({
    ipcMain,
    sendBackendCommandWithRetry,
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
    sendBackendCommandWithRetry,
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
    sendPythonCommand: sendBackendCommand,
  });

  registerDialogIpcHandlers({
    ipcMain,
    getMainWindow,
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
    sendBackendCommandWithRetry,
  });

  registerWatchIpcHandlers({
    ipcMain,
    getMainWindow,
    log,
  });
}
