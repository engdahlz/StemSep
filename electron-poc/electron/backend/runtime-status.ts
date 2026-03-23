import type { BrowserWindow } from "electron";

type AudioSourceProfile = {
  [key: string]: any;
};

type SourceStagingDecision = {
  [key: string]: any;
};

type SendBackendCommandWithRetry = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
  maxRetries?: number,
) => Promise<any>;

type ModelInfoByJobState = {
  requestedModelId: string;
  effectiveModelId: string;
  inputFile: string;
  finalOutputDir: string;
  previewDir: string;
  sourceAudioProfile?: AudioSourceProfile;
  stagingDecision?: SourceStagingDecision;
  outputFiles?: Record<string, string>;
};

const SYSTEM_RUNTIME_TTL_MS = 30_000;

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

export function createBackendRuntimeController({
  getMainWindow,
  activeSelectionJobIds,
  lastProgressByJobId,
  modelInfoByJobId,
  playbackCaptureSessions,
  emitPlaybackCaptureProgress,
  lastPlaybackCaptureProgressById,
  resolvePlaybackStems,
  sendBackendCommandWithRetry,
  isPlaybackCaptureActive,
  getPreviewCacheBaseDir,
  previewCacheKeepLast,
  previewCacheMaxAgeDays,
}: {
  getMainWindow: () => BrowserWindow | null;
  activeSelectionJobIds: Set<string>;
  lastProgressByJobId: Map<string, number>;
  modelInfoByJobId: Map<string, ModelInfoByJobState>;
  playbackCaptureSessions: Map<string, any>;
  emitPlaybackCaptureProgress: (payload: any) => void;
  lastPlaybackCaptureProgressById: Map<string, any>;
  resolvePlaybackStems: (
    stems: Record<string, string>,
    options: Record<string, any>,
  ) => { stems: Record<string, string> };
  sendBackendCommandWithRetry: SendBackendCommandWithRetry;
  isPlaybackCaptureActive: () => boolean;
  getPreviewCacheBaseDir: () => string;
  previewCacheKeepLast: number;
  previewCacheMaxAgeDays: number;
}) {
  let getGpuDevicesInflight: Promise<any> | null = null;
  let getRuntimeFingerprintInflight: Promise<any> | null = null;
  let getSystemRuntimeInfoInflight: Promise<any> | null = null;

  let gpuDevicesCache: { value: any; expiresAt: number } | null = null;
  let runtimeFingerprintCache: { value: any; expiresAt: number } | null = null;

  function emitNormalizedSeparationEvent(msg: any) {
    const mainWindow = getMainWindow();
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
        step && Number.isFinite(Number(step.index))
          ? Number(step.index)
          : undefined,
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
    const mainWindow = getMainWindow();
    if (msg?.type === "bridge_ready") {
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
            Object.keys(resolved.stems).length > 0
              ? resolved.stems
              : normalizedOutputFiles;
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

  function getGpuDevicesDeduped(): Promise<any> {
    if (getGpuDevicesInflight) return getGpuDevicesInflight;
    getGpuDevicesInflight = sendBackendCommandWithRetry(
      "get-gpu-devices",
      {},
      30_000,
    ).finally(() => {
      getGpuDevicesInflight = null;
    });
    return getGpuDevicesInflight;
  }

  function getRuntimeFingerprintDeduped(): Promise<any> {
    if (getRuntimeFingerprintInflight) return getRuntimeFingerprintInflight;
    getRuntimeFingerprintInflight = sendBackendCommandWithRetry(
      "get_runtime_fingerprint",
      {},
      20_000,
      1,
    ).finally(() => {
      getRuntimeFingerprintInflight = null;
    });
    return getRuntimeFingerprintInflight;
  }

  async function getGpuDevicesCached(): Promise<{
    data: any;
    fromCache: boolean;
  }> {
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
    if (
      runtimeFingerprintCache &&
      runtimeFingerprintCache.expiresAt > Date.now()
    ) {
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
          keepLast: previewCacheKeepLast,
          maxAgeDays: previewCacheMaxAgeDays,
          ephemeral: true,
        },
      };
    })().finally(() => {
      getSystemRuntimeInfoInflight = null;
    });

    return getSystemRuntimeInfoInflight;
  }

  return {
    routeBackendMessage,
    getGpuDevicesCached,
    getSystemRuntimeInfoCached,
  };
}
