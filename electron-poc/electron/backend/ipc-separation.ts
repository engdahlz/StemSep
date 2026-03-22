import { randomUUID } from "crypto";
import type { BrowserWindow, IpcMain } from "electron";

type CatalogSelectionType = "model" | "recipe" | "workflow";

type ModelJobState = {
  requestedModelId: string;
  effectiveModelId: string;
  inputFile: string;
  finalOutputDir: string;
  previewDir: string;
  sourceAudioProfile?: any;
  stagingDecision?: any;
  outputFiles?: Record<string, string>;
};

type EnsureBackend = () => unknown;
type LogFn = (message: string, ...args: any[]) => void;
type ResolveSelectionExecutionPlan = (
  selectionEnvelope: Record<string, any> | undefined,
  context: Record<string, any>,
) => Promise<any>;
type EnsureWavInput = (inputFile: string, previewDir: string) => Promise<any>;
type NormalizeSelectionType = (
  selectionType?: string | null,
) => CatalogSelectionType | null;
type SendBackendCommand = (
  command: string,
  payload?: Record<string, any>,
  timeoutMs?: number,
) => Promise<any>;
type SubscribeBackendEvent = (
  eventType: string,
  handler: (payload: any) => void,
) => () => void;

export function registerSeparationIpcHandlers({
  ipcMain,
  log,
  ensureBackend,
  createPreviewDirForInput,
  resolveEffectiveModelId,
  resolveSelectionExecutionPlan,
  ensureWavInput,
  normalizeSelectionType,
  sendBackendCommand,
  subscribeBackendEvent,
  resolvePlaybackStems,
  modelInfoByJobId,
  lastProgressByJobId,
  activeSelectionJobIds,
  getMainWindow,
}: {
  ipcMain: IpcMain;
  log: LogFn;
  ensureBackend: EnsureBackend;
  createPreviewDirForInput: (inputFile: string) => string;
  resolveEffectiveModelId: (modelId: string, ensembleConfig?: any) => string;
  resolveSelectionExecutionPlan: ResolveSelectionExecutionPlan;
  ensureWavInput: EnsureWavInput;
  normalizeSelectionType: NormalizeSelectionType;
  sendBackendCommand: SendBackendCommand;
  subscribeBackendEvent: SubscribeBackendEvent;
  resolvePlaybackStems: (
    outputFiles?: Record<string, string>,
    playback?: Record<string, any>,
  ) => { stems: Record<string, string>; issues: Record<string, any> };
  modelInfoByJobId: Map<string, ModelJobState>;
  lastProgressByJobId: Map<string, number>;
  activeSelectionJobIds: Set<string>;
  getMainWindow: () => BrowserWindow | null;
}) {
  ipcMain.handle(
    "separation-preflight",
    async (
      _event,
      {
        inputFile,
        modelId,
        outputDir,
        selectionType,
        selectionId,
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
        pipelineConfig,
        workflow,
        runtimePolicy,
        exportPolicy,
        selectionEnvelope,
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
        volumeCompensation?: {
          enabled: boolean;
          stage?: "export" | "blend" | "both";
          dbPerExtraModel?: number;
        };
        pipelineConfig?: any[];
        workflow?: Record<string, any>;
        runtimePolicy?: Record<string, any>;
        exportPolicy?: Record<string, any>;
        selectionType?: CatalogSelectionType;
        selectionId?: string;
        selectionEnvelope?: Record<string, any>;
      },
    ) => {
      const previewDir = createPreviewDirForInput(inputFile);
      const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);
      const explicitSelectionEnvelope =
        selectionEnvelope ||
        (selectionType && selectionId
          ? {
              selectionType,
              selectionId,
              selection_type: selectionType,
              selection_id: selectionId,
            }
          : undefined);
      const executionContext = await resolveSelectionExecutionPlan(
        explicitSelectionEnvelope,
        {
          modelId,
          selectionType,
          selectionId,
          workflow,
          pipelineConfig,
          runtimePolicy,
          exportPolicy,
        },
      ).catch((error) => {
        log("Failed to resolve selection execution plan for preflight", error);
        return null;
      });
      const stagedInput = await ensureWavInput(inputFile, previewDir);
      const effectiveInputFile = stagedInput.effectiveInputFile;

      const result = await sendBackendCommand(
        "separation_preflight",
        {
          file_path: effectiveInputFile,
          model_id: effectiveModelId,
          selection_type:
            normalizeSelectionType(selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selection_type) ||
            null,
          selection_id:
            String(
              selectionId ||
                explicitSelectionEnvelope?.selectionId ||
                explicitSelectionEnvelope?.selection_id ||
                "",
            ).trim() || null,
          selection_envelope: explicitSelectionEnvelope || null,
          selectionType:
            normalizeSelectionType(selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selection_type) ||
            null,
          selectionId:
            String(
              selectionId ||
                explicitSelectionEnvelope?.selectionId ||
                explicitSelectionEnvelope?.selection_id ||
                "",
            ).trim() || null,
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
          pipeline_config: pipelineConfig,
          workflow,
          runtime_policy: runtimePolicy,
          export_policy: exportPolicy,
          execution_plan: executionContext || null,
          resolved_bundle:
            executionContext?.resolvedBundle ??
            executionContext?.resolved_bundle ??
            null,
        },
        30000,
      );

      return {
        ...result,
        sourceAudioProfile: stagedInput.sourceAudioProfile,
        stagingDecision: stagedInput.stagingDecision,
      };
    },
  );

  ipcMain.handle(
    "separate-audio",
    async (
      _event,
      {
        inputFile,
        modelId,
        outputDir,
        selectionType,
        selectionId,
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
        pipelineConfig,
        workflow,
        runtimePolicy,
        exportPolicy,
        selectionEnvelope,
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
        volumeCompensation?: {
          enabled: boolean;
          stage?: "export" | "blend" | "both";
          dbPerExtraModel?: number;
        };
        pipelineConfig?: any[];
        workflow?: Record<string, any>;
        runtimePolicy?: Record<string, any>;
        exportPolicy?: Record<string, any>;
        selectionType?: CatalogSelectionType;
        selectionId?: string;
        selectionEnvelope?: Record<string, any>;
      },
    ) => {
      const requestId = randomUUID().slice(0, 8);
      log("[separate-audio] request", {
        requestId,
        inputFile,
        modelId,
        splitFreq,
        hasEnsemble: Boolean(
          ensembleConfig &&
            Array.isArray(ensembleConfig.models) &&
            ensembleConfig.models.length > 0,
        ),
      });
      const process = ensureBackend();
      if (!process) {
        return Promise.reject(new Error("Backend not available."));
      }

      const previewDir = createPreviewDirForInput(inputFile);
      const effectiveModelId = resolveEffectiveModelId(modelId, ensembleConfig);
      const explicitSelectionEnvelope =
        selectionEnvelope ||
        (selectionType && selectionId
          ? {
              selectionType,
              selectionId,
              selection_type: selectionType,
              selection_id: selectionId,
            }
          : undefined);
      const executionContext = await resolveSelectionExecutionPlan(
        explicitSelectionEnvelope,
        {
          modelId,
          selectionType,
          selectionId,
          workflow,
          pipelineConfig,
          runtimePolicy,
          exportPolicy,
        },
      ).catch((error) => {
        log("Failed to resolve selection execution plan for separation", error);
        return null;
      });

      const stagedInput = await ensureWavInput(inputFile, previewDir);
      const effectiveInputFile = stagedInput.effectiveInputFile;
      log("[separate-audio] staging-ready", {
        requestId,
        effectiveModelId,
        previewDir,
        effectiveInputFile,
        sourceAudioProfile: stagedInput.sourceAudioProfile,
        stagingDecision: stagedInput.stagingDecision,
      });

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          cleanup();
          reject(new Error("Separation timeout (60 minutes)"));
        }, 60 * 60 * 1000);

        let myJobId: string | null = null;
        const payload = {
          file_path: effectiveInputFile,
          model_id: effectiveModelId,
          selection_type:
            normalizeSelectionType(selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selection_type) ||
            null,
          selection_id:
            String(
              selectionId ||
                explicitSelectionEnvelope?.selectionId ||
                explicitSelectionEnvelope?.selection_id ||
                "",
            ).trim() || null,
          selection_envelope: explicitSelectionEnvelope || null,
          selectionType:
            normalizeSelectionType(selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selectionType) ||
            normalizeSelectionType(explicitSelectionEnvelope?.selection_type) ||
            null,
          selectionId:
            String(
              selectionId ||
                explicitSelectionEnvelope?.selectionId ||
                explicitSelectionEnvelope?.selection_id ||
                "",
            ).trim() || null,
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
          pipeline_config: pipelineConfig,
          workflow,
          runtime_policy: runtimePolicy,
          export_policy: exportPolicy,
          execution_plan: executionContext || null,
          resolved_bundle:
            executionContext?.resolvedBundle ??
            executionContext?.resolved_bundle ??
            null,
        };

        const onDone = (msg: any) => {
          if (!myJobId || msg?.job_id !== myJobId) return;
          log("[separate-audio] complete", { requestId, jobId: myJobId });
          cleanup();
          const resolvedPlayback = resolvePlaybackStems(msg.output_files || {}, {
            sourceKind: "preview_cache",
            previewDir,
            savedDir: outputDir,
          });
          const normalizedOutputFiles =
            Object.keys(resolvedPlayback.stems).length > 0
              ? resolvedPlayback.stems
              : msg.output_files || {};
          resolve({
            success: true,
            outputFiles: normalizedOutputFiles,
            jobId: msg.job_id,
            outputDir: previewDir,
            sourceAudioProfile: stagedInput.sourceAudioProfile,
            stagingDecision: stagedInput.stagingDecision,
            playbackSourceKind: "preview_cache",
          });
        };

        const onError = (msg: any) => {
          if (!myJobId || msg?.job_id !== myJobId) return;
          log("[separate-audio] backend-error", {
            requestId,
            jobId: myJobId,
            error: msg?.error || "Separation failed",
          });
          cleanup();
          reject(new Error(msg.error || "Separation failed"));
        };

        const onCancelled = (msg: any) => {
          if (!myJobId || msg?.job_id !== myJobId) return;
          log("[separate-audio] cancelled", { requestId, jobId: myJobId });
          cleanup();
          reject(new Error("Separation cancelled"));
        };

        const unsubComplete = subscribeBackendEvent("separation_complete", onDone);
        const unsubError = subscribeBackendEvent("separation_error", onError);
        const unsubCancelled = subscribeBackendEvent(
          "separation_cancelled",
          onCancelled,
        );

        const cleanup = () => {
          clearTimeout(timeout);
          unsubComplete();
          unsubError();
          unsubCancelled();
        };

        sendBackendCommand("run_selection_job", payload)
          .then((response) => {
            if (!response?.job_id) {
              cleanup();
              reject(new Error("Backend did not return a job ID"));
              return;
            }
            myJobId = String(response.job_id);
            log("[separate-audio] job-started", {
              requestId,
              jobId: myJobId,
              effectiveModelId,
            });
            activeSelectionJobIds.add(myJobId);
            modelInfoByJobId.set(myJobId, {
              requestedModelId: String(modelId || ""),
              effectiveModelId: String(effectiveModelId || ""),
              inputFile: String(inputFile || ""),
              finalOutputDir: String(outputDir || ""),
              previewDir,
              sourceAudioProfile: stagedInput.sourceAudioProfile,
              stagingDecision: stagedInput.stagingDecision,
            });
            lastProgressByJobId.set(myJobId, 0);
            const mainWindow = getMainWindow();
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
          })
          .catch((err) => {
            log("[separate-audio] dispatch-failed", {
              requestId,
              error: err?.message || String(err),
            });
            cleanup();
            reject(err);
          });
      });
    },
  );
}
