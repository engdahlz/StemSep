import type { IpcMain } from "electron";

type CatalogSelectionType = "model" | "recipe" | "workflow";

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
  createPreviewDirForInput,
  resolveEffectiveModelId,
  resolveSelectionExecutionPlan,
  ensureWavInput,
  normalizeSelectionType,
  sendBackendCommand,
}: {
  ipcMain: IpcMain;
  log: LogFn;
  createPreviewDirForInput: (inputFile: string) => string;
  resolveEffectiveModelId: (modelId: string, ensembleConfig?: any) => string;
  resolveSelectionExecutionPlan: ResolveSelectionExecutionPlan;
  ensureWavInput: EnsureWavInput;
  normalizeSelectionType: NormalizeSelectionType;
  sendBackendCommand: SendBackendCommand;
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
        batchSize,
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
        batchSize?: number;
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
          batch_size: batchSize,
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
}
