import type { SeparationConfig, SeparationWorkflow } from '@/types/separation'
import type { SeparationPlan } from '@/lib/separation/resolveSeparationPlan'
import type { VolumeCompensation } from '@/types/separation'
import type { CatalogSelectionType, ModelSelectionEnvelope } from "@/types/modelCatalog"
import type { SourceAudioProfile, StagingDecision } from "@/types/media"

export interface SeparationBackendPayload {
  inputFile: string
  modelId: string
  outputDir: string
  selectionType?: CatalogSelectionType
  selectionId?: string
  stems?: string[]
  device?: string
  overlap?: number
  segmentSize?: number
  batchSize?: number
  shifts?: number
  outputFormat: string
  bitrate?: string
  tta?: boolean
  ensembleConfig?: SeparationConfig['ensembleConfig']
  ensembleAlgorithm?: string
  invert?: boolean
  splitFreq?: number
  phaseParams?: SeparationConfig['ensembleConfig'] extends infer _T
    ? { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number }
    : never
  postProcessingSteps?: SeparationConfig['postProcessingSteps']
  volumeCompensation?: VolumeCompensation
  pipelineConfig?: SeparationWorkflow['steps']
  workflow?: SeparationWorkflow
  runtimePolicy?: SeparationConfig['runtimePolicy']
  exportPolicy?: SeparationConfig['exportPolicy']
  selectionEnvelope?: ModelSelectionEnvelope
}

type SelectionJobSnapshotLike = {
  job_id?: string
  jobId?: string
  status?: string
  requested_at?: string
  requestedAt?: string
  started_at?: string
  startedAt?: string
  finished_at?: string
  finishedAt?: string
  progress?: number
  message?: string
  error?: string
  model_id?: string
  modelId?: string
  selection_type?: CatalogSelectionType
  selectionType?: CatalogSelectionType
  selection_id?: string
  selectionId?: string
  file_path?: string
  filePath?: string
  output_dir?: string
  outputDir?: string
  output_files?: Record<string, string>
  outputFiles?: Record<string, string>
  sourceAudioProfile?: SourceAudioProfile
  source_audio_profile?: SourceAudioProfile
  stagingDecision?: StagingDecision
  staging_decision?: StagingDecision
}

export function toBackendOverlap(value: unknown): number | undefined {
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return undefined
  if (n <= 0) return undefined
  if (n < 1) return Math.min(0.99, Math.max(0, n))
  const ratio = (n - 1) / n
  return Math.min(0.99, Math.max(0, ratio))
}

export function toBackendSegmentSize(value: unknown): number | undefined {
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return undefined
  if (n <= 0) return undefined
  return Math.trunc(n)
}

export function buildSeparationBackendPayload(args: {
  inputFile: string
  outputDir: string
  config: SeparationConfig
  plan: SeparationPlan
}): SeparationBackendPayload {
  const { inputFile, outputDir, config, plan } = args
  const pipelineConfig =
    plan.effectiveWorkflow?.kind === 'pipeline' &&
    Array.isArray(plan.effectiveWorkflow.steps) &&
    plan.effectiveWorkflow.steps.length > 0
      ? plan.effectiveWorkflow.steps
      : undefined

  return {
    inputFile,
    modelId: plan.effectiveModelId,
    outputDir,
    selectionType:
      config.selectionEnvelope?.selectionType ||
      plan.effectiveWorkflow?.selectionEnvelope?.selectionType ||
      (plan.effectiveWorkflow ? "workflow" : "model"),
    selectionId:
      config.selectionEnvelope?.selectionId ||
      plan.effectiveWorkflow?.selectionEnvelope?.selectionId ||
      plan.effectiveWorkflow?.id ||
      plan.effectiveModelId,
    stems: plan.effectiveStems,
    device:
      config.device && config.device !== 'auto' ? config.device : undefined,
    overlap: toBackendOverlap(
      plan.effectiveAdvancedParams?.overlap ?? config.advancedParams?.overlap,
    ),
    segmentSize: toBackendSegmentSize(
      plan.effectiveAdvancedParams?.segmentSize ??
        config.advancedParams?.segmentSize,
    ),
    batchSize:
      plan.effectiveAdvancedParams?.batchSize ?? config.advancedParams?.batchSize,
    shifts: plan.effectiveAdvancedParams?.shifts ?? config.advancedParams?.shifts,
    outputFormat: config.outputFormat || 'wav',
    bitrate: config.advancedParams?.bitrate,
    tta: plan.effectiveAdvancedParams?.tta ?? config.advancedParams?.tta,
    ensembleConfig: plan.effectiveEnsembleConfig,
    ensembleAlgorithm: plan.effectiveEnsembleConfig?.algorithm,
    invert: config.invert,
    splitFreq: config.splitFreq,
    phaseParams: plan.effectiveGlobalPhaseParams,
    postProcessingSteps: plan.effectivePostProcessingSteps,
    volumeCompensation: config.volumeCompensation,
    pipelineConfig,
    workflow: plan.effectiveWorkflow,
    runtimePolicy: config.runtimePolicy,
    exportPolicy: config.exportPolicy,
    selectionEnvelope: config.selectionEnvelope ?? plan.effectiveWorkflow?.selectionEnvelope,
  }
}

function resolveTransportSelection(
  payload: SeparationBackendPayload,
): {
  selectionType?: CatalogSelectionType
  selectionId?: string
  selectionEnvelope?: ModelSelectionEnvelope
} {
  const derivedSelectionType =
    payload.selectionType ||
    payload.selectionEnvelope?.selectionType ||
    payload.workflow?.selectionEnvelope?.selectionType ||
    (payload.workflow ? "workflow" : undefined)
  const derivedSelectionId =
    payload.selectionId ||
    payload.selectionEnvelope?.selectionId ||
    payload.workflow?.selectionEnvelope?.selectionId ||
    payload.workflow?.id

  return {
    selectionType: derivedSelectionType,
    selectionId: derivedSelectionId,
    selectionEnvelope:
      payload.selectionEnvelope ||
      payload.workflow?.selectionEnvelope ||
      (derivedSelectionType && derivedSelectionId
        ? {
            selectionType: derivedSelectionType,
            selectionId: derivedSelectionId,
          }
        : undefined),
  }
}

export function executeSeparationPreflight(
  api: Window['electronAPI'],
  payload: SeparationBackendPayload,
) {
  const selection = resolveTransportSelection(payload)
  return api.separationPreflight(
    payload.inputFile,
    payload.modelId,
    payload.outputDir,
    selection.selectionType,
    selection.selectionId,
    payload.stems,
    payload.device,
    payload.overlap,
    payload.segmentSize,
    payload.batchSize,
    payload.shifts,
    payload.outputFormat,
    payload.bitrate,
    payload.tta,
    payload.ensembleConfig,
    payload.ensembleAlgorithm,
    payload.invert,
    payload.splitFreq,
    payload.phaseParams,
    payload.postProcessingSteps,
    payload.volumeCompensation,
    payload.pipelineConfig,
    payload.workflow,
    payload.runtimePolicy,
    payload.exportPolicy,
    selection.selectionEnvelope,
  )
}

function normalizeSelectionJobSnapshot(snapshot: SelectionJobSnapshotLike | null | undefined) {
  if (!snapshot) return null
  const outputFiles = snapshot.outputFiles || snapshot.output_files || {}
  return {
    jobId: snapshot.jobId || snapshot.job_id || "",
    status: snapshot.status || "",
    requestedAt: snapshot.requestedAt || snapshot.requested_at,
    startedAt: snapshot.startedAt || snapshot.started_at,
    finishedAt: snapshot.finishedAt || snapshot.finished_at,
    progress: typeof snapshot.progress === "number" ? snapshot.progress : undefined,
    message: snapshot.message,
    error: snapshot.error,
    modelId: snapshot.modelId || snapshot.model_id,
    selectionType: snapshot.selectionType || snapshot.selection_type,
    selectionId: snapshot.selectionId || snapshot.selection_id,
    filePath: snapshot.filePath || snapshot.file_path,
    outputDir: snapshot.outputDir || snapshot.output_dir,
    outputFiles,
    sourceAudioProfile: snapshot.sourceAudioProfile || snapshot.source_audio_profile,
    stagingDecision: snapshot.stagingDecision || snapshot.staging_decision,
  }
}

function isFinalSelectionJobStatus(status?: string) {
  return ["completed", "failed", "cancelled", "discarded"].includes(
    String(status || "").toLowerCase(),
  )
}

async function waitForSelectionJobCompletion(
  api: Window["electronAPI"],
  jobId: string,
  initialSnapshot?: SelectionJobSnapshotLike | null,
) {
  const timeoutMs = 24 * 60 * 60 * 1000
  const startedAt = Date.now()
  let snapshot = normalizeSelectionJobSnapshot(initialSnapshot)

  while (Date.now() - startedAt < timeoutMs) {
    if (
      snapshot?.status &&
      isFinalSelectionJobStatus(snapshot.status) &&
      Object.keys(snapshot.outputFiles || {}).length > 0
    ) {
      return snapshot
    }

    const next = await api.getSelectionJob(jobId)
    snapshot = normalizeSelectionJobSnapshot(next)
    if (!snapshot) {
      throw new Error("Selection job not found")
    }

    if (
      isFinalSelectionJobStatus(snapshot.status) &&
      Object.keys(snapshot.outputFiles || {}).length > 0
    ) {
      return snapshot
    }

    if (isFinalSelectionJobStatus(snapshot.status) && snapshot.status !== "completed") {
      throw new Error(snapshot.error || snapshot.message || `Selection job ${snapshot.status}`)
    }

    await new Promise((resolve) => window.setTimeout(resolve, 1000))
  }

  throw new Error("Timed out waiting for selection job completion")
}

export function executeSeparation(
  api: Window['electronAPI'],
  payload: SeparationBackendPayload,
) {
  const selection = resolveTransportSelection(payload)
  return api
    .runSelectionJob({
      inputFile: payload.inputFile,
      modelId: payload.modelId,
      outputDir: payload.outputDir,
      selectionType: selection.selectionType,
      selectionId: selection.selectionId,
      stems: payload.stems,
      device: payload.device,
      overlap: payload.overlap,
      segmentSize: payload.segmentSize,
      batchSize: payload.batchSize,
      shifts: payload.shifts,
      outputFormat: payload.outputFormat,
      bitrate: payload.bitrate,
      tta: payload.tta,
      ensembleConfig: payload.ensembleConfig,
      ensembleAlgorithm: payload.ensembleAlgorithm,
      invert: payload.invert,
      splitFreq: payload.splitFreq,
      phaseParams: payload.phaseParams,
      postProcessingSteps: payload.postProcessingSteps,
      volumeCompensation: payload.volumeCompensation,
      pipelineConfig: payload.pipelineConfig,
      workflow: payload.workflow,
      runtimePolicy: payload.runtimePolicy,
      exportPolicy: payload.exportPolicy,
      selectionEnvelope: selection.selectionEnvelope,
    })
    .then(async (response) => {
      const initialSnapshot = normalizeSelectionJobSnapshot(
        response?.job || response?.selection_job || response,
      )
      const jobId =
        initialSnapshot?.jobId ||
        response?.job_id ||
        response?.jobId ||
        response?.selection_job_id ||
        response?.selectionJobId

      if (!jobId) {
        throw new Error("Selection job did not return a job id")
      }

      const finalSnapshot =
        initialSnapshot &&
        isFinalSelectionJobStatus(initialSnapshot.status) &&
        Object.keys(initialSnapshot.outputFiles || {}).length > 0
          ? initialSnapshot
          : await waitForSelectionJobCompletion(api, jobId, initialSnapshot)

      return {
        success: true,
        jobId,
        outputFiles: finalSnapshot.outputFiles || {},
        outputDir: finalSnapshot.outputDir,
        sourceAudioProfile: finalSnapshot.sourceAudioProfile,
        stagingDecision: finalSnapshot.stagingDecision,
        playbackSourceKind: "selection_job",
      }
    })
}
