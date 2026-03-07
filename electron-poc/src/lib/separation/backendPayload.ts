import type { SeparationConfig } from '@/types/separation'
import type { SeparationPlan } from '@/lib/separation/resolveSeparationPlan'
import type { VolumeCompensation } from '@/types/separation'

export interface SeparationBackendPayload {
  inputFile: string
  modelId: string
  outputDir: string
  stems?: string[]
  device?: string
  overlap?: number
  segmentSize?: number
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

  return {
    inputFile,
    modelId: plan.effectiveModelId,
    outputDir,
    stems: plan.effectiveStems,
    device:
      config.device && config.device !== 'auto' ? config.device : undefined,
    overlap: toBackendOverlap(config.advancedParams?.overlap),
    segmentSize: toBackendSegmentSize(config.advancedParams?.segmentSize),
    shifts: config.advancedParams?.shifts,
    outputFormat: config.outputFormat || 'wav',
    bitrate: config.advancedParams?.bitrate,
    tta: config.advancedParams?.tta,
    ensembleConfig: plan.effectiveEnsembleConfig,
    ensembleAlgorithm: plan.effectiveEnsembleConfig?.algorithm,
    invert: config.invert,
    splitFreq: config.splitFreq,
    phaseParams: plan.effectiveGlobalPhaseParams,
    postProcessingSteps: plan.effectivePostProcessingSteps,
    volumeCompensation: config.volumeCompensation,
  }
}

export function executeSeparationPreflight(
  api: Window['electronAPI'],
  payload: SeparationBackendPayload,
) {
  return api.separationPreflight(
    payload.inputFile,
    payload.modelId,
    payload.outputDir,
    payload.stems,
    payload.device,
    payload.overlap,
    payload.segmentSize,
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
  )
}

export function executeSeparation(
  api: Window['electronAPI'],
  payload: SeparationBackendPayload,
) {
  return api.separateAudio(
    payload.inputFile,
    payload.modelId,
    payload.outputDir,
    payload.stems,
    payload.device,
    payload.overlap,
    payload.segmentSize,
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
  )
}
