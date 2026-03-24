import type { ModelSelectionEnvelope } from "./modelCatalog"
import type { PostProcessingStep } from '../presets'

export type VolumeCompensationStage = 'export' | 'blend' | 'both'
export type WorkflowKind = 'single' | 'ensemble' | 'pipeline'
export type QualityProfile =
    | 'fast'
    | 'balanced'
    | 'maximum_quality'
    | 'long_file_safe'
export type WorkflowSurface =
    | 'single'
    | 'ensemble'
    | 'workflow'
    | 'restoration'
    | 'special_stem'
export type WorkflowModelRole =
    | 'primary'
    | 'ensemble_partner'
    | 'phase_reference'
    | 'post_process'
    | 'special_stem'
    | 'fullness_source'
    | 'bleedless_reference'
    | 'karaoke_base'
    | 'lead_back_refiner'
    | 'harmony_recovery'
    | 'restoration_step'
    | 'low_band_source'
    | 'high_band_source'
    | 'secondary_cleanup_pass'
export type WorkflowBlendAlgorithm =
    | 'average'
    | 'avg_wave'
    | 'max_spec'
    | 'min_spec'
    | 'phase_fix'
    | 'frequency_split'

export interface WorkflowPhaseFixParams {
    lowHz: number
    highHz: number
    highFreqWeight: number
}

export interface WorkflowModelRef {
    model_id: string
    weight?: number
    role?: WorkflowModelRole | string
    stage?: string
    required?: boolean
}

export interface WorkflowStep {
    id?: string
    name?: string
    action?: string
    model_id?: string
    source_model?: string
    input_source?: string
    output?: string | string[]
    apply_to?: string
    role?: WorkflowModelRole | string
    weight?: number
    optional?: boolean
    params?: Record<string, unknown>
}

export interface WorkflowBlendConfig {
    algorithm?: WorkflowBlendAlgorithm
    stemAlgorithms?: {
        vocals?: 'average' | 'max_spec' | 'min_spec'
        instrumental?: 'average' | 'max_spec' | 'min_spec'
    }
    splitFreq?: number
    splitFade?: 'linear' | 'equal_power' | 'soft'
    phaseFixEnabled?: boolean
    phaseFixParams?: WorkflowPhaseFixParams
}

export interface WorkflowRuntimePolicy {
    required?: string[]
    fallbacks?: string[]
    allowManualModels?: boolean
    preferredRuntime?: string
}

export interface WorkflowExportPolicy {
    stems?: string[]
    outputFormat?: 'wav' | 'mp3' | 'flac'
    preservePreviewCache?: boolean
    intermediateOutputs?: string[]
}

export interface WorkflowFallbackPolicy {
    mode?: 'none' | 'runtime_fallback' | 'workflow_fallback' | 'profile_fallback'
    reason?: string
    runtimeOrder?: string[]
    fallbackWorkflowId?: string
    fallbackOperatingProfile?: string
}

export interface SeparationWorkflow {
    version: 1
    id?: string
    name?: string
    kind: WorkflowKind
    selectionEnvelope?: ModelSelectionEnvelope
    surface?: WorkflowSurface
    family?: string
    description?: string
    stems?: string[]
    models?: WorkflowModelRef[]
    steps?: WorkflowStep[]
    blend?: WorkflowBlendConfig
    postprocess?: PostProcessingStep[]
    runtimePolicy?: WorkflowRuntimePolicy
    exportPolicy?: WorkflowExportPolicy
    intermediateOutputs?: string[]
    fallbackPolicy?: WorkflowFallbackPolicy
    operatingProfile?: string
}

export interface VolumeCompensation {
    enabled: boolean
    stage?: VolumeCompensationStage
    dbPerExtraModel?: number
}

export interface SeparationConfig {
    mode: 'simple' | 'advanced'
    presetId?: string
    workflowId?: string
    modelId?: string
    selectionEnvelope?: ModelSelectionEnvelope
    device: string
    outputFormat: 'wav' | 'mp3' | 'flac'
    exportMixes?: string[]
    stems?: string[]
    invert?: boolean
    normalize?: boolean
    bitDepth?: string
    volumeCompensation?: VolumeCompensation
    advancedParams?: {
        overlap?: number
        segmentSize?: number
        batchSize?: number
        shifts?: number
        tta?: boolean
        bitrate?: string
    }
    qualityProfile?: QualityProfile
    ensembleConfig?: {
        models: { model_id: string; weight?: number }[]
        algorithm: 'average' | 'avg_wave' | 'max_spec' | 'min_spec' | 'phase_fix' | 'frequency_split'
        stemAlgorithms?: {
            vocals?: 'average' | 'max_spec' | 'min_spec'
            instrumental?: 'average' | 'max_spec' | 'min_spec'
        }
        phaseFixEnabled?: boolean
        phaseFixParams?: {
            lowHz: number
            highHz: number
            highFreqWeight: number
        }
    }
    splitFreq?: number
    postProcessingSteps?: PostProcessingStep[]
    workflow?: SeparationWorkflow
    runtimePolicy?: WorkflowRuntimePolicy
    exportPolicy?: WorkflowExportPolicy
}
