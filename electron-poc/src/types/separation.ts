import type { PostProcessingStep } from '../presets'

export type VolumeCompensationStage = 'export' | 'blend' | 'both'

export interface VolumeCompensation {
    enabled: boolean
    stage?: VolumeCompensationStage
    dbPerExtraModel?: number
}

export interface SeparationConfig {
    mode: 'simple' | 'advanced'
    presetId?: string
    modelId?: string
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
        shifts?: number
        tta?: boolean
        bitrate?: string
    }
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
}
