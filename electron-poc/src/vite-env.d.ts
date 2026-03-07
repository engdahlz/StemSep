/// <reference types="vite/client" />

type VolumeCompensation = import('./types/separation').VolumeCompensation
type Recipe = import('./types/recipes').Recipe
type SourceAudioProfile = import('./types/media').SourceAudioProfile
type StagingDecision = import('./types/media').StagingDecision
type SeparationProgressEvent = import('./types/media').SeparationProgressEvent
type ExportProgressEvent = import('./types/media').ExportProgressEvent

type MissingAudioCode = 'MISSING_CACHE_FILE' | 'STALE_SESSION' | 'MISSING_SOURCE_FILE'

type ReadAudioFileResult =
    | { success: true; data: string; mimeType: string; resolvedPath?: string }
    | { success: false; error: string; code?: MissingAudioCode; hint?: string }

type ExportFilesResult =
    | {
        success: true
        exported: Record<string, string>
        requestId?: string
        outputFiles?: Record<string, string>
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }
    | { success: false; error: string; code?: MissingAudioCode; hint?: string; requestId?: string }

interface SystemRuntimeInfo {
    fetchedAt: string
    cache?: {
        ttlMs?: number
        gpuSource?: 'cache' | 'fresh' | string
        runtimeFingerprintSource?: 'cache' | 'fresh' | 'error' | string
    }
    gpu?: any
    runtimeFingerprint?: {
        version?: string
        platform?: string
        torch?: {
            version?: string
            cuda_available?: boolean
            cuda_device_count?: number
            cuda_device_name_0?: string
        }
        neuralop?: {
            version?: string
            fno1d_import_ok?: boolean
            fno1d_import_error?: string
            fno_import_ok?: boolean
            fno_import_error?: string
        }
    } | null
    runtimeFingerprintError?: string | null
    previewCachePolicy?: {
        baseDir?: string
        keepLast?: number
        maxAgeDays?: number
        ephemeral?: boolean
    }
}

interface ElectronAPI {
    selectOutputDirectory: () => Promise<string | null>
    openAudioFileDialog: () => Promise<string[] | null>
    openModelFileDialog: () => Promise<string[] | null>
    scanDirectory: (folderPath: string) => Promise<string[]>
    separationPreflight: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        stems?: string[],
        device?: string,
        overlap?: number,
        segmentSize?: number,
        shifts?: number,
        outputFormat?: string,
        bitrate?: string,
        tta?: boolean,
        ensembleConfig?: any,
        ensembleAlgorithm?: string,
        invert?: boolean,
        splitFreq?: number,
        phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number },
        postProcessingSteps?: any[],
        volumeCompensation?: VolumeCompensation
    ) => Promise<any & { sourceAudioProfile?: SourceAudioProfile; stagingDecision?: StagingDecision }>
    separateAudio: (
        inputFile: string,
        modelId: string,
        outputDir: string,
        stems?: string[],
        device?: string,
        overlap?: number,
        segmentSize?: number,
        shifts?: number,
        outputFormat?: string,
        bitrate?: string,
        tta?: boolean,
        ensembleConfig?: any,
        ensembleAlgorithm?: string,
        invert?: boolean,
        splitFreq?: number,
        phaseParams?: { enabled: boolean; lowHz: number; highHz: number; highFreqWeight: number },
        postProcessingSteps?: any[],
        volumeCompensation?: VolumeCompensation
    ) => Promise<{
        success: boolean
        outputFiles: Record<string, string>
        jobId?: string
        error?: string
        outputDir?: string
        playbackSourceKind?: string
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }>
    cancelSeparation: (jobId: string) => Promise<any>
    saveJobOutput: (jobId: string) => Promise<{
        success: boolean
        outputFiles?: Record<string, string>
        error?: string
        sourceAudioProfile?: SourceAudioProfile
        stagingDecision?: StagingDecision
      }>
    discardJobOutput: (jobId: string) => Promise<{ success: boolean; error?: string }>
    pauseQueue: () => Promise<void>
    resumeQueue: () => Promise<void>
    reorderQueue: (jobIds: string[]) => Promise<void>
    exportOutput: (jobId: string, exportPath: string, format: string, bitrate: string, requestId?: string) => Promise<ExportFilesResult>
    exportFiles: (sourceFiles: Record<string, string>, exportPath: string, format: string, bitrate: string, requestId?: string) => Promise<ExportFilesResult>
    onSeparationProgress: (callback: (data: { progress: number; message: string; jobId?: string; meta?: Record<string, any> }) => void) => () => void
    onSeparationEvent: (callback: (data: SeparationProgressEvent) => void) => () => void
    onSeparationStarted: (callback: (data: { jobId: string }) => void) => () => void
    onSeparationComplete: (callback: (data: { outputFiles: Record<string, string> }) => void) => () => void
    onSeparationError: (callback: (data: { error: string }) => void) => () => void
    onExportProgress: (callback: (data: ExportProgressEvent) => void) => () => void

    // Model operations
    getModels: () => Promise<any[]>
    getModelTech: (modelId: string) => Promise<any>
    getRecipes: () => Promise<Recipe[]>
    qualityBaselineCreate: (payload: Record<string, any>) => Promise<any>
    qualityCompare: (payload: Record<string, any>) => Promise<any>
    downloadModel: (modelId: string) => Promise<boolean>
    pauseDownload: (modelId: string) => Promise<any>
    resumeDownload: (modelId: string) => Promise<any>
    removeModel: (modelId: string) => Promise<any>
    importCustomModel: (filePath: string, modelName: string, architecture?: string) => Promise<any>
    openFolder: (folderPath: string) => Promise<void>
    readAudioFile: (filePath: string) => Promise<ReadAudioFileResult>

    // YouTube URL -> local temp audio file
    resolveYouTubeUrl: (url: string) => Promise<
        | { success: true; file_path: string; title: string; source_url?: string }
        | { success: false; code?: string; error: string; hint?: string }
    >
    onYouTubeProgress: (callback: (data: { status: string; percent?: string; speed?: string; eta?: string; error?: string }) => void) => () => void
    getGpuDevices: () => Promise<any>
    getSystemRuntimeInfo: () => Promise<SystemRuntimeInfo>
    getWorkflows: () => Promise<{ workflows: Record<string, any> }>
    checkPresetModels: (presetMappings: Record<string, string>) => Promise<Record<string, boolean>>
    onDownloadProgress: (callback: (data: { modelId: string; progress: number }) => void) => () => void
    onDownloadComplete: (callback: (data: { modelId: string }) => void) => () => void
    onDownloadError: (callback: (data: { modelId: string; error: string }) => void) => () => void
    onDownloadPaused: (callback: (data: { modelId: string }) => void) => () => void
    openDirectoryDialog: () => Promise<{ canceled: boolean; filePaths: string[] }>
    onBackendError: (callback: (data: { error: string }) => void) => () => void
    onBridgeReady: (callback: (data: { capabilities: string[]; modelsCount: number; recipesCount: number }) => void) => () => void
    onQualityProgress: (callback: (data: any) => void) => () => void
    onQualityComplete: (callback: (data: any) => void) => () => void
    saveQueue: (queue: any[]) => Promise<{ success: boolean; error?: string }>
    loadQueue: () => Promise<any[] | null>
    startWatchMode: (path: string) => Promise<boolean>
    stopWatchMode: () => Promise<boolean>
    onWatchFileDetected: (callback: (path: string) => void) => () => void

    // App Config
    saveAppConfig: (config: Record<string, any>) => Promise<boolean>
    getAppConfig: () => Promise<Record<string, any>>

    // Hugging Face auth (optional)
    setHuggingFaceToken: (token: string) => Promise<{ success: boolean; error?: string }>
    clearHuggingFaceToken: () => Promise<{ success: boolean; error?: string }>
    getHuggingFaceAuthStatus: () => Promise<{ configured: boolean }>

    // External links
    openExternalUrl: (url: string) => Promise<boolean>
    resolveMediaUrl?: (filePath: string) => string
}

interface Window {
    electronAPI: ElectronAPI
}
